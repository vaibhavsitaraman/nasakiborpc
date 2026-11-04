package jp.jaxa.iss.kibo.rpc.sampleapk;
import org.opencv.aruco.DetectorParameters;
import org.opencv.aruco.Dictionary;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.aruco.Aruco;

import java.util.*;

import gov.nasa.arc.astrobee.types.Quaternion;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

public class ImageProcessor {

    // Camera intrinsics and distortion coefficients
    private Mat cameraMatrix;
    private MatOfDouble distCoeffs;

    // ArUco detector
    private Dictionary arucoDict;

    public ImageProcessor() {
        // Initialize camera matrix
        cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        cameraMatrix.put(0, 0, new float[]{
                523.105750f, 0.000000f, 635.434258f,
                0.000000f, 534.765913f, 500.335102f,
                0.000000f, 0.000000f, 1.000000f
        });

        // Initialize distortion coefficients
        distCoeffs = new MatOfDouble(new double[]{-0.164787, 0.020375, -0.001572, -0.000369, 0.000000});

        // ArUco detector setup
        arucoDict = Aruco.getPredefinedDictionary(Aruco.DICT_5X5_250);
    }

    /**
     * Step 1: Preprocessing - undistort image
     */
    public Mat preprocessImage(Mat image) {
        Mat undistorted = new Mat();
        Calib3d.undistort(image, undistorted, cameraMatrix, distCoeffs);
        return undistorted;
    }

    /**
     * Detect ArUco marker and return its corners.
     * The returned MatOfPoint2f contains the 4 corners of the first detected marker.
     */
    public MatOfPoint2f detectArucoMarker(Mat image, int locationIndex) {
        Mat gray = new Mat();
        if (image.channels() == 1) {
            gray = image.clone(); // Already grayscale
        } else {
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY); // Convert to grayscale
        }

        // Create detection parameters optimized for small markers and poor lighting
        DetectorParameters detectorParams = DetectorParameters.create();

        // Most important: Allow very small markers
        detectorParams.set_minMarkerPerimeterRate(0.003);    // Allow very small markers (default: 0.03)

        // Essential: Corner refinement for accuracy
        detectorParams.set_cornerRefinementMethod(Aruco.CORNER_REFINE_SUBPIX);

        // Important: Better adaptive thresholding for poor lighting
        detectorParams.set_adaptiveThreshConstant(7);       // Lower constant for poor lighting
        detectorParams.set_adaptiveThreshWinSizeMin(3);     // Smaller window for small markers

        List<Mat> cornersList = new ArrayList<>();
        Mat ids = new Mat();
        Aruco.detectMarkers(gray, arucoDict, cornersList, ids, detectorParams);
        MatOfPoint2f arucoCorners = new MatOfPoint2f();

        if (cornersList.isEmpty()) {
            return arucoCorners;
        }

        // Handle location index for multiple markers
        if ((locationIndex == 1 || locationIndex == 2) && cornersList.size() >= 2) {
            System.out.println("oopsies");
            System.out.println(cornersList);

            // Calculate center X coordinates for each marker to determine left/right position
            double[] centerXCoords = new double[cornersList.size()];
            for (int i = 0; i < cornersList.size(); i++) {
                Mat corners = cornersList.get(i);
                // Calculate center X by averaging the 4 corner X coordinates
                double centerX = 0;
                for (int j = 0; j < 4; j++) {
                    double[] point = corners.get(0, j);
                    centerX += point[0]; // X coordinate
                }
                centerXCoords[i] = centerX / 4.0;
            }

            // Find the leftmost (index 1) or rightmost (index 2) marker
            int selectedIndex = 0;
            if (locationIndex == 1) {
                // Find leftmost marker (minimum center X)
                double minX = centerXCoords[0];
                for (int i = 1; i < centerXCoords.length; i++) {
                    if (centerXCoords[i] < minX) {
                        minX = centerXCoords[i];
                        selectedIndex = i;
                    }
                }
            } else { // locationIndex == 2
                // Find rightmost marker (maximum center X)
                double maxX = centerXCoords[0];
                for (int i = 1; i < centerXCoords.length; i++) {
                    if (centerXCoords[i] > maxX) {
                        maxX = centerXCoords[i];
                        selectedIndex = i;
                    }
                }
            }

            cornersList.get(selectedIndex).convertTo(arucoCorners, CvType.CV_32FC2);
        } else {
            // Default behavior: return first marker
            cornersList.get(0).convertTo(arucoCorners, CvType.CV_32FC2);
        }

        gray.release(); // Release intermediate Mat
        ids.release();
        for(Mat m : cornersList) m.release(); // Release mats in list

        return arucoCorners; // This is a MatOfPoint2f representing the 4 corners
    }

    public static class RotatedImageInfo {
        public Mat rotatedImage;
        public MatOfPoint2f rotatedArucoCorners;

        public RotatedImageInfo(Mat rotatedImage, MatOfPoint2f rotatedArucoCorners) {
            this.rotatedImage = rotatedImage;
            this.rotatedArucoCorners = rotatedArucoCorners;
        }
    }

    public RotatedImageInfo rotateImageToCorrectOrientation(Mat image, MatOfPoint2f arucoCornersMat) {
        Point[] arucoCorners = arucoCornersMat.toArray(); // Easier to work with Point array

        // ArUco corners are ordered: top-left, top-right, bottom-right, bottom-left
        Point topLeft = arucoCorners[0];
        Point topRight = arucoCorners[1];

        // Get the top edge vector (from top-left to top-right)
        Point topVector = new Point(topRight.x - topLeft.x, topRight.y - topLeft.y);

        // Calculate angle of the top edge relative to horizontal
        double angleRad = Math.atan2(topVector.y, topVector.x);
        double angleDeg = Math.toDegrees(angleRad);

        while (angleDeg > 180) angleDeg -= 360;
        while (angleDeg < -180) angleDeg += 360;

        // Determine the rotation needed to make the marker upright
        // We want the top edge to be horizontal (0 degrees)
        double rotationAngle = angleDeg; // Python code uses angle_deg directly

        System.out.printf("ArUco top edge angle: %.1f°, applying rotation: %.1f°%n", angleDeg, rotationAngle);

        int h = image.rows();
        int w = image.cols();
        Point center = new Point(w / 2.0, h / 2.0);

        Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, rotationAngle, 1.0);

        // Calculate new image size to ensure no content is lost
        double cosAbs = Math.abs(rotationMatrix.get(0, 0)[0]);
        double sinAbs = Math.abs(rotationMatrix.get(0, 1)[0]);
        int newW = (int) (h * sinAbs + w * cosAbs);
        int newH = (int) (h * cosAbs + w * sinAbs);

        newW = Math.max(newW, w);
        newH = Math.max(newH, h);

        // Adjust rotation matrix for new center
        rotationMatrix.put(0, 2, rotationMatrix.get(0, 2)[0] + (newW - w) / 2.0);
        rotationMatrix.put(1, 2, rotationMatrix.get(1, 2)[0] + (newH - h) / 2.0);

        Mat rotated = new Mat();
        // Using BORDER_CONSTANT with black, as BORDER_TRANSPARENT might not be directly available/behave identically.
        Imgproc.warpAffine(image, rotated, rotationMatrix, new Size(newW, newH), Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

        // Manually transform ArUco corners to preserve proper 2D point structure
        Point[] rotatedArucoCornersArray = new Point[arucoCorners.length];
        for (int i = 0; i < arucoCorners.length; i++) {
            Point originalPoint = arucoCorners[i];

            // Apply 2D transformation matrix manually: [x', y'] = [x, y, 1] * rotationMatrix^T
            double newX = rotationMatrix.get(0, 0)[0] * originalPoint.x +
                    rotationMatrix.get(0, 1)[0] * originalPoint.y +
                    rotationMatrix.get(0, 2)[0];
            double newY = rotationMatrix.get(1, 0)[0] * originalPoint.x +
                    rotationMatrix.get(1, 1)[0] * originalPoint.y +
                    rotationMatrix.get(1, 2)[0];

            rotatedArucoCornersArray[i] = new Point(newX, newY);
        }

        // Create properly structured MatOfPoint2f from the transformed points
        MatOfPoint2f rotatedArucoCornersMat = new MatOfPoint2f(rotatedArucoCornersArray);

        Point newTopVector = new Point(rotatedArucoCornersArray[1].x - rotatedArucoCornersArray[0].x,
                rotatedArucoCornersArray[1].y - rotatedArucoCornersArray[0].y);
        double newAngle = Math.toDegrees(Math.atan2(newTopVector.y, newTopVector.x));
        System.out.printf("After rotation, ArUco top edge angle: %.1f°%n", newAngle);

        rotationMatrix.release();

        return new RotatedImageInfo(rotated, rotatedArucoCornersMat);
    }

    public Mat cropPaperFromImage(Mat rotatedImage, MatOfPoint2f rotatedArucoCorners) {
        // Known measurements in cm
        final double ARUCO_SIZE_CM = 5.0;
        final double TOP_PADDING_CM = 4.25;
        final double BOTTOM_PADDING_CM = 11.75;
        final double LEFT_PADDING_CM = 22.35;
        final double RIGHT_PADDING_CM = 2.35;

        Point[] arucoCorners = rotatedArucoCorners.toArray();

        // Calculate ArUco marker size in pixels to determine scale
        Point topLeft = arucoCorners[0];
        Point topRight = arucoCorners[1];
        Point bottomRight = arucoCorners[2];
        Point bottomLeft = arucoCorners[3];

        double markerWidthPx = Math.hypot(topRight.x - topLeft.x, topRight.y - topLeft.y);
        double markerHeightPx = Math.hypot(bottomLeft.x - topLeft.x, bottomLeft.y - topLeft.y);
        double markerSizePx = (markerWidthPx + markerHeightPx) / 2.0;
        double pixelsPerCm = markerSizePx / ARUCO_SIZE_CM;

        // Calculate marker center
        double markerCenterX = (topLeft.x + topRight.x + bottomRight.x + bottomLeft.x) / 4.0;
        double markerCenterY = (topLeft.y + topRight.y + bottomRight.y + bottomLeft.y) / 4.0;

        // Compute initial crop boundaries
        double paperLeftX = markerCenterX - (LEFT_PADDING_CM + ARUCO_SIZE_CM/2.0) * pixelsPerCm;
        double paperTopY = markerCenterY - (TOP_PADDING_CM + ARUCO_SIZE_CM/2.0) * pixelsPerCm;
        double paperRightX = markerCenterX + (RIGHT_PADDING_CM + ARUCO_SIZE_CM/2.0) * pixelsPerCm;
        double paperBottomY = markerCenterY + (BOTTOM_PADDING_CM + ARUCO_SIZE_CM/2.0) * pixelsPerCm;

        int imageWidth = rotatedImage.cols();
        int imageHeight = rotatedImage.rows();
        int cropX = (int) Math.max(0, Math.round(paperLeftX));
        int cropY = (int) Math.max(0, Math.round(paperTopY));
        int cropW = (int) Math.min(imageWidth - cropX, Math.round(paperRightX - paperLeftX));
        int cropH = (int) Math.min(imageHeight - cropY, Math.round(paperBottomY - paperTopY));
        cropW = Math.max(1, cropW);
        cropH = Math.max(1, cropH);

        // First crop: the full paper
        Rect initialRect = new Rect(cropX, cropY, cropW, cropH);
        Mat croppedPaper = new Mat(rotatedImage, initialRect);

        // SECOND CROP: remove top 10%, bottom 10%, left 10%, right 30%
        int cw = croppedPaper.cols();
        int ch = croppedPaper.rows();
        int offsetX = (int) Math.round(cw * 0.065);
        int offsetY = (int) Math.round(ch * 0.10);
        int innerW = (int) Math.round(cw * (1.0 - 0.065 - 0.27)); // 60%
        int innerH = (int) Math.round(ch * (1.0 - 0.010 - 0.08)); // 80%

        offsetX = Math.max(0, offsetX);
        offsetY = Math.max(0, offsetY);
        innerW = Math.max(1, Math.min(innerW, cw - offsetX));
        innerH = Math.max(1, Math.min(innerH, ch - offsetY));

        Rect innerRect = new Rect(offsetX, offsetY, innerW, innerH);
        Mat finalCropped = new Mat(croppedPaper, innerRect);

        return finalCropped;
    }

    public class ProcessedResult {
        public Mat image;
        public gov.nasa.arc.astrobee.types.Point position;
        public boolean detectedAruco;

        public ProcessedResult(Mat i, gov.nasa.arc.astrobee.types.Point p, boolean b) {
            image = i;
            position = p;
            detectedAruco = b;
        }
    }

    public ProcessedResult processSingleImage(Mat image, gov.nasa.arc.astrobee.types.Point currentRobotPos, Quaternion currentRobotQuat, int locationIndex, KiboRpcApi api, String filename) {
        Mat undistorted = null;
        RotatedImageInfo rotatedInfo = null;
        MatOfPoint2f arucoCornersInitial = null;
        gov.nasa.arc.astrobee.types.Point targetPosition = null;
        Mat test = null;

        try {
            // Step 1: Preprocessing
            undistorted = preprocessImage(image);
            //api.saveMatImage(undistorted, filename+"_undistorted.png");
            System.out.println("after undistort");

            // Step 2a: Detect ArUco marker
            arucoCornersInitial = detectArucoMarker(undistorted, locationIndex); // Returns MatOfPoint2f
            if(arucoCornersInitial.empty()) {
                System.out.println("Failed to detect Aruco");
                ProcessedResult result = new ProcessedResult(image, targetPosition, false);
                return result;
            }
            System.out.println("after aruco initial");
            System.out.println(arucoCornersInitial);

            if(locationIndex != -1) targetPosition = EstimatePose.computeTargetPoint(currentRobotPos, currentRobotQuat, cameraMatrix, undistorted, arucoCornersInitial, locationIndex);
            System.out.println("after compute target point");

            // Step 2b: Rotate image to correct orientation
            rotatedInfo = rotateImageToCorrectOrientation(undistorted, arucoCornersInitial);
            System.out.println("after rotate");
            //api.saveMatImage(rotatedInfo.rotatedImage, filename + "_rotated.png");

            test = cropPaperFromImage(rotatedInfo.rotatedImage, rotatedInfo.rotatedArucoCorners);

            System.out.println("after crop");

            ProcessedResult result = new ProcessedResult(test, targetPosition, true);
            return result;

        } catch (Exception e) {
            e.printStackTrace();
            ProcessedResult result = new ProcessedResult(image, targetPosition, false);
            return result;
        } finally {
            // Release all intermediate Mats
            if (image != null) image.release();
            if(arucoCornersInitial != null && arucoCornersInitial.empty()) {
                System.out.println("Failed to detect Aruco");
                ProcessedResult result = new ProcessedResult(image, targetPosition, false);
                return result;
            }
            if (undistorted != null && rotatedInfo != null && undistorted != rotatedInfo.rotatedImage) undistorted.release();
            if (arucoCornersInitial != null) arucoCornersInitial.release();
            if (rotatedInfo != null) {
                if (rotatedInfo.rotatedImage != null && rotatedInfo.rotatedImage != test) rotatedInfo.rotatedImage.release();
                if (rotatedInfo.rotatedArucoCorners != null) rotatedInfo.rotatedArucoCorners.release();
            }
        }
    }
}