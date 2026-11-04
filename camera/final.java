import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;
import org.opencv.objdetect.ArucoDetector;
import org.opencv.objdetect.Dictionary;
import org.opencv.objdetect.DetectorParameters;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CameraCorrectionModel {

    // Load OpenCV native library
    static {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load. Make sure OpenCV is configured correctly.\n" + e);
            System.exit(1);
        }
    }

    private String inputFolder;
    private String outputFolder;

    // Camera intrinsics and distortion coefficients
    private Mat cameraMatrix;
    private MatOfDouble distCoeffs; // OpenCV Java uses MatOfDouble for distortion coefficients

    // ArUco detector
    private Dictionary arucoDict;
    private DetectorParameters arucoParams;
    private ArucoDetector arucoDetector;

    // Sheet dimensions (in cm) - used for ratios
    private double sheetWidth = 29.7;
    private double sheetHeight = 21.0;
    private double arucoSize = 5.0; // cm
    private double arucoOffsetX = 2.60; // from right edge cm
    private double arucoOffsetY = 4.25; // from top edge cm

    // ArUco marker size in output image (pixels)
    private int targetArucoSize = 100;
    private int padding;

    public CameraCorrectionModel(String inputFolder, String outputFolder) {
        this.inputFolder = inputFolder;
        this.outputFolder = outputFolder;

        // Create output folder if it doesn't exist
        try {
            Files.createDirectories(Paths.get(this.outputFolder));
        } catch (IOException e) {
            System.err.println("Error creating output directory: " + e.getMessage());
            // It's often better to throw a RuntimeException or handle this more gracefully
        }
        
        this.padding = (int) (this.targetArucoSize * 1.5);

        // Initialize camera matrix
        cameraMatrix = new Mat(3, 3, CvType.CV_32F);
        cameraMatrix.put(0, 0, new float[]{
                523.105750f, 0.000000f, 635.434258f,
                0.000000f, 534.765913f, 500.335102f,
                0.000000f, 0.000000f, 1.000000f
        });

        // Initialize distortion coefficients
        // Note: OpenCV Java typically expects CV_64F for distCoeffs in functions like undistort
        distCoeffs = new MatOfDouble(new double[]{-0.164787, 0.020375, -0.001572, -0.000369, 0.000000});

        // ArUco detector setup
        this.arucoDict = Objdetect.getPredefinedDictionary(Objdetect.DICT_5X5_250);
        this.arucoParams = DetectorParameters.create();
        this.arucoDetector = new ArucoDetector(this.arucoDict, this.arucoParams);
    }

    /**
     * Step 1: Preprocessing - undistort image
     */
    public Mat preprocessImage(Mat image, String filename) {
        System.out.println("Step 1: Preprocessing " + filename);
        Mat undistorted = new Mat();
        Imgproc.undistort(image, undistorted, cameraMatrix, distCoeffs);

        // Save checkpoint (optional)
        // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_1_undistorted.jpg").toString(), undistorted);
        return undistorted;
    }

    /**
     * Detect ArUco marker and return its corners.
     * The returned MatOfPoint2f contains the 4 corners of the first detected marker.
     */
    private MatOfPoint2f detectArucoMarker(Mat image, String filename) {
        System.out.println("Detecting ArUco marker in " + filename);
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

        List<Mat> cornersList = new ArrayList<>();
        Mat ids = new Mat();
        arucoDetector.detectMarkers(gray, cornersList, ids);

        if (cornersList.isEmpty()) {
            throw new RuntimeException("No ArUco marker detected in " + filename);
        }

        // Use the first detected marker
        // cornersList.get(0) is a Mat of size 4x1, type CV_32FC2 (4 points, each with 2 float coordinates)
        MatOfPoint2f arucoCorners = new MatOfPoint2f(cornersList.get(0));

        // Draw detected marker for visualization (optional)
        // Mat markedImage = image.clone();
        // Objdetect.drawDetectedMarkers(markedImage, cornersList, ids);
        // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_2a_aruco_detected.jpg").toString(), markedImage);
        
        gray.release(); // Release intermediate Mat
        ids.release();
        for(Mat m : cornersList) m.release(); // Release mats in list

        return arucoCorners; // This is a MatOfPoint2f representing the 4 corners
    }


    /**
     * Rotates the image so ArUco marker is in correct orientation (top-right).
     * Returns a class containing the rotated image and the transformed ArUco corners.
     */
    public static class RotatedImageInfo {
        public Mat rotatedImage;
        public MatOfPoint2f rotatedArucoCorners;

        public RotatedImageInfo(Mat rotatedImage, MatOfPoint2f rotatedArucoCorners) {
            this.rotatedImage = rotatedImage;
            this.rotatedArucoCorners = rotatedArucoCorners;
        }
    }

    public RotatedImageInfo rotateImageToCorrectOrientation(Mat image, MatOfPoint2f arucoCornersMat, String filename) {
        System.out.println("Orienting image " + filename);

        Point[] arucoCorners = arucoCornersMat.toArray(); // Easier to work with Point array

        // ArUco corners are ordered: top-left, top-right, bottom-right, bottom-left
        Point topLeft = arucoCorners[0];
        Point topRight = arucoCorners[1];
        
        // Get the top edge vector (from top-left to top-right)
        Point topVector = new Point(topRight.x - topLeft.x, topRight.y - topLeft.y);

        // Calculate angle of the top edge relative to horizontal
        double angleRad = Math.atan2(topVector.y, topVector.x);
        double angleDeg = Math.toDegrees(angleRad);

        // Normalize angle to [-180, 180] range (already handled by atan2 and toDegrees for typical ranges)
        // The Python code's normalization loop is generally not needed if using atan2 correctly.
        // However, to match Python's specific logic if it handles >360 or <-360:
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

        // Rotate ArUco corners accordingly
        MatOfPoint2f rotatedArucoCornersMat = new MatOfPoint2f();
        Core.transform(arucoCornersMat, rotatedArucoCornersMat, rotationMatrix);
        
        Point[] rotatedArucoCornersArray = rotatedArucoCornersMat.toArray();
        Point newTopVector = new Point(rotatedArucoCornersArray[1].x - rotatedArucoCornersArray[0].x, rotatedArucoCornersArray[1].y - rotatedArucoCornersArray[0].y);
        double newAngle = Math.toDegrees(Math.atan2(newTopVector.y, newTopVector.x));
        System.out.printf("After rotation, ArUco top edge angle: %.1f°%n", newAngle);

        // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_2b_rotated.jpg").toString(), rotated);
        
        rotationMatrix.release();

        return new RotatedImageInfo(rotated, rotatedArucoCornersMat);
    }
    
    /**
     * Create debug visualizations for edge detection process
     */
    public Mat debugEdgeDetection(Mat image, String filename, String stepName) {
        Path debugFolder = Paths.get(outputFolder, "debug_edges");
        try {
            Files.createDirectories(debugFolder);
        } catch (IOException e) { /* ignored */ }

        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        // Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_01_gray.jpg").toString(), gray);

        Mat blurred = new Mat();
        Imgproc.GaussianBlur(gray, blurred, new Size(0, 0), 1.0);
        
        Mat enhanced = new Mat();
        Core.addWeighted(gray, 1.5, blurred, -0.5, 0, enhanced);
        
        // Clip values, similar to np.clip(enhanced, 0, 255).astype(np.uint8)
        Core.MinMaxLocResult mmr = Core.minMaxLoc(enhanced);
        enhanced.convertTo(enhanced, CvType.CV_8U, 255.0/(mmr.maxVal - mmr.minVal), -mmr.minVal * 255.0/(mmr.maxVal - mmr.minVal));
        // A simpler way if values are generally within a reasonable range before scaling:
        // Core.normalize(enhanced, enhanced, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        // Or, if direct clipping is needed:
        // Mat maskMax = new Mat(); Mat maskMin = new Mat();
        // Imgproc.threshold(enhanced, maskMax, 255, 255, Imgproc.THRESH_TRUNC); // cap at 255
        // Imgproc.threshold(maskMax, maskMin, 0, 255, Imgproc.THRESH_TOZERO); // floor at 0
        // maskMin.convertTo(enhanced, CvType.CV_8U);
        // For simplicity, assuming addWeighted output can be directly used by Canny or converted.
        // If enhanced is CV_32F or CV_64F, it needs conversion to CV_8U for Canny.
        // Let's assume enhanced is already suitable or convert it.
        if (enhanced.depth() != CvType.CV_8U) {
            Core.normalize(enhanced, enhanced, 0, 255, Core.NORM_MINMAX);
            enhanced.convertTo(enhanced, CvType.CV_8U);
        }


        Mat edges2 = new Mat();
        Imgproc.Canny(enhanced, edges2, 20, 60);
        // Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_04_canny_enhanced.jpg").toString(), edges2);
        
        gray.release();
        blurred.release();
        enhanced.release();

        return edges2; // Enhanced Canny seems most promising
    }

    /**
     * Debug contour detection process
     */
    public Map<String, List<MatOfPoint>> debugContourDetection(Mat edges, Mat originalImage, String filename, String stepName) {
        Path debugFolder = Paths.get(outputFolder, "debug_contours");
        try {
            Files.createDirectories(debugFolder);
        } catch (IOException e) { /* ignored */ }

        Mat kernelSmall = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2, 2));
        Mat kernelMedium = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        Mat kernelLarge = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));

        Mat closed1 = new Mat(); Imgproc.morphologyEx(edges, closed1, Imgproc.MORPH_CLOSE, kernelSmall);
        Mat closed2 = new Mat(); Imgproc.morphologyEx(edges, closed2, Imgproc.MORPH_CLOSE, kernelMedium);
        Mat closed3 = new Mat(); Imgproc.morphologyEx(edges, closed3, Imgproc.MORPH_CLOSE, kernelLarge);

        // Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_01_closed_small.jpg").toString(), closed1);
        // Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_02_closed_medium.jpg").toString(), closed2);
        // Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_03_closed_large.jpg").toString(), closed3);
        
        Mat dilated1 = new Mat(); Imgproc.dilate(edges, dilated1, kernelSmall, new Point(-1,-1), 1);
        Mat dilated2 = new Mat(); Imgproc.dilate(edges, dilated2, kernelMedium, new Point(-1,-1), 2);

        // Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_04_dilated_light.jpg").toString(), dilated1);
        // Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_05_dilated_strong.jpg").toString(), dilated2);

        Map<String, Mat> edgeVariantsMap = new LinkedHashMap<>(); // Preserve order
        edgeVariantsMap.put("original", edges);
        edgeVariantsMap.put("closed_medium", closed2);
        edgeVariantsMap.put("dilated_light", dilated1);
        edgeVariantsMap.put("dilated_strong", dilated2);
        
        Map<String, List<MatOfPoint>> allContours = new HashMap<>();

        for (Map.Entry<String, Mat> entry : edgeVariantsMap.entrySet()) {
            String variantName = entry.getKey();
            Mat edgeImg = entry.getValue();
            
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(edgeImg, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            allContours.put(variantName, contours);
            hierarchy.release();

            // Mat contourVis = originalImage.clone();
            // Imgproc.drawContours(contourVis, contours, -1, new Scalar(0, 255, 0), 2);
            // Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_contours_" + variantName + ".jpg").toString(), contourVis);
            // contourVis.release();
            System.out.println("Found " + contours.size() + " contours using " + variantName + " edges");
        }
        
        kernelSmall.release(); kernelMedium.release(); kernelLarge.release();
        closed1.release(); closed2.release(); closed3.release();
        dilated1.release(); dilated2.release();
        // 'edges' is an input, should not be released here. Other edge variants are released if not 'edges'.
        if (edgeVariantsMap.get("closed_medium") != edges) edgeVariantsMap.get("closed_medium").release();
        if (edgeVariantsMap.get("dilated_light") != edges) edgeVariantsMap.get("dilated_light").release();
        if (edgeVariantsMap.get("dilated_strong") != edges) edgeVariantsMap.get("dilated_strong").release();


        return allContours;
    }

    // Helper class for contour scoring
    private static class ContourScoreInfo {
        MatOfPoint2f contour; // approxPolyDP result
        double score;
        double area;
        double rectangularity;
        double ratioScore;
        double convexity;
        double epsilonFactor;
        Rect bbox;
        String variant;

        ContourScoreInfo(MatOfPoint2f contour, double score, double area, double rectangularity, double ratioScore, double convexity, double epsilonFactor, Rect bbox, String variant) {
            this.contour = contour;
            this.score = score;
            this.area = area;
            this.rectangularity = rectangularity;
            this.ratioScore = ratioScore;
            this.convexity = convexity;
            this.epsilonFactor = epsilonFactor;
            this.bbox = bbox;
            this.variant = variant;
        }
    }
    
    /**
     * Analyze contours and score them for sheet detection
     */
    public ContourScoreInfo analyzeAndScoreContours(Map<String, List<MatOfPoint>> contoursDict, Mat originalImage, String filename, String stepName, MatOfPoint2f expectedSheetCorners) {
        Path debugFolder = Paths.get(outputFolder, "debug_scoring");
        try {
            Files.createDirectories(debugFolder);
        } catch (IOException e) { /* ignored */ }

        double cropArea = originalImage.rows() * originalImage.cols();
        double minArea = cropArea * 0.2;
        double maxArea = cropArea * 0.9;
        double expectedRatio = this.sheetWidth / this.sheetHeight;

        List<ContourScoreInfo> bestContoursOverall = new ArrayList<>();

        for (Map.Entry<String, List<MatOfPoint>> entry : contoursDict.entrySet()) {
            String variantName = entry.getKey();
            List<MatOfPoint> contours = entry.getValue();
            System.out.println("\nAnalyzing " + variantName + " contours:");

            List<ContourScoreInfo> scoredContoursVariant = new ArrayList<>();

            for (int i = 0; i < contours.size(); i++) {
                MatOfPoint contour = contours.get(i);
                double area = Imgproc.contourArea(contour);

                if (area > minArea && area < maxArea) {
                    MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                    double perimeter = Imgproc.arcLength(contour2f, true);

                    for (double epsilonFactor : new double[]{0.01, 0.02, 0.03, 0.05}) {
                        double epsilon = epsilonFactor * perimeter;
                        MatOfPoint2f approx = new MatOfPoint2f();
                        Imgproc.approxPolyDP(contour2f, approx, epsilon, true);

                        if (approx.rows() >= 4) {
                            double rectArea = Imgproc.contourArea(approx);
                            Rect bbox = Imgproc.boundingRect(approx); // MatOfPoint instead of MatOfPoint2f
                            double bboxArea = (double) bbox.width * bbox.height;
                            double rectangularity = (bboxArea > 0) ? rectArea / bboxArea : 0;

                            double actualRatio = (bbox.height > 0) ? (double) bbox.width / bbox.height : 0;
                            double ratioScore = 1.0 - Math.min(Math.abs(expectedRatio - actualRatio) / expectedRatio, 1.0);

                            MatOfPoint hullPoints = new MatOfPoint();
                            MatOfInt hullInt = new MatOfInt(); // For convexHull indices
                            Imgproc.convexHull(new MatOfPoint(approx.toArray()), hullInt); // Needs MatOfPoint

                            // Create MatOfPoint from hull indices
                            Point[] approxPoints = approx.toArray();
                            Point[] hullContourPoints = new Point[hullInt.rows()];
                            int[] hullIntArray = hullInt.toArray();
                            for(int k=0; k < hullInt.rows(); k++){
                                hullContourPoints[k] = approxPoints[hullIntArray[k]];
                            }
                            hullPoints.fromArray(hullContourPoints);
                            
                            double hullArea = Imgproc.contourArea(hullPoints);
                            double convexity = (hullArea > 0) ? rectArea / hullArea : 0;

                            double areaScore = Math.min(area / maxArea, 1.0);
                            double score = areaScore * rectangularity * ratioScore * convexity;
                            
                            scoredContoursVariant.add(new ContourScoreInfo(approx, score, area, rectangularity, ratioScore, convexity, epsilonFactor, bbox, variantName));
                            hullPoints.release();
                            hullInt.release();
                        }
                        approx.release();
                    }
                    contour2f.release();
                }
                contour.release(); // Release individual contour MatOfPoint
            }

            scoredContoursVariant.sort(Comparator.comparingDouble(cs -> -cs.score)); // Sort descending
            bestContoursOverall.addAll(scoredContoursVariant.stream().limit(3).collect(Collectors.toList()));

            // Visualize top contours for this variant (optional)
            /*
            if (!scoredContoursVariant.isEmpty()) {
                Mat visImage = originalImage.clone();
                for (int j = 0; j < Math.min(5, scoredContoursVariant.size()); j++) {
                    ContourScoreInfo item = scoredContoursVariant.get(j);
                    Scalar color = new Scalar(0,0,0); // Placeholder
                    if(j==0) color = new Scalar(255,0,0); else if(j==1) color = new Scalar(0,255,0); // ...
                    Imgproc.drawContours(visImage, Collections.singletonList(new MatOfPoint(item.contour.toArray())), -1, color, 3);
                    Imgproc.putText(visImage, String.format("#%d S:%.3f", j + 1, item.score),
                                    new Point(item.bbox.x, item.bbox.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
                    System.out.printf("  Contour %d: Score=%.3f, Area=%.0f, Rect=%.3f, Ratio=%.3f, Conv=%.3f, Eps=%.2f%n",
                                      j + 1, item.score, item.area, item.rectangularity, item.ratioScore, item.convexity, item.epsilonFactor);
                }
                Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_scored_" + variantName + ".jpg").toString(), visImage);
                visImage.release();
            }
            */
        }

        bestContoursOverall.sort(Comparator.comparingDouble(cs -> -cs.score));

        // Create final comparison image (optional)
        /*
        if (!bestContoursOverall.isEmpty()) {
            Mat finalVis = originalImage.clone();
            for (int j = 0; j < Math.min(10, bestContoursOverall.size()); j++) {
                ContourScoreInfo item = bestContoursOverall.get(j);
                // ... drawing logic ...
            }
            Imgcodecs.imwrite(debugFolder.resolve(filename + "_" + stepName + "_final_candidates.jpg").toString(), finalVis);
            finalVis.release();
        }
        */
        
        // Clean up MatOfPoint2f in ContourScoreInfo if they are not the chosen one
        ContourScoreInfo bestOne = bestContoursOverall.isEmpty() ? null : bestContoursOverall.get(0);
        for(ContourScoreInfo csi : bestContoursOverall){
            if(csi != bestOne && csi.contour != null) {
                csi.contour.release();
            }
        }


        return bestOne;
    }

    /**
     * Find the 4 most extreme points from a set of points.
     * Input points should be MatOfPoint2f.
     */
    private MatOfPoint2f findExtremePoints(MatOfPoint2f pointsMat) {
        Point[] points = pointsMat.toArray();
        if (points.length <= 4) {
            return pointsMat; // Return original if 4 or less points
        }

        Point topLeft = points[0], topRight = points[0], bottomRight = points[0], bottomLeft = points[0];
        double sumMin = points[0].x + points[0].y;         // Top-left: min(x+y)
        double diffMax = points[0].x - points[0].y;        // Top-right: max(x-y)
        double sumMax = points[0].x + points[0].y;         // Bottom-right: max(x+y)
        double diffMin = -points[0].x + points[0].y;       // Bottom-left: max(-x+y) -> min(x-y) in Python was incorrect, should be max(-x+y) or min(y-x)

        // Corrected logic for bottomLeft based on Python's np.argmax(-points[:, 0] + points[:, 1])
        // This means we want to maximize (y - x).
        double yMinusXMax = points[0].y - points[0].x;


        for (Point p : points) {
            if (p.x + p.y < sumMin) { sumMin = p.x + p.y; topLeft = p; }
            if (p.x - p.y > diffMax) { diffMax = p.x - p.y; topRight = p; }
            if (p.x + p.y > sumMax) { sumMax = p.x + p.y; bottomRight = p; }
            if (p.y - p.x > yMinusXMax) { yMinusXMax = p.y - p.x; bottomLeft = p; }
        }
        return new MatOfPoint2f(topLeft, topRight, bottomRight, bottomLeft);
    }

    /**
     * Sort corners in order: top-left, top-right, bottom-right, bottom-left.
     * Input corners should be MatOfPoint2f.
     */
    private MatOfPoint2f sortCorners(MatOfPoint2f cornersMat) {
        Point[] corners = cornersMat.toArray();
        if (corners.length != 4) return cornersMat; // Expect 4 corners

        // Calculate centroid
        Point centroid = new Point(0, 0);
        for (Point p : corners) {
            centroid.x += p.x;
            centroid.y += p.y;
        }
        centroid.x /= corners.length;
        centroid.y /= corners.length;

        // Sort by angle from centroid
        List<Point> cornerList = new ArrayList<>(Arrays.asList(corners));
        cornerList.sort((p1, p2) -> {
            double angle1 = Math.atan2(p1.y - centroid.y, p1.x - centroid.x);
            double angle2 = Math.atan2(p2.y - centroid.y, p2.x - centroid.x);
            return Double.compare(angle1, angle2);
        });

        // Find top-left corner (smallest sum of coordinates) among sorted
        int topLeftIdx = 0;
        double minSum = Double.MAX_VALUE;
        for (int i = 0; i < cornerList.size(); i++) {
            double sum = cornerList.get(i).x + cornerList.get(i).y;
            if (sum < minSum) {
                minSum = sum;
                topLeftIdx = i;
            }
        }
        
        // Reorder starting from top-left
        Collections.rotate(cornerList, -topLeftIdx);
        
        return new MatOfPoint2f(cornerList.toArray(new Point[0]));
    }


    public MatOfPoint2f detectSheetCorners(Mat image, MatOfPoint2f arucoCornersRotated, String filename) {
        System.out.println("Attempting to detect sheet corners in " + filename);
        Point[] arucoCornersArray = arucoCornersRotated.toArray();

        // Calculate the scale and expected sheet position based on ArUco marker
        Point arucoCenter = new Point(0,0);
        for(Point p : arucoCornersArray) { arucoCenter.x += p.x; arucoCenter.y += p.y; }
        arucoCenter.x /= arucoCornersArray.length;
        arucoCenter.y /= arucoCornersArray.length;

        double arucoWidthPx = Core.norm(new MatOfPoint2f(arucoCornersArray[1]), new MatOfPoint2f(arucoCornersArray[0]));
        double arucoHeightPx = Core.norm(new MatOfPoint2f(arucoCornersArray[3]), new MatOfPoint2f(arucoCornersArray[0]));
        double avgArucoSizePx = (arucoWidthPx + arucoHeightPx) / 2.0;
        double scaleFactorCmPerPx = this.arucoSize / avgArucoSizePx; // cm per pixel

        // Calculate sheet dimensions in pixels
        double sheetWidthPx = this.sheetWidth / scaleFactorCmPerPx;
        double sheetHeightPx = this.sheetHeight / scaleFactorCmPerPx;

        // Calculate sheet position based on ArUco marker position
        double arucoToRightEdgePx = this.arucoOffsetX / scaleFactorCmPerPx;
        double arucoToTopEdgePx = this.arucoOffsetY / scaleFactorCmPerPx;

        // Estimate sheet corners based on ArUco position (assuming ArUco is top-right on sheet)
        // Note: The Python script implies ArUco is at top-right of sheet.
        // The corners are TL, TR, BR, BL for ArUco.
        // If ArUco's top-left is arucoCornersArray[0], its top-right is arucoCornersArray[1] etc.
        // Let's use ArUco center for estimation.
        double estSheetTopRightX = arucoCenter.x + arucoToRightEdgePx; // If ArUco is on right of sheet center
        double estSheetTopRightY = arucoCenter.y - arucoToTopEdgePx; // If ArUco is on top of sheet center
        // This depends on how aruco_offset_x/y are defined relative to ArUco marker's own frame.
        // Python: aruco_offset_x from right edge, aruco_offset_y from top edge.
        // This means the sheet's right edge is aruco_offset_x to the right of ArUco's *right edge*.
        // And sheet's top edge is aruco_offset_y above ArUco's *top edge*.
        // Let's use ArUco's top-right corner (arucoCornersArray[1]) as a reference.
        // Sheet top-right X = ArUco TR X + aruco_offset_x_px
        // Sheet top-right Y = ArUco TR Y - aruco_offset_y_px
        
        // Using ArUco's center and its known size to find its edges first:
        // ArUco's own top-right point in the image is arucoCornersArray[1]
        // ArUco's own top-left point is arucoCornersArray[0]
        // ArUco's own bottom-right point is arucoCornersArray[2]
        // ArUco's own bottom-left point is arucoCornersArray[3]

        // The Python code's estimation:
        // sheet_top_right_x = aruco_center[0] + aruco_to_right_edge_px
        // sheet_top_right_y = aruco_center[1] - aruco_to_top_edge_px
        // This implies aruco_offset_x/y are from aruco_center to sheet_top_right.
        // This seems more plausible.
        
        double sheetTR_X = arucoCenter.x + (this.arucoOffsetX / scaleFactorCmPerPx);
        double sheetTR_Y = arucoCenter.y - (this.arucoOffsetY / scaleFactorCmPerPx);
        // This assumes aruco_offset_x is distance from aruco center to sheet's right edge,
        // and aruco_offset_y is distance from aruco center to sheet's top edge.
        // The original Python code is slightly ambiguous here. Let's re-evaluate:
        // "aruco_offset_x = 2.60  # from right edge"
        // "aruco_offset_y = 4.25  # from top edge"
        // This usually means from the ArUco marker's edge to the sheet's edge.
        // If ArUco is top-right on the sheet:
        // Sheet TopRight X = ArUco TopRight X + (aruco_offset_x / scaleFactorCmPerPx) (if offset is outside marker)
        // Or Sheet TopRight X = ArUco TopLeft X + sheet_width_px (if offset is from sheet edge to marker edge)

        // Let's follow Python's calculation for estimated_corners:
        // sheet_top_right_x = aruco_center[0] + aruco_to_right_edge_px
        // sheet_top_right_y = aruco_center[1] - aruco_to_top_edge_px
        // This means the sheet's top-right corner is estimated relative to the ArUco center.
        
        Point estimatedSheetTopRight = new Point(
            arucoCenter.x + (this.arucoOffsetX / scaleFactorCmPerPx),
            arucoCenter.y - (this.arucoOffsetY / scaleFactorCmPerPx)
        );

        MatOfPoint2f estimatedCornersMat = new MatOfPoint2f(
            new Point(estimatedSheetTopRight.x - sheetWidthPx, estimatedSheetTopRight.y),                 // top-left
            new Point(estimatedSheetTopRight.x, estimatedSheetTopRight.y),                               // top-right
            new Point(estimatedSheetTopRight.x, estimatedSheetTopRight.y + sheetHeightPx),               // bottom-right
            new Point(estimatedSheetTopRight.x - sheetWidthPx, estimatedSheetTopRight.y + sheetHeightPx) // bottom-left
        );
        Point[] estimatedCorners = estimatedCornersMat.toArray();

        System.out.printf("Estimated sheet size: %.1fx%.1f pixels%n", sheetWidthPx, sheetHeightPx);
        System.out.printf("Scale factor: %.4f cm/pixel%n", scaleFactorCmPerPx);

        double paddingFactor = 0.5;
        int paddingX = (int) (sheetWidthPx * paddingFactor);
        int paddingY = (int) (sheetHeightPx * paddingFactor);

        double minXEst = Arrays.stream(estimatedCorners).mapToDouble(p -> p.x).min().orElse(0);
        double minYEst = Arrays.stream(estimatedCorners).mapToDouble(p -> p.y).min().orElse(0);
        double maxXEst = Arrays.stream(estimatedCorners).mapToDouble(p -> p.x).max().orElse(image.cols());
        double maxYEst = Arrays.stream(estimatedCorners).mapToDouble(p -> p.y).max().orElse(image.rows());

        int xMin = Math.max(0, (int) (minXEst - paddingX));
        int yMin = Math.max(0, (int) (minYEst - paddingY));
        int xMax = Math.min(image.cols(), (int) (maxXEst + paddingX));
        int yMax = Math.min(image.rows(), (int) (maxYEst + paddingY));
        
        if (xMax <= xMin || yMax <= yMin) {
            System.err.println("Invalid crop region for " + filename + ", cannot detect sheet corners. Crop: " + xMin + "," + yMin + " to " + xMax + "," + yMax);
            estimatedCornersMat.release();
            return null;
        }
        System.out.printf("Crop region: %d,%d to %d,%d (size: %dx%d)%n", xMin, yMin, xMax, yMax, (xMax - xMin), (yMax - yMin));

        Rect cropROI = new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        Mat cropped = new Mat(image, cropROI);

        // Adjust estimated corners for the crop
        Point[] adjEstCornersArray = new Point[4];
        for(int i=0; i<4; i++){
            adjEstCornersArray[i] = new Point(estimatedCorners[i].x - xMin, estimatedCorners[i].y - yMin);
        }
        MatOfPoint2f adjustedEstimatedCorners = new MatOfPoint2f(adjEstCornersArray);

        // Save crop region with estimated corners (optional)
        // Mat cropVis = cropped.clone();
        // for (int i = 0; i < adjEstCornersArray.length; i++) {
        //     Point corner = adjEstCornersArray[i];
        //     if (corner.x >= 0 && corner.x < cropVis.cols() && corner.y >= 0 && corner.y < cropVis.rows()) {
        //        Imgproc.circle(cropVis, corner, 8, new Scalar(255, 0, 255), -1);
        //        Imgproc.putText(cropVis, "E" + i, new Point(corner.x + 10, corner.y + 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 0, 255), 2);
        //     }
        // }
        // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_2c_crop_with_estimates.jpg").toString(), cropVis);
        // cropVis.release();
        
        Mat bestEdges = debugEdgeDetection(cropped, filename, "crop");
        Map<String, List<MatOfPoint>> contoursDict = debugContourDetection(bestEdges, cropped, filename, "crop");
        ContourScoreInfo bestContourInfo = analyzeAndScoreContours(contoursDict, cropped, filename, "crop", adjustedEstimatedCorners);

        MatOfPoint2f detectedSheetCorners = null;
        if (bestContourInfo != null) {
            MatOfPoint2f cornersApprox = bestContourInfo.contour; // This is already an approxPolyDP result

            // If more than 4 points, find the 4 most extreme ones
            if (cornersApprox.rows() > 4) {
                MatOfPoint2f extremePoints = findExtremePoints(cornersApprox);
                cornersApprox.release(); // release the old one
                cornersApprox = extremePoints;
            }
            
            // Sort corners properly: TL, TR, BR, BL
            MatOfPoint2f sortedCorners = sortCorners(cornersApprox);
            if (cornersApprox != sortedCorners) cornersApprox.release(); // release if sortCorners returned new Mat

            // Adjust for crop offset
            Point[] finalCornerPoints = sortedCorners.toArray();
            for(Point p : finalCornerPoints) {
                p.x += xMin;
                p.y += yMin;
            }
            detectedSheetCorners = new MatOfPoint2f(finalCornerPoints);
            sortedCorners.release(); // release intermediate

            System.out.printf("Successfully detected sheet corners for %s with score %.3f%n", filename, bestContourInfo.score);
            // Visualization (optional)
        } else {
            System.out.println("Could not reliably detect sheet corners for " + filename);
        }
        
        cropped.release();
        bestEdges.release();
        adjustedEstimatedCorners.release();
        estimatedCornersMat.release();
        // contoursDict's Mats are released within analyzeAndScoreContours or debugContourDetection
        
        return detectedSheetCorners;
    }


    /**
     * Step 3: Apply homography transformation.
     * Uses ArUco for initial warp, then attempts to refine with sheet corners if detected.
     */
    public Mat applyHomography(Mat image, MatOfPoint2f rotatedArucoCorners, String filename) {
        System.out.println("Step 3: Applying homography to " + filename);

        double pixelsPerCm = (double) this.targetArucoSize / this.arucoSize;
        int targetWidth = (int) (this.sheetWidth * pixelsPerCm);
        int targetHeight = (int) (this.sheetHeight * pixelsPerCm);
        System.out.printf("Target image size: %dx%d pixels%n", targetWidth, targetHeight);

        int arucoOffsetX_px = (int) (this.arucoOffsetX * pixelsPerCm);
        int arucoOffsetY_px = (int) (this.arucoOffsetY * pixelsPerCm);

        // Target ArUco corners in the output image (top-right of sheet)
        // The Python code's target_aruco_corners implies ArUco is placed relative to padding.
        // And its position is defined by its own size and offsets from sheet edges.
        // ArUco marker's top-left in its local coordinate system on the sheet.
        // If ArUco is at top-right of sheet:
        // Aruco TopLeft X on sheet = sheet_width - aruco_offset_x_from_right_edge - aruco_width
        // Aruco TopLeft Y on sheet = aruco_offset_y_from_top_edge
        
        // Python's target_aruco_corners:
        // aruco_right = target_width - aruco_offset_x_px
        // aruco_left  = aruco_right - self.target_aruco_size
        // aruco_top   = aruco_offset_y_px
        // aruco_bottom= aruco_top + self.target_aruco_size
        // This means aruco_offset_x_px is from the *right edge of the sheet* to the *right edge of ArUco*.
        // And aruco_offset_y_px is from the *top edge of the sheet* to the *top edge of ArUco*.

        int arucoTargetRight = targetWidth - arucoOffsetX_px;
        int arucoTargetLeft = arucoTargetRight - this.targetArucoSize;
        int arucoTargetTop = arucoOffsetY_px;
        int arucoTargetBottom = arucoTargetTop + this.targetArucoSize;

        MatOfPoint2f targetArucoCorners = new MatOfPoint2f(
                new Point(this.padding + arucoTargetLeft, this.padding + arucoTargetTop),       // Top-left
                new Point(this.padding + arucoTargetRight, this.padding + arucoTargetTop),      // Top-right
                new Point(this.padding + arucoTargetRight, this.padding + arucoTargetBottom),   // Bottom-right
                new Point(this.padding + arucoTargetLeft, this.padding + arucoTargetBottom)    // Bottom-left
        );

        Mat homographyMatrix = Calib3d.findHomography(rotatedArucoCorners, targetArucoCorners, Calib3d.RANSAC, 5.0);
        // Mat homographyMatrix = Imgproc.getPerspectiveTransform(rotatedArucoCorners, targetArucoCorners); // Simpler if RANSAC not strictly needed for ArUco

        Mat warped = new Mat();
        Imgproc.warpPerspective(image, warped, homographyMatrix, new Size(targetWidth + 2 * this.padding, targetHeight + 2 * this.padding));

        // Optional: Draw corners on original for visualization
        // Mat cornerVis = image.clone(); // ... draw rotatedArucoCorners ...
        // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_3a_all_corners.jpg").toString(), cornerVis);
        // cornerVis.release();

        // Target sheet corners for the final crop (no padding)
        MatOfPoint2f targetSheetCornersMat = new MatOfPoint2f(
                new Point(0, 0),
                new Point(targetWidth, 0),
                new Point(targetWidth, targetHeight),
                new Point(0, targetHeight)
        );

        // Try to detect sheet corners on the *warped* image
        // The ArUco corners passed to detectSheetCorners should be targetArucoCorners (now in warped image coords)
        MatOfPoint2f detectedSheetCornersInWarped = detectSheetCorners(warped, targetArucoCorners, filename + "_warped_stage");
        
        Mat finalWarped;

        if (detectedSheetCornersInWarped != null && detectedSheetCornersInWarped.checkVector(2) == 4) {
             // Optional: Visualize detected sheet corners on 'warped'
            // Mat ocornerVis = warped.clone(); // ... draw detectedSheetCornersInWarped ...
            // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_3b_warped_with_sheet_corners.jpg").toString(), ocornerVis);
            // ocornerVis.release();

            Mat cornerHomography = Calib3d.findHomography(detectedSheetCornersInWarped, targetSheetCornersMat, Calib3d.RANSAC, 5.0);
            // Mat cornerHomography = Imgproc.getPerspectiveTransform(detectedSheetCornersInWarped, targetSheetCornersMat);
            
            finalWarped = new Mat();
            Imgproc.warpPerspective(warped, finalWarped, cornerHomography, new Size(targetWidth, targetHeight));
            
            cornerHomography.release();
            detectedSheetCornersInWarped.release();
        } else {
            System.out.println("Could not detect sheet corners on warped image, using ArUco warp and cropping padding.");
            if (this.padding > 0 && 
                warped.cols() > 2 * this.padding && warped.rows() > 2 * this.padding) {
                Rect roi = new Rect(this.padding, this.padding, targetWidth, targetHeight);
                finalWarped = new Mat(warped, roi); // Creates a sub-matrix, clone if modification is needed later
            } else if (warped.cols() == targetWidth && warped.rows() == targetHeight) {
                 finalWarped = warped.clone(); // Already correct size
            }
            else {
                // Fallback: resize if padding crop is not possible but dimensions mismatch
                System.out.println("Warning: Warped image size mismatch after ArUco warp, resizing to target.");
                finalWarped = new Mat();
                Imgproc.resize(warped, finalWarped, new Size(targetWidth, targetHeight));
            }
        }
        
        warped.release();
        homographyMatrix.release();
        targetArucoCorners.release();
        targetSheetCornersMat.release();

        return finalWarped;
    }


    /**
     * Step 4: Greyscale enhancement with edge sharpening and final crop
     */
    public Mat enhanceImage(Mat image, String filename) {
        System.out.println("Step 4: Enhancing image " + filename);
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_4a_grayscale.jpg").toString(), gray);

        Mat sharpened = new Mat();
        Imgproc.GaussianBlur(gray, sharpened, new Size(5,5), 0); // Python used (5,5) before adaptiveThreshold

        Mat binary = new Mat();
        Imgproc.adaptiveThreshold(sharpened, binary, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
        // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_4c_binary.jpg").toString(), binary);
        
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5,5));
        Mat opened = new Mat();
        Imgproc.morphologyEx(binary, opened, Imgproc.MORPH_OPEN, kernel);
        // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_4d_opened.jpg").toString(), opened);

        // Crop out specified margins
        int h = opened.rows();
        int w = opened.cols();
        double pixelsPerCm = (double) this.targetArucoSize / this.arucoSize;

        double leftCropCm = 1.35;
        double rightCropCm = 8.35;
        double topCropCm = 3.0;
        double bottomCropCm = 3.0;

        int leftCropPx = (int) (leftCropCm * pixelsPerCm);
        int rightCropPx = (int) (rightCropCm * pixelsPerCm);
        int topCropPx = (int) (topCropCm * pixelsPerCm);
        int bottomCropPx = (int) (bottomCropCm * pixelsPerCm);

        int finalX = leftCropPx;
        int finalY = topCropPx;
        int finalWidth = w - leftCropPx - rightCropPx;
        int finalHeight = h - topCropPx - bottomCropPx;

        if (finalWidth <=0 || finalHeight <=0 || finalX < 0 || finalY < 0 || (finalX + finalWidth) > w || (finalY + finalHeight) > h) {
            System.err.println("Error: Invalid crop dimensions for " + filename + ". Final image will be empty or out of bounds. Skipping final crop.");
            // Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_4e_final_PRE_CROP_ERROR.jpg").toString(), opened);
            gray.release(); sharpened.release(); binary.release(); kernel.release();
            return opened; // Return 'opened' if crop is invalid
        }
        
        Rect cropRect = new Rect(finalX, finalY, finalWidth, finalHeight);
        Mat finalImage = new Mat(opened, cropRect); // This creates a submatrix header

        // Clone to make it a standalone Mat if it's going to be returned and 'opened' released
        Mat finalCroppedStandalone = finalImage.clone(); 

        Imgcodecs.imwrite(Paths.get(outputFolder, filename + "_4e_final.jpg").toString(), finalCroppedStandalone);
        
        gray.release();
        sharpened.release();
        binary.release();
        kernel.release();
        opened.release(); 
        // finalImage is a header for opened, no need to release if opened is released.
        // finalCroppedStandalone is a new Mat, will be returned.

        return finalCroppedStandalone;
    }


    public Mat processSingleImage(String imagePath) {
        Path path = Paths.get(imagePath);
        String filename = path.getFileName().toString();
        // Remove extension for base filename
        int dotIndex = filename.lastIndexOf('.');
        if (dotIndex > 0) {
            filename = filename.substring(0, dotIndex);
        }
        System.out.println("\n=== Processing " + filename + " ===");

        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.err.println("Error: Could not load image " + imagePath);
            return null;
        }
        
        Mat finalResult = null;
        Mat undistorted = null;
        RotatedImageInfo rotatedInfo = null;
        MatOfPoint2f arucoCornersInitial = null;
        Mat warped = null;

        try {
            // Step 1: Preprocessing
            undistorted = preprocessImage(image, filename);

            // Step 2a: Detect ArUco marker
            arucoCornersInitial = detectArucoMarker(undistorted, filename); // Returns MatOfPoint2f

            // Step 2b: Rotate image to correct orientation
            rotatedInfo = rotateImageToCorrectOrientation(undistorted, arucoCornersInitial, filename);
            // rotatedInfo.rotatedImage and rotatedInfo.rotatedArucoCorners are now available

            // Step 3: Apply homography
            warped = applyHomography(rotatedInfo.rotatedImage, rotatedInfo.rotatedArucoCorners, filename);

            // Step 4: Enhancement
            finalResult = enhanceImage(warped, filename);

            System.out.println("Successfully processed " + filename);
            return finalResult;

        } catch (Exception e) {
            System.err.println("Error processing " + filename + ": " + e.getMessage());
            e.printStackTrace();
            return null; // Or rethrow as a custom processing exception
        } finally {
            // Release all intermediate Mats
            if (image != null) image.release();
            if (undistorted != null && undistorted != rotatedInfo.rotatedImage) undistorted.release(); // if not same obj
            if (arucoCornersInitial != null) arucoCornersInitial.release();
            if (rotatedInfo != null) {
                if (rotatedInfo.rotatedImage != null && rotatedInfo.rotatedImage != warped) rotatedInfo.rotatedImage.release();
                if (rotatedInfo.rotatedArucoCorners != null) rotatedInfo.rotatedArucoCorners.release();
            }
            if (warped != null && warped != finalResult) warped.release();
            // finalResult is returned, so it should not be released here.
        }
    }

    public void processAllImages() {
        List<String> imageFiles = new ArrayList<>();
        String[] extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"};

        for (String extPattern : extensions) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(this.inputFolder), extPattern)) {
                for (Path entry : stream) {
                    imageFiles.add(entry.toString());
                }
            } catch (IOException e) {
                System.err.println("Error reading input folder: " + e.getMessage());
            }
            // For case-insensitive matching (less straightforward with DirectoryStream glob, might need custom filter or list and then filter)
             try (DirectoryStream<Path> stream = Files.newDirectoryStream(Paths.get(this.inputFolder), extPattern.toUpperCase())) {
                for (Path entry : stream) {
                    if(!imageFiles.contains(entry.toString())) imageFiles.add(entry.toString());
                }
            } catch (IOException e) { /* ignore */ }
        }
        
        if (imageFiles.isEmpty()) {
            System.out.println("No image files found in " + this.inputFolder);
            return;
        }

        System.out.println("Found " + imageFiles.size() + " image(s) to process");

        for (String imagePath : imageFiles) {
            Mat result = processSingleImage(imagePath);
            if (result != null) {
                result.release(); // Release the final image after processing is done for this image.
            }
        }

        System.out.println("\nProcessing complete. Check " + this.outputFolder + " for results.");
        System.out.println("Debug images can be found in (if uncommented):");
        System.out.println("  - " + Paths.get(this.outputFolder, "debug_edges"));
        System.out.println("  - " + Paths.get(this.outputFolder, "debug_contours"));
        System.out.println("  - " + Paths.get(this.outputFolder, "debug_scoring"));
    }


    public static void main(String[] args) {
        System.out.println("Starting Camera Correction Model processing...");
        // Ensure OpenCV is loaded (already handled by static block)
        // For testing, you might pass folders as arguments or hardcode them.
        String inputDir = "input"; // Default input folder
        String outputDir = "output"; // Default output folder

        if (args.length == 2) {
            inputDir = args[0];
            outputDir = args[1];
        } else if (args.length != 0) {
            System.out.println("Usage: java CameraCorrectionModel <input_folder> <output_folder>");
            System.out.println("Using default folders: input, output");
        }
        
        // Create dummy input folder and a test image if they don't exist for basic runnability
        Path inputPath = Paths.get(inputDir);
        if (!Files.exists(inputPath)) {
            try {
                Files.createDirectories(inputPath);
                System.out.println("Created dummy input folder: " + inputDir);
                // Create a dummy image (e.g., a simple black image with a white square for ArUco)
                // This is complex to do programmatically without more setup.
                // For now, assume user provides images.
            } catch (IOException e) {
                System.err.println("Could not create dummy input folder: " + e.getMessage());
            }
        }


        CameraCorrectionModel model = new CameraCorrectionModel(inputDir, outputDir);
        model.processAllImages();
        System.out.println("Processing finished.");
    }
}
