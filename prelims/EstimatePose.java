package jp.jaxa.iss.kibo.rpc.sampleapk;

import gov.nasa.arc.astrobee.types.Quaternion;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import java.util.Arrays;

public class EstimatePose {

    /**
     * @param locationIndex
     *   0 → Y‐plane at y = −10.58, face +Y
     *   1,2 → Z‐plane at z = 3.76, face +Z
     *   3 → X‐plane at x = 9.867, face −X
     */
    public static gov.nasa.arc.astrobee.types.Point computeTargetPoint(
            gov.nasa.arc.astrobee.types.Point currentPos,
            Quaternion currentOri,
            Mat cameraMatrix,
            Mat image,
            MatOfPoint2f arucoCorners,
            int locationIndex) {

        // --- REFLECT top‐right over bottom‐left ---
        Point[] pts = arucoCorners.toArray();
        Point bl = pts[3];      // bottom‐left
        Point tr = pts[1];      // top‐right
        double u = 2*bl.x - tr.x;
        double v = 2*bl.y - tr.y;

        // Unproject to camera ray (same as before)…
        double fx = cameraMatrix.get(0,0)[0],
                fy = cameraMatrix.get(1,1)[0],
                cx = cameraMatrix.get(0,2)[0],
                cy = cameraMatrix.get(1,2)[0];
        double x_norm = (u - cx) / fx;
        double y_norm = (v - cy) / fy;
        double dirCamX = 1.0, dirCamY = x_norm, dirCamZ = y_norm;

        // Rotate into world frame…
        double qx = currentOri.getX(), qy = currentOri.getY(),
                qz = currentOri.getZ(), qw = currentOri.getW();
        double R00 = 1 - 2*(qy*qy + qz*qz),
                R01 = 2*(qx*qy - qz*qw),
                R02 = 2*(qx*qz + qy*qw);
        double R10 = 2*(qx*qy + qz*qw),
                R11 = 1 - 2*(qx*qx + qz*qz),
                R12 = 2*(qy*qz - qx*qw);
        double R20 = 2*(qx*qz - qy*qw),
                R21 = 2*(qy*qz + qx*qw),
                R22 = 1 - 2*(qx*qx + qy*qy);

        double dirW_X = R00*dirCamX + R01*dirCamY + R02*dirCamZ;
        double dirW_Y = R10*dirCamX + R11*dirCamY + R12*dirCamZ;
        double dirW_Z = R20*dirCamX + R21*dirCamY + R22*dirCamZ;

        // Intersect with plane + apply 0.75 m positive offset
        double t, targetX, targetY, targetZ;
        final double off = 0.75;
        switch (locationIndex) {
            case 0: {
                double planeY = -10.58;
                t = (planeY - currentPos.getY()) / dirW_Y;
                targetX = currentPos.getX() + t * dirW_X;
                targetY = planeY + off;
                targetZ = currentPos.getZ() + t * dirW_Z;
                break;
            }
            case 1: case 2: {
                double planeZ = 3.76;
                t = (planeZ - currentPos.getZ()) / dirW_Z;
                targetX = currentPos.getX() + t * dirW_X;
                targetY = currentPos.getY() + t * dirW_Y;
                targetZ = planeZ + off;
                break;
            }
            case 3: {
                double planeX = 9.867;
                t = (planeX - currentPos.getX()) / dirW_X;
                targetX = planeX + off;
                targetY = currentPos.getY() + t * dirW_Y;
                targetZ = currentPos.getZ() + t * dirW_Z;
                break;
            }
            default:
                throw new IllegalArgumentException("Invalid locationIndex: " + locationIndex);
        }

        return new gov.nasa.arc.astrobee.types.Point(targetX, targetY, targetZ);
    }

}
