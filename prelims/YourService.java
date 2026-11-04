package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import gov.nasa.arc.astrobee.Kinematics;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcService;

import gov.nasa.arc.astrobee.types.Point;
import gov.nasa.arc.astrobee.types.Quaternion;

import org.opencv.android.Utils;
import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.photo.Photo;


import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.HashMap;
import java.util.Map;


/**
 * Class meant to handle commands from the Ground Data System and execute them in Astrobee.
 */
public class YourService extends KiboRpcService {

    private void move(Point p, Quaternion q) {
        double x = p.getX();
        double y  = p.getY();
        double z = p.getZ();
        x = Math.min(11.40d, x);
        x = Math.max(10.45d, x);
        y = Math.min(-6.20d, y);
        y = Math.max(-10.05d, y);
        z = Math.min(5.4d, z);
        z = Math.max(4.4d, z);
        Point newPoint = new Point(x, y, z);
        api.moveTo(newPoint, q, true);
    }

    private void wait(int ms) {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }

    }

    private final String TAG = this.getClass().getSimpleName();

        private final String[] TEMPLATE_NAME = {
                "coin",
                "compass",
                "coral",
                "crystal",
                "diamond",
                "emerald",
                "fossil",
                "key",
                "letter",
                "shell",
                "treasure_box"
        };



    // Define the 4 candidate locations for lost items
    private final Point[] CANDIDATE_LOCATIONS = {
            new Point(10.95d, -9.73d, 5.195d),     // Location 1
            new Point(10.925d, -8.875d, 4.61d),          // Location 2 (example coordinates)
            new Point(10.925d, -7.925d, 4.61d),          // Location 3 (example coordinates)
            new Point(10.816d, -6.8525d, 4.945d)           // Location 4 (example coordinates)
    };

    private Point CANDIDATE_23_LOCATION = new Point (10.925d, -8.35d, 4.86d);

    private final Quaternion[] CANDIDATE_ORIENTATIONS = {
            new Quaternion(0f, 0f, -0.7071f, 0.7071f), // Orientation 1
            new Quaternion(0f, 0.7071f, 0f, 0.7071f),  // Orientation 2
            new Quaternion(0f, 0.7071f, 0f, 0.7071f),  // Orientation 3
            new Quaternion(
                    0.0f,       // x
                    1.0f, 0.0f, 0.0f
            )  // Orientation 4
    };

    private Point[] FINAL_TARGET_LOCATIONS = {
            new Point(10.95d, -9.73d, 5.195d),     // Location 1
            new Point (10.825d, -8.875d, 4.61d),          // Location 2 (example coordinates)
            new Point (10.825d, -7.92d, 4.61d),     // Location 3 (example coordinates)
            new Point(10.816d, -6.8525d, 4.945d)           // Location 4 (example coordinates)
    };

    private Point[] BACKUP_LOCATIONS = {
            new Point(10.95d, -9.88d, 5.195d),     // Location 1
            new Point (10.825d, -8.875d, 4.46d),          // Location 2 (example coordinates)
            new Point (10.825d, -7.92d, 4.46d),     // Location 3 (example coordinates)
            new Point(10.566d, -6.8525d, 4.945d)           // Location 4 (example coordinates)
    };
    // Astronaut location
    private final Point ASTRONAUT_LOCATION = new Point(11.143d, -6.7607d, 4.9654d);
    private final Point AFTER_ASTRONAUT_LOCATION = new Point(11d, -6.8d,4.9d);
    private final Quaternion ASTRONAUT_ORIENTATION = new Quaternion(0f, 0f, 0.7071f, 0.7071f);



    // Store found items at each location


    private Map<Integer, treasureDetection> foundTreasures = new HashMap<>();
    private int targetTreasure = -1;
    private int targetLocation = -1;

    private int[] strategy = {1, 1, 1, 1};

    public class treasureDetection {
        int index;
        float confidence;
        public treasureDetection(int i, float c) {
            confidence = c;
            index = i;
        }
    }


    private ImageProcessor processor;
    private YoloTFLiteRunner runner;

    public boolean isTreasure(int x) {
        if(x == 3 || x == 4|| x == 5) return true;
        else return false;
    }

    public double dist(Point a, Point b, int locationIndex) {
        if(locationIndex == 0) {
            return Math.sqrt((a.getX() - b.getX()) * (a.getX() - b.getX()) + (a.getZ() - b.getZ()) * (a.getZ() - b.getZ()));
        }
        else if (locationIndex == 1 || locationIndex == 2) {
            return Math.sqrt((a.getX() - b.getX()) * (a.getX() - b.getX()) + (a.getY() - b.getY()) * (a.getY() - b.getY()));
        }
        else if(locationIndex == 3) {
            return Math.sqrt((a.getZ() - b.getZ()) * (a.getZ() - b.getZ()) + (a.getY() - b.getY()) * (a.getY() - b.getY()));
        }
        else return 99.9;
    }

    @Override
    protected void runPlan1(){
        Log.i(TAG, "start mission!");
        api.startMission();

        new Thread(() -> {
            Log.i(TAG, "Initializing processor & runner…");
            processor = new ImageProcessor();
            runner    = new YoloTFLiteRunner(YourService.this); //new YoloONNXRunner(YourService.this);
            Log.i(TAG, "…init done");
        }).start();


        // Visit all 4 candidate locations
        for(int locationIndex = 0; locationIndex < CANDIDATE_LOCATIONS.length; locationIndex++) {
            Log.i(TAG, "Moving to location " + (locationIndex + 1));

            // Move to the location
            if(locationIndex == 0) {
                move(new Point(10.95d, -9.575d, 4.87), new Quaternion(-0.07f, -0.07f, -0.7071f, 0.7071f));
                wait(500);
            }
            else if(locationIndex == 1) {
                move(new Point(10.995, -8.875, 5.15), CANDIDATE_ORIENTATIONS[locationIndex]);

            }
            else if(locationIndex == 2) {
                move(new Point(10.825, -7.77, 5.4), CANDIDATE_ORIENTATIONS[locationIndex]);
               wait(3000);
                move(new Point(10.825, -7.77, 5.05), CANDIDATE_ORIENTATIONS[locationIndex]);
            }
            else {
                move(new Point(11.15, -6.8525d, 4.9d), CANDIDATE_ORIENTATIONS[locationIndex]);
                wait(500);
            }


            Kinematics k = api.getRobotKinematics();
            System.out.println(k.getPosition());
            System.out.println(k.getOrientation());
            // Get camera image

            Mat image = api.getMatNavCam();
            //api.saveMatImage(image, "location_" + (locationIndex + 1) + "_raw.png");

            // Process the image and detect items
            System.out.println("before process");
            ImageProcessor.ProcessedResult pResult = processor.processSingleImage(image, k.getPosition(), k.getOrientation(), locationIndex, api,"location_" + (locationIndex + 1));
            System.out.println("after process");

            if(pResult.detectedAruco == false) {
                move(CANDIDATE_LOCATIONS[locationIndex], CANDIDATE_ORIENTATIONS[locationIndex]);
                Kinematics newK = api.getRobotKinematics();
                System.out.println(k.getPosition());
                System.out.println(k.getOrientation());
                image = api.getMatNavCam();
                pResult = processor.processSingleImage(image, newK.getPosition(), newK.getOrientation(), locationIndex, api,"location_" + (locationIndex + 1));
            }
            Mat processed = pResult.image;

            if(pResult.position != null ) {
                FINAL_TARGET_LOCATIONS[locationIndex] = pResult.position;
            }

            /*
            if(pResult.position != null)  {
                FINAL_TARGET_LOCATIONS[locationIndex] = pResult.position;
                if(dist(pResult.position, k.getPosition(), locationIndex) <= 0.35 && locationIndex != 2 && locationIndex != 1) {
                    strategy[locationIndex] = 1;
                }

                if(strategy[locationIndex] == 0) {
                    move(pResult.position, CANDIDATE_ORIENTATIONS[locationIndex]);
                   // wait(1000);
                    Mat temp = api.getMatNavCam();

                    api.saveMatImage(temp, "temp" + locationIndex + ".png");

                    Kinematics newK = api.getRobotKinematics();

                    pResult = processor.processSingleImage(temp, newK.getPosition(), newK.getOrientation(), locationIndex, api, "location_" + (locationIndex + 1));
                    processed = pResult.image;
                    if(pResult.position != null) FINAL_TARGET_LOCATIONS[locationIndex] = pResult.position;
                }
            }
            */

            if(processed.cols() > 0 && processed.rows() > 0) {
                // 1) Clone mat so our worker thread has its own copy
                final Mat processedCopy = processed.clone();
                // 2) Copy index into an effectively final var
                final int locIdx = locationIndex;

                // 3) Launch the heavy work on a background thread
                int finalLocationIndex = locationIndex;
                new Thread(() -> {
                    // Convert Mat → Bitma
                    // Persist the processed image
                    //api.saveMatImage(processedCopy, "location_" + (locIdx + 1) + "_processed.png");

                    // Run YOLO detection
                    List<YoloTFLiteRunner.Detection> detections =
                            runner.detect(processedCopy, 0.4F, 0.5F, "location_" + (locIdx + 1), api);

                    // Compute counts + best non‐treasure class
                    int[] items = new int[TEMPLATE_NAME.length];
                    float maxConfidence = 0.0F;
                    int maxConfidenceId = 0;
                    for (YoloTFLiteRunner.Detection d : detections) {
                        items[d.classId]++;
                        System.out.println(TEMPLATE_NAME[d.classId]);
                        if (!isTreasure(d.classId) && d.confidence > maxConfidence) {
                            maxConfidence = d.confidence;
                            maxConfidenceId = d.classId;
                        }
                        else if(isTreasure(d.classId)) {
                            if(foundTreasures.get(d.classId) != null) {
                                if(foundTreasures.get(d.classId).confidence <= d.confidence) {
                                    foundTreasures.put(d.classId, new treasureDetection(locIdx, d.confidence));
                                }
                            }
                            else {
                                foundTreasures.put(d.classId, new treasureDetection(locIdx, d.confidence));

                            }

                        }
                    }

                    // Update area info
                    api.setAreaInfo(
                            locIdx + 1,
                            TEMPLATE_NAME[maxConfidenceId],
                            items[maxConfidenceId]
                    );

                    // Record found treasures (thread‐safe)

                }).start();
            }

            if(locationIndex == 0) {
                move(new Point(10.995d, -9.575d, 4.87), new Quaternion(
                        -0.028082f,
                        0.503727f,
                        -0.283668f,
                        0.815477f
                ));
            }
            else if(locationIndex == 1) {
                move(new Point(10.995, -8.5d, 5.4d), CANDIDATE_ORIENTATIONS[locationIndex]);
                wait(3000);

            }
            else if(locationIndex == 2) {
                //nothing
            }
            else {
                //nothing
            }

        }


        System.out.println(foundTreasures);

        // Move to astronaut location

            Log.i(TAG, "Moving to astronaut location ");
            //move(ASTRONAUT_LOCATION, ASTRONAUT_ORIENTATION);
        new Thread(() -> {
            move(new Point(11.363, -6.761, 4.875), ASTRONAUT_ORIENTATION);
        }).start();
        wait(13000);



            api.reportRoundingCompletion();
            wait(1000);

            // Process astronaut's clue image
            Mat processedAstronaut = new Mat();
            while(processedAstronaut.rows() == 0) {
                wait(500);
                Mat astronautImage = api.getMatNavCam();
                //api.saveMatImage(astronautImage, "astronaut_clue.png");


                ImageProcessor.ProcessedResult processedAstronautResult = processor.processSingleImage(astronautImage, ASTRONAUT_LOCATION, ASTRONAUT_ORIENTATION, -1, api, "astronaut");
                processedAstronaut = processedAstronautResult.image;
            }






            final Mat astroCopy = processedAstronaut.clone();


                // Run YOLO on astronaut clue
                List<YoloTFLiteRunner.Detection> astronautDetections =
                        runner.detect(astroCopy, 0.5F, 0.5F, "astronaut", api);

                // Look for treasure in the detections
                for (YoloTFLiteRunner.Detection d : astronautDetections) {
                    System.out.println(TEMPLATE_NAME[d.classId]);
                    if (isTreasure(d.classId)) {
                        // If you worry about cross‐thread visibility,
                        // you can make targetTreasure volatile or synchronize.
                        targetTreasure = d.classId;
                        break;
                    }
                }            System.out.println("thread done");





        if(targetTreasure != -1) {
            Log.i(TAG, "Target treasure identified: " + targetTreasure);

            // Find which location has this treasure
            for(Map.Entry<Integer, treasureDetection> entry : foundTreasures.entrySet()) {
                System.out.println(entry.getKey());
                System.out.println(entry.getValue());
                if(entry.getKey() == targetTreasure) {
                    targetLocation = entry.getValue().index + 1;
                }
            }
        }

        api.notifyRecognitionItem();

        // Move to target treasure location
        if(targetLocation <= 0) targetLocation = 4;
        if(targetLocation > 0 && targetLocation <= CANDIDATE_LOCATIONS.length) {
            Log.i(TAG, "Moving to target treasure at location " + targetLocation);

            //move(CANDIDATE_LOCATIONS[targetLocation -1], CANDIDATE_ORIENTATIONS[targetLocation - 1]);
            move(FINAL_TARGET_LOCATIONS[targetLocation-1], CANDIDATE_ORIENTATIONS[targetLocation-1]);
            wait(500);
            Mat targetImage = api.getMatNavCam();
            //api.saveMatImage(targetImage, "target_normal.png");



            Kinematics k = api.getRobotKinematics();
            System.out.println(k.getPosition());
            System.out.println(k.getOrientation());

            ImageProcessor.ProcessedResult temp = processor.processSingleImage(targetImage, k.getPosition(), k.getOrientation(), targetLocation-1, api, "t");
            /*
            if(temp.position == null || temp.detectedAruco == false) {
                move(CANDIDATE_LOCATIONS[targetLocation-1], CANDIDATE_ORIENTATIONS[targetLocation-1]);
                Mat target2Image = api.getMatNavCam();

                ImageProcessor.ProcessedResult temp2 = processor.processSingleImage(target2Image, k.getPosition(), k.getOrientation(), targetLocation-1, api, "t");
                temp = temp2;

            }
            */

            if(temp.position != null && dist(k.getPosition(), temp.position, targetLocation-1) >= 0.11) {
                move(new Point(temp.position.getX() * 0.66 + k.getPosition().getX() * 0.34, temp.position.getY() * 0.66 + k.getPosition().getY() * 0.34, temp.position.getZ() * 0.66 + k.getPosition().getZ() * 0.34), CANDIDATE_ORIENTATIONS[targetLocation-1]);

                    // wait(1000);
                    Mat doubletemp = api.getMatNavCam();
                    //api.saveMatImage(doubletemp, "temp.png");
            }



            api.takeTargetItemSnapshot();





        } else {
            Log.e(TAG, "Target treasure location not found!");
            api.takeTargetItemSnapshot();
        }

    }

    @Override
    protected void runPlan2(){
        // write your plan 2 here.
    }

    @Override
    protected void runPlan3(){
        // write your plan 3 here.
    }






}