package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;

public class YoloONNXRunner {
    private static final String[] CLASS_NAMES = {
            "coin","compass","coral","crystal","diamond",
            "emerald","fossil","key","letter","shell","treasure_box"
    };

    private OrtEnvironment environment;
    private OrtSession session;
    private String inputName;
    private String outputName;

    public static class Detection {
        public int classId;
        public float confidence;
        public RectF box;
    }

    public YoloONNXRunner(Context context) {
        try {
            environment = OrtEnvironment.getEnvironment();

            File modelFile = new File(context.getFilesDir(), "model.onnx");
            if (!modelFile.exists()) {
                try (InputStream is = context.getAssets().open("model.onnx");
                     FileOutputStream os = new FileOutputStream(modelFile)) {
                    byte[] buf = new byte[4 * 1024 * 1024];
                    int len;
                    while ((len = is.read(buf)) > 0) {
                        os.write(buf, 0, len);
                    }
                    os.getFD().sync();
                }
            }

            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
            session = environment.createSession(modelFile.getAbsolutePath(), opts);

            inputName = session.getInputNames().iterator().next();
            outputName = session.getOutputNames().iterator().next();

        } catch (OrtException | IOException e) {
            e.printStackTrace();
        }
    }

    // === NEW METHOD: Takes Grayscale Mat and Converts to RGB ===
    public List<Detection> detect(Mat grayMat, float threshold, float iouThreshold, String filename, KiboRpcApi api) {
        try {
            // 1) Convert single‑channel grayMat to 3‑channel RGB
            Mat rgbMat = new Mat();
            Imgproc.cvtColor(grayMat, rgbMat, Imgproc.COLOR_GRAY2RGB);

            // 2) Resize to model input size
            int inSize = 640;
            Mat resized = new Mat();
            Imgproc.resize(rgbMat, resized, new Size(inSize, inSize));

            // 3) Normalize and convert to [1,3,640,640]
            float[][][][] input = new float[1][3][inSize][inSize];
            for (int y = 0; y < inSize; y++) {
                for (int x = 0; x < inSize; x++) {
                    double[] pix = resized.get(y, x);  // pix[0]=R, pix[1]=G, pix[2]=B
                    input[0][0][y][x] = (float) (pix[0] / 255.0);
                    input[0][1][y][x] = (float) (pix[1] / 255.0);
                    input[0][2][y][x] = (float) (pix[2] / 255.0);
                }
            }

            // 4) Run the ONNX model
            OnnxTensor tensor = OnnxTensor.createTensor(environment, input);
            OrtSession.Result res = session.run(Collections.singletonMap(inputName, tensor));
            float[][][] output = (float[][][]) res.get(outputName).get().getValue();
            tensor.close();
            res.close();

            // 5) Post‑processing uses the original image dimensions
            int imgW = grayMat.width(), imgH = grayMat.height();
            List<Detection> dets = new ArrayList<>();
            int preds = output[0][0].length;

            for (int i = 0; i < preds; i++) {
                float cx = output[0][0][i], cy = output[0][1][i];
                float w  = output[0][2][i], h  = output[0][3][i];
                float bestC = -1; int bestI = -1;
                for (int c = 0; c < CLASS_NAMES.length; c++) {
                    float conf = output[0][4 + c][i];
                    if (conf > bestC) { bestC = conf; bestI = c; }
                }
                if (bestC < threshold) continue;

                RectF box = new RectF(
                        (cx - w/2) * imgW, (cy - h/2) * imgH,
                        (cx + w/2) * imgW, (cy + h/2) * imgH
                );

                // corner‑filtering logic
                float boxW = box.width(), boxH = box.height();
                boolean wide = boxW / boxH > 2.5f;
                boolean top = box.top <= 10, bottom = imgH - box.bottom <= 10;
                boolean left = box.left <= 10, right = imgW - box.right <= 10;
                boolean inCorner = wide && ((top && left) || (top && right) || (bottom && left) || (bottom && right));
                if (inCorner && bestC <= 0.9f) continue;

                Detection d = new Detection();
                d.classId = bestI;
                d.confidence = bestC;
                d.box = box;
                dets.add(d);
            }

            return nonMaximumSuppression(dets, iouThreshold, grayMat, filename, api);

        } catch (OrtException e) {
            e.printStackTrace();
            return new ArrayList<>();
        }
    }


    private List<Detection> nonMaximumSuppression(
            List<Detection> dets,
            float iouThreshold,
            Mat grayMat,
            String filename,
            KiboRpcApi api
    ) {
        Collections.sort(dets, (a,b) -> Float.compare(b.confidence, a.confidence));
        List<Detection> kept = new ArrayList<>();
        boolean[] skip = new boolean[dets.size()];
        for (int i = 0; i < dets.size(); i++) {
            if (skip[i]) continue;
            Detection A = dets.get(i);
            kept.add(A);
            for (int j = i+1; j < dets.size(); j++) {
                if (!skip[j] && iou(A.box, dets.get(j).box) > iouThreshold) {
                    skip[j] = true;
                }
            }
        }
        //Mat drawn = drawDetections(grayMat, kept);
        //api.saveMatImage(drawn, filename + "_detections.png");
        return kept;
    }

    private float iou(RectF a, RectF b) {
        float l = Math.max(a.left, b.left);
        float t = Math.max(a.top, b.top);
        float r = Math.min(a.right, b.right);
        float btm = Math.min(a.bottom, b.bottom);
        float inter = Math.max(0, r - l) * Math.max(0, btm - t);
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        float areaB = (b.right - b.left) * (b.bottom - b.top);
        return inter / (areaA + areaB - inter);
    }

    private Mat drawDetections(Mat inputMat, List<Detection> dets) {
        Bitmap bmp = Bitmap.createBitmap(inputMat.cols(), inputMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(inputMat, bmp);
        Bitmap m = bmp.copy(Bitmap.Config.ARGB_8888, true);

        Canvas c = new Canvas(m);
        Paint p = new Paint();
        p.setColor(Color.RED);
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(3);

        Paint tp = new Paint();
        tp.setColor(Color.RED);
        tp.setTextSize(32);

        for (Detection d : dets) {
            c.drawRect(d.box, p);
            c.drawText(CLASS_NAMES[d.classId] + String.format(": %.2f", d.confidence),
                    d.box.left, d.box.top - 10, tp);
        }

        Mat out = new Mat();
        Utils.bitmapToMat(m, out);
        return out;
    }

    public void close() {
        try {
            if (session != null) session.close();
            if (environment != null) environment.close();
        } catch (OrtException e) {
            e.printStackTrace();
        }
    }
}
