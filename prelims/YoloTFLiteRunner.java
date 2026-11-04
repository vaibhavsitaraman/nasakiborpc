package jp.jaxa.iss.kibo.rpc.sampleapk;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
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
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import jp.jaxa.iss.kibo.rpc.api.KiboRpcApi;
import org.tensorflow.lite.Interpreter;

public class YoloTFLiteRunner {
    private static final String[] CLASS_NAMES = {
            "coin","compass","coral","crystal","diamond",
            "emerald","fossil","key","letter","shell","treasure_box"
    };
    private static final int INPUT_SIZE = 640;      // same as ONNX
    private static final int NUM_CLASSES = CLASS_NAMES.length;
    private static final int NUM_CHANNELS = 3;
    private static final int MAX_DETECTIONS = 8400; // adjust if your model differs

    private Interpreter interpreter;

    public static class Detection {
        public int classId;
        public float confidence;
        public RectF box;
    }

    public YoloTFLiteRunner(Context context) {
        try {
            interpreter = new Interpreter(loadModelFile(context));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        // 1) copy out of assets into filesDir (only does it once)
        File outFile = new File(context.getFilesDir(), "model.tflite");
        if (!outFile.exists()) {
            try (InputStream is = context.getAssets().open("model.tflite");
                 FileOutputStream os = new FileOutputStream(outFile)) {
                byte[] buf = new byte[4 * 1024];        // small, fixed 4 KB chunk
                int len;
                while ((len = is.read(buf)) > 0) {
                    os.write(buf, 0, len);
                }
                os.getFD().sync();
            }
        }

        // 2) memory‑map the file — no large heap allocations
        try (FileInputStream fis = new FileInputStream(outFile)) {
            FileChannel fc = fis.getChannel();
            return fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size());
        }
    }

    /** Detect on a single‑channel grayMat, same signature as ONNX runner */
    public List<Detection> detect(Mat grayMat, float threshold, float iouThreshold,
                                  String filename, KiboRpcApi api) {
        // 1) gray → RGB
        Mat rgb = new Mat();
        Imgproc.cvtColor(grayMat, rgb, Imgproc.COLOR_GRAY2RGB);

        // 2) resize to INPUT_SIZE
        Mat resized = new Mat();
        Imgproc.resize(rgb, resized, new Size(INPUT_SIZE, INPUT_SIZE));

        // 3) build input buffer [1,3,640,640]
        ByteBuffer inputBuf = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS);
        inputBuf.order(ByteOrder.nativeOrder());
        for (int y = 0; y < INPUT_SIZE; y++) {
            for (int x = 0; x < INPUT_SIZE; x++) {
                double[] pix = resized.get(y, x);
                inputBuf.putFloat((float)(pix[0] / 255.0));  // R
                inputBuf.putFloat((float)(pix[1] / 255.0));  // G
                inputBuf.putFloat((float)(pix[2] / 255.0));  // B
            }
        }
        inputBuf.rewind();

        // 4) run TFLite
        // assuming output shape [1, 4+NUM_CLASSES, MAX_DETECTIONS]
        float[][][] output = new float[1][4 + NUM_CLASSES][MAX_DETECTIONS];
        interpreter.run(inputBuf, output);

        // 5) postprocess on original dims
        int imgW = grayMat.width(), imgH = grayMat.height();
        List<Detection> dets = new ArrayList<>();

        for (int i = 0; i < MAX_DETECTIONS; i++) {
            float cx = output[0][0][i];
            float cy = output[0][1][i];
            float w  = output[0][2][i];
            float h  = output[0][3][i];

            // find best class
            float bestC = -1f; int bestI = -1;
            for (int c = 0; c < NUM_CLASSES; c++) {
                float conf = output[0][4 + c][i];
                if (conf > bestC) { bestC = conf; bestI = c; }
            }
            if (bestC < threshold) continue;

            RectF box = new RectF(
                    (cx - w/2) * imgW, (cy - h/2) * imgH,
                    (cx + w/2) * imgW, (cy + h/2) * imgH
            );

            // corner filtering (same as ONNX)
            float boxW = box.width(), boxH = box.height();
            boolean wide = boxW/boxH > 2.5f;
            boolean top = box.top <= 10, bottom = imgH - box.bottom <= 10;
            boolean left = box.left <= 10, right = imgW - box.right <= 10;
            boolean inCorner = wide && ((top&&left)||(top&&right)||(bottom&&left)||(bottom&&right));
            if (inCorner && bestC <= 0.9f) continue;

            Detection d = new Detection();
            d.classId = bestI;
            d.confidence = bestC;
            d.box = box;
            dets.add(d);
        }

        return nonMaximumSuppression(dets, iouThreshold, grayMat, filename, api);
    }

    private List<Detection> nonMaximumSuppression(
            List<Detection> dets, float iouThreshold,
            Mat grayMat, String filename, KiboRpcApi api) {

        Collections.sort(dets, new Comparator<Detection>() {
            @Override
            public int compare(Detection a, Detection b) {
                return Float.compare(b.confidence, a.confidence);
            }
        });

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

        // draw & save exactly as ONNX
       // Mat drawn = drawDetections(grayMat, kept);
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
        Bitmap mutable = bmp.copy(Bitmap.Config.ARGB_8888, true);

        Canvas c = new Canvas(mutable);
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
        Utils.bitmapToMat(mutable, out);
        return out;
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
        }
    }
}
