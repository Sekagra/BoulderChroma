/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package de.tum.hack.BoulderChroma;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.JsonReader;
import android.util.JsonToken;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import de.tum.hack.BoulderChroma.customview.OverlayView;
import de.tum.hack.BoulderChroma.customview.OverlayView.DrawCallback;
import de.tum.hack.BoulderChroma.env.BorderedText;
import de.tum.hack.BoulderChroma.env.ImageUtils;
import de.tum.hack.BoulderChroma.env.Logger;
import de.tum.hack.BoulderChroma.tflite.Classifier;
import de.tum.hack.BoulderChroma.tflite.TFLiteObjectDetectionAPIModel;
import de.tum.hack.BoulderChroma.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 416;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "model.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labels.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.2f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();


            // Send web request to our """backend""" an get info about where to draw the boxes
            ByteArrayOutputStream bao = new ByteArrayOutputStream();
            croppedBitmap.compress(Bitmap.CompressFormat.JPEG, 100, bao);

            byte[] data = bao.toByteArray();

            List<Classifier.Recognition> rects = new ArrayList<>();
            try {
              LOGGER.i("Attempting to send an image.");
              URL url = new URL("http://131.159.226.43:5000/");
              // multipart file
              HttpURLConnection con = (HttpURLConnection) url.openConnection();
              con.setRequestMethod("POST");
              con.setUseCaches(false);
              con.setDoOutput(true);
              con.setRequestProperty("Content-Size", "" + data.length);
              String boundary = "===" + System.currentTimeMillis() + "===";
              con.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

              OutputStream out = con.getOutputStream();

              OutputStreamWriter writer = new OutputStreamWriter(out);
              writer.append("--" + boundary + "\r\n");
              writer.append(
                      "Content-Disposition: form-data; name=\"file\"; filename=\"image.jpg\"\r\n");
              writer.append(
                      "Content-Type: " + URLConnection.guessContentTypeFromName("image.jpg") + "\r\n");
              writer.append("Content-Transfer-Encoding: binary\r\n\r\n");
              writer.flush();

              out.write(data);
              out.flush();

              writer.append("\r\n");
              writer.flush();

              writer.append("\r\n").flush();
              writer.append("--" + boundary + "--\r\n");
              writer.close();


              int code = con.getResponseCode();
              if (code < 200 || code >= 300) {
                throw new Exception("Image upload failed with return code " + code);
              }

              JsonReader reader = new JsonReader(new InputStreamReader(con.getInputStream(), "UTF-8"));

              // Parse JSON response
              reader.beginArray();
              while (reader.hasNext()) {
                rects.add(readRecognition(reader));
              }
              reader.endArray();

              reader.close();
              con.disconnect();
            } catch (Exception e) {
              e.printStackTrace();
              computingDetection = false;
              return;
            }

            // TODO return
            LOGGER.i(String.format("Received %d rects.", rects.size()));

            // DRIVE BY
            /*AssetManager assetManager = DetectorActivity.this.getAssets();

            InputStream istr;
            Bitmap testBitmap = null;
            try {
              istr = assetManager.open("0026.png");
              testBitmap = BitmapFactory.decodeStream(istr);
            } catch (IOException e) {
              // handle exception
            }*/



            /*final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {*/

                // find color of the location
                /*if (location.centerX() < cropCopyBitmap.getWidth() && location.centerY() < cropCopyBitmap.getHeight()) {
                  int pixel = cropCopyBitmap.getPixel((int)location.centerX(), (int)location.centerY());
                  int redValue = Color.red(pixel);
                  int blueValue = Color.blue(pixel);
                  int greenValue = Color.green(pixel);
                  result.setColor(Color.rgb(redValue, greenValue, blueValue));
                  LOGGER.i(String.format("Found pixel with color #%06X", (0xFFFFFF & result.getColor())));
                }*/

                /*canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }*/

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);

            tracker.trackResults(rects, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
            }
        });
  }

  private Classifier.Recognition readRecognition(JsonReader reader) throws IOException {
    String id = null;
    String title = null;
    float confidence = 0.0f;
    RectF boundingBox = null;

    reader.beginObject();
    while (reader.hasNext()) {
      String name = reader.nextName();
      if (name.equals("probability")) {
        confidence = (float) reader.nextDouble();
      } else if (name.equals("tagId")) {
        id = reader.nextString();
      } else if (name.equals("tagName")) {
        title = reader.nextString();
      } else if (name.equals("boundingBox")) {
        boundingBox = readRect(reader);
      } else {
        reader.skipValue();
      }
    }
    reader.endObject();
    return new Classifier.Recognition(id, title, confidence, boundingBox);
  }

  private RectF readRect(JsonReader reader) throws IOException {
    float left = 0.0f;
    float top = 0.0f;
    float width = 0.0f;
    float height = 0.0f;

    reader.beginObject();
    while (reader.hasNext()) {
      String name = reader.nextName();
      if (name.equals("left")) {
        left = (float) reader.nextDouble();// * croppedBitmap.getWidth();
      } else if (name.equals("top")) {
        top = (float) reader.nextDouble();// * croppedBitmap.getHeight();
      } else if (name.equals("width")) {
        width = (float) reader.nextDouble();// * croppedBitmap.getWidth();
      } else if (name.equals("height")) {
        height = (float) reader.nextDouble();// * croppedBitmap.getHeight();
      } else {
        reader.skipValue();
      }
    }
    reader.endObject();
    return new RectF(left, top, left + width, top + height);
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
