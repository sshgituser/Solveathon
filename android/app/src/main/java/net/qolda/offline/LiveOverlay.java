package net.qolda.offline;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.media.MediaFormat;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Build;
import android.os.SystemClock;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import org.json.JSONObject;

/** Native Camera2 plumbing and a minimal technical preview surface; no MediaRecorder. */
final class LiveOverlay {
    private final Context context;
    private final MainActivity host;
    private final LiveAnalyzer cycle;
    private final Consumer<JSONObject> progress;
    private final Consumer<Analyzer.Outcome> done;
    private final LinearLayout layout;
    private final TextureView preview;
    private final TextView status;
    private final Button finish, cancel;
    private final Handler main;
    private final AtomicBoolean closing = new AtomicBoolean();
    private Runnable onRemoved;
    private volatile long captureStarted;
    private volatile int rotation;
    private volatile Size dimensions;
    private int seen;
    private boolean visible;
    private String preliminary = "";

    // The camera callbacks keep a live executor while an asynchronous open is in flight. A callback
    // arriving after stop can therefore close that device instead of leaking it on a dead handler.
    private static final class CameraThread {
        static final Handler HANDLER = create();
        private static Handler create() {
            HandlerThread t = new HandlerThread("qolda-camera-io");
            t.start();
            return new Handler(t.getLooper());
        }
    }
    // The fields below are confined to CameraThread.HANDLER.
    private CameraDevice camera;
    private CameraCaptureSession cameraSession;
    private ImageReader reader;
    private Surface screen;
    private long lastConverted;
    private boolean disableRotateAndCrop;

    LiveOverlay(MainActivity host, Analyzer analyzer, String language, Consumer<JSONObject> progress,
                Consumer<Analyzer.Outcome> done) {
        this.host = host;
        Configuration cfg = new Configuration(host.getResources().getConfiguration());
        cfg.setLocale(Locale.forLanguageTag(language));
        context = host.createConfigurationContext(cfg);
        this.progress = progress;
        this.done = done;
        this.main = analyzer.main;
        cycle = new LiveAnalyzer(analyzer, this::update, this::complete);

        layout = new LinearLayout(host);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setBackgroundColor(Color.BLACK);
        layout.setPadding(dp(8), dp(8), dp(8), dp(8));
        status = new TextView(host);
        status.setTextColor(Color.WHITE);
        status.setTextSize(16);
        status.setText(context.getString(R.string.live_preparing));
        layout.addView(status, new LinearLayout.LayoutParams(-1, -2));
        preview = new TextureView(host);
        preview.setContentDescription(context.getString(R.string.live_preview));
        layout.addView(preview, new LinearLayout.LayoutParams(-1, 0, 1));
        LinearLayout controls = new LinearLayout(host);
        finish = makeButton(R.string.live_finish, this::finish);
        finish.setEnabled(false);
        cancel = makeButton(R.string.live_cancel, this::cancel);
        controls.addView(finish, new LinearLayout.LayoutParams(0, -2, 1));
        controls.addView(cancel, new LinearLayout.LayoutParams(0, -2, 1));
        layout.addView(controls, new LinearLayout.LayoutParams(-1, -2));
    }

    View view() { return layout; }
    void open(Runnable removed) {
        onRemoved = removed;
        if (!cycle.start(this::openPreview)) {
            complete(Analyzer.Outcome.error("engine_busy", 0));
            return;
        }
        main.postDelayed(openTimeout, 15_000);
    }

    private final Runnable openTimeout = () -> { if (captureStarted == 0) stop("camera_unavailable"); };
    private final Runnable tick = new Runnable() {
        @Override public void run() {
            if (closing.get() || captureStarted == 0) return;
            if (Analyzer.elapsed(captureStarted) >= LiveAnalyzer.LIMIT_MS) { finish(); return; }
            paint();
            main.postDelayed(this, 250);
        }
    };

    private void openPreview() {
        if (closing.get()) return;
        TextureView.SurfaceTextureListener listener = new TextureView.SurfaceTextureListener() {
            @Override public void onSurfaceTextureAvailable(SurfaceTexture texture, int w, int h) { launchCamera(texture); }
            @Override public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int w, int h) { transform(); }
            @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
                if (!closing.get()) cancel();
                return true;
            }
            @Override public void onSurfaceTextureUpdated(SurfaceTexture texture) { }
        };
        preview.setSurfaceTextureListener(listener);
        if (preview.isAvailable() && preview.getSurfaceTexture() != null) launchCamera(preview.getSurfaceTexture());
    }

    private void launchCamera(SurfaceTexture texture) {
        final int display = host.getWindowManager().getDefaultDisplay().getRotation() * 90;
        CameraThread.HANDLER.post(() -> openCamera(texture, display));
    }

    @SuppressLint("MissingPermission") // The activity checks and requests CAMERA before creating this overlay.
    private void openCamera(SurfaceTexture texture, int display) {
        if (closing.get()) return;
        try {
            CameraManager manager = context.getSystemService(CameraManager.class);
            String rear = null;
            CameraCharacteristics info = null;
            for (String id : manager.getCameraIdList()) {
                CameraCharacteristics candidate = manager.getCameraCharacteristics(id);
                Integer lens = candidate.get(CameraCharacteristics.LENS_FACING);
                if (lens != null && lens == CameraCharacteristics.LENS_FACING_BACK) { rear = id; info = candidate; break; }
            }
            if (rear == null || info == null) { failed("camera_unavailable"); return; }
            StreamConfigurationMap map = info.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Integer orientation = info.get(CameraCharacteristics.SENSOR_ORIENTATION);
            if (map == null || orientation == null) { failed("camera_unavailable"); return; }
            Size[] yuv = map.getOutputSizes(ImageFormat.YUV_420_888), screens = map.getOutputSizes(SurfaceTexture.class);
            if (yuv == null || screens == null) { failed("camera_unavailable"); return; }
            Size size = Arrays.stream(yuv).filter(s -> s.getWidth() <= 1280 && s.getHeight() <= 960)
                    .filter(s -> Arrays.asList(screens).contains(s))
                    .min(Comparator.comparingLong(s -> Math.abs((long) s.getWidth() * s.getHeight() - 640L * 480)))
                    .orElse(null);
            if (size == null) { failed("camera_unavailable"); return; }
            dimensions = size;
            rotation = Math.floorMod(orientation - display, 360);
            if (rotation % 90 != 0) { failed("camera_unavailable"); return; }
            if (Build.VERSION.SDK_INT >= 31) {
                int[] modes = info.get(CameraCharacteristics.SCALER_AVAILABLE_ROTATE_AND_CROP_MODES);
                disableRotateAndCrop = modes != null && Arrays.stream(modes)
                        .anyMatch(mode -> mode == CaptureRequest.SCALER_ROTATE_AND_CROP_NONE);
            }
            texture.setDefaultBufferSize(size.getWidth(), size.getHeight());
            screen = new Surface(texture);
            reader = ImageReader.newInstance(size.getWidth(), size.getHeight(), ImageFormat.YUV_420_888, 3);
            PoseInput geometry = new PoseInput(size.getWidth(), size.getHeight(), rotation);
            MediaFormat color = new MediaFormat();
            reader.setOnImageAvailableListener(source -> frame(source, geometry, color), CameraThread.HANDLER);
            main.post(this::transform);
            manager.openCamera(rear, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice device) {
                    if (closing.get()) { device.close(); return; }
                    camera = device;
                    configure(device);
                }
                @Override public void onDisconnected(CameraDevice device) { device.close(); failed("camera_unavailable"); }
                @Override public void onError(CameraDevice device, int error) { device.close(); failed("camera_unavailable"); }
            }, CameraThread.HANDLER);
        } catch (SecurityException e) { failed("forbidden"); }
        catch (Exception e) { failed("camera_unavailable"); }
    }

    private void configure(CameraDevice device) {
        try {
            CaptureRequest.Builder request = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            if (Build.VERSION.SDK_INT >= 31 && disableRotateAndCrop) {
                request.set(CaptureRequest.SCALER_ROTATE_AND_CROP, CaptureRequest.SCALER_ROTATE_AND_CROP_NONE);
            }
            request.addTarget(screen);
            request.addTarget(reader.getSurface());
            device.createCaptureSession(Arrays.asList(screen, reader.getSurface()), new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession session) {
                    if (closing.get()) { session.close(); return; }
                    cameraSession = session;
                    try {
                        session.setRepeatingRequest(request.build(), null, CameraThread.HANDLER);
                        cycle.beginCapture();
                        captureStarted = SystemClock.elapsedRealtime();
                        main.post(() -> {
                            if (closing.get()) return;
                            main.removeCallbacks(openTimeout);
                            finish.setEnabled(true);
                            tick.run();
                        });
                    } catch (Exception e) { failed("camera_unavailable"); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession session) {
                    session.close(); failed("camera_unavailable");
                }
            }, CameraThread.HANDLER);
        } catch (Exception e) { failed("camera_unavailable"); }
    }

    private void frame(ImageReader source, PoseInput geometry, MediaFormat color) {
        float[] rgb = null;
        try (Image im = source.acquireLatestImage()) {
            if (im == null || closing.get() || captureStarted == 0) return;
            long stamp = im.getTimestamp();
            if (stamp <= 0 || stamp - lastConverted < 70_000_000L) return;
            rgb = cycle.acquire();
            if (rgb == null) return;
            geometry.fill(im, color, rgb);
            lastConverted = stamp;
            cycle.submit(rgb, geometry, stamp);
            rgb = null;
        } catch (Exception e) {
            if (!closing.get()) failed("camera_unavailable");
        } finally { if (rgb != null) cycle.recycle(rgb); }
    }

    private void transform() {
        Size size = dimensions;
        int w = preview.getWidth(), h = preview.getHeight(), turn = rotation;
        if (size == null || w == 0 || h == 0 || closing.get()) return;
        int sourceW = turn % 180 == 0 ? size.getWidth() : size.getHeight();
        int sourceH = turn % 180 == 0 ? size.getHeight() : size.getWidth();
        float scale = Math.min(w / (float) sourceW, h / (float) sourceH);
        float left = (w - sourceW * scale) / 2, top = (h - sourceH * scale) / 2;
        float right = w - left, bottom = h - top;
        float[] target;
        switch (turn) {
            case 90: target = new float[]{right, top, right, bottom, left, top}; break;
            case 180: target = new float[]{right, bottom, left, bottom, right, top}; break;
            case 270: target = new float[]{left, bottom, left, top, right, bottom}; break;
            default: target = new float[]{left, top, right, top, left, bottom};
        }
        Matrix matrix = new Matrix();
        matrix.setPolyToPoly(new float[]{0, 0, w, 0, 0, h}, 0, target, 0, 3);
        preview.setTransform(matrix);
    }

    private void update(JSONObject detail) {
        if (closing.get()) return;
        seen = detail.optInt("uniqueFrames");
        visible = detail.optBoolean("visibleNow");
        JSONObject p = detail.optJSONObject("provisional");
        JSONObject attempt = p == null || p.optJSONArray("attempts") == null ? null : p.optJSONArray("attempts").optJSONObject(0);
        preliminary = attempt == null ? "" : attempt.optString("label");
        paint();
        progress.accept(detail);
    }

    private void paint() {
        int seconds = (int) Math.min(30, Analyzer.elapsed(captureStarted) / 1000);
        String arm = context.getString(visible ? R.string.live_visible : R.string.live_not_visible);
        String text = context.getString(R.string.live_progress, seconds, seen, arm);
        int verdict = "complete".equals(preliminary) ? R.string.live_complete
                : "incomplete".equals(preliminary) ? R.string.live_incomplete
                : "uncertain".equals(preliminary) ? R.string.live_uncertain : 0;
        if (verdict != 0) text += "\n" + context.getString(R.string.live_preliminary, context.getString(verdict));
        status.setText(text);
    }

    void finish() { if (captureStarted == 0) cancel(); else stop("finish"); }
    void cancel() { stop("cancelled"); }
    private void failed(String code) { main.post(() -> stop(code)); }
    private void stop(String code) {
        if (closing.compareAndSet(false, true)) {
            finish.setEnabled(false); cancel.setEnabled(false);
            status.setText(context.getString(R.string.live_finishing));
            main.removeCallbacks(tick);
            main.removeCallbacks(openTimeout);
            CameraThread.HANDLER.post(this::releaseCamera);
        }
        cycle.terminate(code);
    }

    private void releaseCamera() {
        if (cameraSession != null) { cameraSession.close(); cameraSession = null; }
        if (camera != null) { camera.close(); camera = null; }
        if (reader != null) { reader.close(); reader = null; }
        if (screen != null) { screen.release(); screen = null; }
    }

    private void complete(Analyzer.Outcome result) {
        if (closing.compareAndSet(false, true)) CameraThread.HANDLER.post(this::releaseCamera);
        main.removeCallbacks(tick);
        main.removeCallbacks(openTimeout);
        if (layout.getParent() instanceof ViewGroup) ((ViewGroup) layout.getParent()).removeView(layout);
        onRemoved.run();
        done.accept(result);
    }

    private Button makeButton(int label, Runnable action) {
        Button b = new Button(host);
        b.setText(context.getString(label)); b.setAllCaps(false); b.setMinHeight(dp(48));
        b.setOnClickListener(v -> action.run());
        return b;
    }
    private int dp(int value) { return Math.round(context.getResources().getDisplayMetrics().density * value); }
}
