package net.qolda.offline;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.os.OperationCanceledException;
import android.os.SystemClock;
import android.provider.OpenableColumns;
import android.util.Log;
import java.util.LinkedHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import org.json.JSONException;
import org.json.JSONObject;

/** One worker owns both learned graphs for file and camera. The common client owns history. */
final class Analyzer {
    static final String TAG = "QoldaReceipt";
    private static Analyzer instance;
    private final Context app;
    final Handler main = new Handler(Looper.getMainLooper());
    final ExecutorService worker = Executors.newSingleThreadExecutor(r -> new Thread(r, "qolda-analysis"));
    final AtomicBoolean busy = new AtomicBoolean();
    private final LinkedHashMap<String, Result> recent = new LinkedHashMap<>();
    private volatile boolean ready;
    private long lastOpenAttempt;
    private Engine engine;

    static synchronized Analyzer get(Context context) {
        if (instance == null) instance = new Analyzer(context.getApplicationContext());
        return instance;
    }

    private Analyzer(Context context) {
        app = context;
        worker.execute(() -> ClipSource.clearAbandonedSpools(app.getCacheDir()));
    }

    void prepare(Consumer<String> done) {
        if (!BuildConfig.ENGINE_LINKED) { done.accept("nomodel"); return; }
        if (ready) { done.accept("ok"); return; }
        worker.execute(() -> {
            try { ensureEngine(); } catch (ClipException ignored) { }
            main.post(() -> done.accept(ready ? "ok" : "nomodel"));
        });
    }

    Cancel stage(Uri uri, Consumer<Selection> done) {
        Cancel cancel = new Cancel();
        worker.execute(() -> {
            Selection result;
            try (ClipSource source = ClipSource.open(app, uri, cancel)) {
                String name = "";
                try (Cursor cursor = app.getContentResolver().query(uri, new String[]{OpenableColumns.DISPLAY_NAME},
                        null, null, null, cancel.signal)) {
                    if (cursor != null && cursor.moveToFirst() && !cursor.isNull(0)) name = cursor.getString(0);
                } catch (RuntimeException ignored) { }
                cancel.check();
                name = name.substring(0, Math.min(name.length(), 120)).replaceAll("[\\p{Cc}\\p{Cf}]", "").trim();
                result = new Selection(uri, name, source.mime, source.size, source.sha256, null);
            } catch (Exception e) { result = Selection.error(code(e, cancel)); }
            Selection value = result;
            main.post(() -> done.accept(cancel.cancelled() ? Selection.error("cancelled") : value));
        });
        return cancel;
    }

    Cancel start(Selection clip, Consumer<Outcome> done) {
        if (!busy.compareAndSet(false, true)) return null;
        Cancel cancel = new Cancel();
        worker.execute(() -> {
            Outcome out = run(clip, cancel);
            main.post(() -> {
                busy.set(false);
                done.accept(cancel.cancelled() ? Outcome.error("cancelled", out.ms) : out);
            });
        });
        return cancel;
    }

    void clearTransient() { worker.execute(recent::clear); }

    private Outcome run(Selection clip, Cancel cancel) {
        long start = SystemClock.elapsedRealtime();
        if (!BuildConfig.ENGINE_LINKED) return Outcome.error("model_unavailable", 0);
        try (ClipSource source = ClipSource.open(app, clip.uri, cancel)) {
            if (!clip.sha.equals(source.sha256)) throw new ClipException("invalid_video");
            Result prior = recent.get(source.sha256);
            if (prior != null && ready) return new Outcome(prior, null, true, null, elapsed(start));
            ClipDecoder.Probe probe = ClipDecoder.probe(source, cancel);
            cancel.check();
            boolean cold = !ready;
            long openMs = ensureEngine();
            Forearm forearm = new Forearm();
            float[] sx = new float[17 * 384], sy = new float[17 * 512];
            ClipDecoder.frames(source, probe, cancel, (i, n, im, rgb) -> {
                cancel.check();
                pose(rgb, sx, sy);
                forearm.add(i, n, Forearm.observe(im, sx, sy));
            });
            source.checkUnchanged(cancel);
            cancel.check();
            Forearm.Features f = forearm.finish();
            Result result = classify(f);
            cancel.check();
            if (result.label != null) {
                recent.put(source.sha256, result);
                if (recent.size() > 32) recent.remove(recent.keySet().iterator().next());
            }
            JSONObject details = new JSONObject().put("input_bytes", source.size)
                    .put("container", source.mime).put("codec", probe.codec)
                    .put("width", probe.width).put("height", probe.height).put("rotation", probe.rotation)
                    .put("duration_ms", probe.durationUs / 1000.0).put("native_frames", probe.timestamps.length)
                    .put("unique_samples", Math.min(Forearm.PHASES, probe.timestamps.length))
                    .put("left_pair_frames", f.leftCount).put("right_pair_frames", f.rightCount)
                    .put("usable_pair_frames", f.usableCount)
                    .put("policy_refusal", f.refusal == null ? JSONObject.NULL : f.refusal)
                    .put("engine_open_cold", cold).put("engine_open_ms", openMs).put("classifier_ran", f.refusal == null);
            return new Outcome(result, null, false, details, elapsed(start));
        } catch (Exception e) { return Outcome.error(code(e, cancel), elapsed(start)); }
    }

    long ensureEngine() throws ClipException {
        if (ready) return 0;
        long start = SystemClock.elapsedRealtime();
        if (!BuildConfig.ENGINE_LINKED || (lastOpenAttempt > 0 && start - lastOpenAttempt < 2000)) {
            throw new ClipException("model_unavailable");
        }
        lastOpenAttempt = start;
        try { engine = Engine.open(app.getAssets()); }
        catch (Exception | LinkageError failure) {
            if (BuildConfig.DEBUG) Log.w(TAG, "gpu_open_failed_ms=" + elapsed(start));
            throw new ClipException("model_unavailable");
        }
        ready = true;
        long ms = elapsed(start);
        if (BuildConfig.DEBUG) Log.i(TAG, "gpu_ready_ms=" + ms + " apk_version=" + BuildConfig.VERSION_NAME);
        return ms;
    }

    void pose(float[] rgb, float[] sx, float[] sy) throws ClipException {
        try {
            engine.pose(rgb, sx, sy);
            for (float f : sx) if (!Float.isFinite(f)) throw new IllegalStateException();
            for (float f : sy) if (!Float.isFinite(f)) throw new IllegalStateException();
        } catch (Exception | LinkageError failure) {
            retire();
            throw new ClipException("model_unavailable");
        }
    }

    Result classify(Forearm.Features f) throws ClipException {
        if (f.refusal != null) return Result.refused(f.refusal);
        try {
            float logit = engine.logit(f.values);
            if (!Float.isFinite(logit)) throw new IllegalStateException();
            return Result.score(logit);
        } catch (Exception | LinkageError failure) {
            retire();
            throw new ClipException("model_unavailable");
        }
    }

    private void retire() {
        ready = false;
        recent.clear();
        if (engine != null) {
            try { engine.close(); } catch (Exception | LinkageError ignored) { }
            engine = null;
        }
    }

    private static String code(Exception e, Cancel cancel) {
        if (cancel.cancelled() || e instanceof OperationCanceledException) return "cancelled";
        if (e instanceof ClipException) return ((ClipException) e).code;
        return e instanceof SecurityException ? "forbidden" : "invalid_video";
    }

    static long elapsed(long start) { return SystemClock.elapsedRealtime() - start; }

    static void report(Outcome outcome, String source) {
        if (!BuildConfig.DEBUG) return;
        try {
            JSONObject wire = outcome.error == null ? outcome.result.wire() : Result.wireError(outcome.error);
            JSONObject receipt = new JSONObject().put("schema", "qolda-android-run-v1")
                    .put("package", BuildConfig.APPLICATION_ID).put("apk_version", BuildConfig.VERSION_NAME)
                    .put("analysis_ms", outcome.ms).put("source", source).put("dedup_cache", outcome.cached)
                    .put("result", wire);
            if (outcome.details != null) receipt.put("clip_aggregates", outcome.details);
            Log.i(TAG, receipt.toString());
        } catch (JSONException e) { Log.w(TAG, "Receipt serialization failed"); }
    }

    static final class Selection {
        final Uri uri;
        final String name, mime, sha, error;
        final long size;

        Selection(Uri uri, String name, String mime, long size, String sha, String error) {
            this.uri = uri; this.name = name; this.mime = mime; this.size = size; this.sha = sha; this.error = error;
        }

        static Selection error(String code) { return new Selection(null, null, null, 0, null, code); }
    }

    static final class Outcome {
        final Result result;
        final String error;
        final boolean cached;
        final JSONObject details;
        final long ms;

        Outcome(Result result, String error, boolean cached, JSONObject details, long ms) {
            this.result = result; this.error = error; this.cached = cached; this.details = details; this.ms = ms;
        }

        static Outcome error(String code, long ms) { return new Outcome(null, code, false, null, ms); }
    }
}
