package net.qolda.offline;

import android.os.SystemClock;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.function.Consumer;
import org.json.JSONException;
import org.json.JSONObject;

/** Latest-only RGB handoff; only compact measured landmarks survive a GPU frame. */
final class LiveAnalyzer {
    static final long LIMIT_MS = 30_000;
    private final Analyzer owner;
    private final Consumer<JSONObject> progress;
    private final Consumer<Analyzer.Outcome> done;
    private final ArrayDeque<float[]> pool = new ArrayDeque<>(3);
    private final ArrayList<Forearm.Observation> observed = new ArrayList<>();
    private final float[] sx = new float[17 * 384], sy = new float[17 * 512];
    private final long began = SystemClock.elapsedRealtime();
    private Frame latest;
    private boolean queued, finished, opened;
    private String terminal;
    private long captureMs, lastTimestamp, lastProgress, lastProvisional;
    private int rawRight;
    private Result provisional;

    LiveAnalyzer(Analyzer owner, Consumer<JSONObject> progress, Consumer<Analyzer.Outcome> done) {
        this.owner = owner;
        this.progress = progress;
        this.done = done;
        for (int i = 0; i < 3; i++) pool.add(new float[PoseInput.W * PoseInput.H * 3]);
    }

    boolean start(Runnable ready) {
        if (!owner.busy.compareAndSet(false, true)) return false;
        owner.worker.execute(() -> {
            try { owner.ensureEngine(); }
            catch (ClipException e) { terminate(e.code); return; }
            synchronized (this) { if (terminal != null) return; opened = true; }
            owner.main.post(() -> {
                synchronized (this) { if (terminal != null) return; }
                ready.run();
            });
        });
        return true;
    }

    synchronized void beginCapture() { if (opened && terminal == null && captureMs == 0) captureMs = SystemClock.elapsedRealtime(); }

    synchronized float[] acquire() {
        if (terminal != null || captureMs == 0 || Analyzer.elapsed(captureMs) >= LIMIT_MS) return null;
        return pool.pollFirst();
    }

    synchronized void recycle(float[] rgb) { pool.add(rgb); }

    synchronized void submit(float[] rgb, PoseInput input, long timestamp) {
        if (terminal != null || captureMs == 0 || Analyzer.elapsed(captureMs) >= LIMIT_MS || timestamp <= lastTimestamp) {
            pool.add(rgb); return;
        }
        lastTimestamp = timestamp;
        if (latest != null) pool.add(latest.rgb);
        latest = new Frame(rgb, input);
        schedule();
    }

    /** Only the "finish" transition can score one bounded cycle. */
    synchronized void terminate(String code) {
        if (finished || (terminal != null && !"finish".equals(terminal))) return;
        if (terminal != null && "finish".equals(code)) return;
        terminal = code;
        if (!"finish".equals(code) && latest != null) { pool.add(latest.rgb); latest = null; }
        schedule();
    }

    private void schedule() {
        if (!queued) { queued = true; owner.worker.execute(this::drain); }
    }

    private void drain() {
        Frame frame;
        String terminalNow;
        synchronized (this) {
            frame = latest; latest = null;
            terminalNow = terminal;
        }
        if (frame != null) {
            try {
                if (terminalNow == null || "finish".equals(terminalNow)) process(frame);
            } catch (ClipException e) { terminate(e.code); }
            finally { recycle(frame.rgb); }
        }
        synchronized (this) {
            if (latest != null) { owner.worker.execute(this::drain); return; }
            if (terminal == null) { queued = false; return; }
            terminalNow = terminal;
        }
        Analyzer.Outcome outcome = "finish".equals(terminalNow) ? score() : Analyzer.Outcome.error(terminalNow, Analyzer.elapsed(began));
        synchronized (this) {
            if (!terminalNow.equals(terminal)) outcome = Analyzer.Outcome.error(terminal, Analyzer.elapsed(began));
            finished = true;
        }
        Analyzer.Outcome finalValue = outcome;
        owner.main.post(() -> {
            owner.busy.set(false);
            done.accept(finalValue);
        });
    }

    private void process(Frame frame) throws ClipException {
        owner.pose(frame.rgb, sx, sy);
        Forearm.Observation sample = Forearm.observe(frame.input, sx, sy);
        observed.add(sample);
        if (sample.right) rawRight++;
        long now = SystemClock.elapsedRealtime();
        if (!sample.right) provisional = null;
        if (sample.right && rawRight >= Forearm.MIN_FRAMES && now - lastProvisional >= 750) {
            Forearm.Features f = aggregate();
            provisional = f.refusal == null ? owner.classify(f) : null;
            lastProvisional = now;
        }
        if (now - lastProgress < 200) return;
        lastProgress = now;
        try {
            long elapsed = Math.min(LIMIT_MS, Analyzer.elapsed(captureMs));
            JSONObject detail = new JSONObject().put("version", "seriq.live.v1").put("elapsedMs", elapsed)
                    .put("phase", provisional == null ? "collecting" : "ready").put("uniqueFrames", observed.size())
                    .put("rightVisibleFrames", rawRight).put("visibleNow", sample.right)
                    .put("provisional", provisional == null ? JSONObject.NULL : provisional.wire());
            owner.main.post(() -> {
                synchronized (this) { if (terminal != null) return; }
                progress.accept(detail);
            });
        } catch (JSONException ignored) { }
    }

    private Forearm.Features aggregate() {
        Forearm forearm = new Forearm();
        int[] indices = Forearm.phaseIndices(observed.size());
        for (int i = 0; i < indices.length; i++) forearm.add(i, indices.length, observed.get(indices[i]));
        return forearm.finish();
    }

    private Analyzer.Outcome score() {
        try {
            Forearm.Features f = aggregate();
            Result result = owner.classify(f);
            JSONObject evidence = new JSONObject().put("duration_ms", captureMs == 0 ? 0 : Math.min(LIMIT_MS, Analyzer.elapsed(captureMs)))
                    .put("processed_unique_frames", observed.size()).put("unique_samples", Math.min(Forearm.PHASES, observed.size()))
                    .put("right_pair_frames", f.rightCount).put("left_pair_frames", f.leftCount)
                    .put("usable_pair_frames", f.usableCount).put("classifier_ran", f.refusal == null)
                    .put("policy_refusal", f.refusal == null ? JSONObject.NULL : f.refusal);
            return new Analyzer.Outcome(result, null, false, evidence, Analyzer.elapsed(began));
        } catch (ClipException e) { return Analyzer.Outcome.error(e.code, Analyzer.elapsed(began)); }
        catch (JSONException e) { return Analyzer.Outcome.error("invalid_video", Analyzer.elapsed(began)); }
    }

    private static final class Frame {
        final float[] rgb;
        final PoseInput input;
        Frame(float[] rgb, PoseInput input) { this.rgb = rgb; this.input = input; }
    }
}
