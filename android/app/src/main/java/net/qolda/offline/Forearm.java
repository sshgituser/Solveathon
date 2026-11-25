package net.qolda.offline;

import java.util.Arrays;

/** Image-plane COCO left/right elbow and wrist; no side substitution or gap filling. */
final class Forearm {
    static final int PHASES = 64;
    static final int MIN_FRAMES = 40;
    // COCO joint indices; ml/extract.py maps them to (13,14,15,16) in MediaPipe order.
    private static final int[] JOINTS = {7, 9, 8, 10};
    private final float[][] xy = new float[PHASES][8];
    private final boolean[] left = new boolean[PHASES];
    private final boolean[] right = new boolean[PHASES];

    static int[] phaseIndices(int count) {
        int[] indices = new int[Math.min(count, PHASES)];
        for (int i = 0; i < indices.length; i++) indices[i] = index(i, count, indices.length);
        return indices;
    }

    private static int index(int i, int extent, int count) {
        if (count <= 1) return 0;
        // np.linspace then np.rint: ties to even, unlike Math.round.
        return i == count - 1 ? extent - 1 : (int) Math.rint(i * ((extent - 1.0) / (count - 1)));
    }

    static Observation observe(PoseInput image, float[] sx, float[] sy) {
        float[] xy = new float[8];
        int span = Math.max(image.width, image.height);
        boolean[] observed = new boolean[4];
        for (int j = 0; j < JOINTS.length; j++) {
            int joint = JOINTS[j];
            int bx = argmax(sx, joint * 384, 384);
            int by = argmax(sy, joint * 512, 512);
            if (bx < 0 || by < 0) continue;
            float score = Math.min(sx[joint * 384 + bx], sy[joint * 512 + by]);
            double x = (bx * .5 - image.left) / (image.scale * image.width);
            double y = (by * .5 - image.top) / (image.scale * image.height);
            observed[j] = score >= .50f && x >= .01 && x <= .99 && y >= .01 && y <= .99;
            xy[j * 2] = (float) (x * image.width / span);
            xy[j * 2 + 1] = (float) (-y * image.height / span);
        }
        return new Observation(xy, observed[0] && observed[1], observed[2] && observed[3]);
    }

    void add(int sample, int count, Observation observation) {
        int slot = index(sample, PHASES, count);
        xy[slot] = observation.xy;
        left[slot] = observation.left;
        right[slot] = observation.right;
    }

    private static int argmax(float[] values, int start, int length) {
        int best = 0;
        float high = values[start];
        if (!Float.isFinite(high)) return -1;
        for (int i = 1; i < length; i++) {
            float value = values[start + i];
            if (!Float.isFinite(value)) return -1;
            if (value > high) { high = value; best = i; }
        }
        return best;
    }

    // Faithful projection of ml/data.py::forearm_sequence onto A's five input channels.
    Features finish() {
        boolean[] keep = new boolean[PHASES];
        float[] lengths = new float[PHASES], goodLengths = new float[PHASES];
        int nl = 0, nr = 0, validLengths = 0;
        for (int i = 0; i < PHASES; i++) {
            if (left[i]) nl++;
            if (!right[i]) continue;
            nr++;
            lengths[i] = distance(xy[i], 4, xy[i], 6);
            keep[i] = Float.isFinite(lengths[i]) && lengths[i] > 1e-5f;
            if (keep[i]) goodLengths[validLengths++] = lengths[i];
        }
        if (nr == 0) return new Features("no_visible_forearm_pair", nl, nr, 0, null);
        if (validLengths == 0) return new Features("degenerate_forearm_pair", nl, nr, 0, null);
        if (nr < MIN_FRAMES) return new Features("insufficient_visible_forearm_pair", nl, nr, 0, null);

        Arrays.sort(goodLengths, 0, validLengths);
        float scale = (goodLengths[(validLengths - 1) / 2] + goodLengths[validLengths / 2]) / 2f;
        float low = .60f * scale, high = 1.45f * scale;
        int[] keptInQuarter = new int[4];
        int count = 0, separate = 0, first = -1, prev = -1;
        boolean jump = false;
        for (int i = 0; i < PHASES; i++) {
            keep[i] &= lengths[i] >= low && lengths[i] <= high;
            if (!keep[i]) continue;
            count++;
            keptInQuarter[i / 16]++;
            if (first < 0) first = i;
            if (left[i]) {
                float length = distance(xy[i], 0, xy[i], 2);
                float separation = Math.max(distance(xy[i], 0, xy[i], 4), distance(xy[i], 2, xy[i], 6));
                if (length >= low && length <= high && separation > .45f * scale) separate++;
            }
            if (prev >= 0) {
                float step = Math.max(distance(xy[i], 4, xy[prev], 4), distance(xy[i], 6, xy[prev], 6));
                if (step > scale * Math.min(.20 * (i - prev), .55)) jump = true;
            }
            prev = i;
        }

        String refusal = null;
        if (nr - count > 6 || jump) refusal = "inconsistent_forearm_trajectory";
        else if (separate >= 8) refusal = "ambiguous_forearm_tracks";
        else if (count < MIN_FRAMES || Arrays.stream(keptInQuarter).min().orElse(0) < 6) {
            refusal = "incomplete_forearm_coverage";
        } else if (movement(keep, count) < .10 * scale) refusal = "static_forearm_trajectory";
        if (refusal != null) return new Features(refusal, nl, nr, 0, null);

        float[] features = new float[PHASES * 5];
        float gain = (float) (.30 / scale);
        for (int i = 0; i < PHASES; i++) if (keep[i]) {
            for (int k = 0; k < 4; k++) features[i * 5 + k] = (xy[i][4 + k] - xy[first][4 + k % 2]) * gain;
            features[i * 5 + 4] = 1f;
        }
        return new Features(null, nl, nr, count, features);
    }

    private double movement(boolean[] keep, int count) {
        float[] values = new float[count];
        double[] spans = new double[4];
        for (int k = 0; k < 4; k++) {
            int n = 0;
            for (int i = 0; i < PHASES; i++) if (keep[i]) values[n++] = xy[i][4 + k];
            Arrays.sort(values);
            spans[k] = percentile(values, .90) - percentile(values, .10);
        }
        return Math.max(Math.hypot(spans[0], spans[1]), Math.hypot(spans[2], spans[3]));
    }

    private static double percentile(float[] values, double p) {
        double pos = (values.length - 1) * p;
        int low = (int) pos;
        return values[low] + (values[(int) Math.ceil(pos)] - (double) values[low]) * (pos - low);
    }

    private static float distance(float[] a, int i, float[] b, int j) {
        float x = a[i] - b[j], y = a[i + 1] - b[j + 1];
        return (float) Math.sqrt(x * x + y * y);
    }

    static final class Observation {
        final float[] xy;
        final boolean left, right;
        Observation(float[] xy, boolean left, boolean right) { this.xy = xy; this.left = left; this.right = right; }
    }

    static final class Features {
        final String refusal;
        final int leftCount, rightCount, usableCount;
        final float[] values;

        Features(String refusal, int leftCount, int rightCount, int usableCount, float[] values) {
            this.refusal = refusal;
            this.leftCount = leftCount;
            this.rightCount = rightCount;
            this.usableCount = usableCount;
            this.values = values;
        }
    }
}