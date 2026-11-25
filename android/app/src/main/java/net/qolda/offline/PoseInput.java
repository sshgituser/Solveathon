package net.qolda.offline;

import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.media.Image;
import android.media.MediaFormat;
import java.nio.ByteBuffer;

/** Deterministic host preprocessing for ml/gpu.py::CudaPose; not a learned graph. */
final class PoseInput {
    static final int W = 192, H = 256;
    private static final float[] MEAN = {123.675f, 116.28f, 103.53f};
    private static final float[] STD = {58.395f, 57.12f, 57.375f};
    final int width, height;
    final double scale, left, top;
    private final int rawWidth, rawHeight, rotation;
    private final int[] xs = new int[W], ys = new int[H];

    PoseInput(int rawWidth, int rawHeight, int rotation) {
        this.rawWidth = rawWidth;
        this.rawHeight = rawHeight;
        this.rotation = rotation;
        width = rotation % 180 == 0 ? rawWidth : rawHeight;
        height = rotation % 180 == 0 ? rawHeight : rawWidth;
        scale = Math.min(W / (double) width, H / (double) height) / 1.25;
        left = (W - scale * width) / 2;
        top = (H - scale * height) / 2;
        // warpAffine receives float32 transform coefficients, with INTER_LINEAR's 1/32 pixel grid.
        double inverse = 1.0 / (float) scale;
        map(xs, inverse, -(float) left * inverse);
        map(ys, inverse, -(float) top * inverse);
    }

    private static void map(int[] values, double gain, double translate) {
        int base = (int) Math.rint(translate * 1024) + 16;
        for (int i = 0; i < values.length; i++) values[i] = (base + (int) Math.rint(gain * i * 1024)) >> 5;
    }

    void fill(Image image, MediaFormat format, float[] rgb) throws ClipException {
        if (image.getFormat() != ImageFormat.YUV_420_888) throw new ClipException("unsupported_media");
        Rect crop = image.getCropRect();
        if (crop.width() != rawWidth || crop.height() != rawHeight) throw new ClipException("invalid_video");
        PlaneYuv pixels = new PlaneYuv(image, format, crop);
        int out = 0;
        for (int yy : ys) for (int xx : xs) {
            int x = xx >> 5, y = yy >> 5, fx = xx & 31, fy = yy & 31;
            int a = pixel(pixels, x, y), b = pixel(pixels, x + 1, y);
            int c = pixel(pixels, x, y + 1), d = pixel(pixels, x + 1, y + 1);
            int wa = (32 - fx) * (32 - fy), wb = fx * (32 - fy), wc = (32 - fx) * fy, wd = fx * fy;
            for (int k = 0; k < 3; k++) {
                int shift = 16 - 8 * k;
                int v = (((a >> shift) & 255) * wa + ((b >> shift) & 255) * wb
                        + ((c >> shift) & 255) * wc + ((d >> shift) & 255) * wd + 512) >> 10;
                rgb[out++] = (v - MEAN[k]) / STD[k];
            }
        }
    }

    private int pixel(PlaneYuv pixels, int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0;
        switch (rotation) {
            case 90: return pixels.rgb(y, rawHeight - 1 - x);
            case 180: return pixels.rgb(rawWidth - 1 - x, rawHeight - 1 - y);
            case 270: return pixels.rgb(rawWidth - 1 - y, x);
            default: return pixels.rgb(x, y);
        }
    }

    private static final class PlaneYuv {
        private final ByteBuffer y, u, v;
        private final int yr, ur, vr, us, vs, left, top;
        private final int yOffset, yg, rv, gu, gv, bu;

        PlaneYuv(Image image, MediaFormat format, Rect crop) {
            Image.Plane[] p = image.getPlanes();
            y = p[0].getBuffer().slice(); u = p[1].getBuffer().slice(); v = p[2].getBuffer().slice();
            yr = p[0].getRowStride(); ur = p[1].getRowStride(); vr = p[2].getRowStride();
            us = p[1].getPixelStride(); vs = p[2].getPixelStride();
            left = crop.left; top = crop.top;
            int standard = value(format, MediaFormat.KEY_COLOR_STANDARD);
            double kr = .299, kb = .114;  // unspecified follows BT.601, like default swscale
            if (standard == MediaFormat.COLOR_STANDARD_BT709) { kr = .2126; kb = .0722; }
            else if (standard == MediaFormat.COLOR_STANDARD_BT2020) { kr = .2627; kb = .0593; }
            boolean full = value(format, MediaFormat.KEY_COLOR_RANGE) == MediaFormat.COLOR_RANGE_FULL;
            double cg = full ? 1.0 : 255.0 / 224;
            double kg = 1 - kr - kb;
            yOffset = full ? 0 : 16;
            yg = coefficient(full ? 1.0 : 255.0 / 219);
            rv = coefficient(2 * (1 - kr) * cg);
            gu = coefficient(2 * kb * (1 - kb) / kg * cg);
            gv = coefficient(2 * kr * (1 - kr) / kg * cg);
            bu = coefficient(2 * (1 - kb) * cg);
        }

        int rgb(int x, int row) {
            x += left; row += top;
            int luma = ((y.get(row * yr + x) & 255) - yOffset) * yg;
            int cu = (u.get((row / 2) * ur + (x / 2) * us) & 255) - 128;
            int cv = (v.get((row / 2) * vr + (x / 2) * vs) & 255) - 128;
            return (clamp(luma + rv * cv) << 16) | (clamp(luma - gu * cu - gv * cv) << 8) | clamp(luma + bu * cu);
        }

        private static int coefficient(double v) { return (int) Math.round(v * 16384); }
        private static int clamp(int v) { return Math.max(0, Math.min(255, (v + 8192) >> 14)); }
        private static int value(MediaFormat f, String key) { return f.containsKey(key) ? f.getInteger(key) : 0; }
    }
}
