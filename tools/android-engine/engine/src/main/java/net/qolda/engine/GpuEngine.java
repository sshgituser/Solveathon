package net.qolda.engine;

import android.content.res.AssetManager;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * Strict-GPU Qolda engine: RTMPose (SimCC) and the E02 forearm classifier.
 *
 * <p>Both graphs are compiled with {@code kLiteRtHwAcceleratorGpu} only. {@link #open} refuses to
 * return an engine unless each compiled graph reports {@code LiteRtCompiledModelIsNonCpuFullyAccelerated}
 * and every graph node is delegated to the non-CPU accelerator. There is no CPU or NPU fallback: a
 * partially delegated graph is closed instead of being executed.
 *
 * <p>The caller supplies the verified assets ({@code qolda/engine.json} and the two FP32
 * {@code .tflite} files from the private T-017 kit) and passes this library's {@link AssetManager}.
 * The engine never reads video, never writes files and requests no permission.
 *
 * <p>One instance owns one worker thread's worth of buffers; calls are serialised. {@link #close()} is
 * idempotent and invalidates the engine for further inference.
 */
public final class GpuEngine implements AutoCloseable {
    public static final int POSE_INPUT = 1 * 256 * 192 * 3;
    public static final int SIMCC_X = 1 * 17 * 384;
    public static final int SIMCC_Y = 1 * 17 * 512;
    public static final int FEATURES = 1 * 64 * 5;

    private static final String MANIFEST = "qolda/engine.json";

    static {
        System.loadLibrary("qolda_engine");
    }

    private long handle;

    private GpuEngine(long handle) {
        this.handle = handle;
    }

    /**
     * Verifies the manifest and model bytes, then compiles both graphs on the GPU.
     *
     * @throws EngineException {@code model_unavailable} when an asset is missing, does not match the
     *     recorded SHA-256/size, or the GPU cannot host both graphs completely.
     */
    public static GpuEngine open(AssetManager assets) throws EngineException {
        if (assets == null) {
            throw new EngineException(EngineException.INVALID_ARGUMENT, "assets == null");
        }
        final JSONObject manifest;
        try {
            manifest = new JSONObject(readText(assets, MANIFEST));
        } catch (Exception error) {
            throw new EngineException(EngineException.MODEL_UNAVAILABLE, "engine manifest is unreadable", error);
        }
        byte[] pose = verifiedAsset(assets, manifest, "pose");
        byte[] classifier = verifiedAsset(assets, manifest, "classifier");
        final long created;
        try {
            created = nativeOpen(pose, classifier);
        } catch (EngineException error) {
            throw error;
        } catch (RuntimeException error) {
            throw new EngineException(EngineException.MODEL_UNAVAILABLE, "GPU engine did not start", error);
        }
        if (created == 0L) {
            throw new EngineException(EngineException.MODEL_UNAVAILABLE, "GPU engine did not start");
        }
        return new GpuEngine(created);
    }

    private static byte[] verifiedAsset(AssetManager assets, JSONObject manifest, String name)
            throws EngineException {
        try {
            JSONObject graph = manifest.getJSONObject("graphs").getJSONObject(name);
            String path = graph.getString("path");
            long expectedBytes = graph.getLong("bytes");
            String expectedSha = graph.getString("sha256");
            if (!"float32".equals(graph.getString("dtype")) || !"serving_default".equals(graph.getString("signature"))) {
                throw new EngineException(EngineException.MODEL_UNAVAILABLE, name + ": unexpected dtype or signature");
            }
            // engine.json records an APK member; AssetManager.open is rooted *inside* assets/.
            String member = "assets/qolda/" + ("pose".equals(name) ? "rtmpose-fp32.tflite" : "a-tcn-fp32.tflite");
            if (!member.equals(path) || expectedBytes <= 0 || expectedBytes > (64L << 20)) {
                throw new EngineException(EngineException.MODEL_UNAVAILABLE, name + ": invalid asset declaration");
            }
            byte[] data = new byte[(int) expectedBytes];
            try (java.io.DataInputStream in = new java.io.DataInputStream(assets.open(member.substring("assets/".length())))) {
                in.readFully(data);
                if (in.read() != -1) {
                    throw new EngineException(EngineException.MODEL_UNAVAILABLE, name + ": size does not match the manifest");
                }
            }
            if (!sha256(data).equals(expectedSha)) {
                throw new EngineException(EngineException.MODEL_UNAVAILABLE, name + ": SHA-256 does not match the manifest");
            }
            return data;
        } catch (EngineException error) {
            throw error;
        } catch (Exception error) {
            throw new EngineException(EngineException.MODEL_UNAVAILABLE, name + ": asset cannot be verified", error);
        }
    }

    /** Runs RTMPose on one normalised NHWC RGB frame ({@code 1x256x192x3}). */
    public synchronized void inferPose(float[] rgbNhwc, float[] simccX, float[] simccY) throws EngineException {
        requireOpen();
        if (rgbNhwc == null || rgbNhwc.length != POSE_INPUT || simccX == null || simccX.length != SIMCC_X
                || simccY == null || simccY.length != SIMCC_Y) {
            throw new EngineException(EngineException.INVALID_ARGUMENT, "pose buffer length");
        }
        nativeInferPose(handle, rgbNhwc, simccX, simccY);
    }

    /** Runs the E02 classifier on the compact forearm trajectory ({@code 1x64x5}) and returns logit. */
    public synchronized float inferLogit(float[] features) throws EngineException {
        requireOpen();
        if (features == null || features.length != FEATURES) {
            throw new EngineException(EngineException.INVALID_ARGUMENT, "classifier buffer length");
        }
        return nativeInferLogit(handle, features);
    }

    @Override
    public synchronized void close() {
        if (handle != 0L) {
            nativeClose(handle);
            handle = 0L;
        }
    }

    private void requireOpen() throws EngineException {
        if (handle == 0L) {
            throw new EngineException(EngineException.CLOSED, "engine is closed");
        }
    }

    private static String readText(AssetManager assets, String path) throws IOException {
        return new String(readAll(assets, path), StandardCharsets.UTF_8);
    }

    private static byte[] readAll(AssetManager assets, String path) throws IOException {
        try (InputStream stream = assets.open(path); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[1 << 16];
            while (true) {
                int read = stream.read(buffer);
                if (read < 0) break;
                out.write(buffer, 0, read);
            }
            return out.toByteArray();
        }
    }

    private static String sha256(byte[] data) throws NoSuchAlgorithmException {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(data);
        StringBuilder text = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            text.append(Character.forDigit((value >> 4) & 0xF, 16));
            text.append(Character.forDigit(value & 0xF, 16));
        }
        return text.toString();
    }

    private static native long nativeOpen(byte[] poseModel, byte[] classifierModel) throws EngineException;

    private native void nativeInferPose(long handle, float[] input, float[] simccX, float[] simccY)
            throws EngineException;

    private native float nativeInferLogit(long handle, float[] features) throws EngineException;

    private native void nativeClose(long handle);
}
