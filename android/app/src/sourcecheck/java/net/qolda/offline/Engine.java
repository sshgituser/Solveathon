package net.qolda.offline;

import android.content.res.AssetManager;

/** Unlinked source-check APK: never claims readiness or produces a result. */
final class Engine implements AutoCloseable {
    static Engine open(AssetManager assets) { throw unavailable(); }
    void pose(float[] rgb, float[] simccX, float[] simccY) { throw unavailable(); }
    float logit(float[] features) { throw unavailable(); }
    @Override public void close() { }

    private static IllegalStateException unavailable() {
        return new IllegalStateException("model_unavailable");
    }
}