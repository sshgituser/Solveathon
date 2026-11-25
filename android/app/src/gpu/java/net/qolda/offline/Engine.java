package net.qolda.offline;

import android.content.res.AssetManager;
import net.qolda.engine.GpuEngine;

/** Used only on the analysis worker. T-018 owns the GPU placement checks. */
final class Engine implements AutoCloseable {
    private final GpuEngine engine;

    private Engine(GpuEngine engine) { this.engine = engine; }

    static Engine open(AssetManager assets) throws Exception {
        return new Engine(GpuEngine.open(assets));
    }

    void pose(float[] rgb, float[] simccX, float[] simccY) throws Exception {
        engine.inferPose(rgb, simccX, simccY);
    }

    float logit(float[] features) throws Exception { return engine.inferLogit(features); }

    @Override public void close() throws Exception { engine.close(); }
}