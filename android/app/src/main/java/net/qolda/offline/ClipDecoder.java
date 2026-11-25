package net.qolda.offline;

import android.media.Image;
import android.media.MediaCodec;
import android.media.MediaCodecInfo;
import android.media.MediaCodecList;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.os.SystemClock;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

/** Counts source packets, then feeds the first video track once in stream order. */
final class ClipDecoder {
    private static final int MAX_FRAMES = 3600, MAX_PACKETS = 9000;
    private static final long MAX_DURATION_US = 30_000_000L;
    private static final long MAX_PIXELS = 4096L * 2160;
    private static final long STALL_MS = 12_000;
    private static final String AVC = "video/avc", HEVC = "video/hevc", MPEG4 = "video/mp4v-es";
    private static final String AV1 = "video/av01", VP8 = "video/x-vnd.on2.vp8", VP9 = "video/x-vnd.on2.vp9";

    interface FrameConsumer {
        void accept(int index, int total, PoseInput input, float[] rgb) throws ClipException;
    }

    static Probe probe(ClipSource source, Cancel cancel) throws ClipException, IOException {
        MediaExtractor extractor = new MediaExtractor();
        try {
            source.bind(extractor);
            int nTracks = extractor.getTrackCount(), video = -1;
            long duration = 0;
            for (int i = 0; i < nTracks; i++) {
                MediaFormat format = extractor.getTrackFormat(i);
                String type = format.getString(MediaFormat.KEY_MIME);
                if (type != null && type.startsWith("video/") && video == -1) video = i;
                if (format.containsKey(MediaFormat.KEY_DURATION)) duration = Math.max(duration, format.getLong(MediaFormat.KEY_DURATION));
            }
            if (video < 0) throw new ClipException("invalid_video");
            if (duration > MAX_DURATION_US) throw new ClipException("duration_limit");
            MediaFormat format = extractor.getTrackFormat(video);
            String mime = format.getString(MediaFormat.KEY_MIME);
            boolean mp4 = source.mime.equals("video/mp4");
            if (!(AV1.equals(mime) || (mp4 && (AVC.equals(mime) || HEVC.equals(mime) || MPEG4.equals(mime)))
                    || (!mp4 && (VP8.equals(mime) || VP9.equals(mime))))) throw new ClipException("unsupported_media");
            int w = integer(format, MediaFormat.KEY_WIDTH), h = integer(format, MediaFormat.KEY_HEIGHT);
            if (w <= 0 || h <= 0 || w > 4096 || h > 4096 || (long) w * h > MAX_PIXELS) {
                throw new ClipException("invalid_video");
            }
            int rotation = Math.floorMod(integer(format, MediaFormat.KEY_ROTATION), 360);
            if (rotation % 90 != 0) throw new ClipException("unsupported_media");
            for (int i = 0; i < nTracks; i++) extractor.selectTrack(i);
            int packets = 0, nativeFrames = 0;
            long[] timestamps = new long[MAX_FRAMES];
            for (;;) {
                cancel.check();
                int track = extractor.getSampleTrackIndex();
                if (track < 0) break;
                if (++packets > MAX_PACKETS) throw new ClipException("invalid_video");
                if (track == video) {
                    if (nativeFrames == MAX_FRAMES) throw new ClipException("invalid_video");
                    long pts = extractor.getSampleTime();
                    if (pts < 0 || extractor.getSampleSize() <= 0) throw new ClipException("invalid_video");
                    int flags = extractor.getSampleFlags();
                    if ((flags & (MediaExtractor.SAMPLE_FLAG_ENCRYPTED | MediaExtractor.SAMPLE_FLAG_PARTIAL_FRAME)) != 0) {
                        throw new ClipException("unsupported_media");
                    }
                    timestamps[nativeFrames++] = pts;
                }
                if (!extractor.advance()) break;
            }
            if (nativeFrames == 0) throw new ClipException("invalid_video");
            timestamps = Arrays.copyOf(timestamps, nativeFrames);
            Arrays.sort(timestamps);
            long span = timestamps[nativeFrames - 1] - timestamps[0];
            if (span > MAX_DURATION_US) throw new ClipException("duration_limit");
            // Check presentation span as well as declared duration; infer a final frame interval if needed.
            double rate = frameRate(format);
            long extra = rate > 0 ? Math.round(1_000_000.0 / rate) : finalInterval(timestamps);
            if (duration == 0) duration = span + extra;
            else duration = Math.max(duration, span);
            if (duration > MAX_DURATION_US) throw new ClipException("duration_limit");
            if (duration <= 0) throw new ClipException("invalid_video");
            return new Probe(video, mime, w, h, rotation, duration, timestamps);
        } finally { extractor.release(); }
    }

    private static long finalInterval(long[] pts) {
        for (int i = pts.length - 1; i > 0; i--) if (pts[i] > pts[i - 1]) return pts[i] - pts[i - 1];
        return 0;
    }

    static void frames(ClipSource source, Probe probe, Cancel cancel, FrameConsumer consumer) throws ClipException, IOException {
        boolean[] inferred = {false};
        FrameConsumer tracked = (i, n, input, rgb) -> {
            inferred[0] = true;
            consumer.accept(i, n, input, rgb);
        };
        try {
            decodeTrack(source, probe, cancel, tracked, null);
        } catch (ClipException e) {
            if (inferred[0] || !"unsupported_media".equals(e.code)) throw e;
            String software = platformSoftwareDecoder(probe.codec);
            if (software == null) throw e;
            cancel.check();
            // The platform decoder can provide a readable YUV plane when a vendor codec can't.
            // No graph falls back, and no frame is sent to the learned extractor twice.
            decodeTrack(source, probe, cancel, tracked, software);
        }
    }

    private static String platformSoftwareDecoder(String mime) {
        for (MediaCodecInfo candidate : new MediaCodecList(MediaCodecList.REGULAR_CODECS).getCodecInfos()) {
            String name = candidate.getName();
            if (candidate.isEncoder() || !(name.startsWith("c2.android.") || name.startsWith("OMX.google."))) continue;
            for (String type : candidate.getSupportedTypes()) if (mime.equalsIgnoreCase(type)) return name;
        }
        return null;
    }

    private static void decodeTrack(ClipSource source, Probe probe, Cancel cancel, FrameConsumer consumer, String codecName)
            throws ClipException, IOException {
        MediaExtractor extractor = new MediaExtractor();
        MediaCodec decoder = null;
        try {
            source.bind(extractor);
            extractor.selectTrack(probe.track);
            MediaFormat format = extractor.getTrackFormat(probe.track);
            format.setInteger(MediaFormat.KEY_ROTATION, 0);
            format.setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible);
            try { decoder = codecName == null ? MediaCodec.createDecoderByType(probe.codec) : MediaCodec.createByCodecName(codecName); }
            catch (IllegalArgumentException | IOException e) { throw new ClipException("unsupported_media"); }
            try { decoder.configure(format, null, null, 0); }
            catch (IllegalArgumentException | MediaCodec.CodecException e) { throw new ClipException("unsupported_media"); }
            decoder.start();
            decode(extractor, decoder, probe, format, cancel, consumer);
        } finally {
            try { if (decoder != null) decoder.release(); }
            finally { extractor.release(); }
        }
    }

    private static void decode(MediaExtractor extractor, MediaCodec decoder, Probe probe, MediaFormat format,
                               Cancel cancel, FrameConsumer consumer) throws ClipException {
        int[] targets = Forearm.phaseIndices(probe.timestamps.length);
        float[] rgb = new float[PoseInput.W * PoseInput.H * 3];
        PoseInput input = new PoseInput(probe.width, probe.height, probe.rotation);
        MediaCodec.BufferInfo info = new MediaCodec.BufferInfo();
        MediaFormat outputFormat = format;
        int frame = 0, sampled = 0, fed = 0;
        long progress = SystemClock.elapsedRealtime();
        boolean inputEnd = false, outputEnd = false;
        while (!outputEnd) {
            cancel.check();
            if (SystemClock.elapsedRealtime() - progress > STALL_MS) throw new ClipException("invalid_video");
            if (!inputEnd) {
                int at = decoder.dequeueInputBuffer(2_000);
                if (at >= 0) {
                    long size = extractor.getSampleSize();
                    if (size < 0) {
                        decoder.queueInputBuffer(at, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM);
                        inputEnd = true;
                    } else {
                        ByteBuffer buffer = decoder.getInputBuffer(at);
                        if (++fed > probe.timestamps.length || buffer == null || size <= 0 || size > buffer.capacity()) {
                            throw new ClipException("invalid_video");
                        }
                        buffer.clear();
                        int sample = extractor.readSampleData(buffer, 0);
                        if (sample != size) throw new ClipException("invalid_video");
                        decoder.queueInputBuffer(at, 0, sample, extractor.getSampleTime(), 0);
                        extractor.advance();
                    }
                    progress = SystemClock.elapsedRealtime();
                }
            }
            int out = decoder.dequeueOutputBuffer(info, 5_000);
            if (out == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                outputFormat = decoder.getOutputFormat();
                int width = integer(outputFormat, MediaFormat.KEY_WIDTH), height = integer(outputFormat, MediaFormat.KEY_HEIGHT);
                // Codec allocation may pad edges. It may not silently switch to a larger stream.
                if (width > probe.width + 64 || height > probe.height + 64) throw new ClipException("invalid_video");
                continue;
            }
            if (out < 0) continue;
            boolean selected = false;
            try {
                if (info.size > 0 && (info.flags & MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0) {
                    if (frame >= probe.timestamps.length || info.presentationTimeUs != probe.timestamps[frame]) {
                        // Never substitute a nearby timestamp or count the same decoder output twice.
                        throw new ClipException("invalid_video");
                    }
                    selected = sampled < targets.length && frame == targets[sampled];
                    if (selected) {
                        try (Image image = decoder.getOutputImage(out)) {
                            if (image == null) throw new ClipException("unsupported_media");
                            input.fill(image, outputFormat, rgb);
                        }
                    }
                    frame++;
                }
                outputEnd = (info.flags & MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0;
            } finally { decoder.releaseOutputBuffer(out, false); }
            if (selected) consumer.accept(sampled++, targets.length, input, rgb);
            progress = SystemClock.elapsedRealtime();
        }
        if (fed != probe.timestamps.length || frame != fed || sampled != targets.length) {
            throw new ClipException("invalid_video");
        }
    }

    private static int integer(MediaFormat f, String key) { return f.containsKey(key) ? f.getInteger(key) : 0; }

    private static double frameRate(MediaFormat format) {
        if (!format.containsKey(MediaFormat.KEY_FRAME_RATE)) return 0;
        try { return format.getInteger(MediaFormat.KEY_FRAME_RATE); }
        catch (ClassCastException e) { return format.getFloat(MediaFormat.KEY_FRAME_RATE); }
    }

    static final class Probe {
        final int track, width, height, rotation;
        final long durationUs;
        final String codec;
        final long[] timestamps;
        Probe(int track, String codec, int width, int height, int rotation, long durationUs, long[] timestamps) {
            this.track = track; this.codec = codec; this.width = width; this.height = height;
            this.rotation = rotation; this.durationUs = durationUs; this.timestamps = timestamps;
        }
    }
}
