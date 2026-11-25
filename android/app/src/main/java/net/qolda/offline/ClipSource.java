package net.qolda.offline;

import android.content.ContentResolver;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.media.MediaExtractor;
import android.net.Uri;
import android.os.ParcelFileDescriptor;
import android.system.ErrnoException;
import android.system.Os;
import android.system.OsConstants;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

/** Reads only the SAF grant. Streams the exact bytes; spools only non-seekable grants. */
final class ClipSource implements AutoCloseable {
    static final long MAX_BYTES = 64L * 1024 * 1024;
    private static final String SPOOL = "qolda-clip-";
    private static final String MP4 = "video/mp4", WEBM = "video/webm";
    final String sha256, mime;
    final long size;
    private final AssetFileDescriptor afd;
    private final File temporary;

    private ClipSource(AssetFileDescriptor afd, File temporary, String mime, Scan scan) {
        this.afd = afd;
        this.temporary = temporary;
        this.mime = mime;
        size = scan.size;
        sha256 = scan.sha256;
    }

    static void clearAbandonedSpools(File cache) {
        File[] abandoned = cache.listFiles((dir, name) -> name.startsWith(SPOOL) && name.endsWith(".tmp"));
        if (abandoned != null) for (File f : abandoned) f.delete();
    }

    static ClipSource open(Context context, Uri uri, Cancel cancel) throws ClipException, IOException {
        if (!ContentResolver.SCHEME_CONTENT.equals(uri.getScheme())) throw new ClipException("unsupported_media");
        ContentResolver resolver = context.getContentResolver();
        String declared = resolver.getType(uri);
        if (declared != null && !MP4.equals(declared) && !WEBM.equals(declared)
                && !"application/octet-stream".equals(declared) && !"video/*".equals(declared)) {
            throw new ClipException("unsupported_media");
        }
        AssetFileDescriptor afd = resolver.openAssetFileDescriptor(uri, "r", cancel.signal);
        if (afd == null) throw new ClipException("invalid_video");
        File temp = null;
        boolean handoff = false;
        try {
            if (afd.getLength() > MAX_BYTES) throw new ClipException("too_large");
            if (!rewind(afd)) temp = File.createTempFile(SPOOL, ".tmp", context.getCacheDir());
            Scan scan;
            try (OutputStream copy = temp == null ? null : new FileOutputStream(temp)) {
                scan = scan(afd, copy, cancel);
            }
            String actual = sniff(scan.prefix);
            if (actual == null) {
                if (scan.size < 16 && (MP4.equals(declared) || WEBM.equals(declared))) throw new ClipException("invalid_video");
                throw new ClipException("unsupported_media");
            }
            if ((MP4.equals(declared) || WEBM.equals(declared)) && !actual.equals(declared)) {
                throw new ClipException("unsupported_media");
            }
            ClipSource source = new ClipSource(afd, temp, actual, scan);
            handoff = true;
            return source;
        } finally {
            if (!handoff) {
                try { afd.close(); } finally { if (temp != null) temp.delete(); }
            }
        }
    }

    void bind(MediaExtractor extractor) throws IOException {
        if (temporary != null) extractor.setDataSource(temporary.getAbsolutePath());
        else extractor.setDataSource(afd.getFileDescriptor(), afd.getStartOffset(), size);
    }

    void checkUnchanged(Cancel cancel) throws IOException, ClipException {
        if (temporary != null) return;
        if (!rewind(afd)) throw new ClipException("invalid_video");
        Scan after = scan(afd, null, cancel);
        if (after.size != size || !after.sha256.equals(sha256)) throw new ClipException("invalid_video");
    }

    private static boolean rewind(AssetFileDescriptor afd) {
        try { Os.lseek(afd.getFileDescriptor(), 0, OsConstants.SEEK_SET); return true; }
        catch (ErrnoException ex) { return false; }
    }

    private static Scan scan(AssetFileDescriptor afd, OutputStream copy, Cancel cancel) throws IOException, ClipException {
        MessageDigest digest;
        try { digest = MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException e) { throw new IllegalStateException(e); }
        long bytes = 0;
        byte[] buf = new byte[64 * 1024], prefix = new byte[4096];
        int header = 0;
        long declared = afd.getDeclaredLength();
        try (ParcelFileDescriptor clone = ParcelFileDescriptor.dup(afd.getFileDescriptor());
             InputStream input = new AssetFileDescriptor(clone, afd.getStartOffset(), declared).createInputStream()) {
            cancel.watch(input);
            try {
                for (;;) {
                    cancel.check();
                    // Read at most one byte beyond the allowed maximum, even on providers with no size metadata.
                    int got = input.read(buf, 0, (int) Math.min(buf.length, MAX_BYTES - bytes + 1));
                    if (got < 0) break;
                    if (got == 0) throw new ClipException("invalid_video");
                    bytes += got;
                    if (bytes > MAX_BYTES) throw new ClipException("too_large");
                    int n = Math.min(prefix.length - header, got);
                    System.arraycopy(buf, 0, prefix, header, n);
                    header += n;
                    digest.update(buf, 0, got);
                    if (copy != null) copy.write(buf, 0, got);
                }
            } finally { cancel.unwatch(); }
        }
        if (bytes == 0 || (declared >= 0 && declared != bytes)) throw new ClipException("invalid_video");
        return new Scan(bytes, hex(digest.digest()), Arrays.copyOf(prefix, header));
    }

    private static String sniff(byte[] prefix) throws ClipException {
        if (prefix.length >= 8 && ascii(prefix, 4, 4).equals("ftyp")) {
            if (prefix.length < 16) throw new ClipException("invalid_video");
            // QuickTime brand is not the supported MP4 container.
            return ascii(prefix, 8, 4).equals("qt  ") ? null : MP4;
        }
        if (prefix.length >= 4 && (prefix[0] & 255) == 0x1a && (prefix[1] & 255) == 0x45
                && (prefix[2] & 255) == 0xdf && (prefix[3] & 255) == 0xa3) {
            int[] offset = {4};
            long size = vint(prefix, offset, false);
            if (size > prefix.length - offset[0]) throw new ClipException("invalid_video");
            int end = offset[0] + (int) size;
            while (offset[0] < end) {
                long id = vint(prefix, offset, true);
                long length = vint(prefix, offset, false);
                if (length > end - offset[0]) throw new ClipException("invalid_video");
                if (id == 0x4282) return length == 4 && ascii(prefix, offset[0], 4).equals("webm") ? WEBM : null;
                offset[0] += (int) length;
            }
            throw new ClipException("invalid_video");
        }
        return null;
    }

    private static long vint(byte[] bytes, int[] offset, boolean keepMarker) throws ClipException {
        if (offset[0] >= bytes.length) throw new ClipException("invalid_video");
        int head = bytes[offset[0]++] & 255;
        if (head == 0) throw new ClipException("invalid_video");
        int width = Integer.numberOfLeadingZeros(head) - 23;
        if (width > bytes.length - offset[0] + 1) throw new ClipException("invalid_video");
        long value = keepMarker ? head : head & (0xff >> width);
        for (int i = 1; i < width; i++) value = (value << 8) | (bytes[offset[0]++] & 255);
        return value;
    }

    private static String ascii(byte[] bytes, int offset, int length) {
        return new String(bytes, offset, length, StandardCharsets.US_ASCII);
    }

    private static String hex(byte[] bytes) {
        final String alphabet = "0123456789abcdef";
        StringBuilder out = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) out.append(alphabet.charAt((b & 255) >>> 4)).append(alphabet.charAt(b & 15));
        return out.toString();
    }

    @Override public void close() throws IOException {
        try { afd.close(); } finally { if (temporary != null) temporary.delete(); }
    }

    private static final class Scan {
        final long size;
        final String sha256;
        final byte[] prefix;
        Scan(long size, String sha256, byte[] prefix) { this.size = size; this.sha256 = sha256; this.prefix = prefix; }
    }
}
