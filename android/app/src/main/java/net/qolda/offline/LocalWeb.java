package net.qolda.offline;

import android.content.res.AssetManager;
import android.net.Uri;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.HashMap;
import java.util.Collections;

/** A packaged, synthetic HTTPS origin. An unhandled request never reaches a network stack. */
final class LocalWeb {
    static final Uri ORIGIN = Uri.parse("https://appassets.androidplatform.net");
    static final String ENTRY = ORIGIN + "/frontend/index.html";
    private static final String ENTRY_PATH = "/frontend/index.html";
    private static final String CSP = "default-src 'self'; connect-src 'self'; media-src 'self' blob:; "
            + "img-src 'self' data: blob:; style-src 'self'; font-src 'self'; script-src 'self'; "
            + "base-uri 'none'; object-src 'none'; frame-src 'none'; frame-ancestors 'none'; form-action 'none'";

    private LocalWeb() { }

    static boolean isEntry(Uri uri) { return own(uri) && ENTRY_PATH.equals(uri.getPath()) && uri.getEncodedQuery() == null; }

    private static boolean own(Uri uri) {
        return "https".equals(uri.getScheme()) && ORIGIN.getEncodedAuthority().equals(uri.getEncodedAuthority());
    }

    static WebResourceResponse serve(AssetManager assets, WebResourceRequest request) {
        Uri uri = request.getUrl();
        if (!"GET".equals(request.getMethod()) || !own(uri) || uri.getEncodedQuery() != null) return blocked(403);
        if (request.isForMainFrame() && !isEntry(uri)) return blocked(403);
        String path = uri.getEncodedPath();
        if (path == null || !path.matches("/[A-Za-z0-9_./-]+") || path.contains("//") || path.contains("..")
                || !(path.startsWith("/frontend/") || path.startsWith("/assets/") || path.equals("/android/bridge.js"))) {
            return blocked(404);
        }
        String mime = mime(path);
        if (mime == null) return blocked(404);
        Map<String, String> headers = new HashMap<>();
        headers.put("X-Content-Type-Options", "nosniff");
        headers.put("Cross-Origin-Resource-Policy", "same-origin");
        if (path.equals(ENTRY_PATH)) {
            headers.put("Content-Security-Policy", CSP);
            headers.put("Cache-Control", "no-store");
        }
        try {
            String charset = mime.startsWith("text/") ? "UTF-8" : null;
            return new WebResourceResponse(mime, charset, 200, "OK", headers, assets.open("web" + path));
        } catch (IOException ex) { return blocked(404); }
    }

    private static String mime(String path) {
        String extension = path.substring(path.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT);
        switch (extension) {
            case "html": return "text/html";
            case "js": return "text/javascript";
            case "css": return "text/css";
            case "svg": return "image/svg+xml";
            case "png": return "image/png";
            case "jpg": case "jpeg": return "image/jpeg";
            case "webp": return "image/webp";
            case "glb": return "model/gltf-binary";
            case "ttf": return "font/ttf";
            case "woff2": return "font/woff2";
            case "json": return "application/json";
            default: return null;
        }
    }

    private static WebResourceResponse blocked(int status) {
        return new WebResourceResponse("text/plain", "UTF-8", status, status == 403 ? "Forbidden" : "Not Found",
                Collections.singletonMap("Cache-Control", "no-store"), new ByteArrayInputStream(new byte[0]));
    }
}
