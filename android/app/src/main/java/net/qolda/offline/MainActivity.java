package net.qolda.offline;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.webkit.PermissionRequest;
import android.webkit.RenderProcessGoneDetail;
import android.webkit.WebChromeClient;
import android.webkit.WebMessage;
import android.webkit.WebMessagePort;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.FrameLayout;
import android.widget.TextView;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.UUID;
import org.json.JSONException;
import org.json.JSONObject;

/** Android host for the existing bundled common frontend; no product screens are implemented here. */
public final class MainActivity extends Activity {
    private static final int PICK = 11, EXPORT = 12, CAMERA = 13;
    private FrameLayout root;
    private WebView web;
    private WebMessagePort bridge;
    private Analyzer analyzer;
    private Analyzer.Selection selection;
    private String selectionToken;
    private Request document, fileRun, liveRequest;
    private Cancel staging, fileCancel;
    private LiveOverlay live;
    private boolean foreground;

    @Override protected void onCreate(Bundle state) {
        super.onCreate(state);
        analyzer = Analyzer.get(this);
        root = new FrameLayout(this);
        web = new WebView(this);
        root.addView(web, new FrameLayout.LayoutParams(-1, -1));
        setContentView(root);
        WebView.setWebContentsDebuggingEnabled(false);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setAllowFileAccessFromFileURLs(false);
        s.setAllowUniversalAccessFromFileURLs(false);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        s.setBlockNetworkLoads(true);
        s.setSupportMultipleWindows(false);
        s.setJavaScriptCanOpenWindowsAutomatically(false);
        web.setWebChromeClient(new WebChromeClient() {
            @Override public void onPermissionRequest(PermissionRequest permission) { permission.deny(); }
        });
        web.setWebViewClient(new WebViewClient() {
            @Override public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest r) {
                return LocalWeb.serve(getAssets(), r);
            }
            @Override public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest r) {
                return !r.isForMainFrame() || !LocalWeb.isEntry(r.getUrl());
            }
            @Override public void onPageStarted(WebView view, String url, Bitmap icon) { disconnect(); }
            @Override public void doUpdateVisitedHistory(WebView view, String url, boolean reload) {
                String route = routeOf(url);
                if (fileRun != null && !route.equals(fileRun.route) && fileCancel != null) fileCancel.cancel();
                if (liveRequest != null && !route.equals(liveRequest.route)) {
                    if (live != null) live.cancel(); else { liveRequest.cancelled = true; answer(liveRequest, cancelled()); }
                }
            }
            @Override public void onPageFinished(WebView view, String url) {
                if (bridge != null || !isEntry(url) || !isEntry(view.getUrl())) return;
                WebMessagePort[] ports = view.createWebMessageChannel();
                bridge = ports[0];
                bridge.setWebMessageCallback(new WebMessagePort.WebMessageCallback() {
                    @Override public void onMessage(WebMessagePort port, WebMessage msg) {
                        if (port == bridge && isEntry(view.getUrl())) message(port, msg.getData());
                    }
                });
                // This API targets the main frame. CSP prohibits embedded frames, and navigation is sealed.
                view.postWebMessage(new WebMessage("qolda-android-v1", new WebMessagePort[]{ports[1]}), LocalWeb.ORIGIN);
                if (BuildConfig.DEBUG) Log.i(Analyzer.TAG, "web_document_ready version=" + BuildConfig.VERSION_NAME
                        + " elapsed_since_process_ms=" + (SystemClock.elapsedRealtime() - Process.getStartElapsedRealtime()));
            }
            @Override public boolean onRenderProcessGone(WebView view, RenderProcessGoneDetail detail) {
                disconnect();
                root.removeView(view);
                view.destroy();
                web = null;
                TextView fallback = new TextView(MainActivity.this);
                fallback.setText(R.string.renderer_unavailable);
                root.addView(fallback);
                return true;
            }
        });
        web.loadUrl(LocalWeb.ENTRY);
    }

    private static boolean isEntry(String url) { return url != null && LocalWeb.isEntry(Uri.parse(url)); }
    private static String routeOf(String url) {
        String fragment = url == null ? null : Uri.parse(url).getFragment();
        return fragment == null ? "" : fragment;
    }

    private void message(WebMessagePort port, String message) {
        if (message == null || message.length() > 2_200_000) return;
        long id = 0;
        try {
            JSONObject obj = new JSONObject(message);
            String op = obj.getString("op");
            if ("cancel".equals(op)) { cancel(obj.optLong("target", -1)); return; }
            id = obj.getLong("id");
            if (id <= 0) return;
            Request r = new Request(port, id);
            r.route = routeOf(web == null ? null : web.getUrl());
            switch (op) {
                case "hello":
                    if (BuildConfig.DEBUG) Log.i(Analyzer.TAG, "main_frame_bridge_ready");
                    answer(r, object("kind", "ok"));
                    analyzer.prepare(status -> { });
                    break;
                case "health": analyzer.prepare(status -> answer(r, status)); break;
                case "rearCameraFeature": answer(r, getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA)); break;
                case "pick": pick(r); break;
                case "analyze": analyze(r, obj.optString("token"), obj.optString("model")); break;
                case "release":
                    if (obj.optString("token").equals(selectionToken)) clearSelection();
                    answer(r, object("kind", "ok"));
                    break;
                case "reset":
                    clearSelection();
                    analyzer.clearTransient();
                    if (document != null) { document.cancelled = true; answer(document, cancelled()); }
                    if (staging != null) staging.cancel();
                    if (fileCancel != null) fileCancel.cancel();
                    if (live != null) live.cancel();
                    else if (liveRequest != null) { liveRequest.cancelled = true; answer(liveRequest, cancelled()); }
                    answer(r, object("kind", "ok"));
                    break;
                case "export": export(r, obj.optString("name"), obj.getString("text")); break;
                case "liveStart": startLive(r, obj.optString("model"), obj.optString("language")); break;
                case "liveFinish":
                    if (live == null) answer(r, error("invalid_request"));
                    else { live.finish(); answer(r, object("kind", "ok")); }
                    break;
                case "liveCancel":
                    if (live != null) live.cancel();
                    else if (liveRequest != null) { liveRequest.cancelled = true; answer(liveRequest, cancelled()); }
                    answer(r, object("kind", "ok"));
                    break;
                default: answer(r, error("invalid_request"));
            }
        } catch (JSONException | IllegalArgumentException e) {
            if (id > 0) answer(new Request(port, id), error("invalid_request"));
        }
    }

    private void pick(Request r) {
        if (!foreground || document != null || staging != null || fileRun != null || liveRequest != null) {
            answer(r, error("engine_busy")); return;
        }
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
                .setType("video/*").putExtra(Intent.EXTRA_LOCAL_ONLY, true)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        launchDocument(r, intent, PICK);
    }

    private void export(Request r, String name, String text) throws JSONException {
        if (!foreground || document != null || staging != null || fileRun != null || liveRequest != null) {
            answer(r, error("engine_busy")); return;
        }
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        if (bytes.length > 1_048_576) { answer(r, error("too_large")); return; }
        if (!"seriq.export.v1".equals(new JSONObject(text).optString("format"))) {
            answer(r, error("invalid_request")); return;
        }
        r.data = bytes;
        if (!name.matches("[A-Za-z0-9_-]{1,60}\\.json")) name = "qolda.json";
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE)
                .setType("application/json").putExtra(Intent.EXTRA_TITLE, name)
                .putExtra(Intent.EXTRA_LOCAL_ONLY, true).addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        launchDocument(r, intent, EXPORT);
    }

    private void launchDocument(Request r, Intent intent, int kind) {
        try { startActivityForResult(intent, kind); document = r; }
        catch (ActivityNotFoundException | SecurityException e) { answer(r, error("picker_unavailable")); }
    }

    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != PICK && request != EXPORT) return;
        Request r = document;
        document = null;
        if (r == null || r.port != bridge || r.cancelled) return;
        if (request == PICK && !r.route.equals(routeOf(web == null ? null : web.getUrl()))) { answer(r, cancelled()); return; }
        Uri uri = data == null ? null : data.getData();
        if (result != RESULT_OK || uri == null) { answer(r, cancelled()); return; }
        if (!"content".equals(uri.getScheme())) { answer(r, error("forbidden")); return; }
        if (request == EXPORT) {
            analyzer.worker.execute(() -> {
                boolean ok = false;
                try (OutputStream out = getContentResolver().openOutputStream(uri, "wt")) {
                    if (out != null) { out.write(r.data); ok = true; }
                } catch (Exception ignored) { ok = false; }
                boolean saved = ok;
                runOnUiThread(() -> answer(r, saved ? object("kind", "ok") : error("export_failed")));
            });
            return;
        }
        clearSelection();
        staging = analyzer.stage(uri, selected -> {
            staging = null;
            if (r.port != bridge || r.cancelled) return;
            if (selected.error != null) { answer(r, replyError(selected.error)); return; }
            selection = selected;
            selectionToken = UUID.randomUUID().toString();
            try {
                JSONObject video = new JSONObject().put("token", selectionToken).put("name", selected.name)
                        .put("type", selected.mime).put("size", selected.size).put("hash", selected.sha);
                answer(r, new JSONObject().put("kind", "ok").put("video", video));
            } catch (JSONException e) { answer(r, error("invalid_video")); }
        });
    }

    private void analyze(Request r, String token, String model) {
        if (!"scratch".equals(model)) { answer(r, error("model_unavailable")); return; }
        if (selection == null || !token.equals(selectionToken)) { answer(r, error("invalid_request")); return; }
        if (!foreground || document != null || staging != null || fileRun != null || liveRequest != null) {
            answer(r, error("engine_busy")); return;
        }
        fileCancel = analyzer.start(selection, outcome -> {
            fileCancel = null;
            fileRun = null;
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            if (r.port != bridge) return;
            Analyzer.Outcome finalResult = foreground ? outcome : Analyzer.Outcome.error("cancelled", outcome.ms);
            Analyzer.report(finalResult, "file");
            answer(r, wire(finalResult, null));
        });
        if (fileCancel == null) { answer(r, error("engine_busy")); return; }
        fileRun = r;
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void startLive(Request r, String model, String language) {
        if (!"scratch".equals(model) || !BuildConfig.ENGINE_LINKED) { answer(r, error("model_unavailable")); return; }
        if (!foreground || liveRequest != null || document != null || staging != null || fileRun != null) {
            answer(r, error("engine_busy")); return;
        }
        liveRequest = r;
        r.language = language.equals("kk") ? "kk" : "ru";
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            try { requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA); }
            catch (RuntimeException failure) { liveRequest = null; answer(r, error("forbidden")); }
        } else showLive(r);
    }

    @Override public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grants) {
        super.onRequestPermissionsResult(requestCode, permissions, grants);
        if (requestCode != CAMERA || liveRequest == null) return;
        Request r = liveRequest;
        if (r.cancelled || r.port != bridge) { liveRequest = null; return; }
        if (grants.length == 0 || grants[0] != PackageManager.PERMISSION_GRANTED) {
            liveRequest = null;
            answer(r, error("forbidden"));
        } else if (foreground) showLive(r);
        else r.permissionApproved = true;
    }

    private void showLive(Request r) {
        if (!foreground || r.cancelled) { liveRequest = null; answer(r, cancelled()); return; }
        String key = "live:" + UUID.randomUUID();
        live = new LiveOverlay(this, analyzer, r.language, progress -> event(r, "live-progress", progress), outcome -> {
            live = null;
            liveRequest = null;
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            if (r.port != bridge) return;
            Analyzer.Outcome result = foreground ? outcome : Analyzer.Outcome.error("cancelled", outcome.ms);
            Analyzer.report(result, "camera");
            answer(r, wire(result, key));
        });
        root.addView(live.view(), new FrameLayout.LayoutParams(-1, -1));
        if (web != null) web.setImportantForAccessibility(WebView.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        live.open(() -> { if (web != null) web.setImportantForAccessibility(WebView.IMPORTANT_FOR_ACCESSIBILITY_AUTO); });
    }

    private void cancel(long id) {
        if (fileRun != null && fileRun.id == id && fileCancel != null) fileCancel.cancel();
        if (liveRequest != null && liveRequest.id == id) {
            liveRequest.cancelled = true;
            if (live != null) live.cancel();
        }
    }

    private void clearSelection() { selection = null; selectionToken = null; }
    private void disconnect() {
        WebMessagePort old = bridge;
        bridge = null;
        if (old != null) try { old.close(); } catch (IllegalStateException ignored) { }
        clearSelection();
        if (staging != null) staging.cancel();
        if (fileCancel != null) fileCancel.cancel();
        if (live != null) live.cancel();
        if (liveRequest != null) liveRequest.cancelled = true;
    }

    @Override protected void onStart() {
        super.onStart();
        foreground = true;
        if (liveRequest != null && liveRequest.permissionApproved && live == null) showLive(liveRequest);
    }
    @Override public void onConfigurationChanged(Configuration config) {
        super.onConfigurationChanged(config);
        if (live != null) live.cancel();
    }
    @Override protected void onStop() {
        foreground = false;
        if (fileCancel != null) fileCancel.cancel();
        if (live != null) live.cancel();
        super.onStop();
    }
    @Override public void onBackPressed() {
        if (live != null) live.cancel();
        else if (fileCancel != null) fileCancel.cancel();
        else if (web != null && web.canGoBack()) web.goBack();
        else super.onBackPressed();
    }
    @Override protected void onDestroy() {
        disconnect();
        if (web != null) {
            ((ViewGroup) web.getParent()).removeView(web);
            web.destroy();
            web = null;
        }
        super.onDestroy();
    }

    private void answer(Request r, Object value) {
        try { send(r, new JSONObject().put("id", r.id).put("result", value)); }
        catch (JSONException ignored) { }
    }
    private void event(Request r, String kind, JSONObject detail) {
        try { send(r, new JSONObject().put("event", kind).put("detail", detail)); }
        catch (JSONException ignored) { }
    }
    private void send(Request r, JSONObject msg) {
        if (r.port != bridge) return;
        try { r.port.postMessage(new WebMessage(msg.toString())); }
        catch (IllegalStateException ignored) { }
    }
    private static JSONObject wire(Analyzer.Outcome out, String key) {
        if (out.error != null) return replyError(out.error);
        try {
            JSONObject result = new JSONObject().put("kind", "ok").put("data", out.result.wire());
            if (key != null) result.put("key", key);
            return result;
        } catch (JSONException e) { return error("invalid_video"); }
    }
    private static JSONObject replyError(String code) { return "cancelled".equals(code) ? cancelled() : error(code); }
    private static JSONObject cancelled() { return object("kind", "canceled"); }
    private static JSONObject object(String name, Object value) {
        try { return new JSONObject().put(name, value); }
        catch (JSONException impossible) { throw new IllegalStateException(impossible); }
    }
    private static JSONObject error(String code) {
        int status;
        switch (code) {
            case "invalid_request": status = 400; break;
            case "forbidden": status = 403; break;
            case "too_large": status = 413; break;
            case "unsupported_media": status = 415; break;
            case "duration_limit": case "invalid_video": status = 422; break;
            default: status = 503;
        }
        try { return new JSONObject().put("kind", "error").put("status", status).put("code", code); }
        catch (JSONException impossible) { throw new IllegalStateException(impossible); }
    }

    private static final class Request {
        final WebMessagePort port;
        final long id;
        boolean cancelled, permissionApproved;
        byte[] data;
        String language;
        String route = "";
        Request(WebMessagePort port, long id) { this.port = port; this.id = id; }
    }
}
