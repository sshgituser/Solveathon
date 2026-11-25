/* Android-only transport, loaded before the common app. No paths, URIs or video bytes cross it. */
(() => {
  "use strict";
  const origin = "https://appassets.androidplatform.net";
  if (window.top !== window || location.origin !== origin) return;
  const timeout = (ms, fallback) => new Promise(resolve => setTimeout(() => resolve(fallback), ms));
  const error = (status, code) => ({ kind: "error", status, code });
  const cancelled = () => ({ kind: "canceled" });
  const pending = new Map();
  let next = 0;
  let attach;
  const connecting = new Promise(resolve => { attach = resolve; });

  function receive(event) {
    // postWebMessage from Android has no window source; a same-origin page is the only reachable document.
    if (event.data !== "qolda-android-v1" || event.ports.length !== 1 || event.source != null) return;
    if (event.origin && event.origin !== "null" && event.origin !== origin) return;
    window.removeEventListener("message", receive);
    const port = event.ports[0];
    port.onmessage = event => {
      try {
        const reply = JSON.parse(event.data);
        if (reply.event === "live-progress") {
          window.dispatchEvent(new CustomEvent("qolda:live-progress", { detail: reply.detail }));
          return;
        }
        const finish = pending.get(reply.id);
        if (finish) finish(reply.result);
      } catch (_) { /* unreadable messages never become a verdict */ }
    };
    port.start();
    attach(port);
    void request("hello");
  }
  window.addEventListener("message", receive);

  async function request(op, params = {}, signal) {
    if (signal?.aborted) return cancelled();
    const port = await Promise.race([connecting, timeout(10000, null)]);
    if (!port) return error(503, "model_unavailable");
    if (signal?.aborted) return cancelled();
    const id = ++next;
    return new Promise(resolve => {
      let timer = 0;
      const finish = result => {
        clearTimeout(timer);
        signal?.removeEventListener("abort", abort);
        pending.delete(id);
        resolve(result);
      };
      const abort = () => {
        try { port.postMessage(JSON.stringify({ op: "cancel", target: id })); } catch (_) { /* closing */ }
        finish(cancelled());
      };
      pending.set(id, finish);
      signal?.addEventListener("abort", abort, { once: true });
      if (op !== "pick" && op !== "export") timer = setTimeout(() => {
        try { port.postMessage(JSON.stringify({ op: "cancel", target: id })); } catch (_) { /* closing */ }
        finish(error(503, "model_unavailable"));
      }, op === "analyze" ? 180000 : op === "liveStart" ? 90000 : 25000);
      try { port.postMessage(JSON.stringify({ id, op, ...params })); }
      catch (_) { finish(error(503, "model_unavailable")); }
    });
  }

  const transport = Object.freeze({
    async health() {
      const res = await request("health");
      return res === "ok" || res === "nomodel" ? res : "down";
    },
    async exercises() {
      // Device capabilities only: no HTTP and no permission or camera prompt.
      const [health, rear] = await Promise.all([request("health"), request("rearCameraFeature")]);
      const file = health === "ok";
      return { version: "qolda.extra-exercise.v1", exercises: ["E01", "E02", "E03"].map(id => {
        const primary = id === "E02";
        return { id, role: primary ? "primary" : "additional", variants: primary ? ["scratch", "transfer"] : ["single"],
          readiness: { web: { file: false, live: false }, windows: { file: false, live: false },
            android: { file: primary && file, live: primary && file && rear === true } } };
      }) };
    },
    pick: () => request("pick"),
    analyze: (token, model, signal) => request("analyze", { token, model }, signal),
    release: token => request("release", { token }),
    liveStart: (model = "scratch", signal) => request("liveStart", { model, language: document.documentElement.lang }, signal),
    liveFinish: () => request("liveFinish"),
    liveCancel: () => request("liveCancel"),
    exportData: (data, name) => request("export", { text: JSON.stringify(data), name })
  });
  Object.defineProperty(window, "QoldaAndroid", { value: transport, writable: false, configurable: false });
  window.addEventListener("qolda:data-reset", () => { void request("reset"); });
})();
