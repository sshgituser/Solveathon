// Contract: android/ANDROID_BRIDGE.md. The Activity owns the picker, GPU, camera and SAF export.
// Resolve the injected facade at call time; its absence must never select a second transport.
const native = () => globalThis.QoldaAndroid || globalThis.window?.QoldaAndroid;
const error = (status, code) => ({ kind: "error", status, code });
const canceled = () => ({ kind: "canceled" });
const ok = () => ({ kind: "ok" });
const badReply = () => error(502, "bad_response");
const validToken = (token) => typeof token === "string" && token.length > 0 && token.length <= 256 && !/[\s:/\\\x00-\x1f]/.test(token);
const validResult = (d) => d?.version === "seriq.v1" && Array.isArray(d.attempts) && d.attempts.length <= 1 &&
  d.attempts.every((a) => a && ["complete", "incomplete", "uncertain"].includes(a.label));

async function command(method, args, { signal, unavailable = "model_unavailable", success = ok } = {}) {
  if (signal?.aborted) return canceled();
  let abort;
  try {
    const bridge = native();
    if (typeof bridge?.[method] !== "function") return error(503, unavailable);
    // Forward the *same* AbortSignal so the native job is canceled too. The local race also
    // invalidates a late success while the Activity is still establishing its message port.
    const stopped = signal && new Promise((resolve) => {
      abort = () => resolve(canceled()); signal.addEventListener("abort", abort, { once: true });
    });
    const job = bridge[method](...args);
    const r = await (stopped ? Promise.race([job, stopped]) : job);
    if (signal?.aborted || r?.kind === "canceled") return canceled();
    if (r?.kind === "error" && Number.isInteger(r.status) && typeof r.code === "string") return error(r.status, r.code);
    return r?.kind === "ok" ? success(r) : badReply();
  } catch (e) {
    return signal?.aborted || e?.name === "AbortError" ? canceled() : error(503, unavailable);
  } finally {
    if (abort) signal.removeEventListener("abort", abort);
  }
}

export const androidBridge = Object.freeze({
  async health() {
    try { const state = await native()?.health?.(); return state === "ok" || state === "nomodel" ? state : "down"; }
    catch (_) { return "down"; }
  },
  pick() {
    return command("pick", [], { unavailable: "picker_unavailable", success: async ({ video }) => {
      if (!validToken(video?.token)) return badReply();
      if (typeof video.name !== "string" || !["video/mp4", "video/webm"].includes(video.type) ||
          !Number.isSafeInteger(video.size) || video.size < 1 || video.size > 67_108_864 || typeof video.hash !== "string" || !/^[a-f0-9]{64}$/.test(video.hash)) {
        await androidBridge.release(video.token); // The UI cannot own or later release a rejected selection.
        return badReply();
      }
      const { token, name, type, size, hash } = video;
      return { kind: "ok", video: { token, name, type, size, hash } };
    } });
  },
  async analyze(token, model, signal) {
    if (signal?.aborted) return canceled();
    if (!validToken(token)) return error(400, "invalid_request");
    if (model !== "scratch") return error(503, "model_unavailable");
    return command("analyze", [token, model, signal], { signal, success: ({ data }) => validResult(data) ? { kind: "ok", data } : badReply() });
  },
  async release(token) { return validToken(token) ? command("release", [token]) : error(400, "invalid_request"); },
  async liveStart(model = "scratch", signal) {
    if (signal?.aborted) return canceled();
    if (model !== "scratch") return error(503, "model_unavailable");
    return command("liveStart", [model, signal], { signal, success: ({ data, key }) => validResult(data) && typeof key === "string" && key.length > 0
      ? { kind: "ok", data, key } : badReply() });
  },
  liveFinish: () => command("liveFinish", []),
  liveCancel: () => command("liveCancel", []),
  async exportData(data, name) {
    if (data?.format !== "seriq.export.v1" || typeof name !== "string" || !/^[A-Za-z0-9_-]{1,60}\.json$/.test(name)) return error(400, "invalid_request");
    // The host applies the same 1 MiB limit. Avoid transferring an oversized message to it.
    try { if (new TextEncoder().encode(JSON.stringify(data)).length > 1_048_576) return error(413, "too_large"); }
    catch (_) { return error(400, "invalid_request"); }
    return command("exportData", [data, name], { unavailable: "export_failed" });
  }
});
