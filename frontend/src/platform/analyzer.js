// Analyzer port. Every implementation answers in the same shape (docs/contracts/api.md, seriq.v1),
// so screens do not know whether the clip was scored on this computer, by a service or in the browser.
//
//   health()  -> "ok" | "nomodel" | "down"
//   analyze(blob, type, model, signal)
//             -> { kind: "ok", data } | { kind: "error", status, code } | { kind: "canceled" } | { kind: "offline" }
//
// No implementation may invent a verdict: anything but a well-formed 200 is an error state.
import { profile } from "./profile.js";
import { androidBridge } from "./android.js";

/** HTTP implementation, used for the local engine (loopback) and for a hosted service (remote). */
function httpAnalyzer(baseUrl, headers = {}) {
  const base = () => String(baseUrl()).replace(/\/+$/, "");
  return {
    async health() {
      if (!base()) return "down";
      try {
        const j = await (await fetch(base() + "/api/v1/health", { cache: "no-store", headers })).json();
        const m = j.models || {};
        return m.scratch || m.transfer ? "ok" : "nomodel";
      } catch (_) { return "down"; }
    },
    async analyze(blob, type, model, signal) {
      if (!base()) return { kind: "offline" };
      try {
        const r = await fetch(base() + "/api/v1/analyze?model=" + encodeURIComponent(model) + "&mode=single", {
          method: "POST", body: blob, signal, cache: "no-store", headers: { ...headers, "Content-Type": type }
        });
        let j = null; try { j = await r.json(); } catch (_) {}
        if (r.ok && j && j.version === "seriq.v1" && Array.isArray(j.attempts)) return { kind: "ok", data: j };
        return { kind: "error", status: r.status, code: j && j.error && j.error.code };
      } catch (e) {
        return e.name === "AbortError" ? { kind: "canceled" } : { kind: "offline" };
      }
    }
  };
}

/** Placeholder for a kind that is planned but not built: it reports itself as unavailable, honestly. */
const notBuilt = { health: async () => "down", analyze: async () => ({ kind: "error", status: 503, code: "model_unavailable" }) };

/**
 * @param {() => string} loopbackUrl current address of the local engine (a user setting in the local profile)
 */
export function createAnalyzer(loopbackUrl) {
  const nativePort = {
    health: () => androidBridge.health(),
    analyze: (selection, _type, model, signal) => androidBridge.analyze(typeof selection === "string" ? selection : selection?.token, model, signal)
  };
  if (profile.analyzer === "ondevice") return nativePort;
  const web = profile.analyzer === "loopback" ? httpAnalyzer(loopbackUrl) : profile.analyzer === "remote" ? httpAnalyzer(() => profile.apiUrl) : notBuilt;
  // Recheck at the point of dispatch if an initially unidentified host attaches its native facade late.
  const active = () => profile.analyzer === "ondevice" ? nativePort : web;
  return { health: () => active().health(), analyze: (...args) => active().analyze(...args) };
}
