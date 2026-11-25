// LIVE transport for the local analyzer. Same-origin loopback only: the page must have been
// served by the same local engine that receives the frames, so no remote or static-origin
// deployment can use this by accident. Raw RGB24 frames, one request in flight, sequence numbers
// start at 1 and advance only after a successful answer. Nothing is recorded or stored here.
const VERSION = "seriq.live.v1";
const JSON_HEADERS = { "Content-Type": "application/json" };

function loopback(host) {
  return host === "127.0.0.1" || host === "localhost" || host === "::1" || host === "[::1]";
}

/** The engine origin when the page and the engine are the same loopback origin. */
export function liveBase(engineUrl) {
  try {
    const url = new URL(String(engineUrl || "").replace(/\/+$/, ""));
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (!loopback(url.hostname)) return null;
    if (url.origin !== location.origin) return null;
    return url.origin;
  } catch (_) {
    return null;
  }
}

function errorOf(status, j) {
  return { kind: "error", status, code: j && j.error && j.error.code ? j.error.code : null };
}

async function call(url, init, signal, timeout) {
  const ac = new AbortController();
  const abort = () => ac.abort();
  if (signal) {
    if (signal.aborted) ac.abort();
    else signal.addEventListener("abort", abort, { once: true });
  }
  const timer = setTimeout(abort, timeout || 10000);
  try {
    return await fetch(url, Object.assign({ cache: "no-store", signal: ac.signal }, init));
  } catch (_) {
    return null;
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener("abort", abort);
  }
}

async function json(r) {
  try { return await r.json(); } catch (_) { return null; }
}

export const live = {
  base: liveBase,

  async start(base, signal) {
    const r = await call(base + "/api/v1/live/start", { method: "POST", headers: JSON_HEADERS, body: "{}" }, signal, 10000);
    if (!r) return { kind: "offline" };
    const j = await json(r);
    if (!r.ok) return errorOf(r.status, j);
    if (!j || j.version !== VERSION || typeof j.session !== "string" || !j.session) return { kind: "error", status: r.status, code: "bad_response" };
    return { kind: "ok", session: j.session, version: j.version };
  },

  async frame(base, session, seq, w, h, rgb, signal) {
    const url = base + "/api/v1/live/frame?session=" + encodeURIComponent(session) + "&seq=" + seq + "&w=" + w + "&h=" + h;
    const r = await call(url, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: rgb
    }, signal, 15000);
    if (!r) return { kind: "offline" };
    const j = await json(r);
    if (!r.ok) return errorOf(r.status, j);
    if (!j || j.version !== VERSION || !Number.isInteger(j.frames) || typeof j.visible !== "boolean" || (j.phase !== "collecting" && j.phase !== "ready")) {
      return { kind: "error", status: r.status, code: "bad_response" };
    }
    return { kind: "ok", data: j };
  },

  async finish(base, session, signal) {
    const r = await call(base + "/api/v1/live/finish?session=" + encodeURIComponent(session), { method: "POST", headers: JSON_HEADERS, body: "{}" }, signal, 30000);
    if (!r) return { kind: "offline" };
    const j = await json(r);
    if (!r.ok) return errorOf(r.status, j);
    if (!j || j.version !== "seriq.v1" || !Array.isArray(j.attempts)) return { kind: "error", status: r.status, code: "bad_response" };
    return { kind: "ok", data: j };
  },

  /** Best effort: closing the session must not keep the preview alive or block navigation. */
  cancel(base, session, keepalive) {
    if (!base || !session) return Promise.resolve(false);
    return fetch(base + "/api/v1/live/cancel?session=" + encodeURIComponent(session), {
      method: "POST",
      headers: JSON_HEADERS,
      body: "{}",
      keepalive: !!keepalive,
      cache: "no-store"
    }).then((r) => r.ok).catch(() => false);
  }
};
