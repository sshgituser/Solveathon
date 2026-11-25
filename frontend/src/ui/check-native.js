// Android owns the SAF grant, decoding and the camera surface. The web view gets only metadata
// and wire results. A grant belongs to one mounted check, never to preferences or history.
import { h } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { MAX_BYTES, TYPES } from "../core/api.js";
import { androidBridge } from "../platform/android.js";
import { ct } from "../i18n/check-atlas.js";
import { icon } from "./icons.js";

const text = (value) => typeof value === "string" && value.trim().length > 0;
const badReply = () => ({ kind: "error", status: 200, code: "bad_response" });
const noEngine = () => ({ kind: "error", status: 503, code: "model_unavailable" });
const bestEffort = (fn) => { try { Promise.resolve(fn()).catch(() => {}); } catch (_) {} };
const release = (token) => { if (text(token)) bestEffort(() => androidBridge.release(token)); };

// Validate the public single-cycle wire contract before a final result is allowed into the store.
export function nativeResult(data) {
  const labels = ["complete", "incomplete", "uncertain"], reasons = ["low_confidence", "low_visibility", "unscorable"];
  if (!data || data.version !== "seriq.v1" || data.model !== "scratch" || !text(data.model_version) ||
      !Array.isArray(data.attempts) || data.attempts.length > 1 || !Array.isArray(data.warnings) ||
      !data.warnings.every((w) => typeof w === "string") || data.counts?.total !== data.attempts.length) return false;
  const a = data.attempts[0];
  if (!labels.every((label) => data.counts[label] === (a?.label === label ? 1 : 0))) return false;
  if (!data.attempts.length) return data.warnings.includes("no_pose");
  return a?.index === 1 && labels.includes(a.label) && !data.warnings.includes("no_pose") &&
    (a.confidence === null || (Number.isFinite(a.confidence) && a.confidence >= 0 && a.confidence <= 1)) &&
    (a.reason === null || reasons.includes(a.reason)) && (a.label !== "uncertain" || reasons.includes(a.reason));
}

export function createNativeSources({ current, onMode, onFile, describeError }) {
  let generation = 0, grant = null, pending = false, shown = false;
  const source = (id, shape, title, hint, fn) => h("button", { type: "button", id, class: "c-phone-source", on: { click: fn } },
    h("span", { class: "c-source-icon" }, icon(shape)),
    h("span", { class: "c-source-copy" }, h("strong", { text: ct(title) }), h("span", { text: ct(hint) })), icon("arrow"));
  const status = h("p", { class: "c-picker-status", role: "status" });
  const camera = source("c-phone-camera", "camera", "phoneCamera", "phoneCameraHint", () => {
    if (!shown || !current() || document.hidden) return;
    clear(); onMode("live"); // Select a source, not a camera permission or an analysis request.
  });
  const picker = source("clip", "upload", "phoneFile", "phoneFileHint", pick);
  const el = h("div", { class: "c-phone-sources" }, camera, picker, status, h("p", { class: "fine", text: ct("phoneLimits") }));
  function clear() {
    ++generation; pending = false; picker.disabled = false; status.textContent = "";
    const old = grant; grant = null; if (old) release(old.token);
  }
  async function pick() {
    if (pending || !shown || !current() || document.hidden) return;
    clear(); const mine = generation; pending = true; picker.disabled = true; status.textContent = ct("phonePicking");
    onMode("file");
    let res;
    try { res = await androidBridge.pick(); } catch (_) { res = { kind: "error", code: "picker_unavailable" }; }
    const v = res?.video;
    if (mine !== generation || !shown || !current()) { release(v?.token); return; }
    pending = false; picker.disabled = false;
    if (res?.kind !== "ok") { release(v?.token); status.textContent = res?.kind === "canceled" ? ct("phoneNoFile") : describeError(res || badReply()); return; }
    if (!text(v?.token) || typeof v.name !== "string" || !TYPES.includes(v.type) || !Number.isSafeInteger(v.size) ||
        v.size < 1 || v.size > MAX_BYTES || typeof v.hash !== "string" || !/^[0-9a-f]{64}$/.test(v.hash)) {
      release(v?.token); status.textContent = t("check.invalidVideo"); return;
    }
    grant = { token: v.token }; status.textContent = "";
    // Only safe display metadata leaves this closure; the file hash is the deduplication key.
    onFile({ name: v.name, type: v.type, size: v.size, hash: v.hash });
  }
  return { el, clear, suspend: clear, get picking() { return pending; },
    show(on) { shown = on; if (!on && pending) clear(); },
    analyze(signal) { return grant && !signal.aborted ? androidBridge.analyze(grant.token, "scratch", signal) : Promise.resolve({ kind: "canceled" }); }
  };
}

export function createNativeLivePanel({ current, onChange, onResult }) {
  let job = null;
  const status = h("p", { class: "c-live-message", role: "status" });
  const clock = h("p", { class: "c-phone-clock", role: "timer", "aria-live": "off" });
  const values = ["liveFrame", "livePhase", "provisional"].map(() => h("dd"));
  const facts = h("dl", { class: "c-live-facts", "aria-live": "polite" }, ["liveFrame", "livePhase", "provisional"].map((k, i) => h("div", null, h("dt", { text: ct(k) }), values[i])));
  const finish = h("button", { type: "button", class: "btn btn--cta", text: ct("liveFinish"), on: { click: finishCycle } });
  const cancel = h("button", { type: "button", class: "btn btn--quiet", text: t("common.cancel"), on: { click: () => {
    if (!job) return; close(true); onResult({ kind: "canceled" });
  } } });
  const el = h("div", { class: "c-live c-phone-live", hidden: true }, status, clock, facts, h("div", { class: "actions" }, finish, cancel));
  function progress(e) {
    const p = e.detail;
    if (!job || !current() || document.hidden || p?.version !== "seriq.live.v1" || !["collecting", "ready"].includes(p.phase) ||
      !Number.isFinite(p.elapsedMs) || p.elapsedMs < 0 || typeof p.visibleNow !== "boolean" || !Number.isSafeInteger(p.uniqueFrames) || p.uniqueFrames < 1) return;
    if (job.finished || p.uniqueFrames < job.last) return;
    job.last = p.uniqueFrames; finish.disabled = false;
    const title = ct("liveRunning"); if (status.textContent !== title) status.textContent = title;
    clock.textContent = ct("elapsed", { n: Math.min(30, Math.floor(p.elapsedMs / 1000)) }) + " · " + ct("phoneFrames", { n: p.uniqueFrames });
    const preview = nativeResult(p.provisional) ? p.provisional : null;
    [ct(p.visibleNow ? "visible" : "notVisible"), ct(p.phase), preview ? (preview.attempts[0] ? t("label." + preview.attempts[0].label) : t("check.noPose")) : ct("noPreview")]
      .forEach((v, i) => { if (values[i].textContent !== v) values[i].textContent = v; });
  }
  function close(cancelRun) {
    if (!job) return;
    const mine = job; job = null; window.removeEventListener("qolda:live-progress", progress);
    if (cancelRun) { bestEffort(() => androidBridge.liveCancel()); mine.ac.abort(); }
    el.hidden = true; clock.textContent = status.textContent = ""; values.forEach((v) => { v.textContent = ""; }); onChange();
  }
  async function finishCycle() {
    const mine = job; if (!mine || mine.finished || finish.disabled) return;
    mine.finished = true; finish.disabled = true; status.textContent = ct("phoneWait");
    let ack; try { ack = await androidBridge.liveFinish(); } catch (_) {}
    // An acknowledgement is NOT a result; the liveStart promise is the only final channel.
    if (job === mine && ack?.kind !== "ok") { mine.finished = false; finish.disabled = false; status.textContent = ct("phoneFinishFailed"); }
  }
  async function start() {
    if (job || !current() || document.hidden) return;
    const mine = job = { ac: new AbortController(), finished: false, last: 0 };
    el.hidden = false; status.textContent = ct("phoneOpen"); finish.disabled = true;
    [ct("waiting"), ct("collecting"), ct("noPreview")].forEach((v, i) => { values[i].textContent = v; });
    window.addEventListener("qolda:live-progress", progress); onChange(); cancel.focus({ preventScroll: true });
    let res; try { res = await androidBridge.liveStart("scratch", mine.ac.signal); } catch (_) { res = noEngine(); }
    if (job !== mine || mine.ac.signal.aborted) return;
    close(false); if (!current()) return;
    onResult(res?.kind === "ok" && !text(res.key) ? badReply() : res || badReply());
  }
  return { el, start, get active() { return !!job; }, stop() { close(true); } };
}
