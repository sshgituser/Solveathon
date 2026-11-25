// A view controller, not a transport. All session requests go through the platform's live port.
import { h } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { ct } from "../i18n/check-atlas.js";
import { capture } from "../platform/camera.js";
import { profile } from "../platform/profile.js";
import { live, liveBase } from "../platform/live.js";
import { icon } from "./icons.js";

const LIMIT = 30000, GRACE = 5000, PERIOD = 100, SIDE = 320;
const stopTracks = (stream) => stream?.getTracks().forEach((track) => track.stop());
const failure = (code) => ({ kind: "error", code });

// An Android WebView needs a native live bridge, not HTTP to a second machine. Without the
// platform or a per-presented-frame callback the file path remains available, without probing.
export function availableLiveBase(url) {
  const proto = globalThis.HTMLVideoElement?.prototype;
  if (profile.analyzer !== "loopback" || /android/i.test(navigator.userAgent + " " + (navigator.userAgentData?.platform || ""))) return null;
  return capture.supported() && proto?.requestVideoFrameCallback && proto?.cancelVideoFrameCallback ? liveBase(url) : null;
}

export function createLivePanel(base, { validResult, onChange, onResult }) {
  let run = null;
  const video = h("video", { playsinline: true, muted: true, "aria-label": ct("liveCamera") });
  video.muted = true;
  const clock = h("p", { class: "c-countdown", role: "timer", "aria-live": "off" });
  const message = h("p", { role: "status", class: "c-live-message" });
  const values = ["liveFrame", "livePhase", "provisional"].map(() => h("dd"));
  const facts = h("dl", { class: "c-live-facts", "aria-live": "polite" }, ["liveFrame", "livePhase", "provisional"].map((key, i) =>
    h("div", null, h("dt", { text: ct(key) }), values[i])));
  const finish = h("button", { type: "button", class: "btn btn--cta", on: { click: () => finishRun(run) } }, ct("liveFinish"), icon("arrow"));
  const cancel = h("button", { type: "button", class: "btn btn--quiet", text: t("common.cancel"), on: { click: () => {
    if (run) report(run, { kind: "canceled" });
  } } });
  const el = h("div", { class: "c-live", hidden: true }, h("div", { class: "c-camera" }, video, clock),
    message, facts, h("div", { class: "actions" }, finish, cancel));

  function stopCapture(j) {
    clearTimeout(j.cutoff); clearInterval(j.clock);
    if (j.callback != null) { video.cancelVideoFrameCallback(j.callback); j.callback = null; }
    stopTracks(j.stream); j.stream = null; video.pause(); video.srcObject = null;
    if (j.canvas) { j.canvas.width = j.canvas.height = 0; j.ctx = j.canvas = null; }
  }
  function close(j, cancelSession) {
    if (run !== j) return;
    run = null; clearTimeout(j.expiry); stopCapture(j); j.ac.abort();
    if (cancelSession && j.session) live.cancel(base, j.session, true);
    j.session = null; el.hidden = true;
    values.forEach((node) => { node.textContent = ""; }); message.textContent = clock.textContent = "";
    onChange();
  }
  function report(j, result) {
    if (run !== j) return;
    close(j, result.kind !== "ok"); onResult(result);
  }
  function feedback(data) {
    const preview = data.preview;
    const label = preview && validResult(preview) ? preview.attempts[0]?.label : null;
    const texts = [ct(data.visible ? "visible" : "notVisible"), ct(data.phase),
      label ? t("label." + label) : preview && validResult(preview) ? t("check.noPose") : ct("noPreview")];
    // Only semantic changes are live announcements, never a fresh set of labels at 10 Hz.
    texts.forEach((text, i) => { if (values[i].textContent !== text) values[i].textContent = text; });
  }
  function queueFrame(j) {
    if (run !== j || j.phase !== "capture") return;
    j.callback = video.requestVideoFrameCallback((now, meta) => {
      j.callback = null;
      if (run !== j || j.phase !== "capture") return;
      if (now >= j.deadline) { limit(j); return; }
      if (now - j.lastSent < PERIOD || j.lastFrame === meta.presentedFrames || !video.videoWidth || !video.videoHeight) { queueFrame(j); return; }
      j.lastSent = now; j.lastFrame = meta.presentedFrames;
      j.pending = sendFrame(j).catch(() => report(j, failure("invalid_frame"))).finally(() => {
        j.pending = null; queueFrame(j);
      });
    });
  }
  async function sendFrame(j) {
    const scale = Math.min(1, SIDE / Math.max(video.videoWidth, video.videoHeight));
    const w = Math.max(1, Math.round(video.videoWidth * scale)), height = Math.max(1, Math.round(video.videoHeight * scale));
    if (j.canvas.width !== w) j.canvas.width = w;
    if (j.canvas.height !== height) j.canvas.height = height;
    j.ctx.drawImage(video, 0, 0, w, height); // unmirrored, aspect-preserving, latest presented frame
    const rgba = j.ctx.getImageData(0, 0, w, height).data;
    const rgb = new Uint8Array(w * height * 3);
    for (let i = 0, k = 0; i < rgba.length; i += 4) { rgb[k++] = rgba[i]; rgb[k++] = rgba[i + 1]; rgb[k++] = rgba[i + 2]; }
    const res = await live.frame(base, j.session, j.seq, w, height, rgb, j.ac.signal);
    if (run !== j) return;
    if (res.kind !== "ok") { report(j, res); return; } // no retry after an ambiguous acceptance
    j.seq++; feedback(res.data);
  }
  function limit(j) {
    if (run !== j || j.phase !== "capture") return;
    j.phase = "limit"; stopCapture(j); clock.textContent = ct("elapsed", { n: 30 }); message.textContent = ct("liveLimit");
    // No automatic verdict. Only the person's Finish may commit the attempt during the grace period.
    j.expiry = setTimeout(() => report(j, failure("session_expired")), Math.max(0, j.deadline + GRACE - performance.now()));
  }
  async function finishRun(j) {
    if (!j || run !== j || !["capture", "limit"].includes(j.phase)) return;
    j.phase = "finishing"; finish.disabled = true; message.textContent = t("check.sending"); stopCapture(j);
    // Do not race the last accepted frame with Finish, or open a second request while it is in flight.
    if (j.pending) await j.pending;
    if (run !== j) return;
    clearTimeout(j.expiry);
    try { report(j, await live.finish(base, j.session, j.ac.signal)); }
    catch (_) { report(j, { kind: "offline" }); }
  }
  async function start() {
    if (run || !base || document.hidden) return;
    const j = run = { phase: "opening", ac: new AbortController(), seq: 1, lastSent: -Infinity, lastFrame: -1 };
    el.hidden = false; finish.disabled = true; message.textContent = ct("openCamera"); clock.textContent = "";
    values[0].textContent = ct("waiting"); values[1].textContent = ct("collecting"); values[2].textContent = ct("noPreview");
    onChange(); cancel.focus();
    try {
      j.canvas = h("canvas"); j.ctx = j.canvas.getContext("2d", { willReadFrequently: true });
      if (!j.ctx) { report(j, failure("cameraUnsupported")); return; }
      const stream = await capture.open();
      if (run !== j) { stopTracks(stream); return; }
      j.stream = stream; video.srcObject = stream;
      const tracks = stream.getVideoTracks();
      if (!tracks.some((track) => track.readyState === "live")) { report(j, failure("cameraLost")); return; }
      tracks.forEach((track) => track.addEventListener("ended", () => report(j, failure("cameraLost")), { once: true }));
      await video.play();
      if (run !== j) return;
      j.deadline = performance.now() + LIMIT;
      const res = await live.start(base, j.ac.signal);
      if (run !== j) { if (res.kind === "ok") live.cancel(base, res.session, true); return; }
      if (res.kind !== "ok") { report(j, res); return; }
      j.session = res.session; j.phase = "capture"; finish.disabled = false; message.textContent = ct("liveRunning");
      const tick = () => { clock.textContent = ct("elapsed", { n: Math.min(30, Math.floor((performance.now() + LIMIT - j.deadline) / 1000)) }); };
      tick(); j.clock = setInterval(tick, 1000); j.cutoff = setTimeout(() => limit(j), Math.max(0, j.deadline - performance.now()));
      queueFrame(j);
    } catch (e) { if (run === j) report(j, failure(capture.reason(e))); }
  }
  return { el, start, get active() { return !!run; }, stop() { if (run) close(run, true); } };
}
