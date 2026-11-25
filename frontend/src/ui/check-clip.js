// The file/recording half of the check. No analyzer calls: a clip stays here until step 3.
import { h } from "../core/dom.js";
import { t, L } from "../core/i18n.js";
import { clipHash, MAX_BYTES, TYPES } from "../core/api.js";
import { capture } from "../platform/camera.js";
import { ct } from "../i18n/check-atlas.js";
import { icon } from "./icons.js";

const COUNTDOWN = 10, MAX_SECONDS = 30;
const stopTracks = (stream) => stream?.getTracks().forEach((track) => track.stop());

function typeOf(blob, name) {
  const mime = (blob.type || "").split(";")[0];
  if (TYPES.includes(mime)) return mime;
  const lower = (name || "").toLowerCase();
  return lower.endsWith(".mp4") ? "video/mp4" : lower.endsWith(".webm") ? "video/webm" : mime;
}

// Some valid containers have no readable duration; the analyzer then enforces its own limit.
function readDuration(blob, signal) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob), probe = h("video", { preload: "metadata", muted: true });
    const finish = (value) => {
      clearTimeout(timer); signal.removeEventListener("abort", abort);
      probe.onloadedmetadata = probe.onerror = null;
      probe.removeAttribute("src"); probe.load(); URL.revokeObjectURL(url); resolve(value);
    };
    const abort = () => finish(null), timer = setTimeout(abort, 4000);
    probe.onloadedmetadata = () => finish(Number.isFinite(probe.duration) ? probe.duration : null);
    probe.onerror = abort;
    signal.addEventListener("abort", abort, { once: true });
    probe.src = url;
  });
}

export function clipSize(value) {
  const bytes = value.size ?? value.blob.size;
  if (bytes < 104858) return ct("smallClip");
  return ct("size", { n: (bytes / 1048576).toLocaleString(L().code, { maximumFractionDigits: 1, minimumFractionDigits: 1 }) });
}

export function createClip(initial, onChange) {
  let value = initial, pending = null, job = null, url = null, visible = false;
  const file = h("input", { type: "file", id: "clip", accept: TYPES.join(",") + ",.mp4,.webm", class: "visually-hidden" });
  const pickLabel = h("strong");
  const drop = h("label", { class: "c-drop", for: "clip" }, icon("upload"), pickLabel, h("span", { text: t("check.dropHint") }));
  const preview = h("video", { controls: true, playsinline: true, preload: "metadata", hidden: true, "aria-label": t("check.s2") });
  const status = h("p", { class: "c-file-status fine", role: "status" });
  const camera = h("video", { playsinline: true, muted: true, "aria-label": ct("liveCamera") });
  const countdown = h("p", { class: "c-countdown", role: "status" });
  const cameraBox = h("div", { class: "c-camera", hidden: true }, camera, countdown);
  const permission = h("input", { type: "checkbox" });
  const recordText = h("span", { text: t("check.record") });
  const rec = h("button", { type: "button", class: "btn btn--quiet", disabled: true, on: { click: toggleRecording } }, icon("camera"), recordText);
  const cancel = h("button", { type: "button", class: "btn btn--quiet", hidden: true, text: t("common.cancel"), on: { click: () => endJob() } });
  const record = h("details", { class: "c-disclosure" }, h("summary", null, ct("cameraRecord")),
    h("div", { class: "c-disclosure-body" }, h("label", { class: "consent" }, permission, h("span", { text: ct("recordConsent") })),
      cameraBox, h("div", { class: "actions" }, rec, cancel)));
  const canRecord = capture.supported() && capture.recorderSupported();
  const el = h("div", { class: "c-file" }, file, drop, preview, status,
    canRecord ? record : h("p", { class: "fine", text: t("check." + (capture.supported() ? "recordUnsupported" : capture.reason())) }));
  camera.muted = true;
  file.addEventListener("change", () => { const picked = file.files?.[0]; if (picked && visible) choose(picked, picked.name); });
  for (const event of ["dragenter", "dragover", "dragleave", "drop"]) drop.addEventListener(event, (e) => {
    e.preventDefault(); drop.classList.toggle("over", event === "dragenter" || event === "dragover");
    if (event === "drop" && visible && e.dataTransfer?.files?.[0]) choose(e.dataTransfer.files[0], e.dataTransfer.files[0].name);
  });
  permission.addEventListener("change", () => { rec.disabled = !permission.checked; });
  record.addEventListener("toggle", () => { if (!record.open) endJob(); });

  function releasePreview() {
    if (url) { preview.pause(); preview.removeAttribute("src"); preview.load(); URL.revokeObjectURL(url); url = null; }
    preview.hidden = true;
  }
  function paint() {
    pickLabel.textContent = value ? ct("replace") : t("check.pick");
    drop.classList.toggle("has-file", !!value);
    if (!value || !visible || job) releasePreview();
    else if (!url) { url = URL.createObjectURL(value.blob); preview.src = url; preview.hidden = false; }
  }
  function resetValue() {
    if (pending) { pending.abort(); pending = null; }
    value = null; paint(); onChange(null);
  }
  async function choose(blob, name) {
    endJob(); resetValue(); file.value = "";
    const ac = new AbortController(); pending = ac;
    const type = typeOf(blob, name);
    let bad = !TYPES.includes(type) ? "badType" : !blob.size ? "empty" : blob.size > MAX_BYTES ? "tooLarge" : null;
    try {
      status.textContent = t("check.checking");
      if (!bad && name) {
        const duration = await readDuration(blob, ac.signal);
        if (ac.signal.aborted) return;
        if (duration !== null && duration > MAX_SECONDS + 0.05) bad = "duration";
      }
      if (bad) { status.textContent = t("check." + bad); return; }
      const hash = await clipHash(blob);
      if (ac.signal.aborted) return;
      value = { blob, type, hash };
      status.textContent = t("check.selected") + " · " + clipSize(value);
      paint(); onChange(value);
    } catch (_) { if (!ac.signal.aborted) status.textContent = t("check.invalidVideo"); }
    finally { if (pending === ac) pending = null; }
  }

  // Sound is supplementary; the countdown and Stop control do not depend on it.
  function beep(j, freq, ms) {
    try {
      j.audio ||= new (window.AudioContext || window.webkitAudioContext)();
      const osc = j.audio.createOscillator(), gain = j.audio.createGain();
      osc.frequency.value = freq; gain.gain.value = 0.08; osc.connect(gain); gain.connect(j.audio.destination);
      osc.start(); osc.stop(j.audio.currentTime + ms / 1000);
    } catch (_) { /* optional */ }
  }
  function endJob() {
    const j = job; job = null;
    if (j) {
      clearTimeout(j.timer);
      if (j.recorder) {
        j.recorder.onstop = j.recorder.ondataavailable = j.recorder.onerror = null;
        if (j.recorder.state !== "inactive") { try { j.recorder.stop(); } catch (_) {} }
      }
      stopTracks(j.stream); camera.pause(); camera.srcObject = null;
      if (j.audio) j.audio.close().catch(() => {});
      j.chunks.length = 0;
    }
    cameraBox.hidden = true; countdown.textContent = ""; cancel.hidden = true;
    permission.checked = permission.disabled = false; rec.disabled = true; recordText.textContent = t("check.record");
  }
  function fail(j, key) { if (job === j) { endJob(); status.textContent = t("check." + key); } }
  function beginRecorder(j) {
    if (job !== j) return;
    const mime = capture.mimeType();
    if (mime === null) { fail(j, "recordUnsupported"); return; }
    try {
      const r = j.recorder = new MediaRecorder(j.stream, mime ? { mimeType: mime } : undefined);
      r.ondataavailable = (e) => { if (job === j && e.data.size) j.chunks.push(e.data); };
      r.onerror = () => fail(j, "recordFailed");
      r.onstop = () => {
        if (job !== j) return;
        const type = (r.mimeType || mime || "video/webm").split(";")[0];
        const blob = new Blob(j.chunks, { type });
        endJob();
        if (blob.size) choose(blob); else status.textContent = t("check.recordEmpty");
      };
      r.start(); beep(j, 990, 220); countdown.textContent = t("check.recording");
      recordText.textContent = t("check.stopRec"); cancel.hidden = false;
      j.timer = setTimeout(() => { if (job === j && r.state === "recording") r.stop(); }, MAX_SECONDS * 1000);
    } catch (_) { fail(j, "recordFailed"); }
  }
  async function toggleRecording() {
    if (job) {
      if (job.recorder?.state === "recording") job.recorder.stop();
      else if (!job.recorder) endJob();
      return;
    }
    if (!visible || document.hidden || !permission.checked || !canRecord) return;
    resetValue(); status.textContent = "";
    const j = job = { chunks: [] };
    permission.disabled = true; recordText.textContent = t("common.cancel");
    countdown.textContent = ct("openCamera"); cameraBox.hidden = false;
    try {
      const stream = await capture.open();
      if (job !== j) { stopTracks(stream); return; }  // permission returned after leaving the step
      j.stream = stream; camera.srcObject = stream;
      stream.getVideoTracks().forEach((track) => track.addEventListener("ended", () => fail(j, "cameraLost"), { once: true }));
      camera.play().catch(() => {});
      let left = COUNTDOWN;
      const tick = () => {
        if (job !== j) return;
        if (!stream.getVideoTracks().some((tr) => tr.readyState === "live")) { fail(j, "cameraLost"); return; }
        if (left > 0) { countdown.textContent = t("check.countdown", { n: left-- }); beep(j, 660, 90); j.timer = setTimeout(tick, 1000); }
        else beginRecorder(j);
      };
      tick();
    } catch (e) { fail(j, capture.reason(e)); }
  }
  function show(on) {
    visible = on;
    if (!visible) { endJob(); record.open = false; }
    paint();
  }
  function suspend() {
    show(false);
    if (pending) { pending.abort(); pending = null; status.textContent = ""; }
  }
  function clear() { suspend(); file.value = ""; status.textContent = ""; resetValue(); }
  if (value) status.textContent = t("check.selected") + " · " + clipSize(value);
  paint();
  return { el, show, suspend, clear };
}
