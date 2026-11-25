// One E02 sheet, one submitted cycle. Only a definitive analyzer answer may enter the history.
import { h, calm } from "../core/dom.js";
import { t, L } from "../core/i18n.js";
import { store } from "../core/store.js";
import { engine } from "../core/api.js";
import { profile } from "../platform/profile.js";
import { dayKey } from "../domain/stats.js";
import { createFigure, playDemo, profileSide } from "../stage/index.js";
import { icon } from "../ui/icons.js";
import { choice, segmented } from "../ui/controls.js";
import { createInstruction } from "../ui/instruction.js";
import { createClip, clipSize } from "../ui/check-clip.js";
import { availableLiveBase, createLivePanel } from "../ui/check-live.js";
import { createNativeSources, createNativeLivePanel, nativeResult } from "../ui/check-native.js";
import { ct } from "../i18n/check-atlas.js";

// Keep copy/step changes local; neither camera permissions nor a running job survive a remount.
const fresh = () => ({ step: 0, mode: "file", video: null, model: null, consent: false, result: null, seen: false });
const S = fresh();
let resetMounted = null;
if (typeof window !== "undefined") window.addEventListener("qolda:data-reset", () => {
  if (resetMounted) resetMounted(); else Object.assign(S, fresh());
});
const LABELS = ["complete", "incomplete", "uncertain"];
const REASONS = ["low_confidence", "low_visibility", "unscorable"];
function validResult(data) {
  return data?.version === "seriq.v1" && Array.isArray(data.attempts) && data.attempts.length <= 1 &&
    data.attempts.every((a) => a && LABELS.includes(a.label));
}
const CODE_KEY = {
  model_unavailable: "unavailable", engine_busy: "busy", live_busy: "busy", invalid_video: "invalidVideo",
  duration_limit: "duration", too_large: "tooLarge", unsupported_media: "badType", bad_response: "badResponse"
};
const STATUS_KEY = {
  400: "badRequest", 403: "forbidden", 404: "notFound", 409: "busy", 411: "lengthRequired",
  413: "tooLarge", 415: "badType", 422: "invalidVideo", 429: "busy", 500: "serverError", 503: "unavailable"
};
const CAMERA_KEYS = ["cameraDenied", "cameraUnsupported", "cameraInsecure", "cameraMissing", "cameraBusy", "cameraLost"];
function errorText(res) {
  const isLive = res.source === "live";
  if (profile.name === "android") {
    if (res.code === "picker_unavailable") return ct("phonePickerDown");
    if (res.code === "camera_unavailable") return ct("phoneCameraDown");
    if (res.code === "forbidden" || res.status === 403) return ct(isLive ? "phoneDenied" : "phoneDeniedFile");
    if (res.kind === "offline" || res.code === "model_unavailable" || (res.status === 503 && !["engine_busy", "live_busy"].includes(res.code))) return ct("phoneEngineDown");
    if (res.code === "invalid_request") return ct(isLive ? "phoneCameraDown" : "phoneDeniedFile");
    if (res.status === 404) return ct("phoneEngineDown");
  }
  if (res.kind === "offline") return isLive ? ct("liveOffline") : profile.analyzer === "remote" ? ct("remoteOffline") : t("check.offline");
  if (CAMERA_KEYS.includes(res.code)) return t("check." + res.code);
  if (["session_expired", "frame_sequence", "empty_session", "invalid_frame", "too_few_frames"].includes(res.code)) return ct(res.code);
  if (isLive && res.code === "duration_limit") return ct("live_duration");
  if (isLive && res.status === 410) return ct("session_expired");
  if (isLive && res.status === 404) return ct("liveMissing");
  const key = CODE_KEY[res.code] || STATUS_KEY[res.status] || (res.status === 200 ? "badResponse" : null);
  if (key === "unavailable" && profile.analyzer === "remote") return ct("remoteUnavailable");
  return key ? t("check." + key) : t("check.generic", { code: res.status || "—" });
}
const action = (label, fn, primary = false) => h("button", { type: "button", class: "btn " + (primary ? "btn--cta" : "btn--quiet"), on: { click: fn } },
  h("span", { text: label }), primary ? icon("arrow") : null);
const fact = (label, value) => h("div", null, h("dt", { text: label }), h("dd", null, value));
const memo = (title, ...rows) => h("aside", { class: "c-memo" }, h("h3", { text: title }), h("dl", null, rows));

export function check(root) {
  const prefs = store.state.prefs, native = profile.name === "android";
  const base = native ? null : availableLiveBase(prefs.engineUrl);
  if (!native && !base) S.mode = "file";
  if (native) S.model = "scratch"; else if (!S.model) S.model = prefs.model;
  let disposed = false, suspended = false, upload = null, showReference = false, figure = null, cancelDemo = null;
  const current = () => !disposed && /^#\/?check(?:\?|$)/.test(location.hash);
  const isLive = () => S.mode === "live" && (native || !!base);
  const implemented = native || profile.analyzer === "loopback" || profile.analyzer === "remote";
  const locationLabel = ct(profile.analyzer === "loopback" ? "local" : profile.analyzer === "remote" ? "service" : "onDevice");
  const names = [...t("check.steps")]; names[2] = ct("stepConsent");

  // Step 1: the illustrated camera plate and its text instructions are usable without WebGL.
  const instruction = createInstruction();
  const frame = instruction.el.querySelector(".c-instr-frame");
  frame.append(h("span", { class: "c-plate-key", "aria-hidden": "true", text: "E02" }),
    h("span", { class: "c-plate-label", text: ct(native ? "diagramSide" : "sheetSide") }));
  const slot = h("div", { class: "c-slot", "data-stage-slot": true });
  const caption = h("p", { class: "fine" });
  const figureBox = h("div", { class: "c-figure figure-box", hidden: true }, slot, h("div", { class: "figure-foot" }, caption));
  const replay = action(t("common.replay"), () => {
    if (!figure || calm()) return;
    if (cancelDemo) cancelDemo();
    cancelDemo = playDemo(figure, 1);
  });
  replay.classList.add("btn--small");
  const swap = h("button", { type: "button", class: "c-text-action", text: ct("reference"), "aria-pressed": "false", on: { click: () => {
    showReference = !showReference; instruction.pause(); drawFigure(false);
  } } });
  const firstHost = h("div", { class: "c-visual" }, instruction.el, figureBox);
  const textSteps = t("check.instr");
  const notes = h("div", { class: "c-fieldnotes" }, native ? null : h("h3", { text: ct("how") }),
    h("ol", { class: "c-howto" }, ["position", "environment", "movement"].map((key, i) =>
      h("li", null, h("strong", { text: ct(key) }), h("span", { text: textSteps[i] })))), h("p", { class: "c-limit", text: textSteps[3] }));
  const stepOne = native ? h("div", { class: "c-phone-instruction" }, instruction.el,
    action(ct("chooseSource"), () => go(1, true), true), h("details", { class: "c-disclosure" }, h("summary", { text: ct("writtenSteps") }), notes)) :
    h("div", { class: "c-instruction-layout" }, h("div", { class: "c-plate-wrap" }, firstHost, swap), notes);

  // Step 2: choosing the live source does not open the camera or create a server session.
  const clip = native ? createNativeSources({ current, describeError: errorText,
    onMode(mode) { S.video = null; S.consent = false; S.result = null; S.mode = mode; if (mode === "live") go(2, true); else update(); },
    onFile(meta) { S.video = meta; S.consent = false; S.result = null; go(2, !document.hidden); }
  }) : createClip(S.video, (value) => {
    S.video = value;
    if (!value) { S.consent = false; S.result = null; }
    update(); renderResult(false);
  });
  const modes = base ? segmented(ct("source"), [["file", ct("clip")], ["live", ct("live")]], S.mode, changeMode) : null;
  const liveInfo = h("div", { class: "c-live-info" }, icon("camera"), h("p", { text: ct("liveIntro") }));
  const fileNote = h("p", { class: "c-lead", text: ct("videoNote") });
  const next = action(ct("nextConsent"), () => { if (S.video || isLive()) go(2, true); }, true);
  const nextReason = h("p", { class: "btn-reason", id: "c-need-clip", text: t("check.reasonNoVideo") });
  const videoMemo = memo(ct("specs"), fact(ct("durationLabel"), ct("limit")), fact(ct("formatLabel"), ct("format")));
  const stepTwo = native ? clip.el : h("div", { class: "c-work" }, h("div", { class: "c-form" }, modes, fileNote, clip.el, liveInfo,
    h("div", { class: "c-card-end" }, next, nextReason)), videoMemo);

  // Step 3: one explicit consent for this source and this destination; no consent carries into a new job.
  const consentText = h("span");
  const consent = h("input", { type: "checkbox", on: { change: () => { S.consent = consent.checked; update(); } } });
  const agree = h("label", { class: "consent c-agreement" }, consent, consentText);
  const send = action(t("check.send"), submit, true);
  const sendReason = h("p", { class: "btn-reason", id: "c-need-consent" });
  const discard = action(t("common.cancel"), newCycle);
  const sendActions = h("div", { class: "actions" }, send, discard);
  const modelTag = h("span");
  const models = native ? null : h("details", { class: "c-disclosure" }, h("summary", null, ct("modelOptions"), modelTag),
    h("div", { class: "c-disclosure-body" }, choice(t("check.model"), [["scratch", t("check.scratch")], ["transfer", t("check.transfer")]], S.model,
      (v) => { S.model = v; S.consent = false; update(); })));
  const liveOptions = { current, validResult, onChange: update, onResult: (res) => accept(res, native ? res?.key : null, "live") };
  const livePanel = native ? createNativeLivePanel(liveOptions) : createLivePanel(base, liveOptions);
  const chosen = h("span");
  const receipt = native ? h("p", { class: "c-phone-receipt" }, h("span", { text: ct("chosen") }), chosen) :
    memo(ct("receipt"), fact(ct("chosen"), chosen), fact(ct("location"), locationLabel));
  if (!native && profile.engineSettings) receipt.append(h("a", { class: "c-text-action", href: "#/cabinet", text: t("cabinet.engine") }));
  const stepThree = h("div", { class: "c-work" }, h("div", { class: "c-form" }, native ? receipt : null, agree, models, sendActions, sendReason, livePanel.el), native ? null : receipt);

  // Step 4: no simulated responses, poses or history entries while waiting or on transport failure.
  const box = h("div", { class: "c-verdict", role: "status", "aria-live": "polite" });
  const stop = action(t("common.cancel"), () => { abortUpload(); update(); renderResult(false); });
  const again = action(t("check.again"), newCycle, true);
  const retry = action(ct("retry"), () => { S.consent = false; go(2, true); });
  const readHow = action(names[0], () => go(0, true));
  const emptyNext = action(ct("nextVideo"), () => go(1, true), true);
  const progress = h("a", { class: "btn btn--quiet", href: "#/progress", text: t("check.toProgress") });
  const resultHost = h("div", { class: "c-result-figure", hidden: true });
  const stepFour = h("div", { class: "c-result-layout" }, h("div", { class: "c-result-copy" }, box,
    h("p", { class: "fine", text: t("check.disclaimer") }), h("div", { class: "actions" }, stop, again, retry, readHow, progress, emptyNext)), resultHost);

  const headings = [ct("prepare"), native ? ct("sourceTitle") : t("check.s2"), ct(native ? "phoneLocal" : "consentTitle"), t("check.s4")].map((text, i) => h("h2", { id: "step-h" + i, tabindex: -1, text }));
  const panels = [stepOne, stepTwo, stepThree, stepFour].map((content, i) => h("section", { class: "c-panel", id: "c-panel-" + i, "aria-labelledby": "step-h" + i, hidden: true },
    h("div", { class: "c-panel-head" }, headings[i], i === 0 && !native ? action(ct("nextVideo"), () => go(1, true), true) : null), content));
  const buttons = names.map((name, i) => h("button", { type: "button", "aria-controls": "c-panel-" + i, on: { click: () => go(i, true) } },
    h("span", { class: "n", "aria-hidden": "true", text: "0" + (i + 1) }), h("span", { class: "s", text: name })));
  const stepOf = h("p", { class: "c-page-step" });
  const backText = h("span", { text: t(S.step ? "common.back" : "nav.home") });
  const back = h("button", { type: "button", class: "c-back", on: { click: () => {
    if (!S.step) location.hash = "#/"; else go(native && S.step === 3 ? 1 : S.step - 1, true);
  } } }, icon("arrow"), backText);
  root.append(h("div", { class: native ? "v-check v-check--phone" : "v-check" },
    h("header", { class: "c-mast" }, h("div", null, h("p", { class: "c-sheet", text: native ? "E02 · " + t("home.exerciseName") : ct("sheet") }), h("h1", { text: t("check.title") }),
      native ? null : h("p", { class: "c-subtitle", text: ct("one") })), native ? null : h("p", { class: "c-exercise" }, icon("check"), t("home.exerciseName"))),
    h("nav", { class: "c-route", "aria-label": ct("steps") }, h("div", { class: "c-step-line" }, back, stepOf),
      h("ol", { class: "c-steps" }, buttons.map((b) => h("li", null, b)))), panels));

  function setModeInputs() { modes?.querySelectorAll("input").forEach((input) => { input.checked = input.value === S.mode; }); }
  function changeMode(mode) {
    if (mode === S.mode || (mode === "live" && !base)) return;
    abortUpload(); livePanel.stop(); clip.clear(); S.mode = mode; S.consent = false; S.result = null;
    clip.show(S.step === 1 && !isLive() && !document.hidden); update(); renderResult(false);
  }
  function clearNativeFile() { if (native) { clip.clear(); S.video = null; } }
  function newCycle() {
    abortUpload(); livePanel.stop(); clip.clear(); S.video = null; S.consent = false; S.result = null;
    if (native) S.mode = "file"; go(1, true);
  }
  function abortUpload() {
    if (!upload) return;
    const ac = upload; upload = null; ac.abort(); clearNativeFile(); S.result = { kind: "canceled", source: "file" }; S.consent = false;
  }
  async function submit() {
    if (!current() || suspended || document.hidden || S.step !== 2 || !S.consent || upload || livePanel.active) return;
    if (isLive()) { S.result = null; livePanel.start(); return; }
    if (!S.video || !implemented) return;
    const video = S.video, model = S.model;
    const ac = upload = new AbortController(); S.result = { kind: "sending", source: "file" };
    go(3, true);
    let res;
    try { res = await (native ? clip.analyze(ac.signal) : engine.analyze(video.blob, video.type, model, ac.signal)); }
    catch (_) { res = { kind: "offline" }; }
    if (disposed || upload !== ac || ac.signal.aborted) return;
    upload = null; clearNativeFile(); accept(res, video.hash, "file");
  }
  function accept(res, hash, source) {
    if (!current()) return;
    if (suspended || document.hidden) { S.result = { kind: "canceled", source }; S.consent = false; update(); renderResult(false); return; }
    if (!res || (res.kind === "ok" && (native ? !nativeResult(res.data) || typeof hash !== "string" || !hash.trim() : !validResult(res.data))))
      res = { kind: "error", status: 200, code: "bad_response" };
    // Preserve only the wire result, not the carrier envelope or any native grant metadata.
    if (res.kind === "ok") {
      const dup = res.data.attempts.length === 1 && store.addAttempts(res.data, hash);
      S.result = { kind: "ok", data: res.data, dup, source };
    } else S.result = { kind: res.kind, status: res.status, code: res.code, source };
    S.consent = false;
    if (S.step !== 3) go(3, true); else { update(); renderResult(true); }
  }
  function go(n, focus) {
    const previous = S.step;
    if (n !== previous) {
      if (previous === 0) { S.seen = true; showReference = false; instruction.pause(); }
      if (previous === 1 && n !== 2) clip.suspend();
      if (previous === 2) {
        if (livePanel.active) { livePanel.stop(); S.result = { kind: "canceled", source: "live" }; }
        if (native && n !== 3) { clearNativeFile(); S.mode = "file"; }
        S.consent = false;
      }
      if (previous === 3) abortUpload();
    }
    S.step = n;
    panels.forEach((p, i) => { p.hidden = i !== n; });
    clip.show(n === 1 && (native || !isLive()) && !document.hidden);
    backText.textContent = t(n ? "common.back" : "nav.home");
    stepOf.textContent = t("check.stepOf", { n: n + 1 }) + (native ? " · " + names[n] : "");
    update(); renderResult(false);
    if (focus) { headings[n].focus({ preventScroll: true }); headings[n].scrollIntoView({ block: "nearest" }); }
  }
  function update() {
    if (disposed) return;
    const streaming = isLive(), active = livePanel.active, noVideo = !streaming && !S.video;
    clip.el.hidden = !native && streaming; fileNote.hidden = streaming; liveInfo.hidden = !streaming; videoMemo.hidden = streaming;
    next.disabled = noVideo; nextReason.hidden = !noVideo;
    if (noVideo) next.setAttribute("aria-describedby", "c-need-clip"); else next.removeAttribute("aria-describedby");
    agree.hidden = active; consent.checked = S.consent;
    consentText.textContent = native ? ct(streaming ? "phoneConsentLive" : "phoneConsentFile") : streaming ? ct("liveConsent", { url: base }) :
      profile.analyzer === "remote" ? t("check.consentRemote") : profile.analyzer === "loopback" ? t("check.consent", { url: prefs.engineUrl }) : ct("engineConsent");
    if (models) models.hidden = streaming; modelTag.textContent = S.model === "scratch" ? "A" : "B";
    sendActions.hidden = active; send.firstChild.textContent = streaming ? ct("liveStart") : native ? ct("phoneAnalyze") : t("check.send");
    send.disabled = noVideo || !S.consent || !!upload || active || !implemented;
    const reason = !implemented ? ct("notBuilt") : noVideo ? (native ? ct("phoneSourceNeed") : t("check.reasonNoVideo")) : !S.consent ? (native ? ct("phoneRequireConsent") : streaming ? ct("liveNeedConsent") : t("check.reasonNoConsent")) : "";
    sendReason.textContent = reason; sendReason.hidden = !reason || active;
    if (!sendReason.hidden) send.setAttribute("aria-describedby", "c-need-consent"); else send.removeAttribute("aria-describedby");
    chosen.textContent = streaming ? ct(native ? "phoneCameraSource" : "live") : S.video ? (native ? (S.video.name || ct("phoneFileSource")) : ct("clip")) + " · " + clipSize(S.video) : "—";
    const completed = [S.seen, !noVideo || !!S.result, S.result?.kind === "ok", false];
    buttons.forEach((b, i) => {
      if (native) b.disabled = i !== S.step && ((i === 2 && noVideo) || (i === 3 && !S.result));
      if (i === S.step) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
      b.classList.toggle("done", i < S.step && completed[i]);
    });
  }
  function renderResult(animate) {
    const res = S.result, a = res?.kind === "ok" ? res.data.attempts[0] : null;
    const put = (cls, ...kids) => box.append(h("p", { class: cls }, kids));
    const title = (shape, text) => put("c-title", icon(shape), h("span", { text }));
    box.replaceChildren();
    stop.hidden = res?.kind !== "sending"; again.hidden = !res || !stop.hidden; emptyNext.hidden = !!res;
    retry.hidden = !(res && res.source === "file" && S.video && ["error", "offline", "canceled"].includes(res.kind));
    readHow.hidden = !(res?.kind === "ok" && a?.label !== "complete"); progress.hidden = !a;
    let state = "idle";
    if (!res) title("guide", t("check.waiting"));
    else if (res.kind === "sending") { state = "sending"; title("linkHalf", t("check.sending")); }
    else if (res.kind === "canceled") { state = "canceled"; title("replay", res.source === "live" ? ct("liveCanceled") : t("check.canceled")); }
    else if (res.kind !== "ok") { state = "error"; title("linkOff", errorText(res)); }
    else if (!a) { state = "nopose"; title("nopose", t("check.noPose")); }
    else {
      state = a.label; title(a.label, t("check." + a.label));
      const meta = [];
      if (Number.isFinite(a.confidence) && a.confidence >= 0 && a.confidence <= 1) meta.push(t("check.confidence", { n: Math.round(a.confidence * 100) }));
      if (REASONS.includes(a.reason)) meta.push(t("check.reason." + a.reason));
      if (res.data.model_version) meta.push([res.data.model, res.data.model_version].filter(Boolean).join(" "));
      if (meta.length) put("fine", meta.join(" · "));
      put("c-note", res.dup ? t("check.duplicate") : t("check.added"));
      const goal = store.state.data.profile.goal, today = dayKey(new Date());
      const done = store.state.data.attempts.filter((x) => x.day === today).length;
      put("c-next", done >= goal ? t("check.nextDone") : t("check.nextLeft", { n: L().cycles(goal - done) }));
    }
    if (["error", "canceled", "nopose"].includes(state)) put("c-note", ct("noCount"));
    box.dataset.state = state; drawFigure(animate);
  }
  function releaseFigure() {
    if (cancelDemo) { cancelDemo(); cancelDemo = null; }
    if (figure) { figure.dispose(); figure = null; }
    figureBox.hidden = true;
  }
  function drawFigure(animate) {
    if (native) return; // the native camera surface and GPU are owned by the Activity
    const attempt = S.step === 3 && S.result?.kind === "ok" ? S.result.data.attempts[0] : null;
    const reference = S.step === 0 && showReference;
    instruction.el.hidden = reference; resultHost.hidden = !attempt;
    swap.textContent = ct(reference ? "diagram" : "reference"); swap.setAttribute("aria-pressed", String(reference));
    if (suspended || document.hidden || (!reference && !attempt)) { releaseFigure(); return; }
    const host = reference ? firstHost : resultHost;
    if (figureBox.parentElement !== host) { releaseFigure(); host.append(figureBox); }
    figureBox.hidden = false;
    if (!figure) figure = createFigure(slot, { shot: reference ? "side" : "arm", labels: reference, bend: reference, glide: false, side: profileSide(store.state.data.profile.arm) });
    caption.textContent = reference ? t("check.frameHint") : t("welcome.illus");
    figure.bendable(reference); figure.labels(reference); figure.shot(reference ? "side" : "arm", 0);
    if (reference) { figure.look(""); figure.set(96); figure.addTool(replay); replay.hidden = calm(); return; }
    replay.hidden = true; figure.look(attempt.label === "uncertain" ? "uncertain" : "");
    const pose = attempt.label === "complete" ? 176 : attempt.label === "incomplete" ? 118 : 120;
    if (animate && !calm() && attempt.label !== "uncertain") { figure.set(62); figure.to(pose, attempt.label === "complete" ? 1500 : 1100); }
    else { figure.stop(); figure.set(pose); }
  }
  function suspend(allowSystemPicker = false) {
    suspended = true; instruction.pause(); releaseFigure();
    // SAF temporarily covers the WebView. Retain only its pending ticket. Pagehide, route,
    // language and data reset still invalidate it, including a token that arrives afterwards.
    if (allowSystemPicker === true && native && S.step === 1 && clip.picking) return;
    const wasLive = livePanel.active, staged = native && S.video && S.step === 2;
    if (wasLive) { livePanel.stop(); S.result = { kind: "canceled", source: "live" }; }
    abortUpload(); clip.suspend(); if (native) S.video = null; S.consent = false;
    if (wasLive) go(3, false); else if (staged) go(1, false); else { update(); renderResult(false); }
  }
  function visibility() {
    if (document.hidden) suspend(true);
    else { suspended = false; clip.show(S.step === 1 && (native || !isLive())); update(); renderResult(false); }
  }
  resetMounted = () => {
    suspend(); clip.clear(); Object.assign(S, fresh(), { model: native ? "scratch" : prefs.model }); setModeInputs();
    models?.querySelectorAll("input").forEach((input) => { input.checked = input.value === S.model; });
    suspended = document.hidden; showReference = false; go(0, false);
  };
  document.addEventListener("visibilitychange", visibility); window.addEventListener("pagehide", suspend); window.addEventListener("pageshow", visibility);
  go(S.step, false);
  return () => {
    disposed = true; resetMounted = null;
    document.removeEventListener("visibilitychange", visibility); window.removeEventListener("pagehide", suspend); window.removeEventListener("pageshow", visibility);
    livePanel.stop(); abortUpload(); clip.suspend(); instruction.dispose(); releaseFigure(); S.consent = false;
    // A web language refresh can reuse a Blob. A native grant never survives any remount.
    if (native) { S.video = null; S.mode = "file"; if (S.step === 2) S.step = 1; }
    if (!/^#\/check(?:\?|$)/.test(location.hash)) { S.video = null; if (!S.result) { S.step = 0; S.mode = "file"; S.seen = false; } }
  };
}
