// What screens use. The 3D engine is loaded on demand; until it is ready, and whenever WebGL or
// the model is unavailable, the same handle drives a flat drawing in the same place.
// The figure is interactive: drag to turn it, arrow keys or buttons to turn it a step, a slider
// (or Up and Down) to bend the elbow, a reset button to return to the screen's view.
import { h } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { createArm as createFlatArm } from "../ui/arm.js";
import { icon } from "../ui/icons.js";

let engine = null, loading = null, failed = false, introduced = false;
const waiting = new Set();

/** Call once at boot with the fixed canvas and the SVG label layer. */
export function initStage(canvas, overlay) {
  if (loading) return loading;
  loading = import("./engine.js")
    .then((m) => m.createEngine(canvas, overlay))
    .then((e) => { engine = e; waiting.forEach((fn) => fn()); waiting.clear(); return e; })
    .catch((err) => { failed = true; waiting.clear(); console.warn("Qolda: 3D is unavailable, the flat drawing is used instead.", err); return null; });
  return loading;
}

/** Resolves to the engine, or null when 3D is unavailable. */
export const stageReady = () => loading || Promise.resolve(null);

const jointTexts = () => [t("welcome.jShoulder"), t("welcome.jElbow"), t("welcome.jWrist")];
const STEP = 30;          // degrees per button press or arrow key
const DRAG = 0.45;        // degrees of turn per pixel dragged
const TILT = 0.12;

/**
 * Put the figure into a screen's slot.
 * @param slot element marked with data-stage-slot
 * @param opts { shot: "quarter" | "side" | "arm" | "front", labels, side: "R" | "L", intro, bend: show the elbow slider }
 * @returns handle: set(deg), to(deg, ms, onTick, done), stop(), look(kind), shot(name), activate(side),
 *          labels(on), bendable(on), dispose()
 */
export function createFigure(slot, opts = {}) {
  const flat = createFlatArm({ labels: !!opts.labels });
  const fallback = h("div", { class: "stage-fallback" }, flat.el);
  let live = false, gone = false, demoStop = null;
  const pend = { angle: 60, look: "" };

  // ----- controls drawn over the slot -----
  const hint = h("span", { class: "fig-hint", text: t("stage.hint") });
  const slider = h("input", { type: "range", class: "fig-range", min: 40, max: 180, step: 1, value: 60, "aria-label": t("stage.bend") });
  const sliderBox = h("label", { class: "fig-bend", hidden: !opts.bend }, h("span", { class: "fig-bend-text", text: t("stage.bend") }), slider);
  const btn = (name, label, fn) => h("button", { type: "button", class: "fig-btn", "aria-label": label, title: label, on: { click: (e) => { e.stopPropagation(); touched(); fn(); } } }, icon(name));
  const turnL = btn("turnLeft", t("stage.rotL"), () => engine && engine.turn(-STEP));
  const turnR = btn("turnRight", t("stage.rotR"), () => engine && engine.turn(STEP));
  const reset = btn("recenter", t("stage.reset"), () => engine && engine.resetView());
  const turnBox = h("div", { class: "fig-turn", hidden: true }, turnL, turnR, reset);
  const bar = h("div", { class: "fig-bar" }, hint, h("div", { class: "fig-tools" }, sliderBox, turnBox));
  slot.classList.add("fig-slot");
  slot.append(fallback);
  // The controls sit just above the figure's place, never over the figure itself. Screens build the
  // slot before attaching it to the page, so the bar is placed once the screen has finished mounting.
  const placeBar = () => { if (!gone && slot.parentNode && !bar.isConnected) slot.before(bar); };
  if (slot.parentNode) placeBar(); else queueMicrotask(placeBar);

  // The slot itself is the thing you grab. It is focusable so the keyboard can do the same.
  slot.setAttribute("role", "group");
  function describe() {
    slot.tabIndex = live || opts.bend ? 0 : -1;
    hint.hidden = !live;
    slot.setAttribute("aria-label", t(live ? (opts.bend ? "stage.label" : "stage.turnLabel") : (opts.bend ? "stage.flatBend" : "welcome.illus")));
  }
  describe();

  function touched() {
    // Any hand on the figure ends a running demonstration and retires the hint.
    if (demoStop) { const f = demoStop; demoStop = null; f(); }
    hint.classList.add("gone");
  }
  slider.addEventListener("input", () => { touched(); const v = +slider.value; pend.angle = v; if (live) engine.set(v); else flat.set(v); });

  // ----- drag to turn -----
  let drag = null;
  const onDown = (e) => {
    if (!live || e.button !== 0 || e.target.closest(".fig-bar")) return;
    touched();
    drag = { id: e.pointerId, x: e.clientX, y: e.clientY, t: performance.now(), v: 0, moved: false };
    slot.setPointerCapture(e.pointerId);
    slot.classList.add("grabbing");
    engine.grab(true);
  };
  const onMove = (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y, now = performance.now(), dt = Math.max(1, now - drag.t);
    if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
    engine.spin(dx * DRAG, dy * TILT);
    // Smoothed speed in degrees per second, for the fling.
    drag.v = drag.v * 0.6 + (dx * DRAG / dt * 1000) * 0.4;
    drag.x = e.clientX; drag.y = e.clientY; drag.t = now;
  };
  const onUp = (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const idle = performance.now() - drag.t > 90;   // a hand that stopped before letting go means "stop here"
    engine.fling(idle ? 0 : drag.v);
    slot.classList.remove("grabbing");
    try { slot.releasePointerCapture(drag.id); } catch (_) { /* already released */ }
    drag = null;
    paintTurn();
  };
  slot.addEventListener("pointerdown", onDown);
  slot.addEventListener("pointermove", onMove);
  slot.addEventListener("pointerup", onUp);
  slot.addEventListener("pointercancel", onUp);
  slot.addEventListener("dblclick", () => { if (live) { touched(); engine.resetView(); } });

  // ----- keyboard -----
  slot.addEventListener("keydown", (e) => {
    if (e.target !== slot) return;
    const k = e.key;
    if (k === "ArrowLeft" || k === "ArrowRight") { if (!live) return; touched(); engine.turn(k === "ArrowLeft" ? -STEP : STEP); }
    else if ((k === "ArrowUp" || k === "ArrowDown") && opts.bend) {
      touched();
      const v = Math.max(40, Math.min(180, pend.angle + (k === "ArrowUp" ? 10 : -10)));
      pend.angle = v; slider.value = v;
      if (live) engine.set(v); else flat.set(v);
    } else if (k === "Home" || k === "Escape") { if (!live) return; touched(); engine.resetView(); }
    else return;
    e.preventDefault();
  });

  const paintTurn = () => { turnBox.hidden = !live; };

  const attach = () => {
    if (gone) return;
    live = true; flat.stop(); fallback.hidden = true; paintTurn(); describe();
    engine.acquire(slot, opts);
    engine.labels(!!opts.labels, jointTexts());
    engine.set(pend.angle); engine.look(pend.look);
    // The opening moment plays in full once per session.
    if (opts.intro && !introduced) { introduced = true; engine.intro(); }
  };
  if (engine) attach();
  else if (!failed) {
    waiting.add(attach);
    // The atlas starts in a complete 2D mode. Fetch WebGL assets only for a view that asks for 3D.
    if (!loading) {
      const canvas = document.getElementById("stage"), overlay = document.getElementById("stage-overlay");
      if (canvas && overlay) initStage(canvas, overlay);
    }
  }

  const handle = {
    el: slot,
    get live() { return live; },
    set(deg) { pend.angle = deg; slider.value = deg; if (live) engine.set(deg); else flat.set(deg); },
    to(goal, ms, onTick, done) {
      pend.angle = goal;
      const tick = (v) => { slider.value = Math.round(v); if (onTick) onTick(v); };
      if (live) engine.to(goal, ms, tick, done); else flat.to(goal, ms, tick, done);
    },
    stop() { if (live) engine.stop(); else flat.stop(); },
    /** "" | "uncertain" | "nopose" | "off" */
    look(kind) { pend.look = kind || ""; if (live) engine.look(pend.look); else flat.look(pend.look); },
    shot(name, ms) { opts.shot = name; if (live) engine.shot(name, ms); },
    activate(side) { opts.side = side; if (live) engine.activate(side); },
    labels(on) { opts.labels = !!on; if (live) engine.labels(on, jointTexts()); },
    /** Show or hide the elbow slider (hidden on a verdict, where the pose is the answer). */
    bendable(on) { opts.bend = !!on; sliderBox.hidden = !on; describe(); },
    /** A demonstration registers how to stop itself, so a hand on the figure can interrupt it. */
    setDemo(stop) { demoStop = stop; },
    /** Put a screen's own control (the replay button) at the start of the figure's tool bar. */
    addTool(el) { bar.querySelector(".fig-tools").prepend(el); },
    dispose() {
      gone = true; waiting.delete(attach); flat.stop();
      if (live) engine.release(slot);
      fallback.remove(); bar.remove();
      slot.classList.remove("fig-slot", "grabbing");
      slot.removeAttribute("tabindex"); slot.removeAttribute("role"); slot.removeAttribute("aria-label");
    }
  };
  return handle;
}

/**
 * Show the exercise a fixed number of times, then rest with the arm straight.
 * Motion never runs on its own for long: the screen offers a replay button afterwards.
 * A hand on the figure stops it early, and onDone still runs so the replay button comes back.
 * @returns a function that cancels the demonstration
 */
export function playDemo(figure, cycles, onDone) {
  let n = 0, timer = 0, off = false;
  const finish = () => { if (off) return; off = true; clearTimeout(timer); figure.setDemo(null); if (onDone) onDone(); };
  const extend = () => { if (off) return; figure.to(176, 1500, null, () => { n += 1; if (n >= cycles) finish(); else timer = setTimeout(bend, 700); }); };
  const bend = () => { if (off) return; figure.to(62, 1100, null, () => { timer = setTimeout(extend, 350); }); };
  figure.set(62);
  timer = setTimeout(extend, 600);
  figure.setDemo(() => { figure.stop(); finish(); });
  return () => { off = true; clearTimeout(timer); figure.setDemo(null); figure.stop(); };
}

export const profileSide = (arm) => (arm === "left" ? "L" : "R");
