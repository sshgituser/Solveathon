// An exercise plate, not a measurement. The drawing and the optional mannequin share the three poses.
import { h, s, calm } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { icon } from "./icons.js";
import { createFigure, profileSide, stageReady } from "../stage/index.js";

const CX = 224, CY = 180, LENGTH = 145, POSE = [-109, -7, -109];
const rad = a => a * Math.PI / 180;
const tip = (a, r) => [CX + r * Math.cos(rad(a)), CY + r * Math.sin(rad(a))];
const pt = xy => xy.map(n => n.toFixed(1)).join(" ");
const path = (cls, d) => s("path", { class: cls, d });
const circle = (cls, cx, cy, r) => s("circle", { class: cls, cx, cy, r });
const bones = "M1 -14C24 -15 45 -8 72 -7S123 -10 139 -8Q149 -7 150 0Q150 8 139 8C107 9 94 5 71 6S22 15 2 15Q-10 14 -12 2Q-13 -10 1 -14Z";
function drawing() {
  const sector = `M${CX} ${CY}L${pt(tip(POSE[0], LENGTH))}A${LENGTH} ${LENGTH} 0 0 1 ${pt(tip(POSE[1], LENGTH))}Z`;
  const ticks = s("g", { class: "ap-ticks" });
  for (let i = 0; i <= 12; i++) {
    const a = POSE[0] + (POSE[1] - POSE[0]) * i / 12;
    ticks.append(path("", `M${pt(tip(a, 148))}L${pt(tip(a, i % 3 === 0 ? 157 : 152))}`));
  }
  const ghost = a => s("g", { transform: `translate(${CX} ${CY}) rotate(${a})`, class: "ap-ghost" }, path("", bones), circle("", LENGTH, 0, 4));
  const moving = s("g", { class: "ap-limb" }, path("ap-bone", bones), path("ap-contour", "M14 6Q52 -3 80 -1T131 1M35 10Q89 2 125 5"), circle("ap-pin", LENGTH, 0, 7));
  const leader = path("ap-lead", "");
  const label = s("text", { class: "ap-label", text: t("welcome.jWrist") });
  const graphic = s("svg", { class: "atlas-sheet", viewBox: "0 0 460 246", role: "img", "aria-label": t("atlas.schemeAlt") },
    path("ap-sector", sector), ticks, ghost(-58), ghost(POSE[1]),
    path("ap-route", `M${pt(tip(POSE[0], 161))}A161 161 0 0 1 ${pt(tip(POSE[1], 161))}`),
    path("ap-upper", "M58 161C74 158 91 169 113 171C140 174 167 170 201 164C217 161 232 168 235 177Q239 194 223 198C201 199 172 189 145 188S91 192 71 197Q47 201 43 183Q40 170 58 161Z"),
    path("ap-contour", "M65 174C93 175 116 184 144 181S196 173 214 174"),
    circle("ap-pin", 62, 181, 9), moving, circle("ap-joint", CX, CY, 17), circle("ap-hub", CX, CY, 8), circle("ap-pin", CX, CY, 3),
    path("ap-lead", "M57 195l-10 17H23"), s("text", { class: "ap-label", x: 22, y: 233, text: t("welcome.jShoulder") }),
    path("ap-lead", "M225 199v13l8 5h20"), s("text", { class: "ap-label", x: 258, y: 223, text: t("welcome.jElbow") }), leader, label);
  return { el: graphic, set(a) {
    moving.setAttribute("transform", `translate(${CX} ${CY}) rotate(${a.toFixed(2)})`);
    const [x, y] = tip(a, LENGTH), right = x > 324, dx = right ? -22 : 18, dy = right ? -24 : -14;
    leader.setAttribute("d", `M${pt([x, y-8])}l${dx} ${dy}h${right ? -10 : 10}`);
    label.setAttribute("x", (x + dx + (right ? -14 : 14)).toFixed(1));
    label.setAttribute("y", Math.max(15, y + dy - 4).toFixed(1));
    label.setAttribute("text-anchor", right ? "end" : "start");
  } };
}

export function movementAtlas(arm = "right") {
  const sketch = drawing(), names = t("atlas.phases");
  let phase = 0, angle = POSE[0], frame = 0, playing = false, figure = null, side = arm, gone = false, request = 0;
  const sheet = h("div", { class: "atlas-paper" }, sketch.el);
  const volume = h("div", { class: "atlas-volume", hidden: true });
  const modeBtns = ["drawing", "figure"].map((key, i) => h("button", { type: "button", "aria-pressed": String(i === 0), text: t("atlas." + key), on: { click: () => mode(i) } }));
  const phaseBtns = names.map((text, i) => h("button", { type: "button", "aria-pressed": String(i === 0), on: { click: () => choose(i) } },
    h("span", { class: "atlas-num", "aria-hidden": "true", text: "0" + (i+1) }), h("span", { text })));
  const play = h("button", { class: "atlas-play", type: "button", on: { click: () => playing ? halt() : run() } });
  const cues = t("atlas.cues");
  const note = h("span", { class: "atlas-note", text: cues[0] });
  const status = h("span", { class: "visually-hidden", role: "status" });
  const plate = h("section", { class: "atlas", "aria-label": t("atlas.map"), "data-mode": "drawing" },
    h("div", { class: "atlas-head" }, h("h2", { class: "atlas-name" }, icon("check"), t("atlas.map")), h("div", { class: "atlas-modes", role: "group", "aria-label": t("atlas.mode") }, modeBtns)),
    sheet, volume, h("div", { class: "atlas-caption" }, play, note),
    h("div", { class: "atlas-phases", role: "group", "aria-label": t("atlas.phaseGroup") }, phaseBtns), status);
  function paint(a) { angle = a; sketch.set(a); if (figure) figure.set(180+a); }
  function mark(i) { phase = i; note.textContent = cues[i]; phaseBtns.forEach((b, n) => b.setAttribute("aria-pressed", String(n === i))); }
  function button() { play.replaceChildren(icon(playing ? "pause" : calm() ? "arrow" : "play"), h("span", { text: t(playing ? "atlas.pause" : calm() ? "atlas.nextPhase" : "atlas.play") })); }
  function halt() { cancelAnimationFrame(frame); frame = 0; playing = false; button(); }
  const ease = n => n*n*(3-2*n);
  function choose(i) {
    halt(); mark(i); const from = angle, to = POSE[i], start = performance.now();
    if (calm() || Math.abs(to-from) < 1) { paint(to); return; }
    const tick = now => { const p = Math.min(1, (now-start)/400); paint(from+(to-from)*ease(p)); if (p < 1) frame = requestAnimationFrame(tick); };
    frame = requestAnimationFrame(tick);
  }
  function run() {
    if (calm()) { choose((phase+1)%3); return; }
    halt(); playing = true; button(); mark(0); paint(POSE[0]); const start = performance.now();
    const tick = now => {
      const ms = now-start, back = ms >= 2000, p = Math.min(1, Math.max(0, (ms-(back ? 2000 : 380))/1250));
      const i = back ? 2 : ms < 380 ? 0 : 1; if (phase !== i) mark(i);
      paint(back ? POSE[1]+(POSE[2]-POSE[1])*ease(p) : POSE[0]+(POSE[1]-POSE[0])*ease(p));
      if (ms < 3300) frame = requestAnimationFrame(tick); else { halt(); status.textContent = names[2]; }
    }; frame = requestAnimationFrame(tick);
  }
  function mode(i) {
    const key = i === 1 ? "figure" : "drawing"; if (plate.dataset.mode === key) return;
    const id = ++request; halt(); plate.dataset.mode = key; modeBtns.forEach((b, k) => b.setAttribute("aria-pressed", String(i === k)));
    sheet.hidden = i === 1; volume.hidden = i === 0;
    if (figure) { figure.dispose(); figure = null; } volume.replaceChildren();
    if (i === 0) return;
    const slot = h("div", { "data-stage-slot": true }); volume.append(slot);
    figure = createFigure(slot, { shot: "quarter", side: profileSide(side), glide: false, labels: false }); figure.set(180+angle);
    stageReady().then(ready => {
      if (gone || request !== id || ready) return;
      mode(0); modeBtns[1].disabled = true; note.textContent = t("atlas.figureOff"); status.textContent = note.textContent; modeBtns[0].focus({ preventScroll: true });
    });
  }
  const visibility = () => { if (document.hidden) halt(); };
  document.addEventListener("visibilitychange", visibility);
  paint(angle); button();
  return { el: plate, setSide(value) { side = value; if (figure) figure.activate(profileSide(side)); },
    dispose() { gone = true; ++request; halt(); if (figure) figure.dispose(); document.removeEventListener("visibilitychange", visibility); } };
}
