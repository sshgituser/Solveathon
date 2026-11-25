// The instruction for E02: an offline vector illustration, not a recording of a person.
// It is deliberately labelled as an illustration (DESIGN.md: the figure shows a category, not
// a measured angle). Four scenes: camera side-on, the whole arm in frame, one extension, and the
// 30-second limit. Captions come from the dictionaries, so RU and KK switch with the app.
// The playback starts only on the person's command and runs once; afterwards the same button
// replays it. With reduced motion the illustration stays still and the steps below describe it.
import { h, s, calm } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { icon } from "./icons.js";

const W = 640;
const H = 360;

// The arm is a side view: the upper arm hangs from the shoulder to the elbow and the forearm
// turns in front of the body. 96° is the bent start, 180° the straight end.
const SHOULDER = [382, 176];
const ELBOW = [360, 236];
const FOREARM = 126;
const BENT = 96;
const STRAIGHT = 180;

const TOTAL = 10400;
const SCENES = [
  { at: 0, name: "setup" },
  { at: 2400, name: "frame" },
  { at: 4600, name: "move" },
  { at: 8800, name: "limit" }
];
const MOVE = [4600, 5800, 6400, 7600, 8800];

const rad = (deg) => (deg * Math.PI) / 180;
const point = (deg, r) => [ELBOW[0] + r * Math.cos(rad(deg)), ELBOW[1] - r * Math.sin(rad(deg))];
const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
const lock = (n) => Number(n.toFixed(1));

function armAt(ms) {
  if (ms < MOVE[0]) return BENT;
  if (ms < MOVE[1]) return BENT + (STRAIGHT - BENT) * ease((ms - MOVE[0]) / (MOVE[1] - MOVE[0]));
  if (ms < MOVE[2]) return STRAIGHT;
  if (ms < MOVE[3]) return STRAIGHT + (BENT - STRAIGHT) * ease((ms - MOVE[2]) / (MOVE[3] - MOVE[2]));
  return BENT;
}

function sceneAt(ms) {
  for (let i = SCENES.length - 1; i >= 0; i -= 1) if (ms >= SCENES[i].at) return i;
  return 0;
}

export function createInstruction() {
  const captions = t("check.instrVideo.captions");

  const svg = s("svg", {
    class: "i-svg", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": t("check.instrVideo.alt")
  });

  // Behind everything: the camera frame and the sight line from the lens to the elbow.
  svg.appendChild(s("rect", { class: "i-frame", x: 44, y: 64, width: 500, height: 250, rx: 12 }));
  svg.appendChild(s("path", { class: "i-sight", d: `M83 188L${ELBOW[0]} ${ELBOW[1]}` }));
  svg.appendChild(s("path", { class: "i-floor", d: "M28 312H608" }));

  // Camera on a tripod: a phone with a lens, at elbow height.
  svg.appendChild(s("g", { class: "i-camera" },
    s("rect", { x: 62, y: 174, width: 42, height: 70, rx: 8 }),
    s("circle", { cx: 83, cy: 188, r: 5 }),
    s("path", { d: "M83 244L58 312M83 244L108 312M83 244V312" })));

  // Chair and a seated person in profile, facing the camera.
  svg.appendChild(s("g", { class: "i-chair" },
    s("rect", { x: 490, y: 150, width: 16, height: 112, rx: 7 }),
    s("rect", { x: 352, y: 252, width: 152, height: 16, rx: 8 }),
    s("path", { d: "M366 268V312M492 268V312" })));
  svg.appendChild(s("g", { class: "i-person" },
    s("circle", { class: "i-head", cx: 398, cy: 112, r: 23 }),
    s("path", { class: "i-neck", d: "M396 136V156" }),
    s("path", { class: "i-torso", d: `M${SHOULDER[0]} ${SHOULDER[1]}L414 252` }),
    s("path", { class: "i-thigh", d: "M414 252L330 256" }),
    s("path", { class: "i-shin", d: "M330 256V312" }),
    s("path", { class: "i-foot", d: "M330 312H368" })));

  // The working arm: drawn over the torso so the movement is unambiguous.
  const upper = s("path", { class: "i-upper", d: `M${SHOULDER[0]} ${SHOULDER[1]}L${ELBOW[0]} ${ELBOW[1]}` });
  const fore = s("path", { class: "i-fore" });
  const hand = s("circle", { class: "i-hand", r: 10 });
  const wrist = s("circle", { class: "i-joint i-wrist", r: 7 });
  const arc = s("path", { class: "i-arc" });
  svg.appendChild(upper);
  svg.appendChild(fore);
  svg.appendChild(hand);
  svg.appendChild(s("circle", { class: "i-joint i-shoulder", cx: SHOULDER[0], cy: SHOULDER[1], r: 7 }));
  svg.appendChild(s("circle", { class: "i-joint i-elbow", cx: ELBOW[0], cy: ELBOW[1], r: 7 }));
  svg.appendChild(wrist);
  svg.appendChild(arc);

  // The 30-second limit: a clock face, not a countdown with digits.
  svg.appendChild(s("g", { class: "i-clock" },
    s("circle", { cx: 560, cy: 108, r: 30 }),
    s("path", { d: "M560 108V90M560 108L574 116" })));

  function setArm(deg) {
    const w = point(deg, FOREARM);
    fore.setAttribute("d", `M${ELBOW[0]} ${ELBOW[1]}L${lock(w[0])} ${lock(w[1])}`);
    hand.setAttribute("cx", lock(w[0]));
    hand.setAttribute("cy", lock(w[1]));
    wrist.setAttribute("cx", lock(w[0]));
    wrist.setAttribute("cy", lock(w[1]));
  }
  function setArc() {
    const a = point(BENT, 80);
    const b = point(STRAIGHT, 80);
    arc.setAttribute("d", `M${lock(a[0])} ${lock(a[1])}A80 80 0 0 0 ${lock(b[0])} ${lock(b[1])}`);
  }
  setArm(BENT);
  setArc();

  const caption = h("p", { class: "c-instr-caption", "aria-live": "polite" });
  const bars = SCENES.map(() => h("span"));
  const bar = h("div", { class: "i-bars", "aria-hidden": "true" }, bars);
  const button = h("button", { type: "button", class: "btn btn--quiet btn--small" });

  let playing = false;
  let ended = false;
  let offset = 0;
  let started = 0;
  let raf = 0;
  let shown = -1;

  function paintButton() {
    button.textContent = "";
    const key = playing ? "check.instrVideo.pause" : ended ? "common.replay" : offset > 0 ? "check.instrVideo.resume" : "check.instrVideo.play";
    button.append(icon(playing ? "pause" : ended ? "replay" : "play"), h("span", { text: t(key) }));
  }

  function paint(ms) {
    const idx = sceneAt(ms);
    svg.dataset.scene = SCENES[idx].name;
    setArm(armAt(ms));
    bars.forEach((b, i) => b.classList.toggle("on", i <= idx));
    if (idx !== shown) {
      shown = idx;
      caption.textContent = captions[idx];
    }
  }

  function finish() {
    playing = false;
    ended = true;
    offset = TOTAL;
    cancelAnimationFrame(raf);
    paint(TOTAL);
    paintButton();
  }

  function frame(now) {
    if (!playing) return;
    if (!started) started = now - offset;
    const ms = Math.min(now - started, TOTAL);
    paint(ms);
    if (ms < TOTAL) raf = requestAnimationFrame(frame);
    else finish();
  }

  function play() {
    if (playing) return;
    if (calm()) {
      ended = true;
      offset = TOTAL;
      paint(TOTAL);
      caption.textContent = t("check.instrVideo.calm");
      paintButton();
      return;
    }
    if (ended || offset >= TOTAL) offset = 0;
    playing = true;
    ended = false;
    started = 0;
    paintButton();
    raf = requestAnimationFrame(frame);
  }

  function pause() {
    if (!playing) return;
    playing = false;
    offset = Math.min(performance.now() - started, TOTAL);
    cancelAnimationFrame(raf);
    paintButton();
  }

  function stop() {
    playing = false;
    cancelAnimationFrame(raf);
  }

  button.addEventListener("click", () => (playing ? pause() : play()));

  const el = h("figure", { class: "c-instr" },
    h("div", { class: "c-instr-frame" }, svg),
    h("figcaption", null,
      h("div", { class: "c-instr-head" },
        h("strong", { text: t("check.instrVideo.title") }),
        h("span", { class: "fine", text: t("check.instrVideo.label") })),
      caption,
      h("div", { class: "c-instr-row" }, button, bar),
      h("p", { class: "fine", text: t("check.instrVideo.note") })));

  paint(0);
  paintButton();

  return { el, play, pause, stop, dispose: stop };
}
