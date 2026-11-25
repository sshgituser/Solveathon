// The arm: shoulder, elbow, wrist and the elbow angle arc. One drawing, used as the
// brand illustration and as the verdict figure. It shows a category, never a measured angle.
import { s, calm } from "../core/dom.js";
import { t } from "../core/i18n.js";

const S = [100, 290], E = [260, 290], LEN = 180, R = 62;

export function createArm({ labels = false } = {}) {
  const svg = s("svg", { class: "arm", viewBox: "0 0 520 400", "aria-hidden": "true" });
  svg.appendChild(s("path", { class: "target", d: `M${E[0]} ${E[1]}H${E[0] + LEN}` }));
  const rest = svg.appendChild(s("path", { class: "arc-rest" }));
  const arc = svg.appendChild(s("path", { class: "arc" }));
  const bone = svg.appendChild(s("path", { class: "bone" }));
  svg.appendChild(s("circle", { class: "joint", r: 13, cx: S[0], cy: S[1] }));
  svg.appendChild(s("circle", { class: "joint", r: 13, cx: E[0], cy: E[1] }));
  const wrist = svg.appendChild(s("circle", { class: "joint", r: 13 }));
  let wristLabel = null;
  if (labels) {
    svg.appendChild(s("text", { x: S[0], y: S[1] + 44, "text-anchor": "middle", text: t("welcome.jShoulder") }));
    svg.appendChild(s("text", { x: E[0], y: E[1] + 44, "text-anchor": "middle", text: t("welcome.jElbow") }));
    wristLabel = svg.appendChild(s("text", { "text-anchor": "middle", text: t("welcome.jWrist") }));
  }

  let raf = 0, theta = 60;
  const pt = (r, deg) => { const a = deg * Math.PI / 180; return [(E[0] + r * Math.cos(a)).toFixed(1), (E[1] - r * Math.sin(a)).toFixed(1)]; };

  function set(th) {
    theta = Math.min(th, 179.5);
    const phi = 180 - theta, w = pt(LEN, phi), a = pt(R, phi);
    bone.setAttribute("d", `M${S[0]} ${S[1]}L${E[0]} ${E[1]}L${w[0]} ${w[1]}`);
    arc.setAttribute("d", `M${E[0] - R} ${E[1]}A${R} ${R} 0 0 1 ${a[0]} ${a[1]}`);
    rest.setAttribute("d", `M${a[0]} ${a[1]}A${R} ${R} 0 0 1 ${E[0] + R} ${E[1]}`);
    wrist.setAttribute("cx", w[0]); wrist.setAttribute("cy", w[1]);
    if (wristLabel) { wristLabel.setAttribute("x", w[0]); wristLabel.setAttribute("y", w[1] - 26); }
  }

  function to(goal, ms, onTick, done) {
    cancelAnimationFrame(raf);
    if (calm() || !ms) { set(goal); if (onTick) onTick(goal); if (done) done(); return; }
    const from = theta, t0 = performance.now();
    const ease = (x) => (x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
    (function frame(now) {
      const k = Math.min(Math.max((now - t0) / ms, 0), 1), v = from + (goal - from) * ease(k);
      set(v); if (onTick) onTick(v);
      if (k < 1) raf = requestAnimationFrame(frame); else if (done) done();
    })(t0);
  }

  set(60);
  return { el: svg, set, to, stop: () => cancelAnimationFrame(raf), look: (v) => svg.setAttribute("data-look", v || "") };
}

/** Bend, extend, hold, repeat. Runs only while visible. Returns a stop function. */
export function loopArm(arm, host, onTick) {
  let timer = 0, on = false;
  const tick = onTick || (() => {});
  function cycle() {
    if (!on) return;
    arm.to(178, 1500, tick, () => { timer = setTimeout(() => arm.to(60, 1100, tick, () => { timer = setTimeout(cycle, 600); }), 1100); });
  }
  function halt() { on = false; clearTimeout(timer); arm.stop(); }
  const io = new IntersectionObserver((en) => {
    if (calm()) { halt(); arm.set(178); tick(178); return; }
    if (en[0].isIntersecting && !on) { on = true; cycle(); } else if (!en[0].isIntersecting) halt();
  }, { threshold: .2 });
  io.observe(host);
  return () => { io.disconnect(); halt(); if (arm.dispose) arm.dispose(); };
}
