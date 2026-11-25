// The stage engine: one renderer and one frame loop for the whole app.
// A screen marks where the figure belongs with a [data-stage-slot] element; the engine frames the
// figure into that rectangle with camera.setViewOffset, so the canvas itself never moves or resizes
// with the layout, and the figure can glide from one screen's slot to the next.
// Nothing here renders unless something changed: there is no idle motion.
import {
  WebGLRenderer, Scene, PerspectiveCamera, Group, Vector3, Quaternion, Matrix4, Plane, Mesh, CylinderGeometry, MeshPhysicalMaterial
} from "./three.js";
import { setupRenderer, buildLights, buildEnvironment } from "./lighting.js";
import { loadMannequin } from "./mannequin.js";
import { createArm, selfTest } from "./rig.js";
import { createDial, DIAL_RADIUS } from "./dial.js";
import { createOverlay } from "./overlay.js";
import { SHOTS, FOV } from "./shots.js";
import { calm } from "../core/dom.js";

const DEG = Math.PI / 180;
const MODEL = new URL("../../../assets/models/mannequin.glb", import.meta.url).href;
const SEAT = 0.075;     // how far below the hip joint the plinth top sits
const FLOOR = 0.54;
const SPIN_FRICTION = 0.9;   // per 1/60 s: how quickly a flung figure slows down
const inOut = (x) => (x < .5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const mix = (a, b, k) => a + (b - a) * k;

export async function createEngine(canvas, overlaySvg) {
  const mobile = window.matchMedia("(max-width: 900px)").matches;
  const renderer = new WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
  setupRenderer(renderer, { mobile });
  renderer.localClippingEnabled = true;

  const scene = new Scene();
  const lights = buildLights(scene, { mobile });
  lights.floor.position.y = FLOOR;
  let disposeEnv = buildEnvironment(renderer, scene);
  const camera = new PerspectiveCamera(FOV, 1, 0.1, 40);
  const pivot = new Group(); scene.add(pivot);

  // ----- figure -----
  const model = await loadMannequin(MODEL);
  const arms = { R: createArm(model, "R"), L: createArm(model, "L") };
  // The rig must reproduce the angles it is asked for; if this ever fails the model changed.
  const worst = Math.max(selfTest(model, arms.R), selfTest(model, arms.L));
  console.assert(worst < 1, "stage: elbow self-test off by " + worst.toFixed(2) + " degrees");
  arms.R.relax(); arms.L.relax();

  // A torso mannequin: the legs are folded away into the hips rather than cut, so there is no
  // ragged edge and no hollow inside to see. Scaling a thigh bone to nothing collapses the whole leg.
  for (const side of ["L", "R"]) { const thigh = model.bone("DEF-thigh" + side); if (thigh) thigh.scale.setScalar(1e-4); }

  // Face the viewer and stand on the turning axis.
  model.root.rotation.y = Math.PI;
  model.root.updateWorldMatrix(true, true);
  const hips = model.bone("DEF-hips").getWorldPosition(new Vector3());
  model.root.position.set(-hips.x, 0, -hips.z);
  pivot.add(model.root);

  // The plinth: a turned stand whose top hides the base of the pelvis.
  const plinthMat = new MeshPhysicalMaterial({ color: 0x0c2a33, roughness: 0.46, metalness: 0.25, clearcoat: 0.45, clearcoatRoughness: 0.35 });
  const top = hips.y - SEAT;
  const plinth = new Mesh(new CylinderGeometry(0.16, 0.2, top - FLOOR, 72), plinthMat);
  plinth.position.y = (top + FLOOR) / 2; plinth.castShadow = true;
  const cap = new Mesh(new CylinderGeometry(0.175, 0.175, 0.018, 72), plinthMat);
  cap.position.y = top - 0.009; cap.castShadow = true;
  pivot.add(plinth, cap);

  const dial = createDial();
  scene.add(dial.group);
  const overlay = createOverlay(overlaySvg);

  // ----- state -----
  const st = {
    slot: null, rect: null, glide: null, shown: false,
    side: "R", elbow: 60, elbowTween: null,
    shot: resolveShot("quarter"), shotTween: null,
    labels: false, dialMix: 1, dialTween: null, releaseTimer: 0,
    // The person's own turn of the figure, added on top of the screen's shot: degrees, degrees per second.
    grabbing: false, spinVel: 0, last: 0
  };
  model.paint(st.side);
  let raf = 0, cw = 0, ch = 0, alive = true;

  const S = new Vector3(), E = new Vector3(), W = new Vector3(), N = new Vector3(), X = new Vector3(), Y = new Vector3();
  const v = new Vector3(), m4 = new Matrix4(), qA = new Quaternion(), qB = new Quaternion(), pA = new Vector3(), pB = new Vector3();

  function resolveShot(name) {
    const d = SHOTS[name] || SHOTS.quarter;
    // uy / ue: the person's turn and tilt on top of the shot. A new shot brings them back to zero.
    const out = { yaw: d.yaw, elev: d.elev, bw: d.box[0], bh: d.box[1], tx: 0, ty: 1.2, tz: 0, uy: 0, ue: 0 };
    if (d.target) { out.tx = d.target[0]; out.ty = d.target[1]; out.tz = d.target[2]; }
    return out;
  }
  /** Where the working arm sits for a given yaw: used by the close-up shot. */
  function armFocus(yawDeg) {
    const a = arms[st.side], keep = pivot.rotation.y;
    pivot.rotation.y = yawDeg * DEG * (st.side === "L" ? -1 : 1);
    pivot.updateWorldMatrix(true, true);
    a.joints(S, E, W);
    pivot.rotation.y = keep;
    return [E.x + 0.04 * (st.side === "L" ? -1 : 1), E.y + 0.03, E.z];
  }

  // ----- frame -----
  const request = () => { if (!raf && alive) raf = requestAnimationFrame(frame); };

  function readSlot() {
    const el = st.slot;
    if (!el || !el.isConnected) return null;
    const r = el.getBoundingClientRect();
    return r.width > 4 && r.height > 4 ? { x: r.left, y: r.top, w: r.width, h: r.height } : null;
  }

  function frame(now) {
    raf = 0;
    if (!alive) return;
    // 1. Layout is read first, before anything is written.
    const target = readSlot();
    if (!target) { if (st.shown) { renderer.clear(); canvas.style.opacity = 0; overlay.show(false); st.shown = false; st.labelsShown = false; } return; }
    let busy = false;
    if (st.glide) {
      const k = clamp01((now - st.glide.t0) / st.glide.ms), e = inOut(k), f = st.glide.from;
      st.rect = { x: mix(f.x, target.x, e), y: mix(f.y, target.y, e), w: mix(f.w, target.w, e), h: mix(f.h, target.h, e) };
      if (k < 1) busy = true; else st.glide = null;
    } else st.rect = target;

    // 2. Tweens.
    if (st.shotTween) {
      const tw = st.shotTween, k = clamp01((now - tw.t0) / tw.ms), e = inOut(k);
      for (const key of Object.keys(tw.to)) st.shot[key] = mix(tw.from[key], tw.to[key], e);
      if (k < 1) busy = true; else st.shotTween = null;
    }
    if (st.elbowTween) {
      const tw = st.elbowTween, k = tw.ms ? clamp01((now - tw.t0) / tw.ms) : 1;
      st.elbow = mix(tw.from, tw.goal, inOut(k));
      if (tw.onTick) tw.onTick(st.elbow);
      if (k < 1) busy = true; else { st.elbowTween = null; if (tw.done) { tw.done(); busy = busy || !!st.elbowTween; } }
    }
    if (st.dialTween) {
      const tw = st.dialTween, k = clamp01((now - tw.t0) / tw.ms);
      st.dialMix = mix(tw.from, tw.to, inOut(k));
      if (k < 1) busy = true; else st.dialTween = null;
    }
    // A flung figure keeps turning and slows down, like a turntable.
    const dt = st.last ? Math.min(0.05, (now - st.last) / 1000) : 0;
    st.last = now;
    if (!st.grabbing && Math.abs(st.spinVel) > 2) {
      st.shot.uy += st.spinVel * dt;
      st.spinVel *= Math.pow(SPIN_FRICTION, dt * 60);
      busy = true;
    } else if (!st.grabbing) st.spinVel = 0;

    // 3. Size. Done right before rendering because resizing clears the buffer.
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (w !== cw || h !== ch) {
      cw = w; ch = h;
      const cap = Math.sqrt(4.2e6 / Math.max(1, w * h));  // keep the drawing buffer near 4 megapixels
      renderer.setPixelRatio(Math.max(1, Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2, cap)));
      renderer.setSize(w, h, false);
      overlay.viewBox(w, h);
    }
    if (!cw || !ch) return;

    // 4. Camera: frame the shot's box into the slot, then widen the frustum to the whole canvas.
    const r = st.rect, sh = st.shot, aspect = r.w / r.h, tan = Math.tan(FOV * DEG / 2);
    const dist = Math.max(sh.bh / (2 * tan), sh.bw / (2 * tan * aspect)), el = (sh.elev + sh.ue) * DEG;
    camera.aspect = aspect;
    camera.position.set(sh.tx, sh.ty + dist * Math.sin(el), sh.tz + dist * Math.cos(el));
    camera.lookAt(sh.tx, sh.ty, sh.tz);
    camera.setViewOffset(r.w, r.h, -r.x, -r.y, cw, ch);
    const mirror = st.side === "L" ? -1 : 1;
    pivot.rotation.y = (sh.yaw * mirror + sh.uy) * DEG;

    // 5. Pose.
    const arm = arms[st.side];
    arm.setElbow(st.elbow);
    pivot.updateWorldMatrix(true, true);
    arm.joints(S, E, W);

    // 6. The goniometer rides on the elbow: ring in the plane of the movement, fixed arm up the upper arm.
    arm.hinge(N);
    X.copy(S).sub(E).normalize();
    // The dial's face looks away from the body, whichever arm it is on.
    N.addScaledVector(X, -N.dot(X)).normalize().multiplyScalar(mirror);
    Y.crossVectors(N, X);
    m4.makeBasis(X, Y, N); qA.setFromRotationMatrix(m4);
    pA.copy(E).addScaledVector(N, 0.052);
    v.copy(W).sub(E);
    const angle = Math.atan2(v.dot(Y), v.dot(X));
    if (st.dialMix >= 1) { dial.setAngle(angle); dial.group.position.copy(pA); dial.group.quaternion.copy(qA); dial.group.scale.setScalar(1); dial.fixedArm.visible = true; }
    else {
      // Free state: facing the viewer in the middle of the slot, as the brand mark (the arm is the tail of the Q).
      const k = st.dialMix, big = Math.min(sh.bw, sh.bh) * 0.3 / DIAL_RADIUS;
      pB.set(sh.tx, sh.ty, sh.tz).lerp(camera.position, 0.25); qB.copy(camera.quaternion);
      dial.setAngle(mix(-45 * DEG, angle, k));
      dial.group.position.copy(pB).lerp(pA, k); dial.group.quaternion.copy(qB).slerp(qA, k); dial.group.scale.setScalar(mix(big, 1, k));
      dial.fixedArm.visible = k > 0.7;
    }

    // 7. Draw, then place the labels from what was just drawn.
    // Keep the canvas local to the exercise card, including while scrolling and switching shots.
    renderer.setScissorTest(false); renderer.clear();
    const left = Math.max(0, Math.floor(r.x)), right = Math.min(cw, Math.ceil(r.x + r.w));
    const upper = Math.max(0, Math.floor(r.y)), lower = Math.min(ch, Math.ceil(r.y + r.h));
    if (right > left && lower > upper) {
      renderer.setScissor(left, ch - lower, right - left, lower - upper);
      renderer.setScissorTest(true); renderer.render(scene, camera); renderer.setScissorTest(false);
    }
    overlaySvg.style.clipPath = `inset(${upper}px ${Math.max(0, cw-right)}px ${Math.max(0, ch-lower)}px ${left}px)`;
    if (!st.shown) { canvas.style.opacity = 1; st.shown = true; }
    // Labels need room: in a small slot they would crowd the figure, so they step out.
    const roomy = st.labels && r.w >= 300 && r.h >= 230;
    if (roomy !== st.labelsShown) { st.labelsShown = roomy; overlay.show(roomy); }
    if (roomy) {
      const pts = [S, E, W].map((p) => { v.copy(p).project(camera); return [(v.x + 1) / 2 * cw, (1 - v.y) / 2 * ch]; });
      // Labels go to the side the arm is on, away from the torso, unless there is no room there.
      v.set(0, 1.2, 0).project(camera);
      let away = pts[1][0] >= (v.x + 1) / 2 * cw ? 1 : -1;
      const edge = Math.min(...pts.map((p) => (away > 0 ? cw - p[0] : p[0])));
      if (edge < 170) away = -away;
      overlay.place(pts, away);
    }
    if (busy) request();
  }

  // ----- outside world -----
  const onLayout = () => request();
  window.addEventListener("resize", onLayout);
  window.addEventListener("scroll", onLayout, { passive: true });
  const ro = new ResizeObserver(onLayout); ro.observe(canvas);
  canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); });
  canvas.addEventListener("webglcontextrestored", () => { disposeEnv = buildEnvironment(renderer, scene); cw = ch = 0; request(); });

  const api = {
    /** A screen takes the stage. The figure glides there if it was somewhere else a moment ago. */
    acquire(slot, { shot = "quarter", labels = false, side, glide = true } = {}) {
      clearTimeout(st.releaseTimer);
      const from = st.rect && st.shown ? st.rect : null;
      st.slot = slot;
      if (side) api.activate(side);
      api.shot(shot, from && glide ? 900 : 0);
      st.glide = from && glide && !calm() ? { from, t0: performance.now(), ms: 900 } : null;
      api.labels(labels);
      request();
    },
    /** A screen leaves. The figure stays for a beat in case the next screen wants it. */
    release(slot) {
      if (st.slot !== slot) return;
      st.elbowTween = null;
      st.releaseTimer = setTimeout(() => { if (st.slot === slot) { st.slot = null; request(); } }, 60);
    },
    shot(name, ms = 900) {
      const to = resolveShot(name);
      if ((SHOTS[name] || {}).focus === "arm") { const f = armFocus(to.yaw); to.tx = f[0]; to.ty = f[1]; to.tz = f[2]; }
      if (calm() || !ms || !st.shown) { Object.assign(st.shot, to); st.shotTween = null; }
      else st.shotTween = { from: { ...st.shot }, to, t0: performance.now(), ms };
      request();
    },
    /** Which arm exercises: "R" or "L". The view mirrors around it. */
    activate(side) {
      if (side !== "R" && side !== "L" || side === st.side) return;
      arms[st.side].relax(); st.side = side; model.paint(side); request();
    },
    labels(on, texts) { st.labels = !!on; if (texts) overlay.setTexts(texts); if (!st.labels) { st.labelsShown = false; overlay.show(false); } request(); },
    look(kind) { model.look(kind || ""); dial.group.visible = !kind; request(); },
    set(deg) { st.elbowTween = null; st.elbow = deg; request(); },
    to(goal, ms, onTick, done) {
      if (calm() || !ms) { st.elbowTween = null; st.elbow = goal; if (onTick) onTick(goal); request(); if (done) done(); return; }
      st.elbowTween = { from: st.elbow, goal, ms, t0: performance.now(), onTick, done }; request();
    },
    stop() { st.elbowTween = null; },

    // ----- the person's hands on the figure -----
    /** The pointer went down on the figure: it follows the hand and stops spinning. */
    grab(on) { st.grabbing = !!on; if (on) { st.spinVel = 0; st.shotTween = null; } request(); },
    /** Turn by a drag: yaw follows the hand, tilt is kept within a calm range. */
    spin(dYaw, dTilt = 0) {
      st.shot.uy += dYaw;
      st.shot.ue = Math.max(-10, Math.min(24, st.shot.ue + dTilt));
      request();
    },
    /** Let go with a speed in degrees per second; the figure coasts to a stop. */
    fling(vel) { st.grabbing = false; st.spinVel = calm() ? 0 : Math.max(-720, Math.min(720, vel)); request(); },
    /** Turn by a fixed step, animated: the buttons and the arrow keys. */
    turn(deg, ms = 520) {
      st.spinVel = 0;
      const to = { ...st.shot, uy: st.shot.uy + deg };
      if (calm()) { Object.assign(st.shot, to); st.shotTween = null; } else st.shotTween = { from: { ...st.shot }, to, t0: performance.now(), ms };
      request();
    },
    /** Back to the view the screen asked for. */
    resetView(ms = 700) {
      st.spinVel = 0;
      // Come back the short way round.
      const uy = ((st.shot.uy % 360) + 540) % 360 - 180;
      st.shot.uy = uy;
      const to = { ...st.shot, uy: 0, ue: 0 };
      if (calm()) { Object.assign(st.shot, to); st.shotTween = null; } else st.shotTween = { from: { ...st.shot }, to, t0: performance.now(), ms };
      request();
    },
    get turned() { return Math.abs(st.shot.uy) > 0.5 || Math.abs(st.shot.ue) > 0.5; },

    /** The opening moment: the dial hangs in front of the figure, then settles on the elbow. */
    intro(ms = 1400) {
      if (calm()) { st.dialMix = 1; request(); return; }
      st.dialMix = 0; st.dialTween = { from: 0, to: 1, t0: performance.now() + 500, ms };
      request();
    },
    get angle() { return st.elbow; },
    dispose() {
      alive = false; cancelAnimationFrame(raf); ro.disconnect();
      window.removeEventListener("resize", onLayout); window.removeEventListener("scroll", onLayout);
      model.dispose(); dial.dispose(); plinth.geometry.dispose(); cap.geometry.dispose(); plinthMat.dispose(); disposeEnv(); renderer.dispose();
    }
  };
  return api;
}
