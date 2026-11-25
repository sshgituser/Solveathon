// The Qolda mark in 3D, "Support": a ceramic cupped palm holding a lacquered teal ball, the joint.
// On arrival, and when the pointer comes near, the ball is lifted and set down into the palm again:
// held, supported. The mark tilts toward the pointer while hovered.
// Rendered on demand only: no frame is drawn while nothing moves.
import {
  WebGLRenderer, Scene, PerspectiveCamera, DirectionalLight, HemisphereLight, NeutralToneMapping,
  Group, Mesh, Shape, ExtrudeGeometry, SphereGeometry, MeshPhysicalMaterial
} from "./three.js";
import { buildEnvironment } from "./lighting.js";
import { calm } from "../core/dom.js";

// The flat mark lives on a 64-unit grid (see ui/icons.js). Same points here, y flipped, centred.
const U = 1 / 64;
const P = (x, y) => [(x - 32) * U, (32 - y) * U];
const STROKE = 8 * U, DEPTH = 6 * U;
const [CX, CY] = P(32, 33);                  // centre of the palm's curve
const R_OUT = (20 + 4) * U, R_IN = (20 - 4) * U, CAP = STROKE / 2;
const TOP = P(0, 22)[1];                      // height of the two fingertips
const BALL_R = 8.5 * U, BALL_Y = P(32, 26)[1];
const LIFT = 0.42;                            // how high the ball is raised before it is set down

/** The palm as one outline: outer curve, a round fingertip, inner curve, the other fingertip. */
function palmShape() {
  const s = new Shape();
  const lx = CX - (R_OUT + R_IN) / 2, rx = CX + (R_OUT + R_IN) / 2;   // stroke centre lines
  s.moveTo(CX - R_OUT, TOP);
  s.lineTo(CX - R_OUT, CY);
  s.absarc(CX, CY, R_OUT, Math.PI, Math.PI * 2, false);                // under the bottom
  s.lineTo(CX + R_OUT, TOP);
  s.absarc(rx, TOP, CAP, 0, Math.PI, false);                            // right fingertip
  s.lineTo(CX + R_IN, CY);
  s.absarc(CX, CY, R_IN, Math.PI * 2, Math.PI, true);                   // inside of the palm
  s.lineTo(CX - R_IN, TOP);
  s.absarc(lx, TOP, CAP, 0, Math.PI, false);                            // left fingertip
  return s;
}

// The ball comes down, slightly overshoots into the palm and settles: a short damped bounce.
const settle = (k) => 1 - Math.pow(1 - k, 3) * Math.cos(k * Math.PI * 1.6) * (1 - k * 0.2);

/**
 * Replace the flat mark inside `box` with a live one.
 * @param trigger element whose hover or focus replays the motion (the rail, the brand block)
 * @param opts { size: CSS pixels, dark: true on the navy rail }
 * @returns dispose function, or null when WebGL refused (the flat mark stays)
 */
export function mountLogo(box, trigger, { size = 56, dark = true } = {}) {
  const canvas = document.createElement("canvas");
  canvas.className = "logo3d"; canvas.setAttribute("aria-hidden", "true");
  let renderer;
  try { renderer = new WebGLRenderer({ canvas, alpha: true, antialias: true }); } catch (_) { return null; }
  renderer.toneMapping = NeutralToneMapping;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(size, size, false);

  const scene = new Scene();
  let disposeEnv = buildEnvironment(renderer, scene);
  const key = new DirectionalLight(0xffffff, 2.4); key.position.set(-1.2, 2, 3);
  const rim = new DirectionalLight(dark ? 0x9fe3ee : 0xffffff, 1.2); rim.position.set(2, -1, -1.5);
  scene.add(key, rim, new HemisphereLight(0xffffff, dark ? 0x2a4e57 : 0xc9d8dc, 0.9));
  const camera = new PerspectiveCamera(26, 1, 0.1, 10);
  camera.position.set(0, 0, 2.35); camera.lookAt(0, 0, 0);

  // A ceramic palm, a lacquered brand-teal joint.
  const palmMat = new MeshPhysicalMaterial({ color: dark ? 0xeef4f5 : 0x0c2a33, roughness: dark ? 0.32 : 0.36, metalness: dark ? 0 : 0.2, clearcoat: 0.8, clearcoatRoughness: 0.18 });
  const ballMat = new MeshPhysicalMaterial({ color: dark ? 0x2fb5cb : 0x0b7285, roughness: 0.18, metalness: 0.1, clearcoat: 1, clearcoatRoughness: 0.08 });

  const palmGeo = new ExtrudeGeometry(palmShape(), { depth: DEPTH, bevelEnabled: true, bevelThickness: 1.6 * U, bevelSize: 1.4 * U, bevelSegments: 6, curveSegments: 48 });
  palmGeo.translate(0, 0, -DEPTH / 2);
  const palm = new Mesh(palmGeo, palmMat);
  const ball = new Mesh(new SphereGeometry(BALL_R, 48, 32), ballMat);
  ball.position.set(0, BALL_Y, DEPTH * 0.15);

  const mark = new Group();
  mark.add(palm, ball);
  mark.position.set(0, 0.03, 0);
  scene.add(mark);

  // ----- motion: render only while something moves -----
  const st = { drop: calm() ? 0 : 1, t0: 0, ms: 1150, tiltX: 0, tiltY: 0, aimX: 0, aimY: 0 };
  let raf = 0, alive = true;
  const BASE_X = -0.26, BASE_Y = 0.32;   // a resting three-quarter view from slightly above, so the cup reads

  function frame(now) {
    raf = 0;
    if (!alive) return;
    let busy = false;
    if (st.t0) {
      const k = Math.min(1, Math.max(0, (now - st.t0) / st.ms));
      st.drop = 1 - settle(k);
      if (k < 1) busy = true; else { st.t0 = 0; st.drop = 0; }
    }
    st.tiltX += (st.aimX - st.tiltX) * 0.14; st.tiltY += (st.aimY - st.tiltY) * 0.14;
    if (Math.abs(st.aimX - st.tiltX) > 0.002 || Math.abs(st.aimY - st.tiltY) > 0.002) busy = true;
    ball.position.y = BALL_Y + LIFT * st.drop;
    mark.rotation.set(BASE_X + st.tiltX, BASE_Y + st.tiltY, 0);
    renderer.render(scene, camera);
    if (busy) raf = requestAnimationFrame(frame);
  }
  const request = () => { if (!raf && alive) raf = requestAnimationFrame(frame); };

  /** Lift the joint and set it down into the palm. */
  function place(delay = 0) {
    if (calm()) { st.drop = 0; request(); return; }
    if (st.t0) return;
    st.drop = 1; st.t0 = performance.now() + delay;
    request();
  }
  const near = (e) => {
    if (calm()) return;
    const r = canvas.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    st.aimY = Math.max(-1, Math.min(1, (e.clientX - cx) / 160)) * 0.5;
    st.aimX = Math.max(-1, Math.min(1, (e.clientY - cy) / 160)) * 0.4;
    request();
  };
  const away = () => { st.aimX = 0; st.aimY = 0; request(); };
  const onEnter = () => place();
  if (trigger) {
    trigger.addEventListener("pointerenter", onEnter);
    trigger.addEventListener("focusin", onEnter);
    trigger.addEventListener("pointermove", near);
    trigger.addEventListener("pointerleave", away);
  }
  canvas.addEventListener("webglcontextlost", (e) => e.preventDefault());
  canvas.addEventListener("webglcontextrestored", () => { disposeEnv = buildEnvironment(renderer, scene); request(); });

  box.textContent = ""; box.appendChild(canvas);
  request();
  place(250);

  return () => {
    alive = false; cancelAnimationFrame(raf);
    if (trigger) {
      trigger.removeEventListener("pointerenter", onEnter); trigger.removeEventListener("focusin", onEnter);
      trigger.removeEventListener("pointermove", near); trigger.removeEventListener("pointerleave", away);
    }
    scene.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    palmMat.dispose(); ballMat.dispose(); disposeEnv(); renderer.dispose();
  };
}
