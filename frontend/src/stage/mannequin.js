// The figure: Quaternius "Animated Base Character" (CC BY 3.0, see assets/models/CREDITS.md),
// re-dressed as a clinical torso mannequin: matte ceramic body, neutral joints, and the joints of
// the working arm in the brand colour.
import { GLTFLoader, MeshPhysicalMaterial, PropertyBinding, BufferAttribute, Color } from "./three.js";

export const BRAND = 0x0f8fa6;   // the brand teal, lifted a little so it reads under studio light
const BODY = 0xdfe4e6;           // below pure white on purpose: highlights need headroom
const NEUTRAL = 0xa9b7bc;        // joints that are not being exercised

function materials() {
  return {
    body: new MeshPhysicalMaterial({ color: BODY, roughness: 0.55, metalness: 0, clearcoat: 0.18, clearcoatRoughness: 0.5 }),
    joints: new MeshPhysicalMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.34, metalness: 0.05, clearcoat: 0.6, clearcoatRoughness: 0.25 })
  };
}

/** Colour joint vertices by the bone that moves them, so one arm can be picked out. */
function jointPainter(mesh, skeleton) {
  const geo = mesh.geometry, n = geo.attributes.position.count;
  const index = geo.attributes.skinIndex, weight = geo.attributes.skinWeight;
  const owner = new Array(n);
  for (let i = 0; i < n; i++) {
    let best = 0, top = -1;
    for (let k = 0; k < 4; k++) { const w = weight.getComponent(i, k); if (w > top) { top = w; best = index.getComponent(i, k); } }
    owner[i] = skeleton.bones[best] ? skeleton.bones[best].name : "";
  }
  const colors = new Float32Array(n * 3);
  geo.setAttribute("color", new BufferAttribute(colors, 3));
  const on = new Color(BRAND), off = new Color(NEUTRAL);
  return (side) => {
    const arm = side ? new RegExp("(upper_arm|forearm|hand)" + side + "$") : null;
    for (let i = 0; i < n; i++) { const c = arm && arm.test(owner[i]) ? on : off; colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b; }
    geo.attributes.color.needsUpdate = true;
  };
}

export async function loadMannequin(url) {
  const gltf = await new GLTFLoader().loadAsync(url);
  const root = gltf.scene, mats = materials();
  let skeleton = null, paint = () => {};
  root.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    skeleton = skeleton || o.skeleton;
    o.frustumCulled = false;      // bones move the mesh outside its rest bounds
    o.castShadow = true;
    if (o.material && o.material.name === "M_Joints") { o.material = mats.joints; paint = jointPainter(o, o.skeleton); }
    else o.material = mats.body;
  });
  paint(null);
  // GLTFLoader strips "." from node names, so "DEF-forearm.R" is "DEF-forearmR" here.
  const bone = (name) => skeleton.getBoneByName(PropertyBinding.sanitizeNodeName(name));
  const grey = new Color(0x9aa8ad), white = new Color(0xffffff);

  return {
    root, skeleton, bone, materials: mats,
    /** Pick out one arm ("R" | "L") or none. */
    paint,
    /** Cut the figure with a plane (the torso stands on a plinth) and keep the shadow consistent. */
    clip(plane) { for (const m of [mats.body, mats.joints]) { m.clippingPlanes = [plane]; m.clipShadows = true; m.needsUpdate = true; } },
    /** Looks for result states: normal, translucent "could not score", emptied "arm not visible", dimmed "no answer". */
    look(kind) {
      const ghost = kind === "uncertain" || kind === "nopose" || kind === "off";
      mats.body.transparent = ghost; mats.body.depthWrite = !ghost;
      mats.body.opacity = kind === "off" ? 0.32 : kind === "nopose" ? 0.2 : kind === "uncertain" ? 0.5 : 1;
      mats.joints.color.copy(kind === "nopose" || kind === "off" ? grey : white);
      mats.joints.transparent = ghost; mats.joints.opacity = kind === "nopose" ? 0.3 : kind === "off" ? 0.45 : kind === "uncertain" ? 0.75 : 1;
      mats.body.needsUpdate = mats.joints.needsUpdate = true;
    },
    dispose() {
      root.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
      mats.body.dispose(); mats.joints.dispose();
    }
  };
}
