// The goniometer: the instrument a therapist lays on a joint to read its angle, and the brand mark.
// A flat machined ring with graduations, a fixed arm along the upper arm and a moving arm on a centre pin.
import {
  Group, Mesh, LatheGeometry, CylinderGeometry, CapsuleGeometry, BoxGeometry, InstancedMesh,
  MeshPhysicalMaterial, MeshStandardMaterial, Vector2, Object3D
} from "./three.js";

export const DIAL_RADIUS = 0.105;

/**
 * @param {{ ring?: number, arm?: number, ticks?: number }} colors hex colours
 * @returns {{ group: Group, setAngle(rad: number): void, fixedArm: Mesh, dispose(): void }}
 */
export function createDial({ ring = 0x0c2a33, arm = 0x0f8fa6, ticks = 0xe9f1f3 } = {}) {
  const group = new Group();
  const mRing = new MeshPhysicalMaterial({ color: ring, roughness: 0.42, metalness: 0.35, clearcoat: 0.4, clearcoatRoughness: 0.3 });
  const mArm = new MeshPhysicalMaterial({ color: arm, roughness: 0.3, metalness: 0.15, clearcoat: 0.7, clearcoatRoughness: 0.2 });
  const mTick = new MeshStandardMaterial({ color: ticks, roughness: 0.6 });

  // Ring: a rectangular profile turned around the axis, then laid flat in the XY plane.
  const t = 0.006, r0 = 0.078, r1 = DIAL_RADIUS;
  const ringGeo = new LatheGeometry([new Vector2(r0, -t), new Vector2(r1, -t), new Vector2(r1, t), new Vector2(r0, t), new Vector2(r0, -t)], 96);
  ringGeo.rotateX(Math.PI / 2);
  const ringMesh = new Mesh(ringGeo, mRing); ringMesh.castShadow = true;

  // Graduations every 10 degrees, longer every 30, raised a hair above the ring face.
  const tickGeo = new BoxGeometry(1, 0.0024, 0.0016);
  const tickMesh = new InstancedMesh(tickGeo, mTick, 36);
  const o = new Object3D();
  for (let i = 0; i < 36; i++) {
    const a = i * 10 * Math.PI / 180, len = i % 3 === 0 ? 0.017 : 0.009, mid = r1 - 0.004 - len / 2;
    o.position.set(Math.cos(a) * mid, Math.sin(a) * mid, t + 0.0006); o.rotation.set(0, 0, a); o.scale.set(len, 1, 1); o.updateMatrix();
    tickMesh.setMatrixAt(i, o.matrix);
  }

  // Arms: flat bars with round ends. The fixed one points along +X; the moving one turns about the pin.
  const bar = (len, mat) => {
    const g = new CapsuleGeometry(0.011, len, 6, 20); g.rotateZ(-Math.PI / 2); g.translate(len / 2, 0, 0); g.scale(1, 1, 0.34);
    const m = new Mesh(g, mat); m.castShadow = true; return m;
  };
  const fixedArm = bar(0.19, mRing); fixedArm.position.z = -0.004;
  const movingArm = bar(0.215, mArm); movingArm.position.z = 0.0105;
  const pinGeo = new CylinderGeometry(0.015, 0.015, 0.03, 32); pinGeo.rotateX(Math.PI / 2);
  const pin = new Mesh(pinGeo, mRing); pin.position.z = 0.004;

  group.add(ringMesh, tickMesh, fixedArm, movingArm, pin);
  return {
    group, fixedArm,
    /** Angle of the moving arm from the fixed arm, counter-clockwise, in radians. */
    setAngle(rad) { movingArm.rotation.z = rad; },
    dispose() { [ringGeo, tickGeo, fixedArm.geometry, movingArm.geometry, pinGeo].forEach((g) => g.dispose()); [mRing, mArm, mTick].forEach((m) => m.dispose()); }
  };
}
