// Elbow control. The hinge axis is measured from the model's own rest pose instead of guessed,
// so the same code drives either arm and survives a different mannequin.
import { Vector3, Quaternion } from "./three.js";

const DEG = Math.PI / 180;

/** @param side "R" | "L" */
export function createArm(model, side) {
  const upper = model.bone("DEF-upper_arm" + side), fore = model.bone("DEF-forearm" + side), hand = model.bone("DEF-hand" + side);
  if (!upper || !fore || !hand) throw new Error("arm bones not found: " + side);
  model.root.updateWorldMatrix(true, true);

  const S = upper.getWorldPosition(new Vector3()), E = fore.getWorldPosition(new Vector3()), W = hand.getWorldPosition(new Vector3());
  const u = E.clone().sub(S), f = W.clone().sub(E);
  const n = u.clone().cross(f);                         // hinge axis; rotating the forearm about it flexes
  const rest = Math.atan2(n.length(), u.dot(f));        // how far the rest pose is from a straight arm
  n.normalize();
  const axisLocal = n.clone().applyQuaternion(fore.getWorldQuaternion(new Quaternion()).invert());
  const foreRest = fore.quaternion.clone(), handRest = hand.quaternion.clone(), upperRest = upper.quaternion.clone();
  const q = new Quaternion(), a = new Vector3(), b = new Vector3(), c = new Vector3();
  let angle = 180 - rest / DEG;

  return {
    side, bones: [upper, fore, hand], upper, fore, hand,
    get angle() { return angle; },
    /** Elbow angle in degrees: 180 is a straight arm. Absolute, never incremental. */
    setElbow(deg) {
      angle = Math.max(35, Math.min(179.5, deg));
      fore.quaternion.copy(foreRest).multiply(q.setFromAxisAngle(axisLocal, (180 - angle) * DEG - rest));
      hand.quaternion.copy(handRest);
    },
    /** Put the arm back to its rest pose (used for the arm that is not exercising). */
    relax() { upper.quaternion.copy(upperRest); fore.quaternion.copy(foreRest); hand.quaternion.copy(handRest); angle = 180 - rest / DEG; },
    /** The angle the model actually shows, from joint positions. Used by the overlay and the self-test. */
    measure() {
      upper.getWorldPosition(a); fore.getWorldPosition(b); hand.getWorldPosition(c);
      const x = a.sub(b), y = c.sub(b);
      return Math.acos(Math.max(-1, Math.min(1, x.dot(y) / (x.length() * y.length())))) / DEG;
    },
    /** World positions of shoulder, elbow, wrist, written into the given vectors. */
    joints(s, e, w) { upper.getWorldPosition(s); fore.getWorldPosition(e); hand.getWorldPosition(w); },
    /** The hinge axis in world space. Stable even when the arm is straight, unlike a cross product of the limbs. */
    hinge(out) { return out.copy(axisLocal).applyQuaternion(fore.getWorldQuaternion(q)).normalize(); }
  };
}

/** Command a few angles and check the model really shows them. Returns the worst error in degrees. */
export function selfTest(model, arm) {
  let worst = 0;
  for (const deg of [60, 90, 135, 178]) {
    arm.setElbow(deg);
    model.root.updateWorldMatrix(true, true);
    worst = Math.max(worst, Math.abs(arm.measure() - deg));
  }
  return worst;
}
