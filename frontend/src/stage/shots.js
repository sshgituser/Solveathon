// Camera shots. The camera never orbits freely: each screen asks for a named shot.
// yaw turns the figure (degrees; positive brings the right arm toward the camera, the engine
// mirrors it for the left arm). box is the width and height in metres that must fit in the slot.
export const FOV = 26;

export const SHOTS = {
  // The default: three-quarter view, working arm toward the viewer.
  quarter: { yaw: 36, elev: 6, target: [0, 1.2, 0], box: [1.2, 1.42] },
  // Strict profile: the way the person should stand to the camera. This is the animated instruction.
  side: { yaw: 90, elev: 2, target: [0.14, 1.2, 0], box: [1.25, 1.42] },
  // Close on the working arm, for the verdict.
  arm: { yaw: 72, elev: 4, focus: "arm", box: [0.86, 0.86] },
  // Facing the viewer, for the cabinet where the working arm is chosen.
  front: { yaw: 0, elev: 4, target: [0, 1.2, 0], box: [1.2, 1.42] }
};
