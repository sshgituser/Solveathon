// One line-icon set: 24px grid, 1.75 stroke, drawn here so nothing is fetched.
import { s } from "../core/dom.js";

const PATHS = {
  // navigation
  home: "M12 3v2.5M12 18.5V21M3 12h2.5M18.5 12H21M5.6 5.6l1.8 1.8M16.6 16.6l1.8 1.8M18.4 5.6l-1.8 1.8M7.4 16.6l-1.8 1.8M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z",
  check: "M4 18h7.5l7-10M7 18a5 5 0 0 1 2.9-4.5",
  progress: "M5 20V11M12 20V4M19 20v-6",
  guide: "M5 4.5h9a4 4 0 0 1 4 4V20H9a4 4 0 0 1-4-4V4.5ZM9 9h5M9 13h5",
  cabinet: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM4.5 20.5a7.5 7.5 0 0 1 15 0",
  // result states: each has its own shape
  complete: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM8 12.5l2.8 2.8L16 9.8",
  incomplete: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 11v5.5M12 7.6v.2",
  uncertain: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM9.6 9.6a2.5 2.5 0 1 1 3.6 2.3c-.8.4-1.2 1-1.2 1.9M12 16.6v.2",
  nopose: "M3 3l18 18M10.6 6.2A9.6 9.6 0 0 1 12 6c5 0 8.5 4.2 9.5 6-.4.8-1.3 2-2.6 3.2M6.5 7.6C4.5 9 3.1 10.900 2.5 12c1 1.800 4.500 6 9.500 6 1.400 0 2.700-.300 3.800-.800M9.900 9.900a3 3 0 0 0 4.200 4.200",
  // safety and connection
  stop: "M12 3l9.500 16.500h-19L12 3ZM12 10v4.500M12 17.400v.200",
  phone: "M6.500 3.500h3l1.500 4-2 1.500a11 11 0 0 0 6 6l1.500-2 4 1.500v3a2 2 0 0 1-2.200 2A16.500 16.500 0 0 1 4.500 5.700a2 2 0 0 1 2-2.200Z",
  linkOk: "M9.500 14.500l5-5M8 11l-2 2a3.500 3.500 0 0 0 5 5l2-2M16 13l2-2a3.500 3.500 0 0 0-5-5l-2 2",
  linkHalf: "M8 11l-2 2a3.500 3.500 0 0 0 5 5l2-2M16 13l2-2a3.500 3.500 0 0 0-5-5l-2 2",
  linkOff: "M8 11l-2 2a3.500 3.500 0 0 0 5 5l2-2M16 13l2-2a3.500 3.500 0 0 0-5-5l-2 2M4 4l16 16",
  // atlas controls and time metadata
  settings: "M3 7h5m4 0h9M3 17h10m4 0h4M8 4v6m8 4v6",
  chevron: "m6 9 6 6 6-6",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 6.5v5.7l3.5 2",
  calendar: "M4 6h16v14H4zM4 10h16M8 4v4M16 4v4",
  // actions
  upload: "M12 16V5m0 0-4.500 4.500M12 5l4.500 4.500M4 17v2h16v-2",
  camera: "M4 8h3l1.500-2h7L17 8h3v11H4V8ZM12 16.500a3.500 3.500 0 1 0 0-7 3.500 3.500 0 0 0 0 7Z",
  replay: "M4 12a8 8 0 1 0 2.600-5.900M4 4v4.500h4.500",
  play: "M8 5.500 19 12 8 18.500V5.500Z",
  pause: "M8 5v14M16 5v14",
  // turning the 3D figure: an ellipse around a vertical axis with an arrow head
  turnLeft: "M12 4v16M20 12c0 2.200-3.600 4-8 4s-8-1.800-8-4 3.600-4 8-4M4 12l2.500-2.200M4 12l2.800 1.800",
  turnRight: "M12 4v16M4 12c0 2.200 3.600 4 8 4s8-1.800 8-4-3.600-4-8-4M20 12l-2.500-2.200M20 12l-2.800 1.800",
  recenter: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v3M12 19v3M2 12h3M19 12h3",
  print: "M7 9V4h10v5M7 17H4v-7h16v7h-3M7 14h10v6H7v-6Z",
  trash: "M5 7h14M10 7V4h4v3M7 7l1 13h8l1-13",
  download: "M12 4v11m0 0-4.500-4.500M12 15l4.500-4.500M4 19h16",
  arrow: "M5 12h14m0 0-6-6m6 6-6 6"
};

/** @param {keyof typeof PATHS} name */
export function icon(name) {
  return s("svg", { class: "ico", viewBox: "0 0 24 24", "aria-hidden": "true", focusable: "false" }, s("path", { d: PATHS[name] || PATHS.incomplete }));
}

/**
 * A cupped palm and a joint. The 3D version (stage/logo3d.js) uses the same points.
 * The palm takes currentColor, the joint the accent.
 */
export function mark(cls = "rail-logo") {
  return s("svg", { class: cls, viewBox: "0 0 64 64", "aria-hidden": "true", focusable: "false" },
    s("path", { d: "M12 22V33A20 20 0 0 0 52 33V22", fill: "none", stroke: "currentColor", "stroke-width": 8, "stroke-linecap": "round" }),
    s("circle", { cx: 32, cy: 26, r: 8.5, fill: "var(--mark-accent, #2FB5CB)" }));
}
