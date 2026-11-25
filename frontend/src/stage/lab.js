// Development page: drives the stage through the same entry point the screens use.
// Query: ?slot=a|b&shot=quarter|side|arm|front&elbow=95&side=R|L&labels=1&look=uncertain&intro=1
// From the console: lab.move("b"), lab.fig.shot("side"), lab.fig.to(176, 1500), lab.demo()
import { initStage, createFigure, playDemo } from "./index.js";

const log = document.getElementById("lab-log");
const say = (...a) => { log.textContent += a.join(" ") + "\n"; };
const p = new URLSearchParams(location.search);
const slots = { a: document.getElementById("slot-a"), b: document.getElementById("slot-b") };
if (p.get("solo") === "1") document.body.classList.add("lab--solo");
// ?nowebgl=1 makes every canvas refuse a context, to exercise the flat fallback.
if (p.get("nowebgl") === "1") HTMLCanvasElement.prototype.getContext = () => null;
if (p.get("calm") === "1") document.documentElement.classList.add("calm");

const opts = () => ({ shot: p.get("shot") || "quarter", labels: p.get("labels") === "1", side: p.get("side") || "R", intro: p.get("intro") === "1" });
let fig = createFigure(slots[p.get("slot") || "a"], opts());
if (p.has("elbow")) fig.set(+p.get("elbow"));
if (p.get("look")) fig.look(p.get("look"));

initStage(document.getElementById("stage"), document.getElementById("stage-overlay")).then((engine) => {
  say(engine ? "engine ready" : "engine FAILED, flat drawing in use");
});

window.lab = {
  get fig() { return fig; },
  /** Hand the figure to the other slot the way a route change does: release, then acquire. */
  move(name) { const o = opts(); fig.dispose(); fig = createFigure(slots[name], o); return fig; },
  demo(n = 1) { return playDemo(fig, n, () => say("demo done")); }
};
