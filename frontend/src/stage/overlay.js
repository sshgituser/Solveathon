// Labels over the 3D figure, drawn in a fixed SVG layer: shoulder, elbow, wrist with leader lines.
// Positions are joint positions projected through the camera after each rendered frame.
import { s } from "../core/dom.js";

export function createOverlay(svg) {
  const g = s("g", { class: "joint-labels" });
  const items = ["shoulder", "elbow", "wrist"].map(() => {
    const line = s("path", { class: "lead-line" }), text = s("text", { "dominant-baseline": "middle" });
    g.append(line, text);
    return { line, text };
  });
  svg.appendChild(g);
  g.style.opacity = 0; g.style.transition = "opacity 260ms ease";

  return {
    setTexts(texts) { items.forEach((it, i) => { it.text.textContent = texts[i] || ""; }); },
    show(on) { g.style.opacity = on ? 1 : 0; },
    /**
     * @param points [[x, y], ...] screen positions of shoulder, elbow, wrist in CSS pixels
     * @param away   +1 when labels should sit to the right of the joints, -1 to the left
     */
    place(points, away) {
      // Labels keep at least a line of space between them even when two joints are close on screen.
      const order = points.map((p, i) => i).sort((a, b) => points[a][1] - points[b][1]);
      const ys = []; let last = -Infinity;
      order.forEach((i) => { const y = Math.max(points[i][1], last + 24); ys[i] = y; last = y; });
      points.forEach(([x, y], i) => {
        const it = items[i], tx = x + away * 68;
        it.line.setAttribute("d", `M${(x + away * 22).toFixed(1)} ${y.toFixed(1)}L${(tx - away * 8).toFixed(1)} ${ys[i].toFixed(1)}`);
        it.text.setAttribute("x", tx.toFixed(1)); it.text.setAttribute("y", ys[i].toFixed(1));
        it.text.setAttribute("text-anchor", away > 0 ? "start" : "end");
      });
    },
    viewBox(w, h) { svg.setAttribute("viewBox", `0 0 ${w} ${h}`); }
  };
}
