// Small SVG charts drawn by hand: no chart library, nothing to download.
// States differ by fill pattern as well as colour, so they survive grayscale print and weak colour vision.
import { h, s } from "../core/dom.js";
import { t, L, fmt } from "../core/i18n.js";
import { dayKey } from "../domain/stats.js";
import { icon } from "./icons.js";

const SEGS = ["complete", "incomplete", "uncertain"];
let uid = 0;

/** Stacked bars of cycles per day. `days` are Date objects, `counts` is the byDay map. */
export function barChart(days, counts) {
  const n = days.length, dense = n > 10, slot = dense ? 24 : 72, W = n * slot, H = 190, base = 150, top = 22;
  const max = Math.max(4, ...days.map((d) => (counts.get(dayKey(d)) || { total: 0 }).total));
  const unit = (base - top) / max, bw = Math.round(slot * (dense ? .6 : .44));
  const total = days.reduce((a, d) => a + (counts.get(dayKey(d)) || { total: 0 }).total, 0);
  const hatch = "hatch" + (++uid);

  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("progress.cycles") + ": " + L().cycles(total) });
  svg.appendChild(s("defs", null, s("pattern", { id: hatch, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    s("rect", { class: "hatch-bg", width: 6, height: 6 }), s("rect", { class: "hatch-ink", width: 2.4, height: 6 }))));
  svg.appendChild(s("path", { class: "axis", d: `M0 ${base + .5}H${W}` }));

  days.forEach((d, i) => {
    const c = counts.get(dayKey(d)), x = i * slot + (slot - bw) / 2;
    const g = s("g", { class: "col" }); g.style.animationDelay = Math.min(i * (dense ? 8 : 30), 240) + "ms";
    let y = base;
    if (c) SEGS.forEach((k) => {
      if (!c[k]) return;
      const hh = c[k] * unit; y -= hh;
      g.appendChild(s("rect", { class: "seg-" + k, x, y: y + 1, width: bw, height: Math.max(hh - 2, 2), rx: 3, fill: k === "incomplete" ? `url(#${hatch})` : null }));
    });
    else g.appendChild(s("rect", { class: "seg-none", x, y: base - 3, width: bw, height: 2, rx: 1 }));
    if (c && !dense) g.appendChild(s("text", { class: "val", x: x + bw / 2, y: y - 7, "text-anchor": "middle", text: c.total }));
    svg.appendChild(g);
    if (!dense || (n - 1 - i) % 5 === 0) svg.appendChild(s("text", { class: "tick", x: x + bw / 2, y: H - 12, "text-anchor": "middle", text: dense ? d.getDate() : fmt.weekdayShort(d) }));
  });
  return svg;
}

export function barLegend() {
  return h("ul", { class: "legend" }, SEGS.map((k) => h("li", null, h("span", { class: "sw sw--" + k, "aria-hidden": "true" }), t("label." + k))));
}

/** Self-reported ease (1..5) per day. Days without an entry leave a gap in the line. */
export function lineChart(days, easeByDay) {
  const n = days.length, W = 640, H = 190, top = 18, bottom = 150;
  const x = (i) => 10 + (W - 20) * (n === 1 ? .5 : i / (n - 1));
  const y = (v) => bottom - (bottom - top) * ((v - 1) / 4);
  const svg = s("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": t("progress.ease") });
  for (let v = 1; v <= 5; v++) svg.appendChild(s("path", { class: "grid", d: `M0 ${y(v)}H${W}` }));
  let d = "", pen = false; const dots = [];
  days.forEach((day, i) => {
    const v = easeByDay.get(dayKey(day));
    if (v == null) { pen = false; return; }
    d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v).toFixed(1); pen = true;
    dots.push(s("circle", { class: "dot", cx: x(i).toFixed(1), cy: y(v).toFixed(1), r: n > 10 ? 4 : 6 }));
  });
  if (d) svg.appendChild(s("path", { class: "trace", d, pathLength: 1 }));
  dots.forEach((c) => svg.appendChild(c));
  svg.appendChild(s("text", { class: "tick", x: 0, y: H - 12, text: fmt.short(days[0]) }));
  svg.appendChild(s("text", { class: "tick", x: W, y: H - 12, "text-anchor": "end", text: fmt.short(days[n - 1]) }));
  return svg;
}

/** Five weeks of days; a filled cell is a day with at least one cycle. */
export function heatGrid(days, counts, todayKey) {
  return h("ol", { class: "heat" }, days.map((d) => {
    const k = dayKey(d), c = counts.get(k), n = c ? c.total : 0, label = fmt.dayMonth(d) + ": " + L().cycles(n);
    return h("li", { class: "cell" + (n ? " on" : "") + (n > 3 ? " hi" : "") + (k === todayKey ? " is-today" : "") + (k > todayKey ? " future" : ""), title: label },
      h("span", { class: "visually-hidden", text: label }));
  }));
}

/** A result state as a shape plus a word: complete, incomplete or uncertain. */
export function statusLabel(kind) {
  return h("span", { class: "status status--" + kind }, icon(kind), h("span", { text: t("label." + kind) }));
}
