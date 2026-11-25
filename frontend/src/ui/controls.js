// Form controls shared by every screen. Markup matches styles/components.css.
import { h } from "../core/dom.js";

let uid = 0;

/** Radio group drawn as one segmented control. options: [[value, label], ...] */
export function segmented(legend, options, value, onChange, { small = false, hideLegend = false } = {}) {
  const name = "seg" + (++uid);
  return h("fieldset", { class: "seg" + (small ? " seg--small" : "") },
    h("legend", { class: hideLegend ? "visually-hidden" : "seg-legend", text: legend }),
    h("div", { class: "seg-row" }, options.map(([v, label]) => h("label", null,
      h("input", { type: "radio", name, value: v, checked: v === value, on: { change: () => onChange(v) } }),
      h("span", { text: label })))));
}

/** Radio group as a list of full-width rows, for options that need a sentence. */
export function choice(legend, options, value, onChange) {
  const name = "choice" + (++uid);
  return h("fieldset", { class: "seg" }, h("legend", { class: "seg-legend", text: legend }),
    h("div", { class: "choice" }, options.map(([v, label]) => h("label", null,
      h("input", { type: "radio", name, value: v, checked: v === value, on: { change: () => onChange(v) } }),
      h("span", { text: label })))));
}

/** A switch with its explanation beside it. */
export function toggle(label, hint, checked, onChange) {
  const input = h("input", { type: "checkbox", role: "switch", checked, on: { change: () => onChange(input.checked) } });
  return h("label", { class: "switch" }, input, h("span", { class: "switch-track", "aria-hidden": "true" }),
    h("span", { class: "switch-text" }, h("strong", { text: label }), hint ? h("small", { text: hint }) : null));
}

/** A number with minus and plus buttons. */
export function stepper(label, hint, value, min, max, onChange) {
  const out = h("output", { text: value });
  const set = (v) => { value = Math.min(max, Math.max(min, v)); out.textContent = value; onChange(value); };
  return h("div", { class: "field" }, h("span", { class: "field-label", text: label }),
    h("div", { class: "stepper", role: "group", "aria-label": label },
      h("button", { type: "button", "aria-label": "−1", text: "−", on: { click: () => set(value - 1) } }), out,
      h("button", { type: "button", "aria-label": "+1", text: "+", on: { click: () => set(value + 1) } })),
    hint ? h("small", { class: "field-hint", text: hint }) : null);
}

export function textField(label, hint, value, attrs, onChange) {
  const input = h("input", Object.assign({ class: "input", id: "field" + (++uid), type: "text", value, on: { change: () => onChange(input.value) } }, attrs));
  return h("label", { class: "field" }, h("span", { class: "field-label", text: label }), input, hint ? h("small", { class: "field-hint", text: hint }) : null);
}

/** The plan bar: one segment per planned cycle while that stays readable, a plain bar beyond 20. */
export function planBar(done, goal, label) {
  if (goal > 20) {
    const fill = h("span", { class: "on" }); fill.style.width = Math.min(100, done / goal * 100) + "%";
    return h("div", { class: "bar bar--plain", role: "img", "aria-label": label }, fill);
  }
  return h("div", { class: "bar", role: "img", "aria-label": label }, Array.from({ length: goal }, (_, i) => h("span", { class: i < done ? "on" : null })));
}
