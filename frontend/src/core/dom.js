// Tiny DOM builders. Everything goes through textContent/attributes, never innerHTML,
// so user-entered text (nickname, notes) cannot inject markup and the CSP stays strict.
const SVG_NS = "http://www.w3.org/2000/svg";

function apply(el, props, kids) {
  if (props) for (const k of Object.keys(props)) {
    const v = props[k];
    if (v == null || v === false) continue;
    if (k === "on") for (const ev of Object.keys(v)) el.addEventListener(ev, v[ev]);
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid == null || kid === false) continue;
    el.appendChild(typeof kid === "object" ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
export const h = (tag, props, ...kids) => apply(document.createElement(tag), props, kids);
export const s = (tag, props, ...kids) => apply(document.createElementNS(SVG_NS, tag), props, kids);

/** Motion is off when the system asks for it or when the person switched it off in the Cabinet. */
export const calm = () => document.documentElement.classList.contains("calm") || window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Destructive actions are confirmed in place, next to the control that asked.
export function confirmInline(button, question, yesLabel, noLabel, onYes) {
  const ask = h("span", { class: "ask" }, h("span", { text: question }),
    h("button", { type: "button", class: "btn btn--stop btn--small", text: yesLabel, on: { click: () => { close(); onYes(); } } }),
    h("button", { type: "button", class: "btn btn--quiet btn--small", text: noLabel, on: { click: () => { close(); button.focus(); } } }));
  function close() { ask.remove(); button.hidden = false; }
  button.hidden = true; button.after(ask); ask.lastChild.focus();
}

// Brief "saved" note beside the thing that was saved.
export function flash(node, text) {
  node.textContent = text; node.classList.remove("on"); void node.offsetWidth; node.classList.add("on");
}
