// Hash router: works from any static origin and inside desktop/Android web views.
import { calm } from "./dom.js";

export function createRouter(root, routes, { guard, onChange }) {
  let cleanup = null, current = null, first = true;

  function path() { return (location.hash.replace(/^#\/?/, "").split("?")[0]) || ""; }

  function mount() {
    const want = path();
    const name = guard(want in routes ? want : "");
    if (cleanup) { cleanup(); cleanup = null; }
    root.textContent = "";
    current = name;
    cleanup = routes[name](root, api) || null;
    onChange(name);
    if (!first) { root.focus({ preventScroll: true }); window.scrollTo(0, 0); }
    first = false;
  }

  function render() {
    // Mount synchronously (navigation must never wait on a frame), then let the new screen settle in.
    const animate = !first && !calm();
    mount();
    if (animate) { root.classList.remove("enter"); void root.offsetWidth; root.classList.add("enter"); }
  }

  const api = {
    go(to) { if (path() === to) render(); else location.hash = "#/" + to; },
    refresh: render,
    get current() { return current; }
  };
  window.addEventListener("hashchange", render);
  api.start = render;
  return api;
}
