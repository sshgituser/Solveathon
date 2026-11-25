// One adaptive masthead and one shared client for all delivery shells.
import { h } from "./core/dom.js";
import { t, setLang, langs } from "./core/i18n.js";
import { store } from "./core/store.js";
import { engine } from "./core/api.js";
import { profile } from "./platform/profile.js";
import { createRouter } from "./core/router.js";
import { icon, mark } from "./ui/icons.js";
import { welcome } from "./views/welcome.js";
import { home } from "./views/home.js";
import { check } from "./views/check.js";
import { progress } from "./views/progress.js";
import { guide } from "./views/guide.js";
import { cabinet } from "./views/cabinet.js";
import { settings, settingsNavigation } from "./views/settings.js";

// Loading an optional exercise page must not block startup of the primary one.
function exercise(root, router) {
  let gone = false, dispose;
  const wait = h("p", { role: "status", text: t("session.checking") }); root.append(wait);
  import("./views/extra.js").then(m => {
    if (!gone) { wait.remove(); dispose = m.extra(root, router); }
  }).catch(() => { if (!gone) root.replaceChildren(h("h1", { text: t("session.extras") }), h("p", { role: "status", text: t("session.off") }), h("a", { class: "session-more", href: "#/settings", text: t("session.back") })); });
  return () => { gone = true; dispose?.(); };
}

const NAV = [["", "home"], ["check", "check"], ["progress", "progress"], ["guide", "guide"], ["cabinet", "cabinet"]];
const STATE_ICON = { ok: "linkOk", nomodel: "linkHalf", down: "linkOff" };
const rail = document.getElementById("rail"), view = document.getElementById("view");
setLang(store.state.prefs.lang);
document.documentElement.classList.toggle("calm", !!store.state.prefs.calm);
let links = new Map(), current = "";
const engineBox = h("span", { class: "engine", role: "status" });
function paintEngine() {
  engineBox.dataset.state = engine.state;
  const onPhone = profile.name === "android";
  engineBox.title = t((onPhone ? "atlas.deviceHealth." : "health.") + engine.state);
  engineBox.replaceChildren(icon(onPhone && engine.state === "ok" ? "complete" : STATE_ICON[engine.state]), h("span", { text: onPhone ? engineBox.title : t("health.short." + engine.state) }));
  engineBox.setAttribute("aria-label", engineBox.title);
}
engine.on(paintEngine);
function buildRail() {
  links = new Map();
  const skip = h("a", { class: "skip", href: "#view", text: t("atlas.skip"), on: { click: e => { e.preventDefault(); view.focus(); } } });
  const brand = h("a", { class: "rail-brand", href: "#/", "aria-label": t("brand.name") },
    h("span", { class: "rail-logo-box" }, mark()), h("span", { class: "rail-word", text: t("brand.name") }));
  const nav = h("nav", { class: "rail-nav", "aria-label": t("atlas.sections") }, NAV.map(([path, key]) => {
    const a = h("a", { href: "#/" + path }, h("span", { class: "nav-glyph", "aria-hidden": "true" }, icon(key)), h("span", { text: t("nav." + key) }));
    links.set(path, a); return a;
  }));
  const language = h("div", { class: "lang", role: "group", "aria-label": "Тіл / Язык" }, langs.map(l =>
    h("button", { type: "button", lang: l.code, "data-language": l.code, "aria-pressed": String(l.code === store.state.prefs.lang),
      "aria-label": l.code === "ru" ? "Русский" : "Қазақша", text: l.name, on: { click: () => switchLang(l.code) } })));
  rail.replaceChildren(skip, h("div", { class: "rail-inner" }, brand, nav, h("div", { class: "rail-foot" }, engineBox, language)));
  paintEngine(); paintCurrent();
}
function paintCurrent() {
  document.body.classList.toggle("bare", current === "welcome");
  links.forEach((a, path) => { if (path === current) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  document.title = (current === "welcome" ? t("atlas.practice") : t("nav." + (current || "home"))) + " · Qolda";
}
function switchLang(code) {
  if (code === store.state.prefs.lang) return;
  store.setPrefs({ lang: code }); setLang(code); buildRail(); router.refresh();
  rail.querySelector('[data-language="' + code + '"]')?.focus({ preventScroll: true });
}
const pingEngine = () => engine.check();
const router = createRouter(view, { "": home, check, progress, guide, cabinet, welcome }, {
  guard: name => store.state.prefs.onboarded ? (name === "welcome" ? "" : name) : "welcome",
  onChange: name => { current = name; paintCurrent(); pingEngine(); }
});
window.addEventListener("focus", pingEngine);
buildRail(); router.start();
