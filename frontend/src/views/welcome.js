// First run is an exercise, not a registration form. The same settings editor remains available later.
import { h } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { store } from "../core/store.js";
import { icon } from "../ui/icons.js";
import { movementAtlas } from "../ui/atlas.js";
import { sessionLink, extraLink } from "../ui/session-link.js";

export function welcome(root, router) {
  const plate = movementAtlas(store.state.data.profile.arm || "right"), title = t("atlas.title");
  function enter(path) {
    // Starting a cycle is not permission for persistent storage, media access or analysis.
    if (!store.state.data.profile.arm) store.setProfile({ arm: "right" });
    store.setPrefs({ onboarded: true }); router.go(path);
  }
  const form = h("form", { class: "w-setup", on: { submit: e => { e.preventDefault(); enter("check"); } } },
    h("div", { class: "w-primary" }, h("button", { type: "submit", class: "btn btn--cta" }, t("atlas.start"), icon("arrow"))),
    sessionLink("welcome"), h("div", { class: "w-tools" }, h("button", { type: "button", class: "w-journal", on: { click: () => enter("progress") } }, icon("progress"), t("atlas.journal")), extraLink("welcome")));
  const prepare = h("details", { class: "w-framing" }, h("summary", null, icon("camera"), h("span", { text: t("atlas.framing") }), icon("chevron")),
    h("ul", null, t("check.instrVideo.captions").slice(0, 2).map(text => h("li", { text }))));
  root.append(h("div", { class: "v-welcome" }, h("div", { class: "w-hero" },
    h("header", { class: "w-copy" },
      h("div", { class: "w-index" }, h("span", { class: "exercise-id", text: "E02" }), h("span", { text: t("atlas.single") }), h("span", { text: t("atlas.duration") })),
      h("h1", { class: "exercise-title" }, title[0], " ", h("em", { text: title[1] })), h("p", { class: "lead", text: t("atlas.lead") })), plate.el, form, prepare)));
  return () => plate.dispose();
}
