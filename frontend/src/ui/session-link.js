// A single entry to practice settings from first run, Today, or Cabinet.
import { h } from "../core/dom.js";
import { t, L } from "../core/i18n.js";
import { store } from "../core/store.js";
import { icon } from "./icons.js";

export function sessionSummary() {
  const { arm, goal } = store.state.data.profile;
  return t("session.preview", { arm: t(arm === "left" ? "session.left" : "session.right"), count: L().cycles(goal) });
}
export function sessionLink(from) {
  return h("a", { class: "session-link", href: "#/settings?from=" + from, "data-session-link": from },
    icon("settings"), h("span", null, h("strong", { text: t("session.title") }), h("small", { text: sessionSummary() })), icon("arrow"));
}
export const extraLink = (from) => h("a", { class: "session-more", href: "#/settings?from=" + from + "&section=exercises" }, t("session.extras"), icon("arrow"));
