// One editor for E02. Editing never starts a check or opts the person into storage.
import { h } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { store } from "../core/store.js";
import { icon } from "../ui/icons.js";

let draft = null, returning = null;
const exits = { welcome: "welcome", home: "", cabinet: "cabinet" };
// Keep an unfinished edit across language changes and the exercise detail pages only.
export function settingsNavigation(route, root) {
  if (route === "settings" || route === "exercise") return;
  draft = null;
  const ret = returning; returning = null;
  if (ret && exits[ret.from] === route) requestAnimationFrame(() => {
    const entry = root.querySelector('[data-session-link="' + ret.from + '"]');
    if (!entry) return;
    if (ret.applied) entry.after(h("span", { class: "session-notice", role: "status", text: t("session.applied") }));
    entry.focus();
  });
}
export function settings(root, router) {
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  const src = params.get("from"), fallback = store.state.prefs.onboarded ? "home" : "welcome";
  const from = store.state.prefs.onboarded && (src === "cabinet" || src === "home") ? src : fallback;
  const saved = store.state.data.profile;
  draft ||= { from, arm: saved.arm === "left" ? "left" : "right", goal: String(saved.goal), jumped: false };
  const edit = draft, ac = new AbortController();
  function exit(apply) {
    if (apply) {
      if (!valid()) { invalid.hidden = false; goal.setAttribute("aria-invalid", "true"); goal.focus(); return; }
      store.setProfile({ arm: edit.arm, goal: Number(goal.value) });
    }
    returning = { from: edit.from, applied: apply }; draft = null; router.go(exits[edit.from]);
  }
  const arms = h("fieldset", { class: "ss-arm", "aria-describedby": "ss-arm-hint" }, h("legend", { text: t("session.side") }),
    h("div", { class: "ss-sides" }, ["left", "right"].map(v => h("label", null,
      h("input", { type: "radio", name: "ss-arm", value: v, checked: edit.arm === v, on: { change: () => { edit.arm = v; } } }),
      h("span", null, t("welcome." + v), h("i", { "aria-hidden": "true" }))))), h("p", { class: "ss-hint", id: "ss-arm-hint", text: t("session.sideHint") }));
  const goal = h("input", { id: "ss-goal", type: "number", min: 1, max: 60, step: 1, required: true, inputmode: "numeric", autocomplete: "off", value: edit.goal,
    "aria-describedby": "ss-unit ss-goal-hint ss-invalid", on: { input: sync } });
  const invalid = h("p", { id: "ss-invalid", class: "ss-invalid", role: "status", hidden: true, text: t("session.invalid") });
  const minus = step(-1), plus = step(1);
  function valid() { return /^\d+$/.test(goal.value) && Number(goal.value) >= 1 && Number(goal.value) <= 60; }
  function sync() { edit.goal = goal.value; const n = Number(goal.value); minus.disabled = valid() && n <= 1; plus.disabled = valid() && n >= 60; invalid.hidden = true; goal.removeAttribute("aria-invalid"); }
  function step(delta) {
    return h("button", { type: "button", class: delta < 0 ? "ss-minus" : "ss-plus", "aria-label": t(delta < 0 ? "session.minus" : "session.plus"),
      text: delta < 0 ? "−" : "+", on: { click: () => { goal.value = String(Math.max(1, Math.min(60, (valid() ? Number(goal.value) : saved.goal) + delta))); sync(); } } });
  }
  sync();
  const plan = h("div", { class: "ss-plan" }, h("label", { for: goal.id, text: t("session.goal") }),
    h("div", { class: "ss-number" }, h("div", { class: "ss-counter" }, minus, goal, plus), h("span", { id: "ss-unit", text: t("session.unit") })),
    invalid, h("p", { class: "ss-hint", id: "ss-goal-hint", text: t("session.goalHint") }));
  const form = h("form", { class: "ss-form", novalidate: true, "aria-labelledby": "ss-primary", on: { submit: e => { e.preventDefault(); exit(true); } } },
    h("header", { class: "ss-primary-head" }, h("div", null, h("span", { class: "exercise-id", text: "E02" }), h("span", { text: t("session.primary") })),
      h("h2", { id: "ss-primary", text: t("home.exerciseName") })),
    h("div", { class: "ss-fields" }, arms, plan, h("div", { class: "ss-actions" }, h("button", { type: "submit", class: "btn btn--cta", text: t("session.apply") }),
      h("button", { type: "button", class: "ss-cancel", text: t("common.cancel"), on: { click: () => exit(false) } }))));
  const flags = new Map();
  const list = ["E01", "E03"].map(id => {
    const state = h("small", { text: t("session.checking") }); flags.set(id, state);
    return h("li", null, h("a", { href: "#/exercise?id=" + id, "data-exercise": id }, h("span", { class: "exercise-id", text: id }),
      h("span", { class: "ss-row-copy" }, h("strong", { text: t("session." + id.toLowerCase()) }), state), icon("arrow")));
  });
  const extras = h("section", { class: "ss-extras", "aria-labelledby": "ss-extras-title" },
    h("h2", { id: "ss-extras-title", tabindex: "-1", text: t("session.extras") }), h("ul", { role: "list" }, list));
  root.append(h("div", { class: "v-settings" }, h("header", { class: "ss-header" }, h("h1", { text: t("session.title") })), h("div", { class: "ss-spread" }, form, extras)));
  // No dependency of the mandatory E02 startup path on the optional catalogue transport.
  const show = ready => { if (!ac.signal.aborted) flags.forEach((n, id) => { const ok = ready?.[id] === true; n.textContent = t(ok ? "session.file" : "session.off"); n.dataset.ready = String(ok); }); };
  import("../platform/exercises.js").then(m => ac.signal.aborted ? null : m.extraAvailability(ac.signal)).then(show, () => show(null));
  if (params.get("section") === "exercises" && !edit.jumped) {
    edit.jumped = true; requestAnimationFrame(() => { if (!ac.signal.aborted) extras.querySelector("h2").focus(); });
  }
  return () => ac.abort();
}
