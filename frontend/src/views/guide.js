// A topic index and two permanent safety entries. Filtering never changes the safety copy.
import { h } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { atlas } from "../i18n/journal-atlas.js";
import { TIPS, CATEGORIES } from "../domain/tips.js";
import { icon } from "../ui/icons.js";

const SAFETY = ["urgent", "pain"];
const ordinary = TIPS.filter((tip) => !SAFETY.includes(tip.id));
let category = "all";

function safetyEntry(id) {
  const [title, text] = t("guide.tips." + id);
  return h("article", { class: "guide-safety-entry", "data-tip": id }, h("h3", { text: title }), h("p", { id: "guide-text-" + id, text }));
}

function topic(cat, toSafety) {
  const go = { record: ["#/check", atlas("toInstructions")], routine: ["#/", atlas("openToday")], doctor: ["#/progress", t("check.toProgress")] }[cat];
  return h("section", { class: "guide-topic", "aria-labelledby": "guide-topic-" + cat },
    h("h2", { id: "guide-topic-" + cat, text: t("guide.cats." + cat) }),
    cat === "safety" ? h("button", { type: "button", class: "guide-jump guide-return", on: { click: toSafety } }, t("guide.tips.urgent")[0], icon("arrow")) : null,
    h("ul", { class: "guide-list", role: "list" }, ordinary.filter((tip) => tip.cat === cat).map(({ id }) => {
      const [title, text] = t("guide.tips." + id);
      return h("li", { class: "guide-tip", "data-tip": id }, h("h3", { text: title }), h("p", { text }));
    })), go ? h("a", { class: "guide-topic-link", href: go[0] }, go[1], icon("arrow")) : null);
}

export function guide(root) {
  const indexTitle = h("h2", { id: "guide-index", text: t("guide.filter"), tabindex: "-1" });
  const list = h("div", { class: "guide-sections", id: "guide-results" });
  const status = h("p", { class: "guide-count", role: "status", "aria-atomic": "true" });
  const buttons = ["all", ...CATEGORIES].map((cat) => h("button", { class: "guide-filter-button", type: "button", "data-cat": cat,
    "aria-controls": "guide-results", on: { click: () => { if (cat !== category) { category = cat; paint(); } } } },
    h("span", { text: cat === "all" ? atlas("allTopics") : t("guide.cats." + cat) }), icon("arrow")));
  function paint() {
    buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.cat === category)));
    list.replaceChildren(...(category === "all" ? CATEGORIES : [category]).map((cat) => topic(cat, () => {
      const title = root.querySelector("#guide-safety-h");
      title.focus({ preventScroll: true }); title.scrollIntoView({ block: "start" });
    })));
    status.textContent = atlas("shown") + " " + ordinary.filter((tip) => category === "all" || category === tip.cat).length;
  }
  function toTopics() {
    indexTitle.focus({ preventScroll: true });
    indexTitle.scrollIntoView({ block: "start" });
  }
  // Keep the optional existing tip-selection hook; no health result is used to recommend a tip.
  let chosen = null;
  function onTip(e) {
    if (!(e.target instanceof Element)) return;
    const id = e.target.closest("[data-tip]")?.dataset.tip || (e.target.closest(".guide-call") ? "urgent" : null);
    if (!id || id === chosen) return;
    chosen = id;
    window.dispatchEvent(new CustomEvent("qolda:tip", { detail: { id: chosen } }));
  }
  paint();
  root.append(h("div", { class: "v-guide", on: { click: onTip, focusin: onTip } },
    h("header", { class: "page-head" }, h("div", null, h("h1", { text: t("guide.title") }), h("p", { class: "guide-lead", text: t("guide.lead") })),
      h("button", { type: "button", class: "guide-jump", on: { click: toTopics } }, atlas("toTopics"), icon("arrow"))),
    h("section", { class: "guide-safety", "aria-labelledby": "guide-safety-h" },
      h("div", { class: "guide-safety-head" }, h("h2", { id: "guide-safety-h", tabindex: "-1" }, icon("stop"), t("guide.cats.safety")),
        h("a", { class: "btn btn--stop guide-call", href: "tel:103", "aria-describedby": "guide-text-urgent" }, icon("phone"), t("guide.call"))),
      h("div", { class: "guide-safety-copy" }, SAFETY.map(safetyEntry))),
    h("div", { class: "guide-book" }, h("div", { class: "guide-index" }, indexTitle,
      h("div", { class: "guide-filter", role: "group", "aria-labelledby": "guide-index" }, buttons), status), list),
    h("button", { type: "button", class: "guide-jump guide-return", on: { click: toTopics } }, atlas("toTopics"), icon("arrow"))));
}
