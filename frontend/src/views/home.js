// Today is the working leaf of the atlas: the exercise, a personal reminder and a local diary.
import { h, flash, calm } from "../core/dom.js";
import { t, L, fmt } from "../core/i18n.js";
import { store } from "../core/store.js";
import { engine } from "../core/api.js";
import { profile } from "../platform/profile.js";
import { dayKey, lastDays, byDay, streak, sum } from "../domain/stats.js";
import { movementAtlas } from "../ui/atlas.js";
import { statusLabel } from "../ui/charts.js";
import { planBar } from "../ui/controls.js";
import { sessionLink, extraLink } from "../ui/session-link.js";
import { icon } from "../ui/icons.js";

function diary(today, entry) {
  const saved = h("span", { class: "flash", role: "status" });
  const labels = t("ease.scale");
  const currentMark = h("p", { class: "t-mark", hidden: !entry.ease, text: entry.ease ? t("atlas.chosenEase", { value: labels[entry.ease - 1] }) : "" });
  const easeButtons = labels.map((label, i) => h("button", { type: "button", role: "radio", "aria-label": (i + 1) + " · " + label,
    "aria-checked": String(entry.ease === i + 1), tabindex: entry.ease === i+1 || (!entry.ease && i === 0) ? 0 : -1,
    on: { click: () => choose(i) } }, h("span", { class: "n", text: i+1 }), h("span", { class: "lbl", text: label })));
  function choose(i) {
    store.setCheckin(today, { ease: i+1 });
    easeButtons.forEach((b, n) => { b.setAttribute("aria-checked", String(n === i)); b.tabIndex = n === i ? 0 : -1; });
    currentMark.textContent = t("atlas.chosenEase", { value: labels[i] }); currentMark.hidden = false;
    flash(saved, t("common.saved"));
  }
  const scale = h("div", { class: "t-scale", role: "radiogroup", "aria-labelledby": "ease-h", on: { keydown: e => {
    const i = easeButtons.indexOf(document.activeElement); if (i < 0) return;
    const moves = { ArrowLeft: (i+4)%5, ArrowRight: (i+1)%5, ArrowUp: (i+4)%5, ArrowDown: (i+1)%5, Home: 0, End: 4 };
    if (!Object.hasOwn(moves, e.key)) return; e.preventDefault(); easeButtons[moves[e.key]].focus(); choose(moves[e.key]);
  } } }, easeButtons);
  const note = h("input", { class: "input", id: "today-note", type: "text", maxlength: 140, value: entry.note,
    placeholder: t("ease.notePh"), on: { change: () => { store.setCheckin(today, { note: note.value }); flash(saved, t("common.saved")); } } });
  return h("section", { class: "panel t-ease", "aria-labelledby": "ease-h" },
    h("div", { class: "panel-head" }, h("h2", { id: "ease-h", text: t("home.easeTitle") }), saved),
    h("p", { class: "field-hint", text: t("ease.hint") }), scale,
    h("div", { class: "t-endpoints", "aria-hidden": "true" }, h("span", { text: labels[0] }), h("span", { text: labels[4] })), currentMark,
    h("label", { class: "field-label", for: "today-note", text: t("ease.note") }), note);
}

export function home(root) {
  const { prefs, data } = store.state, now = new Date(), today = dayKey(now);
  // The daily plan on this exercise card is for E02; legacy records without an exercise were all E02.
  const counts = byDay(data.attempts.filter(a => (a.exercise ?? "E02") === "E02"));
  const c = counts.get(today) || { complete: 0, incomplete: 0, uncertain: 0, total: 0 }, goal = data.profile.goal;
  const week = lastDays(7, now), days = week.map(dayKey), total = sum(counts, days).total;
  const run = streak(counts, now), entry = data.checkins.find(x => x.day === today) || { ease: null, note: "" };
  const easePanel = diary(today, entry), plate = movementAtlas(data.profile.arm);
  const native = profile.name === "android";
  const reasonText = h("span"), reason = h("p", { class: "t-engine" }, reasonText, " ", h("a", { href: "#/cabinet", text: t(native ? "atlas.deviceSettings" : "home.toConnection") }));
  function paintEngine(state) { reason.hidden = state === "ok"; reasonText.textContent = t(native ? "atlas.deviceHealth." + state : state === "nomodel" ? "atlas.noModel" : "atlas.unavailable"); }
  paintEngine(engine.state); const off = engine.on(paintEngine);
  const reached = c.total >= goal, preferDiary = reached && !entry.ease;
  const cta = h("a", { class: "btn btn--cta" + (preferDiary ? " btn--quiet" : ""), href: "#/check" },
    t(c.total ? "home.ctaMore" : "atlas.start"), icon("arrow"));
  const markEase = preferDiary ? h("button", { type: "button", class: "btn btn--cta", text: t("home.markEase"), on: { click: () => {
    easePanel.scrollIntoView({ behavior: calm() ? "auto" : "smooth", block: "center" }); easePanel.querySelector('[tabindex="0"]').focus({ preventScroll: true });
  } } }) : null;
  const planText = t("atlas.planCount", { done: c.total, goal });
  const work = h("section", { class: "t-work", "aria-labelledby": "exercise-title" },
    h("p", { class: "exercise-caption" }, h("span", { class: "exercise-id", text: "E02" }), t("atlas.practice")),
    h("h2", { id: "exercise-title", text: t("home.exerciseName") }),
    h("p", { class: "t-intro", text: t("atlas.todayLead") }),
    h("div", { class: "t-plan" }, h("div", { class: "t-plan-head" }, h("span", { text: t(reached ? "home.goalDone" : "home.dailyGoal") }), h("strong", { text: planText })),
      planBar(c.total, goal, planText)), h("div", { class: "actions t-actions" }, markEase, cta), reason);
  const safety = h("aside", { class: "t-stop", "aria-labelledby": "stop-h" }, icon("stop"),
    h("div", null, h("h2", { id: "stop-h", text: t("home.safety") }), h("p", { text: t("atlas.safetyShort") })),
    h("div", { class: "t-safety-links" }, h("a", { href: "#/guide", text: t("nav.guide") }), h("a", { class: "t-tel", href: "tel:103" }, icon("phone"), t("home.call103"))));
  const weekPanel = h("section", { class: "panel t-week", "aria-labelledby": "week-h" },
    h("div", { class: "panel-head" }, h("h2", { id: "week-h", text: t("home.weekChart") }), h("a", { class: "more", href: "#/progress", text: t("home.toProgress") })),
    h("ol", { class: "t-week-strip" }, week.map((d, i) => {
      const n = counts.get(days[i])?.total || 0;
      return h("li", { class: (n ? "has-cycles " : "") + (i === 6 ? "is-today" : ""), "aria-current": i === 6 ? "date" : null,
        "aria-label": t("atlas.dayCount", { day: fmt.dayMonth(d), n: L().cycles(n) }) },
        h("span", { text: fmt.weekdayShort(d) }), h("b", { text: d.getDate() }), h("span", { class: "t-day-n", text: n > 0 ? n : "—" }));
    })), h("p", { class: "field-hint", text: total ? t("atlas.weekCount", { n: L().cycles(total) }) : t("atlas.weekEmpty") }));
  const stats = h("details", { class: "t-readings" }, h("summary", null, t("atlas.details"), icon("chevron")),
    h("dl", { class: "kv" }, ["complete", "incomplete", "uncertain"].map(k => h("div", null, h("dt", null, statusLabel(k)), h("dd", { text: c[k] }))),
      [["home.streak", L().days(run.current)], ["home.bestStreak", L().days(run.best)], ["home.activeDays", t("home.ofSeven", { n: days.filter(d => counts.has(d)).length })]]
        .map(([label, value]) => h("div", null, h("dt", { text: t(label) }), h("dd", { text: value })))));
  weekPanel.append(stats);
  const storage = !prefs.persist ? h("p", { class: "notice t-storage" }, t("atlas.storageOff"), h("button", { type: "button", class: "btn btn--quiet btn--small", text: t("home.enablePersist"),
    on: { click: () => { store.setPrefs({ persist: true }); storage.remove(); } } })) : null;
  const narrow = matchMedia("(max-width:760px)");
  let phoneOpen = false;
  const recall = h("details", { class: "t-review", open: !narrow.matches }, h("summary", null, icon("replay"), t("atlas.recall"), icon("chevron")), plate.el);
  recall.addEventListener("toggle", () => { if (narrow.matches) phoneOpen = recall.open; });
  const resize = () => { recall.open = !narrow.matches || phoneOpen; }; narrow.addEventListener("change", resize);
  root.append(h("div", { class: "v-home" }, h("header", { class: "page-head t-head" }, h("h1", { text: t("nav.home") }),
    h("p", { class: "t-date" }, icon("calendar"), h("time", { datetime: today, text: fmt.long(now) }))),
    h("div", { class: "t-spread" }, work, safety, recall), h("div", { class: "t-journal" }, easePanel, weekPanel), storage));
  return () => { narrow.removeEventListener("change", resize); off(); plate.dispose(); };
}
