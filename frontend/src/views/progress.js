// A dated local journal. Empty pages never stand in for activity.
import { h, confirmInline } from "../core/dom.js";
import { t, L, fmt } from "../core/i18n.js";
import { atlas } from "../i18n/journal-atlas.js";
import { store } from "../core/store.js";
import { dayKey, addDays, lastDays, byDay, streak, sum, countsOf } from "../domain/stats.js";
import { barChart, barLegend, lineChart, heatGrid, statusLabel } from "../ui/charts.js";
import { icon } from "../ui/icons.js";
import { segmented } from "../ui/controls.js";

const kinds = ["complete", "incomplete", "uncertain"];
const fromDay = (day) => new Date(day + "T00:00");
const fact = (label, value) => h("div", null, h("dt", { text: label }), h("dd", { text: value }));
const sheet = (title, ...children) => h("section", { class: "p-sheet" }, h("h2", { text: title }), children);
const firstCheck = () => h("a", { class: "btn btn--cta p-first", href: "#/check" }, t("home.ctaFirst"), icon("arrow"));
let range = 7, tab = "overview";

function folio(now, latest, counts, entry) {
  const date = latest ? fromDay(latest) : now;
  return h("section", { class: "p-folio", "aria-labelledby": "p-folio-h" },
    h("time", { class: "p-stamp", datetime: dayKey(date) },
      h("span", { class: "p-stamp-label", text: !latest || latest === dayKey(now) ? t("progress.today") : L().weekdays[date.getDay()] }),
      h("b", { text: date.getDate() }), h("span", { text: L().months[date.getMonth()] + " " + date.getFullYear() })),
    h("div", { class: "p-folio-body" }, h("h2", { id: "p-folio-h", tabindex: "-1", text: atlas(latest ? "latest" : "emptyTitle") }),
      latest ? [h("dl", { class: "p-folio-facts" },
        counts.has(latest) ? fact(atlas("checked"), counts.get(latest).total) : null,
        entry?.ease ? fact(atlas("ownMark"), t("ease.scale")[entry.ease - 1]) : null),
        entry?.note ? h("p", { class: "p-user-note" }, h("strong", { text: t("ease.note") + ": " }), entry.note) : null]
        : [h("p", { text: atlas("emptyHint") }), firstCheck()]));
}

function summary(todayCounts, all) {
  const row = (label, k) => h("tr", null, h("th", { scope: "row" }, label), h("td", { text: todayCounts[k] }), h("td", { text: all[k] }));
  return sheet(t("progress.summary"), h("table", { class: "p-table", "aria-label": t("progress.summary") },
    h("thead", null, h("tr", null, [atlas("result"), t("progress.summaryToday"), t("progress.summaryAll")].map((text) => h("th", { scope: "col", text })))),
    h("tbody", null, kinds.map((k) => row(statusLabel(k), k))), h("tfoot", null, row(t("progress.summaryTotal"), "total"))),
    h("p", { class: "fine", text: t("progress.summaryNote") }));
}

function regularity(now, counts, active) {
  const weekEnd = addDays(now, 6 - ((now.getDay() + 6) % 7));
  const days = lastDays(35, weekEnd), run = streak(counts, now);
  const today = dayKey(now), calendar = heatGrid(days, counts, today);
  Array.from(calendar.children).forEach((cell, i) => {
    if (dayKey(days[i]) > today) { // A future date is not a recorded day with zero cycles.
      cell.title = fmt.full(days[i]); cell.firstElementChild.textContent = cell.title;
    }
    cell.append(h("span", { "aria-hidden": "true", text: days[i].getDate() }));
  });
  return sheet(t("progress.regular"), h("div", { class: "p-regular" },
    h("div", { class: "p-calendar" }, h("div", { class: "p-weekdays", "aria-hidden": "true" },
      days.slice(0, 7).map((d) => h("span", { text: fmt.weekdayShort(d) }))), calendar),
    h("dl", { class: "kv" }, fact(t("progress.activeDays"), active + " / " + range),
      fact(t("progress.streakNow"), L().days(run.current)), fact(t("progress.streakBest"), L().days(run.best)))) ,
    h("p", { class: "fine", text: t("progress.regularHint") }));
}

function overview(days, counts, attempts) {
  const keys = days.map(dayKey), total = sum(counts, keys), now = days[days.length - 1];
  const legend = barLegend();
  Array.from(legend.children).forEach((li, i) => li.append(h("strong", { text: total[kinds[i]] })));
  const graph = total.total ? barChart(days, counts) : null;
  if (graph) graph.dataset.period = String(days.length);
  const cycles = sheet(t("progress.cycles"), graph ? [graph, legend]
    : h("p", { class: "p-message", text: atlas("noPeriodCycles") }));
  return attempts.length ? [h("div", { class: "p-spread" }, cycles,
    summary(counts.get(dayKey(now)) || countsOf([]), countsOf(attempts))), regularity(now, counts, keys.filter((k) => counts.has(k)).length)]
    : [sheet(t("progress.cycles"), h("p", { class: "p-message", text: t("progress.cyclesEmpty") }), firstCheck())];
}

function diary(days, entries) {
  const ease = new Map(entries.filter((c) => c.ease).map((c) => [c.day, c.ease]));
  const axis = t("progress.easeAxis");
  return sheet(atlas("movement"), entries.length ? [h("p", { class: "fine", text: t("ease.hint") }),
    ease.size ? h("div", { class: "p-axis" }, h("div", { class: "p-axis-words", "aria-hidden": "true" },
      h("span", { text: axis[1] }), h("span", { text: axis[0] })), lineChart(days, ease)) : null,
    h("ul", { class: "p-notes", role: "list" }, entries.map((c) => h("li", null,
      h("time", { datetime: c.day, text: fmt.dayMonth(fromDay(c.day)) }), h("div", null,
        c.ease ? h("strong", { text: t("ease.scale")[c.ease - 1] }) : null, c.note ? h("p", { text: c.note }) : null))))]
    : h("p", { class: "p-message", text: atlas("noMarks") }),
    h("a", { class: "p-text-link", href: "#/" }, atlas("openToday"), icon("arrow")));
}

function history(attempts, keys, afterDelete) {
  const groups = new Map(), today = dayKey(new Date()), yesterday = dayKey(addDays(new Date(), -1));
  attempts.slice().sort((a, b) => b.ts - a.ts).forEach((a) => {
    if (!groups.has(a.day)) groups.set(a.day, []);
    groups.get(a.day).push(a);
  });
  const reset = h("button", { class: "btn btn--quiet", type: "button" }, icon("trash"), t("progress.clear"));
  reset.addEventListener("click", () => confirmInline(reset, t("progress.clearAsk"), t("common.delete"), t("common.keep"), () => {
    store.clearHistory(); afterDelete(t("progress.cleared"));
  }));
  const records = Array.from(groups, ([day, list]) => h("section", { class: "p-day", "data-in-range": String(keys.has(day)) },
    h("h3", null, h("time", { datetime: day, text: day === today ? t("progress.today") : day === yesterday ? t("progress.yesterday") : fmt.full(fromDay(day)) })),
    h("ul", { class: "p-ledger", role: "list" }, list.map((a) => {
      const time = fmt.time(new Date(a.ts));
      const del = h("button", { type: "button", class: "icon-btn", "aria-label": [t("common.delete"), fmt.dayMonth(fromDay(day)), time, t("label." + a.label)].join(", ") }, icon("trash"));
      del.addEventListener("click", () => confirmInline(del, t("progress.removeAsk"), t("common.delete"), t("common.keep"), () => {
        store.removeAttempt(a.id); afterDelete(atlas("removed"));
      }));
      const model = a.model === "scratch" ? "A" : a.model === "transfer" ? "B" : a.model;
      const meta = [model ? t("check.model") + " " + model : null,
        Number.isFinite(a.confidence) ? t("check.confidence", { n: Math.round(a.confidence * 100) }) : null].filter(Boolean).join(" · ");
      return h("li", null, h("time", { datetime: a.day + "T" + time, text: time }),
        h("div", { class: "p-record" }, statusLabel(a.label), meta ? h("small", { text: meta }) : null), del);
    }))));
  return h("section", { class: "p-sheet p-history" }, h("div", { class: "p-history-head" },
    h("div", null, h("h2", { id: "p-history-h", tabindex: "-1", text: t("progress.history") }),
      h("p", { class: "p-screen-note fine", text: t("progress.summaryAll") })), groups.size ? reset : null),
    groups.size ? records : h("p", { class: "p-message", text: t("progress.historyEmpty") }),
    groups.size && !Array.from(groups.keys()).some((d) => keys.has(d)) ? h("p", { class: "p-print-note", text: atlas("noPeriodCycles") }) : null);
}

export function progress(root) {
  const report = h("div", { class: "report-head" }), folioSlot = h("div"), span = h("p", { class: "p-period-dates" });
  const message = h("p", { class: "p-notice", role: "status" });
  const period = segmented(atlas("period"), [[7, t("progress.range7")], [30, t("progress.range30")]], range,
    (v) => { range = v; render(); }, { hideLegend: true });
  const names = [["overview", atlas("overview")], ["diary", atlas("diary")], ["history", t("progress.history")]];
  const buttons = names.map(([key, text]) => h("button", { type: "button", role: "tab", id: "p-tab-" + key, "aria-controls": "p-pane-" + key,
    text, on: { click: () => select(key), keydown: (e) => {
      const i = buttons.indexOf(e.currentTarget), target = { ArrowLeft: (i + 2) % 3, ArrowRight: (i + 1) % 3, Home: 0, End: 2 }[e.key];
      if (target == null) return;
      e.preventDefault(); select(names[target][0]); buttons[target].focus();
    } } }));
  const tabs = h("div", { class: "p-tabs", role: "tablist", "aria-label": atlas("journalSections") }, buttons);
  const panes = names.map(([key]) => h("div", { class: "p-pane", role: "tabpanel", tabindex: "0", id: "p-pane-" + key, "aria-labelledby": "p-tab-" + key }));
  function select(key) {
    tab = key;
    names.forEach(([k], i) => {
      const active = k === tab;
      buttons[i].setAttribute("aria-selected", String(active)); buttons[i].tabIndex = active ? 0 : -1;
      panes[i].classList.toggle("is-active", active);
    });
  }
  const book = h("div", { class: "p-book" }, tabs, panes);
  const wrap = h("div", { class: "v-progress" }, report, h("header", { class: "page-head" }, h("h1", { text: t("progress.title") }),
    h("button", { type: "button", class: "btn btn--quiet p-print", on: { click: () => window.print() } }, icon("print"), t("progress.print"))),
    message, folioSlot, h("div", { class: "p-period" }, h("div", null, h("strong", { text: atlas("period") }), span), period), book,
    h("div", { class: "p-storage" }, h("span", { text: t(store.state.prefs.persist ? "cabinet.persistOn" : "cabinet.persistOff") }),
      h("a", { class: "p-text-link", href: "#/cabinet" }, atlas("storageSettings"), icon("arrow"))));
  root.append(wrap);

  function afterDelete(text) {
    render(); message.textContent = text;
    const next = !book.hidden && tab === "history" ? wrap.querySelector("#p-history-h") : wrap.querySelector("#p-folio-h");
    next?.focus();
  }
  function render() {
    const { data } = store.state, now = new Date(), days = lastDays(range, now), keys = new Set(days.map(dayKey));
    const counts = byDay(data.attempts), entries = data.checkins.filter((c) => c.ease || c.note).slice().sort((a, b) => b.day.localeCompare(a.day));
    const latest = [...counts.keys(), ...(entries.length ? [entries[0].day] : [])].sort().pop();
    const dates = fmt.full(days[0]) + " – " + fmt.full(now);
    span.textContent = dates;
    folioSlot.replaceChildren(folio(now, latest, counts, entries.find((c) => c.day === latest)));
    report.replaceChildren(h("h1", { text: atlas("reportTitle") }), h("p", { text: dates }),
      ...(data.profile.nick ? [h("p", { text: data.profile.nick })] : []), h("p", { text: t("progress.reportNote") }));
    book.hidden = !latest;
    if (!latest) { tab = "overview"; report.append(h("p", { text: atlas("emptyTitle") })); panes.forEach((p) => p.replaceChildren()); return; }
    panes[0].replaceChildren(...overview(days, counts, data.attempts));
    panes[1].replaceChildren(diary(days, entries.filter((c) => keys.has(c.day))));
    panes[2].replaceChildren(history(data.attempts, keys, afterDelete));
    select(tab);
  }
  if (!store.state.data.attempts.length && store.state.data.checkins.some((c) => c.ease || c.note)) tab = "diary";
  render();
}
