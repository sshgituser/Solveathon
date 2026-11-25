import ru from "../i18n/ru.js";
import kk from "../i18n/kk.js";

const dicts = { ru, kk };
let cur = ru;

export const langs = Object.values(dicts).map((d) => ({ code: d.code, name: d.name }));
export function setLang(code) { cur = dicts[code] || ru; document.documentElement.lang = cur.code; }
export const L = () => cur;

// A one-letter Russian preposition or conjunction must not be left hanging at the end of a line.
const HANG = /(^|[\s(«])([ВвКкСсОоУуИиАаЯя])\s+(?=\S)/g;
const tie = (text) => (cur.code === "ru" ? text.replace(HANG, "$1$2 ") : text);
const tieDeep = (v) => (typeof v === "string" ? tie(v) : Array.isArray(v) ? v.map(tieDeep) : v);

/** t("home.cta") or t("home.goalOf", { goal: 10 }). Returns the key if a string is missing. */
export function t(path, params) {
  let v = cur;
  for (const k of path.split(".")) v = v == null ? v : v[k];
  if (typeof v !== "string") return v == null ? path : tieDeep(v);
  return tie(params ? v.replace(/\{(\w+)\}/g, (_, k) => (k in params ? params[k] : "")) : v);
}

// Dates are formatted from the dictionaries: browsers do not ship Kazakh date names reliably.
const two = (n) => String(n).padStart(2, "0");
export const fmt = {
  weekdayShort: (d) => cur.weekdaysShort[d.getDay()],
  short: (d) => d.getDate() + " " + cur.monthsShort[d.getMonth()],
  dayMonth: (d) => d.getDate() + " " + cur.months[d.getMonth()],
  full: (d) => d.getDate() + " " + cur.months[d.getMonth()] + " " + d.getFullYear(),
  long: (d) => cur.weekdays[d.getDay()] + ", " + d.getDate() + " " + cur.months[d.getMonth()],
  time: (d) => two(d.getHours()) + ":" + two(d.getMinutes())
};
