// Pure functions over the history. No DOM, no storage: easy to test and to reuse in a report.
export function dayKey(d) {
  return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
}
export function addDays(d, n) { const x = new Date(d.getFullYear(), d.getMonth(), d.getDate()); x.setDate(x.getDate() + n); return x; }

/** Last n calendar days, oldest first, ending today. */
export function lastDays(n, today = new Date()) {
  return Array.from({ length: n }, (_, i) => addDays(today, i - n + 1));
}

/** @returns {Map<string,{complete:number,incomplete:number,uncertain:number,total:number}>} */
export function byDay(attempts) {
  const m = new Map();
  for (const a of attempts) {
    const c = m.get(a.day) || { complete: 0, incomplete: 0, uncertain: 0, total: 0 };
    c[a.label] += 1; c.total += 1; m.set(a.day, c);
  }
  return m;
}

/** Consecutive days with at least one cycle. Today without a cycle does not break the run yet. */
export function streak(days, today = new Date()) {
  let cur = 0, d = today;
  if (!days.has(dayKey(d))) d = addDays(d, -1);
  while (days.has(dayKey(d))) { cur += 1; d = addDays(d, -1); }
  let best = 0, run = 0, prev = null;
  for (const k of Array.from(days.keys()).sort()) {
    run = prev && dayKey(addDays(new Date(prev + "T00:00"), 1)) === k ? run + 1 : 1;
    best = Math.max(best, run); prev = k;
  }
  return { current: cur, best: Math.max(best, cur) };
}

export function sum(days, keys) {
  const out = { complete: 0, incomplete: 0, uncertain: 0, total: 0 };
  for (const k of keys) { const c = days.get(k); if (c) for (const f of Object.keys(out)) out[f] += c[f]; }
  return out;
}

/** Totals over a plain list of attempts. Unknown labels are ignored, never counted as a class. */
export function countsOf(attempts) {
  const out = { complete: 0, incomplete: 0, uncertain: 0, total: 0 };
  for (const a of attempts) {
    if (out[a.label] == null) continue;
    out[a.label] += 1;
    out.total += 1;
  }
  return out;
}
