// Single source of truth for the client. Views read `store.state` and call actions;
// nothing else mutates state or touches storage.
import { storage } from "../platform/storage.js";
import { dayKey } from "../domain/stats.js";

const LABELS = ["complete", "incomplete", "uncertain"];
const freshData = () => ({ v: 1, profile: { nick: "", arm: null, goal: 10 }, attempts: [], checkins: [] });
// `calm` is the in-app "reduce motion" switch; it adds to the system setting, never overrides it.
// The loopback engine can serve the app on its own port: keep that UI and its consent
// on the same origin. Documented standalone static dev servers use the desktop default.
function defaultEngineUrl() {
  const loc = globalThis.location;
  if (loc?.protocol === "http:" && ["127.0.0.1", "localhost"].includes(loc.hostname) && !["5173", "8080"].includes(loc.port)) return loc.origin;
  return "http://127.0.0.1:8765";
}
const freshPrefs = () => ({ lang: "ru", engineUrl: defaultEngineUrl(), model: "scratch", onboarded: false, persist: false, calm: false });

const prefs = Object.assign(freshPrefs(), storage.loadPrefs());
const saved = prefs.persist ? storage.loadData() : null;
const state = { prefs, data: saved && saved.v === 1 ? Object.assign(freshData(), saved) : freshData() };

const listeners = new Set();
function commit() {
  storage.savePrefs(state.prefs);
  // History is written to disk only after the person turned persistence on.
  if (state.prefs.persist) storage.saveData(state.data);
  listeners.forEach((fn) => fn(state));
}

export const store = {
  state,
  on(fn) { listeners.add(fn); return () => listeners.delete(fn); },

  setPrefs(patch) {
    const was = state.prefs.persist;
    Object.assign(state.prefs, patch);
    if (was && !state.prefs.persist) storage.clearData();
    commit();
  },
  setProfile(patch) {
    if ("nick" in patch) patch.nick = String(patch.nick).trim().slice(0, 40);
    if ("goal" in patch) patch.goal = Math.min(60, Math.max(1, Math.round(patch.goal) || 1));
    Object.assign(state.data.profile, patch);
    commit();
  },

  /**
   * Add the attempts of one analyzed clip. One physical cycle counts once:
   * re-sending the same clip or comparing models A/B on it adds nothing.
   * @returns {boolean} true when the clip was already counted
   */
  addAttempts(result, hash) {
    if (hash && state.data.attempts.some((a) => a.hash === hash)) return true;
    const ts = Date.now();
    result.attempts.forEach((a, i) => {
      if (!LABELS.includes(a.label)) return;
      state.data.attempts.push({
        id: ts.toString(36) + "-" + i, ts, day: dayKey(new Date(ts)), label: a.label,
        confidence: typeof a.confidence === "number" ? a.confidence : null, reason: a.reason || null,
        model: result.model || null, modelVersion: result.model_version || null, hash: hash || null
      });
    });
    commit();
    return false;
  },
  removeAttempt(id) { state.data.attempts = state.data.attempts.filter((a) => a.id !== id); commit(); },

  /** Remove the record of checks but keep the local profile and the diary. */
  clearHistory() {
    state.data.attempts = [];
    commit();
    if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent("qolda:data-reset", { detail: { scope: "history" } }));
  },

  /** Self-reported ease of movement for a day, 1..5. One entry per day. */
  setCheckin(day, patch) {
    let c = state.data.checkins.find((x) => x.day === day);
    if (!c) { c = { day, ease: null, note: "" }; state.data.checkins.push(c); }
    if ("note" in patch) patch.note = String(patch.note).trim().slice(0, 140);
    Object.assign(c, patch, { ts: Date.now() });
    commit();
  },

  wipe() {
    state.data = freshData();
    storage.clearData();
    commit();
    if (typeof window !== "undefined") window.dispatchEvent(new CustomEvent("qolda:data-reset", { detail: { scope: "all" } }));
  },
  exportData() { return { exported: new Date().toISOString(), format: "seriq.export.v1", ...state.data }; }
};
