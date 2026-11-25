// Storage port: the device copy of the data. The store talks only to this module.
// Both delivery profiles keep a device copy (local-first). In the connected profile a sync
// service may later mirror it to an account by listening to store changes; it does not replace this.
const PREFS = "seriq.prefs.v1"; // language, engine address: nothing personal
const DATA = "seriq.data.v1";   // profile, history, diary: only with explicit consent

function read(key) { try { return JSON.parse(localStorage.getItem(key)); } catch (_) { return null; } }
function write(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); return true; } catch (_) { return false; } }

export const storage = {
  loadPrefs: () => read(PREFS),
  savePrefs: (p) => write(PREFS, p),
  loadData: () => read(DATA),
  saveData: (d) => write(DATA, d),
  clearData: () => { try { localStorage.removeItem(DATA); } catch (_) {} }
};
