// Local settings: the profile, the device copy and the analyzer are distinct things.
import { h, flash, confirmInline } from "../core/dom.js";
import { t } from "../core/i18n.js";
import { atlas } from "../i18n/journal-atlas.js";
import { store } from "../core/store.js";
import { engine } from "../core/api.js";
import { profile as platform } from "../platform/profile.js";
import { androidBridge } from "../platform/android.js";
import { dayKey } from "../domain/stats.js";
import { choice, toggle, textField } from "../ui/controls.js";
import { sessionLink } from "../ui/session-link.js";
import { icon } from "../ui/icons.js";

const LINK_ICON = { ok: "linkOk", nomodel: "linkHalf", down: "linkOff", checking: "replay" };
let wiped = false;

function section(id, label) {
  const title = h("h2", { id: "cb-" + id, tabindex: "-1", text: label });
  const note = h("span", { class: "flash", role: "status" });
  const el = h("section", { class: "cb-section cb-" + id, "aria-labelledby": title.id }, h("div", { class: "cb-section-head" }, title, note));
  return { el, title, saved: () => flash(note, t("common.saved")) };
}
function shortcut(target, value) {
  return h("button", { type: "button", class: "cb-shortcut", "aria-controls": target.id, on: { click: () => {
    target.focus({ preventScroll: true }); target.scrollIntoView({ block: "start" });
  } } }, h("span", null, h("strong", { text: target.textContent }), value), icon("arrow"));
}
// User-editable addresses only apply to the loopback transport. Do not save or probe an invalid draft.
function localOrigin(value) {
  const raw = String(value || "").trim();
  if (!/^https?:\/\//i.test(raw) || /[\s\\]/.test(raw)) return null;
  try {
    const u = new URL(raw);
    if (!["localhost", "127.0.0.1"].includes(u.hostname) || u.username || u.password || u.pathname !== "/" || u.search || u.hash) return null;
    return u.origin;
  } catch (_) { return null; }
}
function download() {
  const url = URL.createObjectURL(new Blob([JSON.stringify(store.exportData(), null, 2)], { type: "application/json" }));
  const a = h("a", { href: url, download: "qolda-" + dayKey(new Date()) + ".json" });
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function cabinet(root, router) {
  const { prefs, data } = store.state, phone = platform.name === "android";
  let disposed = false;
  const profile = section("profile", t("cabinet.profile")), keep = section("storage", t("cabinet.storage"));
  const connection = section("connection", phone ? atlas("phoneAnalysis") : t("cabinet.engine")), motion = section("motion", t("cabinet.motion"));
  const shortKeep = h("span"), shortLink = h("span");
  const nick = textField(t("welcome.nick") + " (" + t("common.optional") + ")", atlas("nickHint"), data.profile.nick,
    { maxlength: 40, autocomplete: "off" }, (v) => {
      store.setProfile({ nick: v }); nick.querySelector("input").value = data.profile.nick; profile.saved();
    });
  profile.el.append(nick, sessionLink("cabinet"));

  // Turning persistence off clears the saved copy, not the records already open in this tab.
  const mode = h("p", { class: "cb-mode" });
  const switchKeep = toggle(t("welcome.persist"), atlas(phone ? "phoneKeepHint" : "storageSwitchHint"), prefs.persist, (v) => {
    if (v) setKeep(true);
    else { keepInput.checked = true; confirmOff.hidden = false; cancelOff.focus(); }
  });
  const keepInput = switchKeep.querySelector("input");
  const cancelOff = h("button", { type: "button", class: "btn btn--quiet", text: t("common.keep"), on: { click: () => {
    confirmOff.hidden = true; keepInput.focus();
  } } });
  const confirmOff = h("div", { class: "cb-confirm", role: "group", hidden: true, "aria-labelledby": "cb-off-q" },
    h("p", { id: "cb-off-q", text: atlas(phone ? "phoneOffAsk" : "storageOffAsk") }), h("button", { type: "button", class: "btn btn--quiet",
      text: atlas("storageOffConfirm"), on: { click: () => { setKeep(false); keepInput.focus(); } } }), cancelOff);
  function paintKeep() {
    shortKeep.textContent = atlas(prefs.persist ? "onDevice" : phone ? "phoneSession" : "onlySession");
    mode.textContent = phone ? atlas(prefs.persist ? "phoneKeepOn" : "phoneKeepOff") : t(prefs.persist ? "cabinet.persistOn" : "cabinet.persistOff");
  }
  function setKeep(v) {
    store.setPrefs({ persist: v }); keepInput.checked = v; confirmOff.hidden = true; paintKeep(); keep.saved();
  }
  paintKeep();
  const wipe = h("button", { type: "button", class: "btn btn--quiet", on: { click: () =>
    confirmInline(wipe, t("cabinet.wipeAsk"), t("common.delete"), t("common.keep"), () => { store.wipe(); wiped = true; router.refresh(); }) } },
    icon("trash"), atlas("wipe"));
  const count = (label, n) => h("div", null, h("dt", { text: label }), h("dd", { text: n }));
  const exportStatus = h("p", { class: "cb-export-status", role: "status", "aria-atomic": "true" });
  const exportButton = h("button", { type: "button", class: "btn btn--quiet", on: { click: exportCopy } }, icon("download"), phone ? atlas("phoneExport") : t("cabinet.export"));
  let exporting = false;
  async function exportCopy() {
    if (platform.name !== "android") { download(); return; }
    if (exporting) return;
    exporting = true; exportButton.setAttribute("aria-disabled", "true"); exportStatus.textContent = atlas("exportWaiting");
    try {
      const r = await androidBridge.exportData(store.exportData(), "qolda-" + dayKey(new Date()) + ".json");
      const note = r.kind === "ok" ? "exportDone" : r.kind === "canceled" ? "exportCanceled" :
        ({ too_large: "exportLimit", engine_busy: "exportBusy", picker_unavailable: "exportPicker" })[r.code] || "exportFailed";
      if (!disposed) exportStatus.textContent = atlas(note);
    } catch (_) { if (!disposed) exportStatus.textContent = atlas("exportFailed"); }
    finally { exporting = false; if (!disposed) exportButton.removeAttribute("aria-disabled"); }
  }
  keep.el.append(switchKeep, confirmOff, mode, h("dl", { class: "cb-counts" }, count(atlas("checked"), data.attempts.length),
    count(atlas("diaryEntries"), data.checkins.filter((c) => c.ease || c.note).length)), h("div", { class: "cb-data" },
    h("h3", { text: t("cabinet.data") }), h("p", { class: "cb-hint", text: atlas("exportHint") }), h("div", { class: "cb-actions" }, exportButton, wipe), exportStatus));

  // Keep the saved endpoint on mount, including ports from older packaged clients. Switching to
  // the page origin is a person's choice; a standalone static development page is not an engine.
  const editable = platform.engineSettings && platform.analyzer === "loopback";
  const pageOrigin = editable && location.protocol === "http:" && !["8080", "5173"].includes(location.port) ? localOrigin(location.origin) : null;
  const error = h("p", { id: "cb-url-error", class: "cb-error", role: "status" });
  const input = editable ? h("input", { id: "cb-url", class: "input", type: "url", value: prefs.engineUrl, inputmode: "url", spellcheck: "false",
    autocomplete: "off", autocapitalize: "off", "aria-describedby": "cb-url-hint cb-url-error", on: {
      change: () => { if (saveAddress()) probe(); }, input: () => { error.textContent = ""; input.removeAttribute("aria-invalid"); }
    } }) : null;
  const usePage = pageOrigin ? h("div", { class: "cb-page-origin", hidden: localOrigin(prefs.engineUrl) === pageOrigin },
    h("button", { type: "button", class: "btn btn--quiet", text: atlas("usePageAddress"), "aria-describedby": "cb-page-address", on: { click: () => {
      input.value = pageOrigin; if (saveAddress()) { input.focus(); probe(); }
    } } }), h("code", { id: "cb-page-address", text: pageOrigin })) : null;
  function saveAddress() {
    if (!input || platform.name === "android") return true;
    const next = localOrigin(input.value);
    if (!next) { input.setAttribute("aria-invalid", "true"); error.textContent = atlas("addressInvalid"); return false; }
    input.value = next; input.removeAttribute("aria-invalid"); error.textContent = "";
    if (next !== prefs.engineUrl) { store.setPrefs({ engineUrl: next }); connection.saved(); }
    if (usePage) usePage.hidden = next === pageOrigin;
    return true;
  }
  const link = h("p", { class: "cb-link", role: "status", "aria-atomic": "true" });
  let shown = "", currentRequest = 0, pending = null;
  function paintLink(state) {
    if (state === shown) return;
    shown = state; link.dataset.state = state;
    const checking = state === "checking";
    const label = phone ? atlas(({ ok: "phoneOk", nomodel: "phoneNoModel", down: "phoneDown", checking: "phoneChecking" })[state]) :
      checking ? atlas("connecting") : t("health." + state);
    link.replaceChildren(icon(LINK_ICON[state]), h("span", { text: label }));
    shortLink.textContent = phone || checking ? label : t("health.short." + state);
  }
  async function probe() {
    const target = editable ? prefs.engineUrl : platform.analyzer;
    if (pending === target) return;
    const n = ++currentRequest; pending = target; paintLink("checking");
    let state = "down";
    try { state = await engine.check(); } catch (_) { /* A failed health check is not a result. */ }
    if (!disposed && n === currentRequest) { pending = null; paintLink(state); }
  }
  paintLink(engine.state);
  const offEngine = engine.on((state) => { if (pending === null) paintLink(state); });
  const test = h("button", { type: "button", class: "btn btn--quiet", text: phone ? atlas("phoneCheck") : t("cabinet.test"), on: { click: () => {
    if (saveAddress()) probe(); else input.focus();
  } } });
  if (phone) connection.el.append(h("div", { class: "cb-device-check" }, h("p", { class: "cb-hint", text: atlas("phoneCheckHint") }),
    h("div", { class: "cb-probe" }, test, link)));
  else connection.el.append(h("div", { class: "cb-connection-fields" }, h("div", { class: "cb-endpoint" },
    h("p", { class: "cb-hint", text: editable ? t("cabinet.engineHint.local") : atlas(platform.analyzer === "ondevice" ? "onDeviceAnalyzer" : "fixedAddress") }),
    input ? h("div", { class: "field" }, h("label", { class: "field-label", for: "cb-url", text: t("cabinet.url") }), input,
      h("p", { class: "field-hint", id: "cb-url-hint", text: atlas("addressHint") }), error) : null,
    usePage, h("div", { class: "cb-probe" }, test, link)),
    choice(t("cabinet.defaultModel"), [["scratch", t("check.scratch")], ["transfer", t("check.transfer")]], prefs.model,
      (v) => { store.setPrefs({ model: v }); connection.saved(); })));

  motion.el.append(toggle(t("cabinet.calm"), t("cabinet.calmHint"), prefs.calm, (v) => {
    store.setPrefs({ calm: v }); document.documentElement.classList.toggle("calm", v); motion.saved();
  }));
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) motion.el.append(h("p", { class: "cb-hint", text: t("cabinet.calmSystem") }));
  const about = t("cabinet.aboutText");
  const info = h("details", { class: "cb-about" }, h("summary", { text: t("cabinet.about") }), h("ul", { role: "list" },
    [atlas("aboutQolda"), about[2], about[3], t("cabinet.aboutCredit"), t("cabinet.aboutCase")].map((text) => h("li", { text }))));
  const notice = wiped ? h("p", { class: "notice", role: "status", text: t("cabinet.wiped") }) : null; wiped = false;
  root.append(h("div", { class: "v-cabinet" }, h("header", { class: "page-head" }, h("h1", { text: t("cabinet.title") }),
    h("p", { class: "cb-lead", text: atlas(phone ? "phoneLead" : "settingsLead") })), notice,
    h("div", { class: "cb-quick", role: "group", "aria-label": atlas("settingsNav") }, shortcut(keep.title, shortKeep), shortcut(connection.title, shortLink)),
    h("div", { class: "cb-spread" }, profile.el, keep.el), connection.el, motion.el, info));
  return () => { disposed = true; offEngine(); };
}
