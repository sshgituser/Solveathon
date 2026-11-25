// The Android package must never use an HTTP analyzer, including while its bridge is unavailable.
// Web/Windows keep their configured transport. These flags do not modify saved preferences.
const meta = (name) => globalThis.document?.querySelector('meta[name="' + name + '"]')?.content?.trim() || "";
const declared = meta("seriq-profile"), selected = meta("seriq-analyzer");
const connected = declared === "connected";
const web = { name: connected ? "connected" : "local", analyzer: selected || (connected ? "remote" : "loopback"),
  engineSettings: !connected, account: connected, installHint: connected, apiUrl: meta("seriq-api") };
const phone = { name: "android", analyzer: "ondevice", engineSettings: false, account: false, installHint: false, apiUrl: "" };
// Recognize an incomplete Android entry (missing meta or script) without hitting a health URL.
// Once this document is identified as Android, bridge loss cannot downgrade it to Web.
let android = declared === "android" || selected === "ondevice" || globalThis.location?.origin === "https://appassets.androidplatform.net";
function current() {
  if (!android && (globalThis.QoldaAndroid || globalThis.window?.QoldaAndroid)) android = true;
  return android ? phone : web;
}
export const profile = Object.freeze({
  get name() { return current().name; },
  get analyzer() { return current().analyzer; },
  get engineSettings() { return current().engineSettings; },
  get account() { return current().account; },
  get installHint() { return current().installHint; },
  get apiUrl() { return current().apiUrl; }
});
