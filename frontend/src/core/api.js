// The one place screens ask about the analyzer. Which analyzer answers (local engine,
// hosted service, in-browser) is decided by the delivery profile in platform/.
import { createAnalyzer } from "../platform/analyzer.js";
import { store } from "./store.js";

export const MAX_BYTES = 67108864;
export const TYPES = ["video/mp4", "video/webm"];

const analyzer = createAnalyzer(() => store.state.prefs.engineUrl);
const subs = new Set();

// Shared status, so the shell pill and the screens agree.
export const engine = {
  state: "down",
  on(fn) { subs.add(fn); return () => subs.delete(fn); },
  async check() { this.state = await analyzer.health(); subs.forEach((fn) => fn(this.state)); return this.state; },
  analyze: (blob, type, model, signal) => analyzer.analyze(blob, type, model, signal)
};

export async function sha256(blob) {
  if (!crypto.subtle) return null;
  const buf = await crypto.subtle.digest("SHA-256", await blob.arrayBuffer());
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * The key that makes one physical clip count once. Normally the SHA-256 of its bytes.
 * On an origin without SubtleCrypto a non-cryptographic fingerprint of size and the first and
 * last 64 KiB is used instead; it is local duplicate detection only, never a security proof.
 */
export async function clipHash(blob) {
  if (globalThis.crypto && crypto.subtle) {
    try {
      const full = await sha256(blob);
      if (full) return full;
    } catch (_) { /* fall through to the local fingerprint */ }
  }
  let h = 0x811c9dc5;
  const parts = [blob.slice(0, 65536)];
  if (blob.size > 65536) parts.push(blob.slice(Math.max(65536, blob.size - 65536)));
  for (const part of parts) {
    const bytes = new Uint8Array(await part.arrayBuffer());
    for (const b of bytes) { h ^= b; h = Math.imul(h, 0x01000193); }
  }
  return "local-fnv1a:" + (h >>> 0).toString(16).padStart(8, "0") + ":" + blob.size;
}
