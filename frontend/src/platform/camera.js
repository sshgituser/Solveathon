// Camera capture is a delivery capability, not a screen concern: it exists only in a secure
// context, only with MediaRecorder, and the browser reports refusals with names that mean
// different things. The screen asks this module for a reason key and shows the matching words;
// choosing a file never touches any of this.
const CANDIDATES = [
  "video/webm;codecs=vp9",
  "video/webm;codecs=vp8",
  "video/webm",
  "video/mp4"
];

export const capture = {
  /** getUserMedia is absent on insecure origins and in some web views. */
  supported() {
    return !!(globalThis.navigator && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  },

  recorderSupported() {
    return typeof globalThis.MediaRecorder === "function";
  },

  /** A supported MIME type, "" to let the browser choose, or null when recording is impossible. */
  mimeType() {
    if (!this.recorderSupported()) return null;
    for (const type of CANDIDATES) {
      try { if (MediaRecorder.isTypeSupported(type)) return type; } catch (_) { /* keep looking */ }
    }
    return "";
  },

  /** One promise around getUserMedia; the caller maps the rejection to words. */
  async open() {
    return navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
  },

  /**
   * Turn a getUserMedia/DOMException into a dictionary key. The person sees the difference
   * between "not allowed", "no camera", "camera busy" and "this address cannot ask".
   */
  reason(error) {
    if (!this.supported()) {
      return globalThis.isSecureContext === false ? "cameraInsecure" : "cameraUnsupported";
    }
    const name = error && error.name;
    if (name === "NotAllowedError" || name === "PermissionDeniedError") return "cameraDenied";
    if (name === "NotFoundError" || name === "DevicesNotFoundError" || name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") return "cameraMissing";
    if (name === "NotReadableError" || name === "TrackStartError" || name === "AbortError") return "cameraBusy";
    if (name === "SecurityError" || name === "TypeError") return "cameraInsecure";
    return "cameraDenied";
  }
};
