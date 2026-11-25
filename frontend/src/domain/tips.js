// General guidance only. Copy lives in the dictionaries under guide.tips.<id>.
// Nothing here depends on analysis results: the app never turns a verdict into advice.
export const CATEGORIES = ["record", "routine", "safety", "doctor"];
export const TIPS = [
  { id: "camera", cat: "record" }, { id: "light", cat: "record" }, { id: "single", cat: "record" },
  { id: "regular", cat: "routine" }, { id: "goal", cat: "routine" }, { id: "diary", cat: "routine" },
  { id: "pain", cat: "safety" }, { id: "posture", cat: "safety" }, { id: "notdx", cat: "safety" }, { id: "urgent", cat: "safety" },
  { id: "ask", cat: "doctor" }, { id: "show", cat: "doctor" }
];
/** Same tip for the whole day, a different one tomorrow. */
export function tipOfDay(date = new Date()) {
  const n = Math.floor(new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime() / 86400000);
  return TIPS[((n % TIPS.length) + TIPS.length) % TIPS.length];
}
