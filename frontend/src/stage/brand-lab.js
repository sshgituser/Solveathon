// Development page: the 3D mark at large sizes on both grounds, next to the flat concepts.
import { mountLogo } from "./logo3d.js";

for (const el of document.querySelectorAll("[data-logo3d]")) {
  const dark = el.dataset.logo3d === "dark";
  mountLogo(el, el, { size: +el.dataset.size || 240, dark });
}
