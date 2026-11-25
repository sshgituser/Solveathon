// Studio light for a white figure on a light page: one soft key with a shadow, a cool rim, sky fill.
import {
  DirectionalLight, HemisphereLight, PMREMGenerator, RoomEnvironment, Mesh, PlaneGeometry, ShadowMaterial,
  NeutralToneMapping, VSMShadowMap
} from "./three.js";

export function setupRenderer(renderer, { mobile = false } = {}) {
  // Neutral keeps whites white; ACES would grey the ceramic.
  renderer.toneMapping = NeutralToneMapping;
  renderer.toneMappingExposure = 1.02;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = VSMShadowMap; // the only map type whose radius softens a directional shadow
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, mobile ? 1.5 : 2));
}

export function buildLights(scene, { mobile = false } = {}) {
  const key = new DirectionalLight(0xffffff, 2.3);
  key.position.set(-2.4, 4.6, 3.4);
  key.castShadow = true;
  key.shadow.mapSize.set(mobile ? 512 : 1024, mobile ? 512 : 1024);
  key.shadow.radius = 7; key.shadow.blurSamples = 14; key.shadow.bias = -0.0004;
  const c = key.shadow.camera; c.left = -1.6; c.right = 1.6; c.top = 2.2; c.bottom = -0.4; c.near = 0.5; c.far = 12;
  const rim = new DirectionalLight(0xd4ecf1, 1.25);
  rim.position.set(3.2, 2.4, -3.6);
  const fill = new HemisphereLight(0xffffff, 0xcddde1, 0.75);
  scene.add(key, rim, fill);

  // The floor exists only to catch the shadow; the page background shows through it.
  const floor = new Mesh(new PlaneGeometry(14, 14), new ShadowMaterial({ opacity: 0.17 }));
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true;
  scene.add(floor);
  return { key, rim, fill, floor };
}

/** Image-based light from a procedural room. Must be rebuilt after a lost WebGL context. */
export function buildEnvironment(renderer, scene) {
  const pmrem = new PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const target = pmrem.fromScene(room, 0.04);
  scene.environment = target.texture;
  scene.environmentIntensity = 0.55;
  room.dispose(); pmrem.dispose();
  return () => target.dispose();
}
