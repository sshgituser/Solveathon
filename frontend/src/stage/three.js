// The one place the stage imports three.js from, so every module shares a single copy of the library.
export * from "../../../assets/vendor/three/three.module.min.js";
export { GLTFLoader } from "../../../assets/vendor/three/GLTFLoader.js";
export { RoomEnvironment } from "../../../assets/vendor/three/RoomEnvironment.js";
export { mergeVertices } from "../../../assets/vendor/three/BufferGeometryUtils.js";
