# 3D models: sources and licenses

| File | Work | Author | Source | License |
|---|---|---|---|---|
| `mannequin.glb` | Animated Base Character | Quaternius | https://poly.pizza/m/cwYvO5UauX | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |

The mannequin is used unmodified as a file. At run time the client replaces its two
materials (body and joints) and drives the arm bones itself.

three.js r170 and its `GLTFLoader` and `BufferGeometryUtils` modules in `assets/vendor/three/`
are MIT licensed (see `assets/vendor/three/LICENSE`). Their `import` paths were rewritten to the
local build so the client needs no bundler and no network.
