# Qolda Desktop for Windows (T-009)

`Qolda.exe` is a WinForms/WebView2 x64 application, not a shortcut to a browser. It opens the existing local Qolda frontend served by `backend.api` and uses that same instance to analyze videos. The desktop distribution ships a self-contained .NET client and WebView2 loader; it requires the installed Microsoft Edge WebView2 Runtime and a separate NVIDIA CUDA engine. No installer, weights, Python, CUDA libraries, raw videos, or app data are included in the client ZIP.

## Run the portable pair

Extract the ZIP into the same parent directory as the private `engine-win64` delivery:

```text
<deliverables>/
  windows/Qolda.exe
  windows/WebView2Loader.dll
  windows/runtimes/win-x64/native/WebView2Loader.dll
  engine-win64/start-engine.cmd          (optional manual way to start the engine)
  engine-win64/python/python.exe
  engine-win64/app/{backend,ml,frontend,assets}/
  engine-win64/runtime/{bundle.json,end2end.b1.onnx,end2end.onnx,A-tcn.safetensors,ffprobe.exe}
```

1. Open `windows/Qolda.exe`. If `127.0.0.1:8765` already returns the `seriq.v1` health schema and serves the frontend, Qolda attaches without touching its process or reading runtime configuration. A conflicting, nonresponding or API-only service is reported, not replaced.
2. Only on an explicit connection refusal during startup, Qolda checks the sibling layout and starts `python.exe -B -E -m backend.api --port 8765 --bundle ... --artifacts ... --pose-model ... --private-scratch ... --ffprobe ...`, with `engine-win64/app` as the working directory. To start it manually instead, run `engine-win64/start-engine.cmd` and wait for its readiness message before opening Qolda. Qolda never stops or restarts the engine, including when the window closes. «Подключиться» retries attachment only. The generated CUDA pose graph must already be next to the bundle; the desktop app does not regenerate or copy private data.
3. «Модель A готова» comes from `/api/v1/health`, only after the embedded local frontend renders. If the service is available but the model isn't, the instruction remains available and the native status shows the failure. If startup files are missing, the status lists their names. Select a permitted MP4/WebM through the existing system file picker and confirm consent before analysis. Only A (`scratch`) is prepared by this strict engine; choosing B/`transfer` is not a fallback. File selection does not require camera access.

Each window stays on the selected `http://127.0.0.1:<port>`; keep that origin in the frontend's engine settings. Top-level and frame navigation, popups, external schemes and WebView2-intercepted off-origin requests are blocked; same-origin `blob:` previews and inline `data:` assets are allowed. The shell exposes no native bridge and denies permission requests except a same-origin, user-initiated camera request, which is left to the WebView2 prompt without saving a host decision. The shared frontend saves session history to disk only if enabled by the user. Default transient clip storage is `%LOCALAPPDATA%/Qolda/scratch`. Nothing from these user-data directories is packaged or logged by the desktop client. A matching health shape is not cryptographic proof of an attached engine's identity or CUDA placement; attach only to a local engine you trust. The provided strict bundle startup delegates CUDA validation to `backend/runtime.py` and `ml/gpu.py`.

## Attach without starting an engine

```powershell
.\Qolda.exe --attach-port 8870
```

The only accepted values are `8765` and `8870` (literal port numbers, not hostnames or URLs). The engine and frontend must already be served on that port; this mode never reads runtime JSON, starts, stops or restarts an engine. `--attach-port` cannot be combined with `--config`. If a service is unavailable, «Подключиться» probes again without launching it.

There is at most one Qolda window per port in the current Windows session. Port `8765` uses the original instance lock and `%LOCALAPPDATA%/Qolda/WebView2` profile, including when attaching explicitly. The `8870` window can run alongside the installed `8765` client and uses the separate `%LOCALAPPDATA%/Qolda/WebView2-8870` profile. Opening `Qolda.exe` without arguments still follows the portable-pair behavior on `8765`.

## Point at an existing private runtime

Copy `qolda.example.json` to `qolda.local.json` next to `Qolda.exe`, or run `Qolda.exe --config <path>` with your own JSON on a local drive. The file configures only runtime startup on `8765`, not the host or port. Relative runtime entries are interpreted relative to the JSON file directory; without a config they are relative to the EXE directory. Absolute local paths and Windows `%VARIABLE%` expansion are allowed for runtime entries. When supplying only some fields, omitted fields keep the defaults in the example and are also relative to the config directory. `attachOnly: true` disables automatic startup; `dllDir` optionally adds the backend's process-local Windows DLL directory. Scratch must be outside the backend app and outside Git. Do not put the local config, WebView2 data or video scratch in the release ZIP or Git.

## Build (without running)

From `desktop/` in Windows PowerShell with the .NET 10 SDK:

```powershell
.\Build-Windows.ps1 -OutputDirectory '<deliverables>\windows'
```

Use a new empty output directory outside the source tree. The build restores the locked WebView2 SDK from NuGet and creates the runnable folder, `SHA256SUMS.txt`, and a ZIP with the `windows/` root. Build-time NuGet access does not imply runtime Internet access. Physically disconnected startup, analysis of a permitted video, a clean laptop, and publisher signing are separate acceptance steps; see [`windows-candidate.md`](../reports/platforms/windows-candidate.md) for what this candidate actually verified.
