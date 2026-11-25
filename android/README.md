# Qolda Android (T-010)

This module supplies an Android host and native file/live E02 engine for the shared frontend; it does not create a separate product UI. For integration see [ANDROID_BRIDGE.md](ANDROID_BRIDGE.md). Status and physical evidence are kept in [android-candidate.md](../reports/platforms/android-candidate.md).

## Build

Use JDK 17, AGP 8.5.2, Gradle 8.7, Android SDK 34 (Build Tools 34.0.0) and the separately built T-018 ARM64 AAR. The Gradle distribution is external: no binary wrapper or private assets are in Git. With `JAVA_HOME` and `ANDROID_HOME` configured, from `android/` in Git Bash:

```sh
private="${USERPROFILE}/source/qolda-gpu"
aar="$private/deliverables/android-engine/qolda-engine-release.aar"
kit="$private/android-runtime/m4-2-engine-input-v1"
gradle --offline --no-daemon :app:assembleDebug \
  "-PqoldaEngineAar=$aar" "-PqoldaEngineKit=$kit"
python verify_apk.py --apk app/build/outputs/apk/debug/app-debug.apk --aar "$aar" \
  --aapt "$ANDROID_HOME/build-tools/34.0.0/aapt.exe" \
  --apksigner "$ANDROID_HOME/build-tools/34.0.0/apksigner.bat"
```

The candidate is `net.qolda.offline`, `0.1.0-t010`, arm64 only. It is debug signed for private device verification, not a store release. The AAR owns all native libraries; the app uses only the allowlisted assets and notices from the kit. Do not add the kit's `jniLibs` or change paths/hashes in `engine.json`. Without both properties Gradle builds a distinct `net.qolda.offline.sourcecheck`; it has no working models and is not a product candidate. A bundled common frontend before the Android transport is wired is a technical build, not the product flow. On the current Windows build host, use `/c/prj/tools/gradle-8.7/bin/gradle` with `JAVA_HOME=C:/prj/tools/jdk-17.0.11+9` and `ANDROID_HOME=C:/prj/android-sdk` if the commands are not already on `PATH`.

## Evidence and acceptance

Use a single authorized install of the linked candidate on CPH2399 and launch it normally. Generate the package receipt with `verify_apk.py --require-wired-frontend --out <apk-receipt.json>`, then read back the exact installed version/SHA, main-frame bridge handshake and GPU delegation with `python collect_device_receipt.py --apk-receipt <apk-receipt.json> --out <device-receipt.json>`. The device reader works only against this package and emits aggregate logs; it never exports the package path, tokens, source clip hashes, device serial or coordinates. Leave settings, persistence and camera permission changes to the user. No `adb reverse`, no resets, no force-stop.

For a fresh package, use `adb install <verified.apk>`; if the technical candidate is already installed, any update (`adb install -r <verified.apk>`) requires the owner's go-ahead. Use `adb shell am start -W -n net.qolda.offline/.MainActivity` only for the agreed normal launch. Keep the APK, its `sha256sum`, and both receipts together under the private `deliverables/android/` directory.

Separate build, installed, main-frame bridge, GPU readiness, whole-file and camera verification. A session with USB attached is not cable-free proof. For the final offline observation the owner manually disables radios, disconnects USB, cold-opens the already-installed app, operates the chosen file/live path, then provides the observation. Use only an approved local cycle, never test split `08`. Do not read other apps or device media to find a clip. `no_pose`, dedupe, refusal and image orientation require their own observed evidence; do not infer them from the build or T-017 probe.
