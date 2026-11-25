# Qolda Android engine (T-018)

Выпускаемая ARM64-библиотека (`AAR`) со строгим GPU-движком Qolda: RTMPose (SimCC) и классификатор
E02 на весах A. Это **не** диагностический APK и не UI: видео, камеры, истории, порогов и HTTP здесь
нет — это слой T-010.

## Что внутри

- `net.qolda.engine.GpuEngine` — единственная точка входа;
- `net.qolda.engine.EngineException` с `code()`: `model_unavailable`, `invalid_argument`, `closed`;
- `jni/arm64-v8a/libqolda_engine.so` — JNI, собран из `engine/src/main/cpp/engine.cc`;
- `jni/arm64-v8a/libLiteRt.so` и `libLiteRtClGlAccelerator.so` — ровно те байты LiteRT 2.4.0,
  которые прошли T-017 на CPH2399 (`libLiteRt.so` SHA-256 `df7fcd5a…`, делегат `23c7f011…`);
- `libc++_shared.so` намеренно отсутствует: движок линкован со статическим libc++, а LiteRT C++
  runtime не требует.

Модели, лицензии и `engine.json` в библиотеку **не входят**: клиент кладёт проверенный комплект
(`m4-2-engine-input-v1/assets/qolda/`) в свои `assets/`.

## Сборка

Нужны JDK 17, Android SDK 34, NDK 26.1.10909125, CMake 3.22.1, Gradle 8.7 (или `./gradlew`).

```bash
cd tools/android-engine
./gradlew :engine:assembleRelease \
  -PqoldaEngineKit=/path/to/m4-2-engine-input-v1 \
  -PqoldaLiteRtSdk=/path/to/litert-2.4.0
# AAR: engine/build/outputs/aar/engine-release.aar
```

`-PqoldaEngineKit` — приватный комплект T-017 (`assets/qolda/engine.json` + два FP32 `.tflite` +
`jniLibs/arm64-v8a`). `-PqoldaLiteRtSdk` — каталог LiteRT 2.4.0 с `sdk/litert_cc_sdk` и
`jni/arm64-v8a/libLiteRt.so`.

На этапе конфигурации `verifyEngineKit` сверяет SHA-256 и размеры обеих моделей, FP32,
`serving_default`, наличие обоих `.so` и совпадение `libLiteRt.so` с SDK. Сборка с чужими байтами
падает, а не собирает «похожий» движок.

## Вызов из T-010

```java
try (GpuEngine engine = GpuEngine.open(context.getAssets())) {
    float[] frame = new float[GpuEngine.POSE_INPUT];       // 1x256x192x3, нормализованный NHWC RGB
    float[] x = new float[GpuEngine.SIMCC_X];              // 1x17x384
    float[] y = new float[GpuEngine.SIMCC_Y];              // 1x17x512
    engine.inferPose(frame, x, y);
    float[] features = new float[GpuEngine.FEATURES];      // 1x64x5: правый локоть xy, запястье xy, маска
    float logit = engine.inferLogit(features);             // ровно один скаляр
}
```

`open` бросает `EngineException` с `model_unavailable`, если ассет отсутствует, не совпал SHA-256,
не сошлась сигнатура или GPU не смог целиком разместить хотя бы одну сеть. Длины буферов
проверяются до Run: `invalid_argument` без сырых путей и тензоров.

## Правила

- Один экземпляр на один рабочий поток. Вызовы сериализованы (Java `synchronized` + мьютекс в C++).
- Обе `CompiledModel` создаются при `open`; Run разрешён только после
  `LiteRtCompiledModelIsNonCpuFullyAccelerated == true` и подтверждения, что все узлы на LiteRT GPU.
  Эталон этого комплекта: 278 узлов позы и 60 классификатора.
- Нет CPU/NPU fallback. Ошибка Run или Lock/Unlock инвалидирует **обе** сети: последующий вызов
  получает `model_unavailable`, а не частично рабочнй движок.
- Сессия владеет окружением, моделями, байтами моделей (LiteRT держит на них указатель) и
  переиспользуемыми буферами; `close` освобождает в обратном порядке и идемпотентен. Частично
  неудачный `open` освобождает всё, что успел создать.
- FP32: `gpu_options` c TOML `precision = 2`; без него Mali считает в FP16 и паритет плывёт.
- Никаких разрешений: манифест AAR пуст.

## Чего этот модуль не делает

Не декодирует видео, не работает с SAF/камерой, не считает видимость и пороги, не ведёт историю,
не ходит в сеть и не пишет файлы. AAR не доказывает вызов из финального APK, полный клип и
офлайн-режим — это приёмка T-010.
