# T-018 · библиотека `net.qolda.engine` — квитанция среза

Дата: 25.11.2026. Модуль: `tools/android-engine` (`:engine`). База: `e43d6ad`, границы — `6ff3557`.
Собран из проверенного комплекта T-017 (`m4-2-engine-input-v1`) и LiteRT 2.4.0.

## Артефакт

| Поле | Значение |
|---|---|
| AAR (относительно приватного каталога `qolda-gpu`, вне Git) | `deliverables/android-engine/qolda-engine-release.aar` |
| SHA-256 | `4e82863a8ea5e7c1ecedbdbce604e3e8f91b748113a33948459caebc462e4022` |
| Размер | 4 224 419 байт |
| ABI | только `arm64-v8a`, ELF64 AArch64 |
| minSdk / target | 26 / 34 (compileSdk 34) |
| Манифест | `net.qolda.engine`, ни одного `uses-permission` |

Содержимое AAR (10 записей): `AndroidManifest.xml` 203, `classes.jar` 4 698, `proguard.txt` 208,
`R.txt` 0, `META-INF/…/aar-metadata.properties`, `jni/arm64-v8a/libqolda_engine.so` 387 464,
`libLiteRt.so` 5 935 696, `libLiteRtClGlAccelerator.so` 3 415 408. Моделей, `.f32`, `engine.json`,
диагностического кода и assets в AAR нет.

## API (ровно зафиксированный, снят `javap` с `classes.jar`)

```
public final class net.qolda.engine.GpuEngine implements java.lang.AutoCloseable {
  public static net.qolda.engine.GpuEngine open(android.content.res.AssetManager) throws EngineException;
  public synchronized void inferPose(float[], float[], float[]) throws EngineException;
  public synchronized float inferLogit(float[]) throws EngineException;
  public synchronized void close();
  public static final int POSE_INPUT, SIMCC_X, SIMCC_Y, FEATURES;
}
public final class net.qolda.engine.EngineException extends java.lang.Exception {
  public String code();   // model_unavailable | invalid_argument | closed
}
```

JNI-экспорты `libqolda_engine.so`: `Java_net_qolda_engine_GpuEngine_nativeOpen`,
`nativeInferPose`, `nativeInferLogit`, `nativeClose`.

## Входные бинарники (пять, SHA-256)

| Файл | Байты | SHA-256 | В AAR |
|---|---:|---|---|
| `assets/qolda/rtmpose-fp32.tflite` | 54 358 484 | `2dd322a0…e5b8cd` | нет (клиент кладёт в assets) |
| `assets/qolda/a-tcn-fp32.tflite` | 22 164 | `99cb4409…d7ec34` | нет |
| `jniLibs/arm64-v8a/libLiteRt.so` | 5 935 696 | `df7fcd5a…101cba` | да |
| `jniLibs/arm64-v8a/libLiteRtClGlAccelerator.so` | 3 415 408 | `23c7f011…dbe465` | да |
| `jniLibs/arm64-v8a/libc++_shared.so` | 1 330 832 | `69e517e6…a86c4` | нет, не нужен |

Почему без `libc++_shared.so`: движок собран с `-DANDROID_STL=c++_static`; `llvm-readelf -d
libqolda_engine.so` показывает `NEEDED` только `libLiteRt.so`, `libandroid.so`, `liblog.so`, `libm.so`,
`libdl.so`, `libc.so`. `libLiteRt.so` и делегат также не требуют `libc++_shared.so`. Дубликатов нет.

Сверка байтов внутри готового AAR: `libLiteRt.so` → `df7fcd5a…` и `libLiteRtClGlAccelerator.so` →
`23c7f011…` совпадают с kit побайтово; `libqolda_engine.so` — 387 464 байта, SHA-256 `11aa8d08…`,
собран из `engine/src/main/cpp/engine.cc` этой сборкой.

## Ворота

- `open(AssetManager)` читает `qolda/engine.json`, сверяет SHA-256, размер, `dtype = float32` и
  сигнатуру `serving_default` обоих графов **до** создания GPU-сессий.
- Обе `CompiledModel` создаются с `kLiteRtHwAcceleratorGpu` (единственный акселератор) и FP32 через
  opaque `gpu_options` (`precision = 2`). Статусы `LiteRtCreateOptions`,
  `LiteRtSetOptionsHardwareAccelerators`, `LiteRtCreateOpaqueOptions`, `LiteRtAddOpaqueOptions`
  проверяются; сбой добавления опций — отказ, не FP16.
- Для каждого графа до первого `Run` требуется `LiteRtCompiledModelIsNonCpuFullyAccelerated == true`
  и все узлы на акселераторе с «GPU» в имени (эталон комплекта: 278 и 60). Частичный делегат
  закрывается, ничего не исполняется.
- Буферы и длины проверяются до `Run` (`invalid_argument`); ошибка `Run`/`Lock`/`Unlock`
  инвалидирует обе сети (`model_unavailable`).
- Сессия владеет окружением, моделями, байтами моделей (LiteRT держит на них указатель) и
  переиспользуемыми буферами; `close` освобождает в обратном порядке и идемпотентен.

## Сборка

```bash
cd tools/android-engine
./gradlew :engine:assembleRelease \
  -PqoldaEngineKit=/path/to/m4-2-engine-input-v1 \
  -PqoldaLiteRtSdk=/path/to/litert-2.4.0
# engine/build/outputs/aar/engine-release.aar
```

`verifyEngineKit` на этапе сборки сверяет SHA и размеры моделей, FP32, `serving_default`, наличие
обоих `.so` и совпадение `libLiteRt.so` с SDK. `libLiteRtClGlAccelerator.so` в SDK и в комплекте —
разные сборки при одинаковом размере; упакована копия из комплекта (`23c7f011…`), прошедшая T-017.

## Границы

Проверка библиотеки сама по себе не доказывает работу финального APK: вызов из клиента, полный клип, камера и офлайн-режим принимаются в T-010. Для этой квитанции телефон, сервисы и тестовые наборы не трогали. Интегратор повторно прочитал API через `javap` из `classes.jar` и таблицу четырёх JNI-экспортов через `llvm-readelf` из AAR после исправления `43e241b`. Не подменять этот файл первым AAR с SHA `f128db1b…0152`.
