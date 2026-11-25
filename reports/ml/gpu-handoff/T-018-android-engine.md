# T-018 — выпускаемый ARM64 AAR строгого GPU-движка

Дата: 25.11.2026. Модуль: `tools/android-engine` (`:engine`, пакет `net.qolda.engine`). База: `e43d6ad`,
уточнение границ — `6ff3557`. Комплект: приватный `m4-2-engine-input-v1` (два FP32 `.tflite` +
`engine.json`), LiteRT 2.4.0.

## Артефакт

| Поле | Значение |
|---|---|
| Путь (вне Git) | `qolda-gpu/deliverables/android-engine/qolda-engine-release.aar` |
| SHA-256 после исправления путей для AssetManager | `4e82863a8ea5e7c1ecedbdbce604e3e8f91b748113a33948459caebc462e4022` |
| Размер | 4 224 419 байт |
| ABI | `arm64-v8a` (ELF64 AArch64), minSdk 26 |
| Манифест | `net.qolda.engine`, ни одного `uses-permission` |
| `classes.jar` | `net/qolda/engine/GpuEngine.class`, `EngineException.class` |
| `jni/arm64-v8a` | `libqolda_engine.so` 387 464; `libLiteRt.so` 5 935 696 (SHA `df7fcd5a…`); `libLiteRtClGlAccelerator.so` 3 415 408 (SHA `23c7f011…`, байты T-017) |
| `libc++_shared.so` | отсутствует: `c++_static`, LiteRT C++ runtime не требует |

JNI-экспорты `libqolda_engine.so`: `nativeOpen`, `nativeInferPose`, `nativeInferLogit`, `nativeClose`.

Первый AAR (`f128db1b…0152`) передавал в `AssetManager.open` путь ZIP-записи `assets/qolda/...` вместо `qolda/...`. Исправлено: загрузчик проверяет точное разрешённое имя записи и читает файл относительно корня AssetManager, без удвоенного `assets/`. Повтор `:engine:assembleRelease` успешен. Эта проверка упаковки не заменяет запуск нового APK.

## API

```
GpuEngine.open(AssetManager assets)            // обе сети или EngineException(model_unavailable)
inferPose(float[] rgbNhwc, float[] simccX, float[] simccY)
inferLogit(float[] features) -> float          // ровно один скаляр logit[1,1]
close()                                        // идемпотентно
```

Длины: `1×256×192×3`, `1×17×384`, `1×17×512`, `1×64×5`. Проверяются до Run; нарушение —
`invalid_argument`.

## Ворота

- `open` читает `qolda/engine.json`, сверяет SHA-256 и размер каждого графа, `dtype = float32`,
  сигнатуру `serving_default` — до создания GPU-сессий.
- Обе `CompiledModel` создаются с `kLiteRtHwAcceleratorGpu` и только с ним; FP32 включается
  opaque-опцией `gpu_options` (`precision = 2`). Статусы `LiteRtCreateOptions`,
  `LiteRtSetOptionsHardwareAccelerators`, `LiteRtCreateOpaqueOptions` и `LiteRtAddOpaqueOptions`
  проверяются; ошибка добавления — отказ, а не молчаливый FP16.
- Для каждого графа требуется `LiteRtCompiledModelIsNonCpuFullyAccelerated == true` **и** все узлы
  на акселераторе с «GPU» в имени (эталон комплекта: 278 и 60 узлов). Частичный делегат закрывается
  без Run.
- Ошибка Run/Lock/Unlock инвалидирует обе сети: следующий вызов получает `model_unavailable`.
- Сессия владеет окружением, моделями, байтами моделей (LiteRT держит указатель) и буферами;
  `close` освобождает в обратном порядке, частично неудачный `open` — всё созданное.

## Воспроизводимая сборка

```bash
cd tools/android-engine
./gradlew :engine:assembleRelease \
  -PqoldaEngineKit=/path/to/m4-2-engine-input-v1 \
  -PqoldaLiteRtSdk=/path/to/litert-2.4.0
```

`verifyEngineKit` на этапе сборки сверяет SHA-256 и размеры моделей, FP32 и `serving_default`,
наличие обоих `.so` и совпадение `libLiteRt.so` с SDK. Замечено и зафиксировано: делегат
`libLiteRtClGlAccelerator.so` в SDK и в проверенном комплекте — разные сборки при одинаковом размере;
упаковывается копия из комплекта (`23c7f011…`), потому что именно она прошла T-017 на телефоне.

## Чего это не доказывает

AAR не подключался к финальному APK и не вызывался из продукта: вызов из T-010, полный клип, камера
и офлайн-режим не проверены. Новый APK и телефонный прогон ради AAR не выполнялись — по T-018 они
не нужны, физическая приёмка остаётся одним согласованным сеансом T-010. Тестовые наборы не
запускались, телефон и сервисы не трогались.
