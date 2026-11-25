# T-017 — технический Android APK: полная GPU-делегация обоих графов

Дата: 25.11.2026. Устройство: OnePlus CPH2399, MediaTek MT6893, GPU Mali-G77 MC9 (OpenGL ES 3.2),
Android 14 / API 34, ABI arm64-v8a. Серийник в отчёт не входит.

## Статусы по отдельности

| Слой | Состояние | Доказательство |
|---|---|---|
| `built` | да | APK `net.qolda.diag` versionName `t017-1`, 59 350 128 байт, SHA-256 `2a42b2cf…68afa` |
| `installed` | да | `adb install -r` → Success; `dumpsys package` versionName `t017-1`, lastUpdate 03:20 |
| `gpu_verified` | да | профиль делегата: pose 278/278 узлов, classifier 60/60, по 1 партиции, `LiteRtCompiledModelIsNonCpuFullyAccelerated` = true до первого `Run` |
| `parity_verified` | да, на dev-входе | pose: `max|Δ|` = 0.003813494, 1 бин из 17 (сдвиг 1), видимость 0 расхождений; classifier: `logit` 13.5313282 против 13.5313292 CUDA, `|Δ|` = 9.54e-7 |
| `e2e` (реальный клип, камера, авиарежим) | нет | отдельная приёмка T-010, не входит в технический APK |

## Что внутри APK

- `assets/rtmpose-fp32.tflite` — 54 358 484 байта, SHA-256 `2dd322a0…ce5b8cd`,
  вход `input[1,256,192,3]` NHWC RGB, выходы `simcc_x[1,17,384]`, `simcc_y[1,17,512]`.
- `assets/a-tcn-fp32.tflite` — 22 164 байта, SHA-256 `99cb4409…fd29b6d7ec34`,
  вход `features[1,64,5]`, выход `logit[1,1]`; веса A `8dd234d4…24e`.
- `lib/arm64-v8a/libLiteRt.so`, `libLiteRtClGlAccelerator.so` (LiteRT v2.4.0), `libqolda_diag.so`.
- Эталоны паритета: кадр клипа 05 (препроцессированный NHWC) и выходы strict-CUDA графа.

`aapt dump permissions`: **ни одного `uses-permission`** — сети у приложения нет.
`aapt dump badging`: `native-code: 'arm64-v8a'`, minSdk 26, targetSdk 34.

## Как получена полная делегация

Первый кандидат (`exported_unverified`) делегировал только 57/262 узлов позы и 60/61 классификатора;
компиляция с GPU отвечала `kLiteRtStatusErrorCompilation` (504). Причины из лога делегата:

| Граф | Отказ делегата | Причина |
|---|---|---|
| pose | `RELU_0_TO_1: Not supported` | `HardSigmoid` из ONNX |
| pose | `SHAPE: Operation is not supported`, `RESHAPE: Expected 1 runtime input` | `Squeeze`/`Unsqueeze` превращались в динамическую Shape→Gather→Reshape-цепочку |
| classifier | `SQUEEZE: Operation is not supported` | финальный `squeeze(-1)` |

Экспорт переписан без подмены модели, с проверкой паритета ONNX на CUDA до/после:

1. `HardSigmoid` → `max(0, min(1, alpha*x + beta))` (MAXIMUM/MINIMUM/MUL/ADD).
2. `Squeeze`/`Unsqueeze` со статическими формами → `Reshape` с константной формой.
3. Финальный `Squeeze` у A → `Reshape [1,1]`; выходной контракт `[1,1]`.
4. ONNX-паритет: pose `max|Δ|` = 6.8e-4 при сдвиге argmax не более 1 бина; A — `0.0`.

После этого: 278/278 узлов позы и 60/60 классификатора на `LiteRT GPU`, по одной партиции.

**Точность.** OpenGL-делегат на Mali по умолчанию считает в FP16: первый прогон дал
`max|Δ|` = 0.0227 и 7 расхождений argmax. Опции `Lrt*` не экспортируются этой сборкой, поэтому
FP32 включён через opaque-опцию `gpu_options` с TOML `precision = 2` (enum
`kLiteRtDelegatePrecisionFp32`). После этого `max|Δ|` упал до 0.0038, расхождений argmax — 1.

## Воспроизводимость

- Репозиторий: `4458b10` (код модели и движка).
- Переписывание ONNX: `tools/export_android_onnx.py` (без весов и без данных).
- Конвертация: `onnx2tf==2.6.9`, `-tb flatbuffer_direct -nuo -dsm -osd -coion` (`-kat features` для A).
- Сборка APK: Gradle 8.7, AGP 8.5.2, NDK 26.1.10909125, CMake 3.22.1, LiteRT 2.4.0 C/C++ SDK.

## Ограничения

- Проверен один разрешённый dev-вход (кадр клипа 05 + его forearm-последовательность), не клип целиком
  и не камера. Решение «complete/incomplete» на телефоне ещё не проверялось.
- Расхождение 1 бин на 1 из 17 точек — в допуске «не больше одного бина», не ноль.
- APK диагностический: экран показывает JSON, UI/истории/камеры в нём нет.
- Веса, видео, скелеты и поклиповый манифест в Git не попадают; APK с весами не публикуется.

## Приёмка и передача T-010

Сверено установленное `net.qolda.diag:t017-1` через `dumpsys` и SHA-256 файла на телефоне с локальным APK: `2a42b2cfc9fe048ad25f3f838049f33918339f8adf4ddbcfc0e7e93ae3d68afa`. Подпись v2 валидна; `aapt` подтвердил только `arm64-v8a` и отсутствие `uses-permission`. Статическое чтение моделей в том APK даёт 278 и 60 операций соответственно и подтверждает `logit[1,1]`. Инференс повторно не запускали; показатели полной делегации и паритета выше — физический прогон Дисика, не результат статического чтения.

В диагностическом APK найден **старый `assets/bundle.json`**: его хэши и `logit[1]` относятся к `m4-1-candidate-v1`, не к фактическим моделям. Также там лежат диагностические тензоры. Для T-010 подготовлен приватный `m4-2-engine-input-v1` с `assets/qolda/engine.json`. Из локального t017-2 с SHA `16263ca0d00c3f87341e9ec4067e94daa8b21792abb292ab31fea19188bafac0` взяты только записи, чьи байты совпадают с `t017-1`: обе модели, `libLiteRt.so`, `libLiteRtClGlAccelerator.so` и `libc++_shared.so`. Это не APK `0dd5228f…a0f` из [последующей проверки на синтетическом эталоне](M4.1-android-device.md), несмотря на одинаковое имя версии. Старый манифест, диагностическая `.so`, входы и эталоны не копируются; финальную сборку это не делает проверенной на устройстве. [Сценарий и инструкция по запуску](../../../docs/SUBMISSION.md).
