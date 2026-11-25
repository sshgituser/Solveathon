# Qolda — отчёт по M0/M1 и обучению A (25.11.2026)

Исполнитель: разработчик модели. База: `main` `0939392`. ПК заменён: прежние
`reports/ml/forearm-v1/*` относятся к другой машине (RTX 3050 Laptop, CUDA 12.6) и не
пересчитывались. Исторический freeze не изменялся.

## 1. Среда (M0)

| Что | Факт |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 283 MiB, compute 12.0, драйвер 617.42 |
| CUDA / cuDNN | 13.0 / 9.19 (из колеса PyTorch) |
| PyTorch | `2.11.0+cu130`, `torch.cuda.is_available() = True` |
| ONNX Runtime | `1.27.0`, провайдеры `Tensorrt/CUDA/CPU` |
| Прочее | onnx 1.20.1, numpy 2.2.6, opencv 5.1.0, safetensors 0.6.2, ffprobe 8.1.2 |
| Android | SDK/ADB/JDK 17/Gradle 8.7 есть (`C:/prj`), NDK и CMake отсутствуют; телефон OnePlus CPH2399 (MT6893, Mali-G77, API 34) подключён |

Закреплено в `ml/requirements-gpu.txt`, машинная запись — `reports/ml/gpu-handoff/environment.json`.

**Найдено на месте:** `onnxruntime-gpu` не находит CUDA-провайдер, пока `<site-packages>/torch/lib`
не добавлен через `os.add_dll_directory` (там `cublasLt64_13.dll`, `cudnn64_9.dll`, `cudart64_13.dll`).
Без этого ORT молча остаётся на CPU. Закрыто в `ml/gpu.enable_cuda_dlls()`.

## 2. Входы

| Вход | Статус | Доказательство |
|---|---|---|
| RTMPose Body7 `end2end.onnx` | есть | SHA-256 `5c0a4bf6…68e3c`, совпал с `freeze.json` |
| SLiM NTU60 XSub база | есть | SHA-256 `c25c70db…5bf`, совпал с `ATTRIBUTION.md` |
| Приватные `A-tcn`/`B-slim-last-block` v1 | **нет** | приватной передачи `seriq-local` на ПК нет |
| Наш размеченный E02 | **нет** | передача отсутствует |
| Расширенный корпус | **нет** | манифест источников отсутствует |
| Публичный E02 (Mendeley V1, CC BY 4.0) | скачан полный архив, приватно | 491 клип, 1.16 GiB; development-часть E02 = 107 файлов (77 + 20 + 10 карантин) |
| Исторический test (08) | не использован | присутствовал в полном архиве; в манифест и оценку A не попал, распакованные E02 `Test/` удалены с диска |
| Физический телефон | подключён | OnePlus CPH2399, API 34, `arm64-v8a`; модели на телефоне не запускались |

## 3. Строгий GPU-движок позы (M1a)

Исходный экспорт: 247 узлов, opset 11, динамический batch, хвост `Shape → Slice → Concat → Reshape`.
Профиль ORT показал 2 узла на CPU (`Slice`, `Concat`) — ровно эта shape-цепочка.

Подготовка графа (`ml/gpu.prepare_static_graph`): batch=1, `Shape` заменён константой `[1,17,8,6]`,
константные `Slice`/`Concat` свёрнуты, `Reshape` получил статичный shape `[1,17,-1]`.
Статическая эквивалентность: `Slice(Shape(x), 0..2) = [1,17]`, `Concat([1,17], [-1]) = [1,17,-1]`,
исходный `Reshape` даёт `[1,17,48]` при `17×8×6`.

| Метрика | Значение |
|---|---|
| узлов до/после | 247 → 244 |
| SHA исходника / экспорта | `5c0a4bf6…` / `2e0a9c0c…` |
| узлов в профиле | 182, все `CUDAExecutionProvider`, `non_cuda = []` |
| паритет к архивному смешанному референсу (64 кадра) | `max|Δ| = 6.55e-4`, SimCC-бины 99.13 %, видимость 100 % |

Численное сравнение выполнено с **архивным смешанным референсом** (исходный граф снят один раз на CUDA,
два shape-узла шли на CPU); автоматического повторного смешанного прогона в проверке больше нет:
`tools/verify_gpu_engine.py` требует готовый `.npz` и сверяет хэш входов. Строгим GPU-паритетом
это не называется.

Приёмка M1a выполнена: `session.disable_cpu_ep_fallback=1`, Python `disable_fallback()` до первого запуска,
профиль подтверждает 0 CPU-узлов. При недоступной CUDA загрузка падает (`CudaUnavailable`), а не
переключается на CPU.

## 4. Обучение A (M2, только исходный E02)

Манифест собран `tools/prepare_e02_development.py`: участники 01–04 train, 05 validation,
`Test` пропущен **до** открытия файлов. SHA манифеста `c725b8bb…`.

| Split | Всего | Scorable | Balanced acc. | F1 complete | Coverage |
|---|---:|---:|---:|---:|---:|
| train | 77 | 71 | 1.0000 | 1.0000 | 0.9221 |
| validation | 20 | 20 | 1.0000 | 1.0000 | 1.0000 |

Отказы train: 5 × `incomplete_forearm_coverage`, 1 × `insufficient_visible_forearm_pair`.
Обучение: 36 эпох, batch 8, lr 2e-3, seed 601, 324 шага, `cuda:0`, peak CUDA 20.4 MB,
8 обучаемых тензоров, все изменены, frozen не было (A с нуля).

Артефакт: `A-tcn.safetensors`, SHA-256 `8dd234d4…24e`, threshold 0.5,
version `seriq-e02-forearm-1.0.0`, `variant=A`, `corpus_scope=mendeley-e02-public-dev`.
Приватный bundle: `seriq-bundle-v1` (модель + поза + манифест по SHA).

**Ограничение, которое нельзя прятать:** validation — один человек (05), 20 клипов из 20.
Это development-число, не перенос на популяцию. `Test` (08) закрыт.

## 5. Движок (M1b)

`backend/runtime.load_bundle_engines` грузит bundle на строгой CUDA, проверяет SHA графа,
SHA весов, метаданные артефакта и CUDA-размещение всех параметров.

| Сценарий | Результат |
|---|---|
| разрешённый development-клип | классификация получена на CUDA; запущенная версия `seriq-e02-forearm-1.0.0+8dd234d4a3e1`; идентификаторы клипа и прогноз — только в приватной квитанции |
| пустой клип (3 с серого экрана) | `warnings=["no_pose"]`, `attempts=[]`, счётчик не растёт |
| битый SHA весов | `ValueError: artifact SHA-256 mismatch` — движок не регистрируется |
| без GPU (`CUDA_VISIBLE_DEVICES=-1`) | `ValueError: strict CUDA pose engine is unavailable` — классификации нет |
| загрузка движка упала | сервер стартует без моделей: Web доступен, анализ → `503 model_unavailable` |

## 6. Что осталось незакрытым

1. **M1 и A нового прогона приняты на этом ПК.** Это новый A, не восстановленный артефакт v1.
2. **B и C** — нет нашего и расширенного корпусов; исходный E02 вместо них не подставляется.
3. **Android (M4)** — инструменты и телефон есть: OnePlus CPH2399, MT6893, Mali-G77 MC9, API 34,
   arm64-v8a (серийник в отчёт не входит). Нет NDK и CMake; LiteRT-делегация не проверялась.
4. **M3:** [Web в Chromium пройден](M3-web.md); Windows `.exe` и проверка на ноутбуке защиты ещё открыты.
5. **Статика shape-цепочки** доказана по графу и паритету; повторного смешанного прогона
   в рабочем режиме нет — референс снят однократно с исходного графа на CUDA.

## 7. Артефакты вне Git (приватно)

- `runs/A-gpu-dev-v1r/` — `A-tcn.safetensors`, `receipt.json`, `report.json`
- `bundles/A-gpu-dev-v1/` — `bundle.json`, `end2end.b1.onnx`
- `gpu-runs/`, `receipts/gpu-full-A-20261125.json` — GPU-receipt
- `manifests/e02-dev-v1.json` — поклиповый манифест (в Git не кладётся)
