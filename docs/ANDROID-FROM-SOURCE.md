# Qolda Android: сборка на ПК Димаса из исходников

Цель — **новый локальный ARM64 debug APK с E02/A**, а не восстановление байтов T-010/T-018. Входы: согласованный клон, официальные upstream-пакеты и свой E02/A, обученный на CUDA на этом же ПК. Не нужны наши APK, AAR, ZIP, LiteRT-«кит», ONNX/TFLite, Gradle wrapper JAR или ключ с другого ПК. Android-цепочка не открывает видео, не обучает, не запускает тесты, не обращается к устройству и не меняет общий клон. Шаг обучения ниже — отдельная операция Димаса при разрешении, не часть команды сборки.

## 1. Windows x64: инструменты и лицензии

Из **корня клона**, в PowerShell. Нужны `git`, `curl.exe`, [uv](https://docs.astral.sh/uv/getting-started/installation/) и NVIDIA-драйвер с поддержкой CUDA 13 для E02/A. Выбрать короткий новый локальный путь вне клона и облачной синхронизации; из-за глубины Gradle/NDK лучше `C:\qolda-local`. Команды выполняются от обычного пользователя. Не брать SDK, его лицензии, Gradle cache или `.android` с другого ПК.

```powershell
$ErrorActionPreference = 'Stop'
$D = 'C:\qolda-local'
if (Test-Path -LiteralPath $D) { throw 'Выберите новый локальный корень' }
New-Item -ItemType Directory -Path "$D\downloads", "$D\toolchains" -Force | Out-Null

function Get-AndroidZip([string]$Url, [string]$Dest, [string]$Pin) {
    & curl.exe --fail --location --retry 2 --output $Dest $Url
    if ($LASTEXITCODE) { throw "Скачивание не удалось: $Url" }
    if ((Get-FileHash -LiteralPath $Dest -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Pin) {
        throw "Неверный SHA-256: $Dest"
    }
}

$JdkZip = "$D\downloads\OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip"
$GradleZip = "$D\downloads\gradle-8.7-bin.zip"
$SdkZip = "$D\downloads\commandlinetools-win-13114758_latest.zip"
Get-AndroidZip 'https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jdk_x64_windows_hotspot_17.0.11_9.zip' $JdkZip 'fdd6664d4131370398fbc8bfbb7b46dbfec4a22a090a511fe5c379dae188c390'
Get-AndroidZip 'https://services.gradle.org/distributions/gradle-8.7-bin.zip' $GradleZip '544c35d6bd849ae8a5ed0bcea39ba677dc40f49df7d1835561582da2009b961d'
Get-AndroidZip 'https://dl.google.com/android/repository/commandlinetools-win-13114758_latest.zip' $SdkZip '98b565cb657b012dae6794cefc0f66ae1efb4690c699b78a614b4a6a3505b003'
Expand-Archive -LiteralPath $JdkZip -DestinationPath "$D\toolchains"
Expand-Archive -LiteralPath $GradleZip -DestinationPath "$D\toolchains"
$env:JAVA_HOME = "$D\toolchains\jdk-17.0.11+9"
$Gradle = "$D\toolchains\gradle-8.7"
$env:ANDROID_HOME = "$D\toolchains\android-sdk"
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$CLT = "$env:ANDROID_HOME\cmdline-tools"
Expand-Archive -LiteralPath $SdkZip -DestinationPath $CLT
Move-Item -LiteralPath "$CLT\cmdline-tools" -Destination "$CLT\19.0"
$SdkManager = "$CLT\19.0\bin\sdkmanager.bat"

# Ознакомиться с каждой лицензией и принять её в своей консоли, если согласен.
& $SdkManager "--sdk_root=$env:ANDROID_HOME" --licenses
if ($LASTEXITCODE) { throw 'Лицензии Android SDK не приняты' }
& $SdkManager "--sdk_root=$env:ANDROID_HOME" 'platforms;android-34' 'build-tools;34.0.0' 'ndk;26.1.10909125' 'cmake;3.22.1'
if ($LASTEXITCODE) { throw 'Android toolchain не установлен' }
```

Источники: [Temurin release/checksum](https://github.com/adoptium/temurin17-binaries/releases/tag/jdk-17.0.11%2B9), [Gradle SHA-256](https://services.gradle.org/distributions/gradle-8.7-bin.zip.sha256), [Android SDK Manager](https://developer.android.com/tools/sdkmanager); пакет cmdline-tools 19.0 дополнительно совпал по SHA-1 `54a582f3bf73e04253602f2d1c80bd5868aac115` с [индексом Google](https://dl.google.com/android/repository/repository2-3.xml). `sdkmanager` проверяет загружаемые SDK-пакеты; принятые лицензии остаются в `$env:ANDROID_HOME\licenses`. AGP 8.5.2 скачивается Gradle из объявленных репозиториев при холодной сборке; **не** передавать `--offline` на пустой среде. Platform Tools/ADB для сборки не нужны.

## 2. Две Python-среды и публичные входы

```powershell
& uv python install 3.11 3.12.13
if ($LASTEXITCODE) { throw 'Установка Python не удалась' }
& uv venv --python 3.11 --seed "$D\venv-gpu"
if ($LASTEXITCODE) { throw 'Не создана CUDA-среда' }
$Gpu = "$D\venv-gpu\Scripts\python.exe"
& $Gpu -m pip install --extra-index-url 'https://download.pytorch.org/whl/cu130' -r ml/requirements-gpu.txt
if ($LASTEXITCODE) { throw 'Не установлены строгие CUDA-зависимости' }
& uv venv --python 3.12.13 "$D\venv-convert"
if ($LASTEXITCODE) { throw 'Не создана среда конвертации' }
$Conv = "$D\venv-convert\Scripts\python.exe"
& uv pip install --python $Conv -r tools/android_from_source_requirements.txt
if ($LASTEXITCODE) { throw 'Не установлены зависимости конвертации' }
& uv pip check --python $Conv
if ($LASTEXITCODE) { throw 'Несовместимые пакеты Python-конвертера; CUDA-окружение не менять' }

& $Conv -B -m tools.android_from_source fetch --kind pose --out "$D\pose"
if ($LASTEXITCODE) { throw 'Не получен официальный RTMPose' }
& $Conv -B -m tools.android_from_source fetch --kind litert --out "$D\litert-upstream"
if ($LASTEXITCODE) { throw 'Не получен официальный LiteRT 2.4.0' }
$Pose = "$D\pose\end2end.onnx"
```

`fetch` проверяет SHA-256 [официального Body7 ZIP](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip), извлекает одну точную запись `end2end.onnx` и проверяет её SHA-256 отдельно. Для LiteRT берёт `litert_cc_sdk.zip` из [релиза GitHub v2.4.0](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.4.0), `com.google.ai.edge.litert:litert:2.4.0` и `com.google.ai.edge.litert:litert-gpu:2.4.0` — из Google Maven. Это AAR **Google**, не наш AAR. Официальные исходные SHA закреплены в `tools/android_from_source.py` и перепроверяются при раскладке runtime. `ai-edge-litert==2.1.2` в PyPI-среде нужен для чтения схемы FlatBuffer; в APK попадают нативные библиотеки **2.4.0**. CPU-зависимости onnx2tf предназначены для офлайн-конвертации, не для исполнения продукта.

## 3. Локальный E02/A (отдельный шаг, только на GPU Димаса)

Если согласованный полный локальный run уже существует, не переобучать: указать его checkpoint, `manifest.json` и **именно** граф `end2end.b1.onnx` из этого run. Иначе нужен собственный [официальный Mendeley V1](https://data.mendeley.com/datasets/49h9dcwx5v/1) и разрешение Димасу на обработку development. Сохранить файл поставщика как `$D\upper-limb-v1.zip`. Не распаковывать его целиком; штатный `--selection E02` отсеивает запрещённые записи до чтения. [Контракт данных и CUDA-конфигурация](TRAINING-ON-ANOTHER-PC.md). На время текущего поручения обучение не запускалось.

Команды **для владельца после разрешения**; только E02/A, без B/C, `--limit`, holdout, камеры или UI:

```powershell
& $Gpu -B -m tools.prepare_extra_development --archive "$D\upper-limb-v1.zip" --selection E02 --out "$D\e02-set"
if ($LASTEXITCODE) { throw 'E02 development intake не прошёл' }
$ASet = "$D\e02-set\E02"
$Commit = (git rev-parse HEAD).Trim() # ревизия именно этой локальной тренировки
if ($LASTEXITCODE) { throw 'Не найдена Git-ревизия' }
& $Gpu -B -m ml.train_gpu --variant A --manifest "$ASet\manifest.json" --manifest-sha256 '8c73b2a35584639f3a2ea057878d39cf105c52ffa4d77a4a41639c2606134931' --root $ASet --pose-model $Pose --private "$D\A-work" --out "$D\A-run"
if ($LASTEXITCODE) { throw 'Нет завершённого локального E02/A' }
if ((Get-Content -Raw -LiteralPath "$D\A-run\receipt.json" | ConvertFrom-Json).status -ne 'trained') {
    throw 'Не использовать неполный run'
}
```

Граф из обучения: `$D\A-work\graphs\end2end.b1.onnx`. Обучение вызывает `ml.gpu.prepare_static_graph`: фиксирует batch=1 и сворачивает статический хвост форм, не переписывая исходник. Не использовать случайный одноимённый ONNX и не подменять локальные pins эталонными. Если A уже обучен в другом **локальном** пути, в блоках дальше заменить соответствующие `$ASet`, `$APose`, `$ARun` и указать действительный training commit (`$Commit`), а не текущий HEAD задним числом.

## 4. Экспорт FP32 и сборка

```powershell
$ARun = "$D\A-run"
$APose = "$D\A-work\graphs\end2end.b1.onnx"
$Bundle = "$D\A-work\android-source-bundle.json"
if (Test-Path -LiteralPath $Bundle) { throw 'Выберите новый локальный bundle.json' }
& $Gpu -B -m tools.build_bundle --artifacts $ARun --name A-tcn.safetensors --pose-model $Pose --graph $APose --manifest "$ASet\manifest.json" --source-commit $Commit --out $Bundle
if ($LASTEXITCODE) { throw 'Не создан локальный bundle' }
$AOnnx = "$D\A-android-export\a-tcn.onnx"
& $Gpu -B -m tools.android_from_source_export_a --bundle $Bundle --artifact "$ARun\A-tcn.safetensors" --out $AOnnx
if ($LASTEXITCODE) { throw 'CUDA-экспорт классификатора не прошёл' }

$Converted = "$D\android-fp32-01"
& $Conv -B -m tools.android_from_source convert --bundle $Bundle --pose $APose --classifier $AOnnx --out $Converted
if ($LASTEXITCODE) { throw 'FP32-конвертация не прошла' }

$Job = "$D\android-local-01"
& $Conv -B -m tools.android_from_source_build --bundle $Bundle --conversion $Converted --litert "$D\litert-upstream" --gradle $Gradle --out $Job
if ($LASTEXITCODE) { throw 'Сборка или проверка APK не прошла' }
```

- **CUDA ONNX.** Перенесена приватная обёртка `export-a-onnx.py`: исходные `conv`/`head`, 64 фазы × 5 каналов, FP32 CUDA trace, ONNX opset 20, `dynamo=False`. Проверяются SHA/размер checkpoint, `scratch/A`, локальные порог и версия, CUDA, pins training manifest и pose graph, полнота training exposure. Sidecar `$AOnnx.json` привязывает экспорт к точному bundle.
- **FP32 LiteRT.** Взяты штатные переписывания `tools/export_android_onnx.py`, перенесённые в проект из приватных `rewrite-pose-android.py` / `rewrite-a-android.py`: `HardSigmoid` → clamp через `Mul/Add/Min/Max`; статические `Squeeze/Unsqueeze` → `Reshape`, логит A → `[1,1]`. Для обеих рабочих копий: `onnx2tf -i <копия> -o <каталог> -tb flatbuffer_direct -nuo -dsm -osd -coion -v warn`; для A ещё `-kat features`. onnx2tf может менять ONNX даже с `-nuo`; проверяются защищённые оригиналы и rewrite до/после. Берутся только `*_float32.tflite`, не FP16 и не диагностические результаты. Схема без вызова интерпретатора: FP32, `serving_default` с корректным сопоставлением имён, поза `input[1,256,192,3]` → `simcc_x[1,17,384]` / `simcc_y[1,17,512]` (278 ops), A `features[1,64,5]` → `logit[1,1]` (60 ops). Это **контракт движка**, не переносимые хэши весов.
- **Локальные входы.** Раскладка повторяет контракт `stage_android_engine.py`, но ничего не извлекает из нашего APK. Две ARM64 `.so` берутся из соответствующих AAR Google, C++-заголовки — из официального SDK ZIP. Проверяются ELF, определённый динамический экспорт `LiteRtCompiledModelIsNonCpuFullyAccelerated`, отсутствие зависимости от `libc++_shared.so`. Из обоих AAR Google извлекаются `LICENSE` и `THIRD_PARTY_NOTICE.txt`; добавляется атрибуция проекта. `engine.json` фиксирует локальные SHA/размеры, но **не** содержит старой квитанции, старого device proof или якобы наблюдённого backend.
- **Две сборки.** Во внешнюю копию попадают только нужные tracked Android/engine/общие web-исходники и штатные ассеты; не копируются `local.properties`, wrapper JAR, `build/` или ключи. Используются `:engine:assembleRelease` и `:app:assembleDebug`. Пакуются исходные байты runtime (не выполнять strip над этими двумя Maven `.so`). Только в копии меняются размер/SHA AAR и SHA трёх ассетов, а `Result.java` берёт версию и порог переобученного A. Нормализованный margin из `ml/api.py` и зона `uncertain < 0.10` сохраняются. APK-проверка использует исходный `android/verify_apk.py` с теми же воротами, но с заранее записанными локальными pins, плюс byte-for-byte контроль лицензий и сертификата свежего local debug key. Gate полной GPU-делегации в JNI **не меняется**: GPU only, opaque `gpu_options` `precision = 2`, полное размещение и ожидаемые счётчики обеих сетей до `Run`, без CPU/NPU fallback.

## Численная проверка перед переносом на телефон

После §4, в CUDA-окружении `$Gpu`, можно сравнить исходные модели с ONNX-графами, переписанными для Android. Указать свои пути и **новую** приватную папку; TFLite здесь только сверяется по SHA, не запускается. Это не замена приёмке LiteRT на CPH2399.

```powershell
& $Gpu -B -m tools.android_local_reference --bundle $Bundle --weights "$ARun\A-tcn.safetensors" --pose $APose --classifier $AOnnx --conversion $Converted --dev-manifest "$ASet\manifest.json" --out "$D\cuda-reference-01"
if ($LASTEXITCODE) { throw 'Не удалось подтвердить численное соответствие ONNX на CUDA' }
```

Для контрольного E02/A на RTX 5060 Ti профиль показал только CUDA: 182 события узлов исходного RTMPose, 190 переписанного RTMPose и 31 классификатора ONNX. На шести заранее выбранных кадрах из двух разрешённых validation-клипов было 18 наблюдений ключевых точек выше порога SimCC: маски совпали, максимальный сдвиг пика — 0 бинов. Для четырёх синтетических входов классификатора решения относительно порога совпали; максимальное изменение выходной вероятности — `7,21e-8` или меньше. Приватная квитанция контрольного прогона: `52a7c395404dfeb0d3a064aa29bfc8c2087fc2ab8ee70fdf62b70e09f3141213`. Реальные кадры и координаты не экспортируются; сохранённые эталонные тензоры синтетические. Не включать эталоны в APK или Git. Без `--dev-manifest` видимые суставы на реальном входе не проверяются.

## 5. Выходы и критерий готовности

| Путь на ПК Димаса | Что такое |
|---|---|
| `$AOnnx` и `$AOnnx.json` | Локальный CUDA-экспорт A и его квитанция. |
| `$Converted\rtmpose-fp32.tflite`, `a-tcn-fp32.tflite`, `conversion.json` | Оба графа и SHA/форма/происхождение; рабочие файлы отдельно в `work/`. |
| `$Job\litert-2.4.0`, `$Job\inputs` | Локальная раскладка SDK; модели, библиотеки и notices для Gradle. |
| `$Job\src\tools\android-engine\engine\build\outputs\aar\engine-release.aar` | ARM64 AAR, собранный здесь. |
| `$Job\pins.json`, `$Job\source-pins.json` | Фактические pins входов и AAR, Git HEAD и SHA содержимого снимка, включая локальные изменения рецепта. Не переносить на другую сборку. |
| `$Job\src\android\app\build\outputs\apk\debug\app-debug.apk` | `net.qolda.offline`, `0.1.0-t010`, arm64-v8a, только разрешение камеры; новый локальный сертификат. |
| `$Job\apk-receipt.json` | Критерий **упакован и статически подтверждён**: SHA APK, AAR, графов, библиотек, manifest и signer; `installed`, `gpu_verified`, `parity_verified`, `e2e_verified` остаются `false`. |
| `$Job\android-user\debug.keystore`, `$Job\gradle-user` | Созданы для этого job; не копировать наружу. |

Если этап упал, сохранить его лог и выбрать новый каталог для повтора (`android-fp32-02`, `android-local-02`); не принимать APK без `apk-receipt.json`, не править checks и не ставить `sourcecheck` вместо linked-сборки. Старые pins T-010/T-018 подтверждают только исторические файлы. Даже официальный делегат Maven `0dfc0b9d…` отличен от исторических проверенных байтов T-017 `23c7f011…` при одинаковом размере, поэтому одного успешного `assembleDebug` недостаточно для GPU/паритета. Историческая подпись не нужна и локальная версия не является совместимым обновлением APK другой подписи. Установку и физическую проверку на целевом устройстве проводить только по отдельному разрешению владельца.

**Контрольная сборка на станции Антона:** новый E02/A обучен на CUDA по чистому development-манифесту (77 train, 20 validation, 324 обновления). Его экспорт прошёл проверку FP32, затем собраны ARM64 AAR (4 224 414 байт, SHA-256 `d4262794b2f9f86140480d21d6a8146a1eed85e1fff12be2dc021a9e90b3ea07`) и подписанный debug APK (62 132 262 байта, SHA-256 `23728fa2cbf260067f3f595366388b57cac1c263850b8564401015cea71786f2`). Оба файла перечитаны и совпали с квитанцией. Из SDK ZIP LiteRT выпал `litert/build_common/build_config.h`; поэтому `fetch --kind litert` дополнительно получает официальный `build_config_gpu.h` из той же ревизии v2.4.0. Если папка `litert-upstream` была скачана по старой инструкции, повторить **только fetch LiteRT в новый каталог** и передать этот каталог через `--litert`. Обучение и конвертацию из-за заголовка повторять не нужно.

Это проверка рецепта, не приёмка ПК Димаса или телефона. На станции Димаса используются собственные веса, ключ и SHA. На CPH2399 этот контрольный APK не ставили: `installed=false`, `gpu_verified=false`, `parity_verified=false`, `e2e_verified=false`.
