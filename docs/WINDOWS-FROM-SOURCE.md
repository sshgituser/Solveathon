# Qolda Windows: сборка на ПК Димаса из исходников

Результат: Web E02/A из своего checkout на `127.0.0.1:8870`, приватная переносимая папка `engine-win64`, соседняя папка `windows` с `Qolda.exe` и, после отдельного допуска, файловый API E03. Всё собирается на целевом ПК. **Не брать наши ZIP, готовые папки, веса, `.npz` или референсы.** Допустимы исходники из Git, публичные upstream-загрузки и свои допущенные development-данные/результаты. Никаких `Test/`, ID 08, карантина, закрытого holdout, CPU/NPU-инференса, подставных ответов или правок UI. Данные, веса, графы, квитанции и выходы сборки — вне Git, на приватном локальном диске.

## 1. Входы и версия исходников

- Нативная Windows x64, полный CPython **3.11 x64** с python.org (не embeddable ZIP и не WSL), Git, драйвер NVIDIA с поддержкой CUDA 13 для своего GPU. Для EXE — [.NET 10 SDK x64](https://dotnet.microsoft.com/download/dotnet/10.0), доступ к NuGet при сборке, установленный [Microsoft Edge WebView2 Runtime x64](https://developer.microsoft.com/microsoft-edge/webview2/) при запуске. Сборка self-contained для .NET, но не включает браузерный Runtime и драйвер NVIDIA. Web не требует Node/npm: backend отдаёт локальные `frontend/` и `assets/`.
- Использовать чистый клон текущей ветки. Portable builder читает **Git-дерево `--ref`**, Web и сборщик EXE читают файлы из клона. Для обоих способов использовать одну ревизию.
- Для A нужны собственные `A-run/A-tcn.safetensors`, `A-run/receipt.json`, исходный допущенный E02 `manifest.json`, upstream `end2end.onnx` и точный `end2end.b1.onnx`, с которым обучали этот A (`A-work/graphs/end2end.b1.onnx` при штатном обучении). Для E03 нужны свои `E03/run/{checkpoint.safetensors,model.json,receipt.json}` и его допущенный development-манифест. Если ещё не обучены: [TRAINING-ON-ANOTHER-PC.md, §§ 1–4](TRAINING-ON-ANOTHER-PC.md). Указанный там `--selection extras` извлекает **допущенный development** E01 и E03; E01 для этого runbook не обучать и не подключать. Для повторной подготовки манифеста пользоваться штатным селектором, не распаковывать все видео.
- RTMPose: [точный Body7 ZIP от OpenMMLab](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip), SHA-256 `f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741`. Извлечь только `end2end.onnx`; его SHA-256 `5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c`. Статический batch-1 граф можно восстановить функцией `ml.gpu.prepare_static_graph(source, dest)` из этого файла, **в новый путь**; принять его только если SHA совпал с `record.pose_graph_sha256` своего A. Исторический SHA графа — ориентир, не замена этой сверки. Сохранить upstream-нотисы и [атрибуцию](../artifacts/seriq/ATTRIBUTION.md).
- SHA сверяются **внутри собственного прогона** (манифест, исходники, графы, checkpoint, упакованные и запущенные байты). SHA A/E03, `model_version`, веса и итоговые manifests между независимо обученными моделями **не обязаны совпадать**. Старый freeze и старый Windows/E03 packer не являются путём допуска нового обучения.

Далее PowerShell, из корня checkout. Пример раскладки; изменить входные пути на свои. `$FitCommit` — **фактический** полный commit обучения A из своей квитанции/записи запуска, не просто текущий `HEAD` (в этом checkout сборка приложения отдельно фиксируется как `$Code`).

```powershell
$ErrorActionPreference = 'Stop'
$Code = (git rev-parse --verify HEAD).Trim()
$D = Join-Path $HOME 'qolda-private'       # вне любого Git, не в облачной папке
$Ship = "$D/windows-local-01"            # новый корень; engine-win64 не должен существовать
$Engine = "$Ship/engine-win64"; $Windows = "$Ship/windows"
$BundleDir = "$D/bundle-win-01"          # новый каталог
$AManifest = "$D/e02-set/E02/manifest.json"
$ARun = "$D/A-run"; $Weights = "$ARun/A-tcn.safetensors"
$Pose = "$D/pose/end2end.onnx"; $TrainGraph = "$D/A-work/graphs/end2end.b1.onnx"
$FitCommit = '<40 lowercase hex из своего журнала обучения>'  # заменить
if ($FitCommit -cnotmatch '^[0-9a-f]{40}$') { throw 'Missing actual A training commit' }
function QHash($File) { (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant() }
function Require-Exit($Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed ($LASTEXITCODE)" } }

New-Item -ItemType Directory -Path $D -Force | Out-Null
# Если уже есть это чистое GPU-окружение, не пересоздавать его; проверить пакеты.
py -3.11 -m venv "$D/venv"
Require-Exit 'venv'
$Py = "$D/venv/Scripts/python.exe"
& $Py -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r ml/requirements-gpu.txt 'setuptools==81.0.0'
Require-Exit 'pinned CUDA stack'
& $Py -m pip check
Require-Exit 'wheel consistency'
& $Py -B -E -c "from ml.gpu import enable_cuda_dlls,cuda_stack; enable_cuda_dlls(); print(cuda_stack())"
Require-Exit 'CUDA preflight'
```

`setuptools` нужен и как копируемый пакет: `build_windows_engine.py` ожидает записи файлов всех `WHEELS` **в этом venv**. Не ставить параллельно CPU `onnxruntime` или другой OpenCV в тот же venv. Рабочий стек: torch `2.11.0+cu130`, ORT GPU `1.27.0`, onnx `1.20.1`, OpenCV contrib `5.1.0.95`; исторический GPU/драйвер — RTX 5060 Ti / `617.42`. При нехватке CUDA DLL, отсутствующем CUDA EP или неподдерживаемом GPU — блокер, не замена провайдера. `cuda_stack()` — preflight, а не доказательство placement.

### FFprobe и лицензии

По [странице FFmpeg](https://ffmpeg.org/download.html) взять конкретный **статический Windows x64** выпуск (например [Gyan release essentials](https://www.gyan.dev/ffmpeg/builds/), у Gyan статические сборки GPLv3). Зафиксировать URL, версию, upstream checksum при наличии, SHA распакованного EXE и архив лицензий от **того же** выпуска. Извлечь в приватный `$D/toolchain`; ниже задаются реальные пути. Историческая рабочая версия `ffprobe 8.1.2` — не pin на чужие байты. `ffprobe.exe` запускается отдельным subprocess с `file`-only whitelist, MOV/Matroska, JSON-выводом о потоках и пакетах; декодирует OpenCV `CAP_FFMPEG`. `ffmpeg.exe` не нужен в portable runtime, только если локально генерируется синтетический контроль. Shared-сборку не брать: builder переносит **только** `ffprobe.exe`, не её DLL.

```powershell
$FFprobe = "$D/toolchain/ffmpeg-release/bin/ffprobe.exe"  # путь к настоящему EXE
$FFLicense = "$D/toolchain/ffmpeg-release/LICENSE"      # настоящий текст лицензии выбранной сборки
if (!(Test-Path -LiteralPath $FFLicense -PathType Leaf)) { throw 'Missing upstream build license' }
& $FFprobe -version
Require-Exit 'ffprobe version'
& $FFprobe -buildconf
Require-Exit 'ffprobe config'
QHash $FFprobe
QHash $FFLicense
```

`-buildconf` сверить с поставщиком, не использовать `--enable-nonfree`. `--ffmpeg-license` копируется как `engine-win64/licenses/FFmpeg-LICENSE.txt` без проверки содержания; не подставлять общий LGPL, страницу загрузки или пустышку. Сохранить полные соответствующие notices, информацию о конфигурации и исходниках; [условия FFmpeg](https://ffmpeg.org/legal.html) зависят от конфигурации, одного текста лицензии для перераспространения недостаточно. Эта сборка остаётся у Димаса; никакого распространения. Штатные пакеты также сохраняют Python `LICENSE.txt`, лицензии/нотисы из wheel manifests, атрибуцию Qolda и `.NET`/WebView2 notices в `windows/licenses/`.

## 2. Связать свой A, граф и Web

Сверить локальный `receipt.json`: `status=trained`, `variant=A`, артефакт `A-tcn.safetensors`, `record.kind=record.variant=A`, `record.runtime_device=cuda:0`, законченный dev-run, не отладочный `--limit`. Не исправлять receipt под найденный файл. Сборщик сверяет SHA манифеста и обоих pose-графов с метаданными внутри checkpoint, а загрузчик повторяет сверку при регистрации. Это предотвращает смешение разных прогонов, но не доказывает качество модели или происхождение произвольно изменённых байтов.

```powershell
$r = Get-Content -Raw -LiteralPath "$ARun/receipt.json" | ConvertFrom-Json
if ($r.status -ne 'trained' -or $r.variant -cne 'A' -or $r.artifact -cne 'A-tcn.safetensors' -or
    $r.record.kind -cne 'A' -or $r.record.variant -cne 'A' -or $r.record.runtime_device -cne 'cuda:0' -or
    $r.sha256 -cne (QHash $Weights) -or $r.record.manifest_sha256 -cne (QHash $AManifest) -or
    $r.record.pose_model_sha256 -cne (QHash $Pose) -or $r.record.pose_graph_sha256 -cne (QHash $TrainGraph) -or
    (QHash $Pose) -cne '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c') { throw 'A provenance mismatch' }
$metadataJson = & $Py -B -E -c "import json,sys; from pathlib import Path; from tools.build_bundle import read_metadata; print(json.dumps(read_metadata(Path(sys.argv[1]))))" $Weights
Require-Exit 'A metadata'
$meta = $metadataJson | ConvertFrom-Json
foreach ($key in 'kind','variant','runtime_device','manifest_sha256','pose_model_sha256','pose_graph_sha256','pose_policy','version','threshold') {
    if ($null -eq $meta.$key -or $meta.$key -cne $r.record.$key) { throw "A metadata mismatch: $key" }
}
New-Item -ItemType Directory -Path $BundleDir | Out-Null
$Bundle = "$BundleDir/bundle.json"; $Graph = "$BundleDir/end2end.b1.onnx"
Copy-Item -LiteralPath $TrainGraph -Destination $Graph
& $Py -B -E -m tools.build_bundle --artifacts $ARun --name A-tcn.safetensors --slot scratch --pose-model $Pose --graph $Graph --manifest $AManifest --source-commit $FitCommit --out $Bundle
Require-Exit 'A bundle'
```

`bundle.json` и `end2end.b1.onnx` должны быть соседями; не давать загрузчику неявно создавать граф при первом запуске. Для **рабочего Web** из корня checkout, с отдельным приватным scratch, только если 8870 свободен и запуск своего процесса разрешён:

```powershell
function Require-FreePort([int]$Port) {
    $listeners = [Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    if ($listeners | Where-Object { $_.Port -eq $Port }) { throw "Port $Port occupied; leave the old process alone" }
}
$WebScratch = "$D/scratch-web-01"
New-Item -ItemType Directory -Path $WebScratch | Out-Null
Require-FreePort 8870
& $Py -B -E -m backend.api --port 8870 --bundle $Bundle --artifacts $ARun --pose-model $Pose --private-scratch $WebScratch --ffprobe $FFprobe
```

Это foreground; ждать `SerIQ http://127.0.0.1:8870/; models: scratch`. `models: none` — здоровый HTTP без модели, не успех. `infra/serve.ps1` без `-PreviewOnly` только сверяет уже работающий Web на `8870`; `-PreviewOnly` не даёт inference. Не запускать ещё один сервер вместо занятого, не менять старый процесс или его модели. Для рабочего Web не нужен отдельный статический сервер, `file://` или `--api-only`.

## 3. Собрать portable engine и EXE

Из того же checkout, в терминале с переменными из §§ 1–2 (не повторять создание уже готового bundle). Это только сборка, без запуска приложений:

```powershell
& $Py -B -E -m tools.build_windows_engine --ref $Code --bundle $Bundle --pose $Pose --weights $Weights --ffprobe $FFprobe --ffmpeg-license $FFLicense --out $Engine
Require-Exit 'portable engine'
powershell.exe -NoProfile -File .\desktop\Build-Windows.ps1 -OutputDirectory $Windows
Require-Exit 'Windows EXE'
```

Engine builder требует **несуществующий** `--out`; Desktop builder допускает новый или пустой `-OutputDirectory` вне checkout. Последний делает `dotnet restore --locked-mode`, затем `publish` Release, `win-x64`, self-contained; промежуточные `desktop/.nuget`, `obj`, `bin` — не поставка. При ошибке lock/restore не редактировать lock/проект, сохранить диагностическое сообщение. Локальный `windows/qolda-windows-win-x64.zip` сборщик создаёт сам, он не нужен для запуска, не передавать и не подменять чужим.

```text
<приватный Ship>/
  engine-win64/
    start-engine.cmd                    # 8765; базовый E02/A, без E03
    manifest.json
    python/python.exe                   # python311._pth изолирует sys.path
    app/{backend,ml,frontend,assets}/
    runtime/{bundle.json,end2end.onnx,end2end.b1.onnx,A-tcn.safetensors,ffprobe.exe}
    licenses/FFmpeg-LICENSE.txt
  windows/
    Qolda.exe
    WebView2Loader.dll
    runtimes/win-x64/native/WebView2Loader.dll
    SHA256SUMS.txt
    licenses/
```

Сохранять относительные пути и достаточно места для staging, wheel-кэша и результата. Использовать локальный диск с правом записи: ORT временно пишет профиль **рядом с `end2end.b1.onnx`** при запуске, а загрузки идут в отдельный приватный scratch. Сначала сверить `engine-win64/manifest.json`: `code_commit=$Code`, `model_source_commit=$FitCommit`, размер и SHA **всех** `files` соответствуют диску; отдельно записать SHA манифеста. В `windows/SHA256SUMS.txt` сверить все перечисленные файлы; `Qolda.exe` должен быть PE AMD64, сохранить SHA и Authenticode-статус (штатный build не подписывает). Для переносимости проверить копию своего результата из другого локального каталога с пробелом, сохранив layout: у её `python/python.exe` `sys.flags.isolated=1`, все `sys.path` под **этой** папкой `engine-win64`, `runtime/ffprobe.exe -version` работает без стороннего FFmpeg в `PATH`, и именно из этой копии проходят HTTP-критерии ниже. Проверку «без установленного Python / offline / на другом ПК» не засчитывать без отдельного такого прогона.

Разрешённый владельцем headless-запуск portable E02 — `engine-win64/start-engine.cmd` на свободном `8765` (scratch `%LOCALAPPDATA%/Qolda/scratch`) либо явная команда ниже с отдельным scratch. Worktree Web остаётся с адресом `8870`; portable `8765` — отдельная проверка, не второй рабочий Web. Не запускать параллельные анализы через разные процессы: GPU guard общий только внутри одного backend. `Qolda.exe` без аргументов пытается подключиться к совместимому `8765` и запускает соседний engine только при явном отказе соединения. Ручной запуск окна **не входит** в headless-приёмку; когда отдельно разрешат UI, к уже подтверждённому backend подключаться `windows\Qolda.exe --attach-port 8765` или `--attach-port 8870`. Attach-only не запускает движок и не комбинируется с `--config`; обычный запуск EXE не передаёт `--extra-model-root`.

## Проверка файлов самой Windows-оболочки

После `Build-Windows.ps1` выполнить `python -B -m tools.verify_windows_shell --windows "$Windows"`. Команда не запускает EXE. Она читает все поставленные файлы, сверяет их с `SHA256SUMS.txt` и байтами локального клиентского ZIP, проверяет формат AMD64 PE32+ у EXE и обеих копий WebView2Loader, отклоняет неожиданные файлы. Это проверка состава, не запуск окна или анализ видео. Не добавлять квитанцию в проверяемый каталог.

На контрольной станции из чистого снимка `desktop/` ревизии `d8b10722302fc4b06d1219ee6641d28c1cbf0bfc` прошли `dotnet restore --locked-mode` и self-contained `publish`. Фактический `Qolda.exe`: 141 063 622 байта, SHA-256 `5ee92b14641f6fa9b5d465f9797dd57b3350cf4477d2cabbfeb377b51d19c0e0`; прочитаны 11 записей ZIP, обе копии загрузчика совпали. Эти хэши не подменяют фактические значения ПК Димаса.

## Ограниченный прогон нового комплекта без окна

Вместо ручного запуска второго постоянно работающего сервера для E02 можно использовать команду ниже. `$Blank` — собственный заранее подготовленный синтетический неподвижный MP4 без человека; `$Scratch` — существующий приватный каталог вне Git и поставки. Запускать после завершения обучения, когда GPU свободна. Команда использует Python из `$Engine`, случайный порт на `127.0.0.1` и завершается сама. Она проверяет точные байты манифеста и передаваемых клипов. Подбирания удачного видео нет: берутся первые validation-клипы каждого класса по порядку канонического манифеста.

```powershell
$BlankPin = (Get-FileHash -Algorithm SHA256 -LiteralPath $Blank).Hash.ToLowerInvariant()
python -B -m tools.receive_windows_e02 --engine $Engine --dev-manifest $AManifest --blank $Blank --blank-sha256 $BlankPin --scratch $Scratch
if ($LASTEXITCODE -ne 0) { throw 'Приёмка не прошла; сохранить фактический результат и не подменять входы' }
```

Контрольная сборка `d8b10722302fc4b06d1219ee6641d28c1cbf0bfc`, A из обучения `0682690bb265c89ee41109c01f02bec519afa805`: независимо перечитаны **25 610** файлов движка (3 787 362 726 байт без `manifest.json`), SHA-256 манифеста `bd463cb9157d9c53e43c630e26990f95dd5f65e0924d37828e6cb7085415896b`. Изолированный комплект выдал через HTTP `incomplete:1`, `complete:1`, на пустой сцене `no_pose` и `total:0`. Во всех трёх ответах модель `seriq-e02-forearm-1.0.0+ce3f55718b2d`, HTTP 200; обслужен `index.html` из комплекта, временный каталог пуст. SHA квитанции контрольного прогона `381654834d0e1d4cb07b9d4674cd517840fcf18d2acf1f7f3b2e0ab058ff0bc2`. Это HTTP-путь на сборочном ПК, **не** запуск окна EXE, физическая камера, офлайн на чистом ноутбуке или выполнение у Димаса.

## 4. Опциональный E03: только новый локальный owner approval

Не переносить старый допуск и не считать `promotion_allowed` автоматическим разрешением. Для собственного E03 **не нужен архив**: штатный реестр `backend.extra_exercises` принимает новый одобренный SHA локальных байтов, а `ml.extra_e03._open` независимо проверяет checkpoint, `model.json` и SHA **байтов загруженного** `ml/extra_e03.py`. Точный LF Git blob важен: не пересохранять файл/JSON ради совпадения с чужим SHA. Portable `app/ml/extra_e03.py` должен быть побайтово тем же исходником, на котором обучен кандидат; иначе согласовать исходный commit или переобучить, не переписывать метаданные.

**Проверить E03 в упакованном Python до подключения к серверу:** нужны свой приватный `E03/run` с исходной квитанцией обучения, канонический E03-манифест, уже одобренная пустая сцена с её SHA и существующий отдельный приватный scratch. Запуск из корня checkout; не требует HTTP, свободного порта или `promotion.json` и не меняет одобрение.

```powershell
& "$Engine/python/python.exe" -B tools/receive_windows_e03.py --manifest "$Extras/E03/manifest.json" --run "$Extras/E03/run" --blank $Blank --blank-sha256 $BlankPin --scratch $Scratch
if ($LASTEXITCODE) { throw 'Прогон E03 не завершён или статичная сцена не дала ожидаемый отказ' }
```

`$Extras` — приватный корень набора дополнительных упражнений. Команда пробует все 20 validation-клипов, а не только успешные; хэширует байты до подачи в FFprobe и CUDA-адаптер. В матрице строки — разметка `incomplete/complete`, столбцы — решения `incomplete/complete/uncertain`. Проверить саму матрицу: нулевой код означает успешное выполнение и ожидаемый отказ на пустой сцене, **не разрешение публиковать модель**. На контрольной сборке `d8b1072`, движок `bd463cb9…`, версии E03 `rgb-motion-cnn-v1+95abbe0a8c77` получены `[[10,0,0],[0,10,0]]`, пустая сцена → `uncertain/low_visibility`, временный каталог удалён. SHA контрольной квитанции `a0c761f2ae27d8edc38c7ee32f2e6aa566ddba08ee48eb9e95ac00336ada17ab`. Это те же клипы **одного** development-участника, проверка переноса исполнения, не новая оценка качества и не экран Windows.

**Критерии для `data-owner` до допуска новых байтов:**

1. Подтвердить provenance, права и SHA допущенного manifest (`125abc289062c15a26a490bbaf7306d73767fcaddbea6c62fff81282a0d2b7ea` для канонического E03); участники 02/03/04/06 только train, 05 только validation, без пересечений/дублей между splits. Для канонического прогона сверить 81 train и 20 validation (10+10), все 20 исходных validation MP4 в replay. Приватные `receipt.json`, `model.json`, SHA весов и исходника должны согласоваться. Требовать `status=trained`, `promotion_allowed=true` и все семь `promotion_checks=true`; код выхода сам по себе не достаточен.
2. Не ослаблять training-gates: сырая BA строго выше динамического baseline (код: `> baseline + 1e-9`), macro-F1 не ниже; unconditional balanced recall с отказами строго выше селективного baseline (тот же допуск `1e-9`) и `0.50`; покрытие ≥`0.70` общее и ≥`0.50` в каждом классе; raw BA ≥ static control + `0.05`; все четыре train-участника прошли `optimizer.step`; reload — 0 расхождений label/reason/version и оба отклонения ≤`1e-6`. Требовать CUDA-аудит reload: `device=cuda:0`, `graph_operator_calls>0`, `non_cuda_tensor_operands_or_outputs=0`, положительное CUDA-время. Порог serving оставить `0.70`; сохранить все отказы модели, включая слабое/глобальное движение и расхождение зеркальных просмотров. Это development, не измерение угла кисти и не клиническая проверка.
3. Для runtime-кандидата зафиксировать **до запуска** первый разрешённый validation MP4 каждого класса в порядке manifest и синтетическую статичную сцену без руки. До регистрации можно проверить их напрямую через строгий `ml.extra_e03_service.open_engine` и `backend.runtime.probe_frames` из собранного Python, без `promotion.json` и без выставления готовности. Требовать на реальных байтах ожидаемые `incomplete` и `complete` с `reason=null`, на пустой сцене `uncertain/low_visibility`; не заменять провалившийся клип удачным. Зафиксировать доказательство fail-closed для отсутствующего/неверного SHA или недоступной CUDA в отдельном разрешённом прогоне/копии, без воздействия на рабочий процесс.
4. Владелец отдельно разрешает каждый file-профиль. Для проверки сначала ограниченный Web-only допуск и HTTP-реплей на целевом portable interpreter, как в [прошлой приёмке](../reports/platforms/shared-wip-e03-file.md). Для `windows=true` нужен явный отдельный допуск владельца по результатам проверки именно этой локальной переносимой сборки с тем же адаптером и FFprobe, а не вывод из `web=true`. После финального решения повторить те же HTTP-запросы из **конечной** локальной раскладки, проверить каталог, версии, что E02 A остался готов, B и E01 не появились, scratch пуст. Если какой-либо этап не выполнен, соответствующий флаг остаётся `false`; при неподтверждённой модели E03 вообще не включать. Это проверка file API, не окна EXE, камеры или Android.

Стадировать копии только своих `checkpoint.safetensors` и `model.json` в отдельный приватный корень `$Ship/extras-approved/E03/run/` вне любого Git, без ссылок/junctions и посторонних писателей (ACL); не тащить media/manifest/receipt обучения в portable app. Файл `promotion.json` там создаёт/передаёт **владелец данных**, не скрипт сборки. Ровно поля из [контракта](contracts/extra-exercises.md#приватная-поставка-и-promotion): `schema=qolda-extra-promotion-v1`, `exercise=E03`, `owner_id=data-owner`, `decision=approved`, `checkpoint=checkpoint.safetensors`, `sha256=<фактический локальный SHA, 64 lowercase hex>`, `low_confidence_below=0.70`, `file_platforms` с ровно двумя boolean `web`, `windows`. UTF-8 без BOM, ≤16 KiB, без дубликатов полей. Квитанция не подпись: ограничить права записи, зафиксировать SHA `model.json` отдельно, не менять используемый каталог/receipt на ходу. Для другого уровня допуска — новый приватный snapshot; не перезапускать существующие процессы в этой работе.

## 5. Headless-приёмка у Димаса

После разрешения оператора: использовать подтверждённую чистую portable раскладку (или её собственную перемещённую копию), **новый приватный scratch с ограниченным ACL**; `--extra-model-root` опустить для базового E02, для одобренного E03 передать **корень**, не `E03/run`. На занятом порту остановиться, старый сервис не проверять как новую сборку. Для отдельного headless HTTP-прогона без EXE можно выбрать `--port 0` и взять напечатанный временный порт; рабочий Web и EXE используют только указанные выше порты.

```powershell
$Port = 8765                            # или 8870 для подтверждённого attach; нужен свободный
$Scratch = "$D/scratch-portable-01"       # новый, вне app и любого Git
$ExtraArgs = @()                          # E02 only; после явного допуска заменить следующей строкой
# $ExtraArgs = @('--extra-model-root', "$Ship/extras-approved")
New-Item -ItemType Directory -Path $Scratch | Out-Null
Require-FreePort $Port
& "$Engine/python/python.exe" -B -E -m backend.api --port $Port --bundle "$Engine/runtime/bundle.json" --artifacts "$Engine/runtime" --pose-model "$Engine/runtime/end2end.onnx" --private-scratch $Scratch --ffprobe "$Engine/runtime/ffprobe.exe" @ExtraArgs
```

Из второго терминала, `$Base = 'http://127.0.0.1:<фактический порт>'`; для HTTP не нужен UI. Примеры для разрешённых MP4 c одним обрезанным циклом (не multipart):

```powershell
curl.exe --noproxy '*' -fsS "$Base/api/v1/health"
curl.exe --noproxy '*' -fsS "$Base/api/v1/exercises"
curl.exe --noproxy '*' -sS -w '\nHTTP %{http_code}\n' -H 'Content-Type: video/mp4' --data-binary "@$AClip" "$Base/api/v1/analyze?model=scratch&mode=single"
# Только после допуска, повторить с обоими классами и статичной сценой:
curl.exe --noproxy '*' -sS -w '\nHTTP %{http_code}\n' -H 'Content-Type: video/mp4' --data-binary "@$E03Clip" "$Base/api/v1/exercises/E03/analyze"
```

| Проверка | Принять, если |
|---|---|
| Web, из checkout и затем из portable | `/`, `/src/main.js`, `/styles/atlas.css` → 200, сохранённые байты совпали по SHA с `frontend/index.html`, `frontend/src/main.js`, `frontend/styles/atlas.css` соответствующего источника. Никакой второй копии фронта для редактирования. |
| E02 GPU-only | При строгой загрузке `ml.gpu.open_cuda_pose`: профиль с ненулевым числом CUDA-узлов и `non_cuda=[]`; у A все параметры и буферы `cuda:0`, finite warm-up. Для отдельной квитанции профиля позы: `tools.verify_gpu_engine --pose-model … --graph … --workdir … --out …` через `$Py -B -E -m`; **без** `--weights/--clip`, чтобы не требовать старого архивного референса. При отсутствии CUDA или неподходящем SHA модель не готова, валидный запрос → `503 model_unavailable`, не CPU. |
| E02 whole clip | `/api/v1/health`: `seriq.v1`, `scratch=true`, `transfer=false`. Заранее выбранный разрешённый непустой development-клип → 200, `model=scratch`, `model_version` оканчивается `+<12 hex SHA своего A>`, ровно одна честная попытка `complete`/`incomplete`; записать соответствие метке, не перебирать входы. Синтетический статичный клип без человека → 200, `counts.total=0`, `attempts=[]`, `warnings=["no_pose"]`. Битый контейнер → 422 `invalid_video`. |
| E03 (одобренный отдельно) | Каталог отражает **ровно** утверждённые Web/Windows file-флаги; E03 live/Android и E01 остаются `false`. Три ответа по §4 → 200 `qolda.extra-exercise.v1`, `exercise=E03`, `model_version=rgb-motion-cnn-v1+<12 hex SHA своего E03>`, корректные `result`/`reason`. Без допуска E03 file `false`, корректный запрос → `503 model_unavailable`. При неожиданном сбое адаптера или CUDA — `503 model_unavailable` и отзыв readiness E03; не подменять это `uncertain`, не вызывать сбой в действующем процессе. Не трактовать `E02.windows.file=false` в каталоге как поломку EXE: каталог намеренно не подтверждает платформу для E02. |
| Scratch и следы | После запросов приватный scratch пуст; приватная квитанция фиксирует Git SHA, версии Python/wheels/драйвера/GPU/FFprobe/.NET, SHA входов и конечных manifests/EXE, статусы HTTP и причины отказа. Никаких путей с личными данными, сырых видео, поз, весов и временных файлов в Git. |

Для пустого контроля можно локально закодировать сплошной цвет своим upstream `ffmpeg.exe`, например `-f lavfi -i color=teal:s=160x120:r=24:d=1 -an -c:v mpeg4 -pix_fmt yuv420p <новый приватный blank.mp4>`; не искать такой файл среди holdout. На всё время прогона закрепить точные байты и не модифицировать работающий runtime. HTTP-приёмка и целостность EXE не доказывают реальный запуск окна, физическую камеру, offline-приёмку, качество на независимых людях или production-готовность.
