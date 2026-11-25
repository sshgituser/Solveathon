# Qolda на другом ПК: от чистого клона до обучения

**Посмотреть интерфейс без анализа:** из корня запустить `./infra/serve.ps1 -PreviewOnly` и открыть `http://127.0.0.1:8870/`. Для анализа требуется отдельно собранный локальный движок.

## 1. Какие данные нужны

Первоисточник: B. Nair, S. Nandana, M. Thenmozhi Dharshini, V. Viswanathan, [An upper limb stroke rehabilitation exercise video dataset, Mendeley Data V1](https://data.mendeley.com/datasets/49h9dcwx5v/1), DOI `10.17632/49h9dcwx5v.1`. [Лицензия данных CC BY 4.0](https://api.datacite.org/dois/10.17632/49h9dcwx5v.1); [статья и таблицы 2/3](https://doi.org/10.1016/j.dib.2026.112819) распространяются отдельно. 491 видео, 10 добровольцев, 4 упражнения; один цикл на файл, RGB MP4; ручная разметка `complete`/`incomplete`. Не диагноз, не угол сустава и не данные о боли. Сохранять атрибуцию, версию, изменения и ссылку на лицензию. Политика проекта строже лицензии: ни исходные клипы, ни приватные модели/кэши/поклиповые записи в Git не отправляем.

| Режим | Опубликованный полный набор: complete / incomplete | Разбиение для **нашего** обучения: train → validation | Что именно учится |
|---|---:|---|---|
| E02, основной, разгибание локтя | 60 / 67 | ID 01–04: 77 (30/47); ID 05: 20 (10/10). Ещё 10 конфликтных E02 не используем | A: временная TCN с нуля, наблюдаемые правые локоть и запястье. 71/77 пригодны для fit; validation 20/20. B и C — отдельные модели на **других** источниках. |
| E01, дополнительный, поднятие предмета | 62 / 61 | ID 01–04: 82 (42/40); ID 05: 21 (10/11) | RTMPose + отдельная TCN по движению рук; не детектор предмета. 74/82 пригодны для fit. |
| E03, дополнительный, разгибание запястья | 60 / 61 | ID 02,03,04,06: 81 (40/41); ID 05: 20 (10/10) | Отдельная RGB-сеть по динамике кадров, без внешних стартовых весов, не измеряет угол запястья. |
| E04, раскрытие кисти | 60 / 60 | Не отбираем | Не входит в текущую поставку. |

Скачать исходный ZIP версии 1 со страницы авторов, поместить **вне Git**. Исторический ZIP на станции: 1 248 733 056 байт; у переупакованной копии могут отличаться имя, размер и хэш. Не распаковывать архив целиком и не перебирать хэши видео: `Test/`, ID 08 и карантин должны оставаться закрытыми. Селектор ниже читает только согласованные участники и каталог `Train`, сверяет код упражнения и совпадение метки в папке и имени **до чтения содержимого**, проверяет дубли SHA и ожидаемые числа. В архиве ожидается `An upper limb stroke rehabilitation exercise video/Exercise/<упражнение>/{Complete,Incomplete}/Train/...`. При несовместимом V1 проверить источник и структуру, а не убирать проверки. Собственные данные B/C не скачиваются из Mendeley.

## 2. Среда и внешние зависимости

Команды ниже — PowerShell, **из корня клона**. Требуются Python 3.11 x64, Git и совместимый NVIDIA-драйвер; рабочая конфигурация — RTX 5060 Ti, драйвер 617.42, PyTorch 2.11.0+cu130, cuDNN 9.19 и ONNX Runtime GPU 1.27.0. Это проверенная конфигурация, а не гарантия для любой GPU. CPU может декодировать видео и считать геометрию; нейросети не переключаются на CPU/NPU. Для Web/Windows с видео нужен настоящий `ffprobe.exe`: [FFmpeg — официальная страница и ссылки на Windows-сборки](https://ffmpeg.org/download.html). Windows-оболочке нужен [WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/), для её сборки — .NET 10 SDK.

```powershell
$D = Join-Path $HOME 'qolda-private'     # не в клоне, не в синхронизируемой облачной папке
New-Item -ItemType Directory -Force $D | Out-Null
py -3.11 -m venv "$D/venv"
$Py = "$D/venv/Scripts/python.exe"
& $Py -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r ml/requirements-gpu.txt
if ($LASTEXITCODE) { throw 'GPU dependencies failed' }
& $Py -B -c "from ml.gpu import enable_cuda_dlls,cuda_stack; enable_cuda_dlls(); print(cuda_stack())"
if ($LASTEXITCODE) { throw 'Strict CUDA unavailable' }
$Commit = (git rev-parse HEAD).Trim()
$Zip = "$D/upper-limb-v1.zip"        # скачанный ZIP Mendeley V1, при необходимости переименовать
```

Чтобы получить исходный датасет **на ПК Димаса**, кнопка `Download All` на странице V1 ведёт на `https://data.mendeley.com/public-api/zip/49h9dcwx5v/download/1`. Не использовать чужой переупакованный архив. На стороне источника проверен GET с диапазоном: `206 bytes 0-3/1248733056`, сигнатура ZIP. HEAD может дать 403 при рабочем GET. PowerShell:

```powershell
if (!(Test-Path -LiteralPath $Zip)) {
  if (Test-Path "$Zip.part") { throw 'Проверь предыдущую незавершённую загрузку, не перезаписывай её' }
  curl.exe --fail --location --retry 2 --user-agent 'Mozilla/5.0' --referer 'https://data.mendeley.com/datasets/49h9dcwx5v/1' --output "$Zip.part" 'https://data.mendeley.com/public-api/zip/49h9dcwx5v/download/1'
  if ($LASTEXITCODE -ne 0 -or (Get-Item "$Zip.part").Length -ne 1248733056) { throw 'Неполная загрузка V1; используй Download All у первоисточника' }
  Move-Item -LiteralPath "$Zip.part" -Destination $Zip
}
```

Не читать и не хэшировать содержимое закрытой части: раздел 3 сверяет допущенные клипы и SHA **итоговых development-манифестов**. Целиком исходный архив не распаковывать.

**RTMPose (E02 и E01):** [исходный проект](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) и [точный ZIP официального экспорта Body7](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip). Проверенный ZIP SHA-256 `f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741`. Из **модельного** ZIP взять `end2end.onnx`, положить в `$D/pose/end2end.onnx`. Не путать с ZIP видео. Отдельный `end2end.b1.onnx` создаётся из этого графа штатным преобразованием, не загружать случайный одноимённый файл.

```powershell
$Pose = "$D/pose/end2end.onnx"; $Graph = "$D/pose/end2end.b1.onnx"
if (!(Test-Path "$D/pose")) { & $Py -B -m tools.prepare_public_pose --out "$D/pose"; if ($LASTEXITCODE) { throw 'Не удалось получить официальный RTMPose или собрать pinned граф' } }
function Assert-QoldaSHA($Path,$Pin) { if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Pin) { throw 'Wrong SHA-256' } }
Assert-QoldaSHA $Pose '5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c'
if (!(Test-Path $Graph)) { & $Py -B -c "import sys; from ml.gpu import prepare_static_graph; print(prepare_static_graph(sys.argv[1],sys.argv[2]))" $Pose $Graph }
Assert-QoldaSHA $Graph '2e0a9c0c5287bfa8897440975c32e98e26be90e88ad9ddd9e72fba8644b68b78'
```

**SLiM (только B/C):** [исходники на закреплённой ревизии](https://github.com/KAIST-VICLab/SLiM/tree/3aaab213883a1e8b06d08fcfa854da50f23889c4), [закреплённое семейство NTU60 XSub](https://huggingface.co/JeonghyeokDo/SLiM/tree/b934cfdbce135479b1df5abe6d1495fcee05e4f5/ntu60_xsub), [непосредственно model.safetensors](https://huggingface.co/JeonghyeokDo/SLiM/resolve/b934cfdbce135479b1df5abe6d1495fcee05e4f5/ntu60_xsub/model.safetensors). 25 484 416 байт; SHA-256 `c25c70db9ca5c1ca3715b80acbd21851fa04f4b3178e4c82f96d45492774b5bf`. Это база для дообучения, **не** готовая модель B/C. Лицензии и изменения: [ATTRIBUTION](../artifacts/seriq/ATTRIBUTION.md).

## 3. Безопасная подготовка development-выборок

```powershell
& $Py -B -m tools.prepare_extra_development --archive $Zip --selection E02 --out "$D/e02-set"
if ($LASTEXITCODE) { throw 'E02 intake failed' }
& $Py -B -m tools.prepare_extra_development --archive $Zip --selection extras --out "$D/extra-set"
if ($LASTEXITCODE) { throw 'E01/E03 intake failed' }
$ASet = "$D/e02-set/E02"; $Extras = "$D/extra-set"
$E01Pin = '287285d9386f26f7d147a9821fdd46b38f32673f801ea7c7df336ddefe7015eb'
$E03Pin = '125abc289062c15a26a490bbaf7306d73767fcaddbea6c62fff81282a0d2b7ea'
$APin = '8c73b2a35584639f3a2ea057878d39cf105c52ffa4d77a4a41639c2606134931'
Assert-QoldaSHA "$ASet/manifest.json" $APin
Assert-QoldaSHA "$Extras/E01/manifest.json" $E01Pin
Assert-QoldaSHA "$Extras/E03/manifest.json" $E03Pin
```

Выходы должны быть новыми. Если каталоги уже существуют, **не удалять** их: использовать подтверждённые манифесты после проверки либо выбрать другие новые каталоги. Новый чистый манифест E02 содержит ровно прежние 97 допущенных троек `(SHA клипа, метка, split)`. Он намеренно имеет другой SHA, чем старый `e02-dev-v1.json`: нет строк карантина и изменены относительные пути. Не использовать старый экстрактор карантина ради воспроизведения старого хэша; новое обучение оформляется отдельным run.

## 4. Обучение и экспорт

**E02/A.** 36 эпох, batch 8, lr 0.002, seed 601; 64 фазы, маски и политика пропусков. Использовать пустые отдельные каталоги вне Git. Базовый A не требует SLiM. Для bundle статический граф должен лежать рядом с его JSON.

```powershell
& $Py -B -m ml.train_gpu --variant A --manifest "$ASet/manifest.json" --manifest-sha256 $APin --root $ASet --pose-model $Pose --private "$D/A-work" --out "$D/A-run"
if ($LASTEXITCODE) { throw 'A fit did not finish' }
& $Py -B -m tools.build_bundle --artifacts "$D/A-run" --name A-tcn.safetensors --pose-model $Pose --graph $Graph --manifest "$ASet/manifest.json" --source-commit $Commit --out "$D/pose/bundle.json"
if ($LASTEXITCODE) { throw 'A bundle failed' }
# На собственном ПК и только если порт свободен; указать установленный ffprobe.exe.
& $Py -B -m backend.api --port 8870 --bundle "$D/pose/bundle.json" --artifacts "$D/A-run" --pose-model $Pose --private-scratch "$D/scratch" --ffprobe ffprobe.exe
```

Последняя команда занимает терминал. Остальное обучение запускать из отдельного терминала с теми же переменными; уже работающий сервер не трогать. Проверка запуска — `GET /api/v1/health` с `models.scratch=true`; это не доказательство полного клипа. [Отчёт A](../reports/ml/gpu-handoff/M0-M1-report.md). Эталонный A с рабочей станции: SHA `8dd234d4a3e18c311038b7c6fdad06244fc4a98254f7515824464faf2d1ae24e`; переобучение с новым манифестом меняет метаданные и не обещает того же SHA весового файла.

**E01/E03.** Две независимые головы в `E01/run/`, `E03/run/`, каждая читает только свой манифест. E01 использует 48 фаз, обе стороны, маски, TCN; CUDA/AdamW, seed 601, 36 эпох, batch 16. E03 использует 24 фазы 96×96, RGB-разности, малую пространственно-временную CNN, seed 20261125, 32 эпохи, batch 12. Для E03 граф RTMPose не нужен.

```powershell
& $Py -B -m ml.extra_e01 --manifest "$Extras/E01/manifest.json" --manifest-sha256 $E01Pin --pose $Graph --source-commit $Commit
if ($LASTEXITCODE) { throw 'Inspect private E01 receipt: blocked or partial' }
& $Py -B -m ml.extra_e03 train --manifest "$Extras/E03/manifest.json" --manifest-sha256 $E03Pin
if ($LASTEXITCODE) { throw 'Inspect private E03 receipt' }
```

**E02/B и C — не запускать на публичном наборе.** B требует собственного E02 с двумя классами и раздельными людьми в train/validation. C стартует независимо от той же SLiM-базы, использует те же собственные строки и только допущенное расширение в train. Изменение B→C не означает дополнительное упражнение. Нужны provenance, права/согласие на локальное обучение и производные веса, закреплённые SHA, общие группы человека/исходной записи/семейства. Для них нет публичной ссылки и нет поставленных новых весов. [Полная схема, команды intake, раздельные manifest pins и единый бюджет B/C](model-intake-bc.md). Без передачи `python -B -m tools.model_intake` → `blocked_inputs`, код 2. [Почему три внешних набора кистей не заменяют E02](../reports/ml/model-intake/corpus-c-hand-sources.md).

## 5. Что сверять и передавать между машинами

- Приватно сохранять исходную Git-ревизию, версию GPU/драйвера, SHA исходников/графов/манифестов/весов, класс и версию модели, seed, число обновлений, split и причины отказов. На **другом** добровольце validation указывать число полных/неполных, confusion, BA, оба F1 с обозначением класса/усреднения и число отказов. Один доброволец — не клиническая точность. Не открывать ID 08 для устранения расхождения.
- [E01, отчёт и baseline](../reports/ml/extra-e01.md): BA `0,8591` против `0,5864`; 21/21 выдано, 3 ошибки. SHA checkpoint `8582ba137631d8ab2fee79a083fbffe102dae0b359e31765b7957799c00ec33a`; **только development**, наличие предмета не подтверждается. [E03, отчёт и два контроля](../reports/ml/extra-e03.md): BA `1,0` против `0,5` на 20/20; SHA `95abbe0a8c77b36d9f319648a6f49490d77600c42479b18d14500cfcaac6f822`. Прохождение локального критерия не включает модуль у пользователя автоматически.
- Готовые веса и архивы с другой станции не используются. Повторить обучение здесь и зафиксировать **свои** SHA checkpoint и версий среды; приведённые выше хэши — контрольный результат другой станции.
- Для интеграции E01/E03 использовать [отдельный API](contracts/extra-exercises.md); отсутствие принятого файла → `503`, не результат E02. Для телефона [сборка APK и требования](../android/README.md), [интерфейс моста](../android/ANDROID_BRIDGE.md), [сборка AAR](../tools/android-engine/README.md). Понадобятся Android SDK 34, JDK 17, Gradle 8.7, [Android NDK](https://developer.android.com/ndk/downloads) и [LiteRT](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.4.0). APK/AAR, `m4-2-engine-input-v1` и проверенные TFLite не возникают из Git clone. Если закрытого комплекта нет, сборка sourcecheck не считается GPU-поставкой. Нужно самостоятельно экспортировать графы из местного обучения, собрать native library и проверить её на телефоне; [квитанция T-018](../reports/ml/gpu-handoff/T-018-library.md) — исторический образец, не источник бинарника и не доказательство новой сборки.
