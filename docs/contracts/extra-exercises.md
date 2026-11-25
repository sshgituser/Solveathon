# Дополнительные упражнения: локальный file API

Контракт `qolda.extra-exercise.v1` — отдельный от `seriq.v1`. Он не является свидетельством обучения, качества, медицинской пригодности или готовности клиента. E01/E03 по умолчанию недоступны. Старые E02 `/api/v1/health`, `/api/v1/analyze` и `/api/v1/live/*`, их выбор `scratch`/`transfer` и реестр не меняются. Новых сервисов, live E01/E03 и Android-интеграции нет.

## Каталог

`GET /api/v1/exercises` без query, ответ 200 с `version` и `exercises` — массив в порядке E01, E02, E03. Пример при пустом реестре:

```json
{"version":"qolda.extra-exercise.v1","exercises":[
  {"id":"E01","role":"additional","variants":["single"],"readiness":{"web":{"file":false,"live":false},"windows":{"file":false,"live":false},"android":{"file":false,"live":false}}},
  {"id":"E02","role":"primary","variants":["scratch","transfer"],"readiness":{"web":{"file":false,"live":false},"windows":{"file":false,"live":false},"android":{"file":false,"live":false}}},
  {"id":"E03","role":"additional","variants":["single"],"readiness":{"web":{"file":false,"live":false},"windows":{"file":false,"live":false},"android":{"file":false,"live":false}}}
]}
```

`variants` — допустимые имена, **не** доказательство готовности каждого варианта. E02 `web.file` означает, что зарегистрирован хотя бы один E02-движок, `web.live` — что установлен live-адаптер и доступен `scratch`; детальный статус моделей остаётся в `/api/v1/health`. E02 Windows/Android здесь консервативно `false`: этот backend не подтверждает нативные платформы. E01/E03 `file` вычисляется независимо для Web и Windows как логическое И загруженного адаптера и соответствующего owner approval в receipt. Их `live` и весь Android всегда `false`. Эти флаги отражают только конфигурацию этого backend и документированный допуск, не факт запуска/проверки интерфейса на платформе; флаги не служат механизмом авторизации. Сервер не устанавливает, от какой платформы пришёл HTTP-запрос. Отсутствие квитанции/движка не заменяется E02.

## Загрузка файла

`POST /api/v1/exercises/E01/analyze` или `/api/v1/exercises/E03/analyze`, без query и без параметров модели/пути. Один уже обрезанный цикл, бинарное тело **не multipart**, `Content-Type: video/mp4` или `video/webm`, один обязательный `Content-Length`, 1–67 108 864 байта, без Transfer-Encoding. Аналогично E02: сигнатура и контейнер проверяются `ffprobe`, максимум 30 секунд, размер кадра до 4096 по стороне и 4096×2160 пикселей, не более 9000 пакетов и 3600 видеопакетов. Поддерживаемые кодеки MP4: h264/hevc/mpeg4/av1; WebM: vp8/vp9/av1. `ffprobe` должен быть установлен и доступен серверу. Совместимость каждого фактического декодера — ответственность адаптера.

200, ровно поля:

```json
{"version":"qolda.extra-exercise.v1","exercise":"E01","model_version":"example-v1","result":"uncertain","confidence":0.4,"reason":"low_visibility"}
```

Пример — схема, не результат модели. `exercise` строго совпадает с маршрутом, `model_version` — ASCII-токен не длиннее 96 символов; `result`: `complete`, `incomplete`, `uncertain` или `no_pose`. `confidence`: конечное число 0–1 или null, это не вероятность болезни. `reason`: null либо `no_pose`, `low_visibility`, `partial_coverage`, `low_confidence`, `unscorable`. `no_pose` принудительно обнуляет confidence и не является попыткой `incomplete`. Любая причина низкой наблюдаемости принудительно даёт `uncertain`, даже если адаптер вернул метку; низкая уверенность ниже порога receipt также даёт `uncertain/low_confidence`. Неопределённая или отсутствующая метка без причины даёт `uncertain/unscorable`. Иначе `complete`/`incomplete` требуют числовой confidence и null reason. На один запрос — один итог, а не счётчик повторений; повторное сохранение контролирует клиент.

Ошибки новых маршрутов имеют вид `{"version":"qolda.extra-exercise.v1","error":{"code":"model_unavailable"}}`: 400 `invalid_request`, 403 `forbidden`, 411 `length_required`, 413 `too_large`, 415 `unsupported_media`, 422 `invalid_video` / `duration_limit`, 503 `model_unavailable` / `engine_busy`. Неизвестные маршруты — старый 404 `not_found`. Нет сырых исключений и локальных путей в ответах. Ограничения origin/Host, loopback, CORS и `Cache-Control: no-store` те же, что в [E02 API](api.md). Общий единственный nonblocking GPU guard охватывает E02 file/live и E01/E03 в одном процессе; занятость даёт 503. Несколько процессов между собой не синхронизированы.

## Приватная поставка и promotion

Настройка только администратором при запуске backend: `--extra-model-root <private-root> --private-scratch <approved-scratch>`; без root дополнения отключены. Оба каталога должны уже существовать вне текущего checkout и любого Git checkout; scratch — одобренный владельцем приватный каталог с ограниченными правами доступа (на Windows — безопасный наследуемый ACL). Проверка пути не устанавливает и не проверяет ACL. Backend создаёт в нём отдельный временный каталог и копию клипа, удаляемые при штатном завершении/исключении; после аварии процесса или отключения питания возможны остатки, за их безопасную очистку отвечает владелец. Не использовать чужие каталоги, не записывать медиа в Git.

Строгие расположения `<private-root>/E01/run/` и `<private-root>/E03/run/` независимы. В каждом нужен отдельный созданный владельцем `promotion.json` и `checkpoint.safetensors` в том же каталоге, без ссылок на файлы и без ссылок/переходов каталогов наружу. Receipt UTF-8 JSON не более 16 KiB, без повторов и лишних полей:

```json
{"schema":"qolda-extra-promotion-v1","exercise":"E01","owner_id":"data-owner","decision":"approved","checkpoint":"checkpoint.safetensors","sha256":"<64 lowercase hex SHA-256 checkpoint bytes>","low_confidence_below":0.1,"file_platforms":{"web":false,"windows":false}}
```

Это шаблон, не действующий допуск: при двух `false` движок не регистрируется. `exercise` должен соответствовать папке, `low_confidence_below` — заранее одобренный конечный порог от 0 до 1; `file_platforms` содержит ровно два булевых поля, каждое отдельно подтверждается владельцем. До регистрации backend проверяет формат, явное решение и фактический SHA-256 checkpoint. Этот локальный receipt **не является криптографической подписью**: кто может писать в приватный каталог, может его подделать. Владелец обязан защитить receipt/веса и адаптер от записи, отдельно подтвердить provenance и проверку платформ, а также сохранять неподвижность всех файлов после сверки и на время работы процесса; проверка на старте не защищает от последующей подмены/TOCTOU. Отзыв или замена требует остановки/перезапуска backend оператором. Не создавать фиктивный approval, не рассчитывать доверенный pin из произвольного найденного файла и не выдавать совпадение SHA за оценку модели.

После прохождения receipt загружается только заранее установленный доверенный Python-модуль `ml.extra_e01` / `ml.extra_e03_service` (строгая обёртка над неизменным `ml.extra_e03`); его `open_engine(model_dir: Path)` должен загрузить именно одобренный checkpoint, инициализировать нужный локальный runtime и вернуть callable `(clip_path: Path) -> dict` с ровно `label`, `confidence`, `reason`, `version`. `label`: `complete`/`incomplete`/`uncertain`/null; остальные значения и версии как выше. Код адаптера является доверенным: backend не может доказать, что он использовал именно проверенные байты, что все вычисления были на GPU, или что информация о видимости правдива. Для безопасного `no_pose`/observability адаптер обязан выдавать указанные причины; без таких сигналов guard не может обнаружить неверный положительный результат. Отсутствующий/ошибочный receipt, SHA, модуль или не-callable загрузка не регистрируют соответствующее упражнение, а неожиданный сбой адаптера/нарушение контракта снимает его с готовности до перезапуска. Сбой одного дополнения не выключает другое или E02. Сам по себе callable не доказывает успешного инференса или проверки качества.
