# Backend ↔ живой фронт Димаса: что сверено и что исправлено

Дата: 25.11.2026. База: `26118df` (`main`, `origin/main`). Фронт не менялся (последний коммит
Димаса `62a70d8`). Проверка — чтение кода и вызовы `backend.runtime` в отдельном процессе;
тесты не запускались.

## Сверенные точки контракта

| Что | Фронт | Backend | Итог |
|---|---|---|---|
| `health` | `models.scratch \|\| models.transfer` → `ok/nomodel/down` | `{"version":"seriq.v1","models":{"scratch":bool,"transfer":bool}}` | совпадает |
| анализ | `POST /api/v1/analyze?model=scratch\|transfer&mode=single`, тело — blob, `Content-Type: video/mp4\|video/webm` | тот же маршрут, обязательный `Content-Length`, бинарное тело | совпадает |
| лимиты | 64 МиБ и 30 с в UI и в записи | 67 108 864 байта, `duration_limit` > 30 с | совпадает |
| конверт | `version`, `model`, `model_version`, `attempts[]`, `counts`, `warnings` | `wire_result` отдаёт ровно эти поля; `validate_result` это требует | совпадает |
| коды отказа | `model_unavailable`, `engine_busy`, `invalid_video`, `duration_limit`, `too_large`, `unsupported_media` | 503 / 503 / 422 / 422 / 413 / 415 с теми же кодами | совпадает |
| `no_pose` | пустые `attempts` → экран «рука не видна», попытка не засчитывается | `warnings:["no_pose"]`, `attempts:[]`, `counts` нулевые | совпадает |
| `uncertain` | отдельный ярлык; причина обязательна | `label:"uncertain"` + `low_visibility` / `low_confidence` / `unscorable` | совпадает |
| история/профиль | локально в браузере, дедупликация по SHA клипа | серверного хранения нет | совпадает |
| loopback | `http://127.0.0.1:8765` | Host/Origin ограничены loopback | совпадает |

## Найденное несхождение (исправлено)

**Слот модели.** `backend/runtime.py::load_bundle_engines` регистрировал любой принятый bundle
под именем `scratch`. Bundle, собранный для B или для другого эксперимента, мог быть выдан за A:
`health.models.scratch = true`, ответ с чужим `model_version`.

Исправление в `backend/runtime.py`:

- `208–210`: bundle обязан содержать `source_commit` из 40 hex;
- `211–212`: `runtime.device_class` обязан быть `nvidia-cuda`;
- `215–222`: принимается только standalone A/scratch (`id = A`, `variant = A`, `slot = scratch`);
- `223–224`: standalone A не может объявлять SLiM-базу.

Отказ происходит до регистрации движка: `api.py` ловит ошибку, поднимает сервер без моделей,
`health` отдаёт `false`, анализ — `503 model_unavailable`. `transfer` при отсутствии весов
остаётся `false` и отвечает 503.

**Сборка bundle.** `tools/build_bundle.py:38–67` теперь пишет `source_commit` (по умолчанию
`git rev-parse HEAD`) и `model.slot`, а для `scratch` проверяет, что артефакт — standalone A.

**Контракт.** `docs/contracts/api.md` описывает gate bundle и поведение при отказе.

## Подтверждено вызовами (read-only, отдельный процесс)

| Bundle | Результат |
|---|---|
| `bundle.26118df.json` (привязан к `26118df`) | загружен: `{'scratch': True, 'transfer': False}` |
| тот же с вариантом `B` и SLiM-базой | `ValueError: bundle is not the agreed A/scratch delivery` |
| без `source_commit` | `ValueError: bundle must be tied to a source commit` |
| `runtime.device_class = "cpu"` | `ValueError: bundle is not a CUDA runtime delivery` |

## Что не проверялось

- HTTP-тесты `backend.test_contract` / `backend.test_api` — без прямого запроса владельца.
- Реальный `.exe` и Android — на момент прогона отложены.
