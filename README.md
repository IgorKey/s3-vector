# S3 Vector Store PoC

Vector Store поверх S3-compatible Object Storage.
Отдельный compute-сервис, который принимает готовые эмбеддинги, хранит их в S3,
строит ANN-индекс и выполняет similarity search.

## Зачем

Проверить гипотезу: можно ли собрать vector search **поверх существующего S3**,
не трогая ядро object storage, а вынеся весь compute в отдельный сервис.

Ответ: **да, можно.** S3 работает как durable storage для векторов, метаданных
и snapshot-ов индекса. Векторный поиск живёт в отдельном процессе с in-memory
HNSW-индексом.

## Архитектура

```mermaid
graph TB
    Client["Client / Demo script"]
    Client -->|"HTTP :8000"| API

    subgraph API["Vector API Service (FastAPI)"]
        direction TB
        Routes["Routes: Collections / Vectors / Search"]
        VS["Vector Service"]
        Routes --> VS

        subgraph Internals[" "]
            direction LR
            S3["S3 Storage<br/>(MinIO SDK)"]
            IDX["Index Engine<br/>(hnswlib HNSW)"]
        end
        VS --> S3
        VS --> IDX
    end

    S3 -->|"read/write JSON & binary"| MinIO["S3-compatible<br/>Object Storage<br/>(MinIO)"]

    style Client fill:#f9f,stroke:#333
    style MinIO fill:#ff9,stroke:#333
    style IDX fill:#9ff,stroke:#333
```

### Что хранится в S3

```mermaid
graph LR
    subgraph Bucket["S3 Bucket: vector-store"]
        direction TB
        Meta["collections/{id}/<b>meta.json</b><br/>name, dimension, metric"]
        Vectors["collections/{id}/vectors/<b>{vid}.json</b><br/>vector + metadata + payload"]
        Snapshot["collections/{id}/index/<b>snapshot.bin</b><br/>сериализованный HNSW-индекс"]
    end

    style Meta fill:#e8f5e9
    style Vectors fill:#e3f2fd
    style Snapshot fill:#fff3e0
```

Индекс — **в памяти процесса**. Периодически (каждые N секунд) сохраняется
в S3 как snapshot. При рестарте загружается из snapshot, а если его нет —
перестраивается из всех vector-объектов в S3.

## Как работает поиск

```mermaid
sequenceDiagram
    participant C as Client
    participant API as Vector API
    participant IDX as HNSW Index
    participant S3 as S3 Storage

    C->>API: POST /collections/{id}/search<br/>{query_vector, top_k, filter}
    API->>IDX: search(query_vector, top_k)
    IDX-->>API: кандидаты [(id, score), ...]
    Note over API: post-filter по metadata<br/>применить min_score
    opt include_payload = true
        API->>S3: get vector object
        S3-->>API: payload
    end
    API-->>C: {results: [{id, score, metadata, payload}]}
```

1. ANN-индекс (HNSW) находит ~top_k ближайших кандидатов — **не перебирает все векторы**
2. Если задан `filter` — post-filter по metadata (exact match)
3. Если задан `min_score` — отсекает кандидатов ниже порога
4. Индекс **один на коллекцию**, обновляется при каждой вставке

## Как работает запись

```mermaid
sequenceDiagram
    participant C as Client
    participant API as Vector API
    participant S3 as S3 Storage
    participant IDX as HNSW Index

    C->>API: PUT /collections/{id}/vectors/{vid}<br/>{vector, metadata, payload}
    API->>API: validate dimension
    API->>S3: put vector object (JSON)
    API->>IDX: add(vid, vector)
    API->>API: update metadata store
    API-->>C: 200 OK
```

S3 — source of truth, индекс — производная структура для быстрого поиска.

## Как работает восстановление

```mermaid
flowchart TD
    Start["Старт сервиса"] --> ListCol["Загрузить список коллекций из S3"]
    ListCol --> CheckSnap{"Есть snapshot.bin?"}
    CheckSnap -->|Да| LoadSnap["Загрузить индекс из snapshot"]
    CheckSnap -->|Нет| Rebuild["Перечитать все vectors/*.json<br/>и пересобрать индекс"]
    LoadSnap --> LoadMeta["Загрузить metadata в память"]
    Rebuild --> LoadMeta
    LoadMeta --> Ready["Сервис готов к работе"]
    Ready --> Periodic["Каждые N секунд:<br/>сохранить snapshot в S3"]

    style Ready fill:#c8e6c9
    style Periodic fill:#fff3e0
```

## API

### Collections

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/collections` | Создать коллекцию (name, dimension, distance_metric) |
| `GET` | `/collections` | Список коллекций |
| `GET` | `/collections/{id}` | Информация о коллекции |
| `DELETE` | `/collections/{id}` | Удалить коллекцию со всеми данными |

### Vectors

| Метод | Путь | Описание |
|-------|------|----------|
| `PUT` | `/collections/{id}/vectors/{vid}` | Добавить/обновить вектор |
| `GET` | `/collections/{id}/vectors/{vid}` | Получить вектор по id |
| `DELETE` | `/collections/{id}/vectors/{vid}` | Удалить вектор |
| `POST` | `/collections/{id}/vectors:batchPut` | Батчевая вставка (до 1000) |

### Search

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/collections/{id}/search` | Similarity search |

**Параметры поиска:**
```json
{
  "query_vector": [0.1, 0.2, ...],
  "top_k": 5,
  "min_score": 0.7,
  "filter": {"space": "ENG"},
  "include_metadata": true,
  "include_payload": true
}
```

### Служебные

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Количество коллекций и векторов |

## Стек

| Компонент | Технология |
|-----------|-----------|
| API | Python 3.11, FastAPI |
| ANN-индекс | hnswlib (HNSW) |
| Object Storage | MinIO (S3-compatible) |
| Модели | Pydantic v2 |
| Логирование | structlog |
| Контейнеризация | Docker Compose |

## Быстрый старт

```bash
# Поднять MinIO + API
docker compose up -d

# Проверить
curl http://localhost:8000/health

# Запустить демо (создаст коллекцию, загрузит векторы, выполнит поиск)
pip install httpx numpy
python -m demo.demo
```

## Конфигурация

Все параметры через переменные окружения с префиксом `S3V_`:

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `S3V_S3_ENDPOINT` | `localhost:9000` | Адрес S3 |
| `S3V_S3_ACCESS_KEY` | `minioadmin` | Access key |
| `S3V_S3_SECRET_KEY` | `minioadmin` | Secret key |
| `S3V_S3_BUCKET` | `vector-store` | Имя бакета |
| `S3V_S3_USE_SSL` | `false` | SSL |
| `S3V_SNAPSHOT_INTERVAL_SECONDS` | `60` | Интервал сохранения snapshot |
| `S3V_LOG_LEVEL` | `INFO` | Уровень логирования |

## Тесты

```bash
# Установить зависимости
pip install -e ".[dev]"

# Запустить (51 тест, ~0.5 сек)
pytest tests/ -v
```

## Ограничения PoC

- Один compute-инстанс, индекс в памяти
- Нет шардирования, репликации, HA
- Metadata filter — только exact match (post-filter)
- Нет IAM, multi-tenancy, billing
- Snapshot раз в N секунд, не real-time
- Это демонстрация концепта, не production-система

## Структура проекта

```
s3-vector/
├── src/
│   ├── main.py                 # FastAPI app, lifespan, health/stats
│   ├── config.py               # Настройки из env
│   ├── models.py               # Pydantic-модели
│   ├── s3_storage.py           # S3 CRUD (MinIO SDK)
│   ├── index_engine.py         # hnswlib wrapper
│   ├── collection_manager.py   # Lifecycle коллекций + snapshot
│   ├── vector_service.py       # CRUD + search orchestration
│   └── routes/
│       ├── collections.py      # Collection endpoints
│       ├── vectors.py          # Vector endpoints
│       └── search.py           # Search endpoint
├── tests/                      # 51 тест (unit + integration)
├── demo/
│   ├── demo.py                 # E2E демо-сценарий
│   └── sample_data.py          # Синтетические embeddings
├── Dockerfile
└── docker-compose.yml          # MinIO + vector-api
```
