# AI-ассистент для прорабов и технических директоров в Telegram

AI-ассистент для прорабов и технических директоров в Telegram. Позволяет быстро находить и интерпретировать нормативные документы (ГОСТ/СНИП) с подтверждёнными ссылками.

## Особенности

- **RAG-архитектура** на базе `LlamaIndex` и `Chroma`
- **Локальная LLM** (`Phi-3-mini`) для отказоустойчивости
- **Ограничение запросов** (3/день для бесплатных пользователей)
- **Проверяемые источники** в ответах
- **Безопасность и надёжность** по умолчанию
- **Готов к продакшену** и интеграции в CI/CD
- **Микросервисная архитектура** (bot, api, rag) для гибкости и масштабируемости

## Архитектура

Проект состоит из трёх основных сервисов, объединённых через `docker-compose`:

- **`bot`**: Telegram-бот, реализованный на `aiogram`. Обрабатывает команды и запросы пользователей.
- **`api`**: Веб-API, реализованный на `FastAPI`. Принимает запросы от бота и взаимодействует с сервисом RAG.
- **`rag`**: Сервис поиска и генерации ответов, использующий `LlamaIndex`, `Chroma` и локальную модель `Phi-3`.

## Требования

- `Docker`
- `Docker Compose` (или `podman` с `podman-compose`)

## Установка и запуск

1.  Склонируйте репозиторий:

    ```bash
    git clone https://github.com/NikGoman/snip_assist.git
    cd snip_assist
    ```

2.  Создайте файл `.env.local` на основе `.env.example` и укажите необходимые переменные:

    ```bash
    cp .env.example .env.local
    # Отредактируйте .env.local, указав TELEGRAM_BOT_TOKEN и другие параметры
    ```

3.  Убедитесь, что у вас установлены `Docker` и `Docker Compose`.

4.  Запустите проект:

    ```bash
    docker-compose up --build
    ```
    Или, если вы используете `podman`:
    ```bash
    podman-compose up --build
    ```

    Эта команда соберёт образы для всех трёх сервисов и запустит их в контейнерах. Бот начнёт принимать сообщения.

## Структура проекта

```
snip_assist/
├── app/                    # Логика Telegram-бота
│   ├── __init__.py
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── handlers/
│   │   │   ├── __init__.py
│   │   │   ├── start.py
│   │   │   ├── query.py
│   │   │   ├── help.py
│   │   │   ├── stats.py
│   │   │   └── limits.py
│   │   ├── keyboards/
│   │   ├── filters/
│   │   └── middleware/
│   │   │   ├── limits.py
│   │   │   ├── logging.py
│   │   │   └── throttling.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py       # Обновлено для новой архитектуры
│   │   └── database.py
│   └── services/
│       ├── __init__.py
│       └── query_service.py # Обновлено для вызова API-сервиса
├── api/                    # Логика веб-API (новый сервис)
│   ├── __init__.py
│   ├── main.py
│   └── config.py
├── rag/                    # Логика RAG (новый сервис)
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   └── rag_engine.py
├── data/                   # Данные (векторные эмбеддинги и документы)
│   ├── docs/
│   │   ├── raw             # "Грязные" документы
│   │   └── text            # Подготовленные документы в txt
│   └── chroma/             # Данные ChromaDB (смонтировано в контейнере rag)
├── logs/                   # Логи сервисов (смонтировано в контейнерах)
├── tests/
│   ├── __init__.py
│   ├── test_bot.py
│   ├── test_rag.py
│   └── conftest.py
├── .env.example            # Шаблон переменных окружения
├── .env.local              # Файл переменных окружения для docker-compose
├── .gitlab-ci.yml
├── .gitgnore
├── Dockerfile.bot          # Dockerfile для сервиса bot
├── Dockerfile.api          # Dockerfile для сервиса api
├── Dockerfile.rag          # Dockerfile для сервиса rag
├── docker-compose.yml      # Оркестрация сервисов
├── requirements.api.txt    # Зависимости для API-сервиса
├── requirements.bot.txt    # Зависимости для бота
├── requirements.rag.txt    # Зависимости для RAG-сервиса
├── pyproject.toml
└── README.md
```

## Разработка

Для локальной разработки и тестирования отдельных сервисов используйте соответствующие `Dockerfile.*` и `docker-compose.yml`.

---

## Лицензия
 GNU AFFERO GENERAL PUBLIC LICENSE
