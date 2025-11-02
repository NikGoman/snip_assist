# Используем официальный образ Python 3.11 как базовый
FROM python:3.11-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Устанавливаем системные зависимости для компиляции Python-пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь исходный код в контейнер
COPY . .

# Убедимся, что файл запуска находится в нужной директории
COPY app/main.py /app/app/main.py

# Указываем команду для запуска приложения
# Предполагается, что запуск будет происходить через python -m app.main
CMD ["python", "-m", "app.main"]