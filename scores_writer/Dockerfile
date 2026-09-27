FROM python:3.12-slim

WORKDIR /app

# Создание директории для логов
RUN mkdir -p /app/logs && touch /app/logs/service.log

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

CMD ["python", "./writer.py"]