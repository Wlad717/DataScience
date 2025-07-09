# Используем официальный образ Python
FROM python:3.11-slim-bookworm

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости и код
COPY requirements.txt .
COPY main.py .
COPY models/ ./models/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт, который будет использовать FastAPI
EXPOSE 8000

# Запускаем сервер
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]