FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .

RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]