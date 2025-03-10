FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY ui.py .

RUN pip install --default-timeout=200 --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]