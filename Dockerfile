FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY mlflow.db /app/mlflow.db 

EXPOSE 5001

CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5001", "--default-artifact-root", "sqlite:///mlflow.db", "--host", "127.0.0.1"]
