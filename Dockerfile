FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY mlflow.db /app/mlflow.db # Copiez votre base de données mlflow.db

EXPOSE 5313

CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5313", "--default-artifact-root", "sqlite:///mlflow.db"]
