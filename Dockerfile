FROM python:3.11.11-slim-buster

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Changed command to use main.py
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:$PORT"]
