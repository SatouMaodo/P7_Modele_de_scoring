# Utilisez une image de base Python
FROM python:3.9-slim

# Créez un répertoire de travail
WORKDIR /app

# Copiez le code de votre application dans l'image Docker
COPY . /app

# Installez les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposez le port sur lequel l'API fonctionnera
EXPOSE 8000

# Commande pour démarrer l'API avec uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
