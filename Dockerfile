# Utiliser l'image de base Python 3.11
FROM python:3.11

# Créer un répertoire de travail dans le conteneur
WORKDIR /app

# Copier tous les fichiers nécessaires dans le conteneur (y compris les fichiers de modèle et données)
COPY . /app/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port 8000 pour l'application
EXPOSE 8000

# Lancer l'application avec Gunicorn et Uvicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
