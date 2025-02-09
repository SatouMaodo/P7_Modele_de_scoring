# Utilisation de la version python 3.11 disponible
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt ./ 

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les autres fichiers de l'application dans le conteneur
COPY . .

# Exposer le port sur lequel l'app Heroku va tourner
EXPOSE $PORT

# Commande pour démarrer l'application avec Gunicorn
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:$PORT"]
