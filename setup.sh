#!/bin/bash

# Installer les dépendances
pip install -r requirements.txt

# Démarrer ngrok
./ngrok http 5000
