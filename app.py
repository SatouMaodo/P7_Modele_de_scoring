import streamlit as st

# Définissez l'URL publique ngrok de votre interface utilisateur MLflow
mlflow_ui_url = "https://d64d-104-197-221-124.ngrok-free.app"

# Affichez l'interface utilisateur MLflow dans un iframe Streamlit
st.title("Comparaison des modèles:Projet 7-Implémenter un modèle de scoring")
st.components.v1.iframe(mlflow_ui_url, width=1000, height=800)
