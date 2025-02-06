import streamlit as st

# Définissez l'URL publique ngrok de votre interface utilisateur MLflow
mlflow_ui_url = "https://ef83-34-125-115-121.ngrok-free.app"

# Affichez l'interface utilisateur MLflow dans un iframe Streamlit
st.title("Comparaison des modèles: P7-Implémenter un modèle de scoring")
st.components.v1.iframe(mlflow_ui_url, width=1500, height=1000)
