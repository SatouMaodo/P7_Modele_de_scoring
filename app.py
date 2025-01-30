import streamlit as st

# Définissez l'URL publique ngrok de votre interface utilisateur MLflow
mlflow_ui_url = "https://6c2e-34-85-241-192.ngrok-free.app"

# Affichez l'interface utilisateur MLflow dans un iframe Streamlit
st.title("Interface utilisateur MLflow")
st.components.v1.iframe(mlflow_ui_url, width=1000, height=800)
