import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from PIL import Image
import plotly.graph_objects as go
import mlflow.pyfunc
import requests
model_path = os.path.join(os.getcwd(), "model")
model = mlflow.pyfunc.load_model(model_path)

# Créer une fonction pour effectuer des prédictions à l'aide de l'API
def predict_with_api(input_data):
    api_url = "https://mon-app-suivi-mlflow-ff2b9c82a88e.herokuapp.com/invocations"  # Remplacez par l'URL de votre API Heroku
    headers = {"Content-Type": "application/json"}
    data = input_data.to_json()  # Convertir les données d'entrée en JSON
    response = requests.post(api_url, headers=headers, data=data)
    prediction = response.json()
    return prediction

def gauge_chart(prediction, threshold):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilité de défaut"},
        gauge = {
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, threshold], 'color': "green"},
                {'range': [threshold, 1], 'color': "red"}],
            'threshold' : {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.76,
                'value': threshold}}))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))  # Remove margins
    return fig
def impute_and_standardize(X):
    """Impute les valeurs manquantes et standardise les données.

    Args:
        X: DataFrame contenant les données à imputer et standardiser.

    Returns:
        DataFrame avec les valeurs manquantes imputées et les données standardisées.
    """

    # Créer une copie de X pour éviter de modifier les données d'origine
    X_imputed = X.copy()

    # Replace infinite values with NaNs
    X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan)

    # Imputer les valeurs manquantes avec la médiane pour les variables numériques
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_imputed)

    # Standardiser les données numériques
    scaler = StandardScaler()
    X_scaled= scaler.fit_transform(X_imputed)

    return X_scaled
# Chargement du modèle et des données
best_model = joblib.load('best_model.joblib')

test_df= pd.read_csv('test_df.csv')
y = test_df['TARGET']
X = test_df.drop(columns=['TARGET'])
X_train, X_val, y_train, y_val = train_test_split( X,y, test_size=0.3, random_state=101)
from sklearn.pipeline import Pipeline
# Créer la pipeline
pipeline = Pipeline([
    ('imputation_standardisation', FunctionTransformer(impute_and_standardize, validate=False)),  # Encapsuler dans FunctionTransformer
    ('model', best_model)
])

# --- Variables les plus influentes ---
influential_features = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'PAYMENT_RATE']

# Charger le logo
logo_image = Image.open("logo.png")  # Remplacez "logo.png" par le chemin d'accès réel à votre fichier de logo

# Afficher le logo en haut à gauche (une seule fois)
st.sidebar.image(logo_image, use_container_width=True)

# Afficher le texte sous le logo (horizontalement)
st.sidebar.markdown(
    """
    <style>
    .wordart-text {
        font-family: 'Arial Black', sans-serif;
        font-size: 20px;
        font-weight: bold;
        color: #800080; /* Violet */
        text-shadow: 2px 2px 4px #000000; /* Ombre */
    }
    .smaller-text {
        font-size: 12px;
        color: black;
    }
    </style>
    <div class="wordart-text">Etudiante: Amsatou NDIAYE - Parcours Data Science </div>
    <div class="smaller-text">Titre du Projet: Implémenter un modèle de scoring-Openclassrooms 2025 </div>
    """,
    unsafe_allow_html=True,
)
# --- Mise en forme de l'application ---
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {
        background-image: linear-gradient(to bottom right, #e0ffff, #cce0ff); 
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Afficher le logo en haut à gauche
#st.sidebar.image(logo_image, use_container_width=True)


# Liste des identifiants uniques des demandeurs
sk_id_curr_list = test_df['SK_ID_CURR'].unique()

# --- Affichage ---
st.markdown("## Prédiction de la probabilité de défaut")

# --- Sélection de l'identité du demandeur ---
selected_sk_id_curr = st.selectbox("Sélectionnez l'identité du demandeur (SK_ID_CURR)", sk_id_curr_list)

# --- Prédiction et importance des variables ---
if selected_sk_id_curr:
    # Filtrer les données pour l'identité sélectionnée
    selected_data = test_df[test_df['SK_ID_CURR'] == selected_sk_id_curr]

    # Supprimer la colonne 'TARGET' si elle est présente
    if 'TARGET' in selected_data.columns:
        selected_data = selected_data.drop(columns=['TARGET'])

    # Prédiction
    prediction = best_model.predict_proba(selected_data)[0, 1]
    st.markdown(f"**Probabilité de défaut :** {prediction:.2f}")

    # Seuil d'acceptation du prêt
    threshold = 0.76
    st.plotly_chart(gauge_chart(prediction, threshold))
    # Jauge
    st.markdown("**Décision :**")
    if prediction < threshold:
        st.success("Accepté")  # Affiche "Accepté" en vert si la prédiction est inférieure au seuil
        st.markdown(
            f'<div style="background-color: green; padding: 10px; border-radius: 5px; color: white; font-weight: bold;">Accepté</div>',
            unsafe_allow_html=True
        )
    else:
        st.error("Refusé")  # Affiche "Refusé" en rouge si la prédiction est supérieure ou égale au seuil
        st.markdown(
            f'<div style="background-color: red; padding: 10px; border-radius: 5px; color: white; font-weight: bold;">Refusé</div>',
            unsafe_allow_html=True
        )

    # Feature importance locale avec shap.plots.waterfall
    feature_names = X.columns
    explainer = shap.TreeExplainer(best_model)
    # Get SHAP values as an Explanation object
    shap_values_explanation = explainer(selected_data)
    # Waterfall plot
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values_explanation[0], show=False)
    plt.title("Importance des variables locales (Waterfall Plot)", fontsize=16)
    plt.xlabel("Impact sur la prédiction", fontsize=12)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()

    # --- Informations sur les variables influentes ---
    st.markdown("### Informations sur les variables influentes")

    # --- Informations personnelles ---
    st.markdown("#### Informations personnelles")
    personal_info = selected_data[['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED']].T.rename(columns={selected_data.index[0]: 'Valeur'})
    personal_info.index = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED'] # Renommer les index
    st.table(personal_info)

    # --- Informations financières ---
    st.markdown("#### Informations financières")
    financial_info = selected_data[['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'PAYMENT_RATE']].T.rename(columns={selected_data.index[0]: 'Valeur'})
    financial_info.index = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'PAYMENT_RATE'] # Renommer les index
    st.table(financial_info)

# --- Boutons pour afficher les groupes ---
if st.button("Afficher les Acceptés"):
    accepted_ids = test_df[best_model.predict_proba(test_df.drop(columns=['TARGET'], errors='ignore'))[:, 1] < threshold]['SK_ID_CURR'].tolist()
    st.write("**Identifiants des Acceptés :**", accepted_ids)

if st.button("Afficher les Refusés"):
    refused_ids = test_df[best_model.predict_proba(test_df.drop(columns=['TARGET'], errors='ignore'))[:, 1] >= threshold]['SK_ID_CURR'].tolist()
    st.write("**Identifiants des Refusés :**", refused_ids)
