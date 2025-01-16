
import streamlit as st
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score,adjusted_rand_score,precision_score, recall_score, f1_score
import shap
from sklearn.impute import SimpleImputer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from imblearn.ensemble import BalancedRandomForestClassifier
import plotly.graph_objects as go
import gc
import time
import plotly.express as px
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  make_scorer, roc_auc_score, accuracy_score,adjusted_rand_score,precision_score, recall_score, f1_score
import shap
from sklearn.impute import SimpleImputer
import time
import skopt
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from imblearn.ensemble import BalancedRandomForestClassifier
from catboost import CatBoostClassifier
import joblib
import mlflow
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt


from mlflow.models.signature import infer_signature
warnings.simplefilter(action='ignore', category=FutureWarning)

#!pip install scikit-optimize

#Reprise de la fonction de standardization et d'imputation
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





#Récupértion du best modèle
best_model =joblib.load('best_model.joblib')
df6= pd.read_csv('df6')
y = df6['TARGET']
X = df6.drop(columns=['TARGET'])
threshold=0.75
X_train, X_val, y_train, y_val = train_test_split( X,y, test_size=0.3, random_state=101)
from sklearn.pipeline import Pipeline
# Créer la pipeline
pipeline = Pipeline([
    ('imputation_standardisation', FunctionTransformer(impute_and_standardize, validate=False)),  # Encapsuler dans FunctionTransformer
    ('model', best_model)
])


# Interface Streamlit
st.markdown(
    """
    <style>
    body {
        background-color: #e0f2f7;  /* Couleur bleu foncé */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Prédiction de la probabilité de défaut")
st.markdown("Cette application utilise un modèle de machine learning pour prédire la probabilité de défaut de crédit d'un client.")

# Variables les plus influentes (à adapter selon votre modèle)
most_influential_features = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'PAYMENT_RATE']

# Regroupement des variables par catégories
grouped_features_by_category = {
    "Informations personnelles": ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH', 'NAME_EDUCATION_TYPE_Highereducation'],
    "Informations financières": ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'PAYMENT_RATE'],
    "Score externe": ['EXT_SOURCE_2', 'EXT_SOURCE_3'],
    # Ajoutez d'autres catégories et variables selon vos besoins
}

# Obtenir toutes les variables de X
all_features = X.columns.to_list()

# Créer la catégorie "Autres"
other_features = [f for f in all_features if f not in sum(grouped_features_by_category.values(), [])]
grouped_features_by_category["Autres"] = other_features


# Regroupement de toutes les variables de X
# grouped_features_by_importance = {
#     "Variables importantes": most_influential_features,
#     "Autres variables": [f for f in all_features if f not in most_influential_features]
# }

# Choix du seuil (en dehors des regroupements)
threshold = st.slider("Seuil de probabilité", 0.0, 1.0, 0.75, 0.01)

# Saisie des valeurs pour les caractéristiques
input_values = {}

# Regroupement par catégories
for category_name, features in grouped_features_by_category.items():
    with st.expander(category_name):
        for feature in features:
            is_important = feature in most_influential_features
            label = f"Valeur pour {feature}"
            if is_important:
                label += " (obligatoire)"
            input_values[feature] = st.number_input(label, key=f"{feature}_input")

# Créer un DataFrame avec les valeurs saisies
input_df = pd.DataFrame([input_values])

# Faire la prédiction avec le pipeline
if st.button("Prédire"):
    # Vérifier si les variables obligatoires sont renseignées
    missing_values = [feature for feature in most_influential_features if feature not in input_values or input_values[feature] is None]
    if missing_values:
        st.error(f"Veuillez renseigner les valeurs obligatoires pour les caractéristiques suivantes : {', '.join(missing_values)}")
    else:
        prediction_proba = pipeline.predict_proba(input_df)[0][1]
        # Appliquer le seuil pour la prédiction finale
        prediction = 1 if prediction_proba >= threshold else 0

        st.write(f"Prédiction : {prediction}")
        st.write(f"Probabilité de défaut : {prediction_proba:.2f}")

        # Afficher la probabilité avec une jauge
        st.markdown("### Probabilité de défaut")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba,
            title={'text': "Probabilité de défaut"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]},
                   'bar': {'color': "red"},
                   'steps': [
                       {'range': [0, threshold], 'color': "red"},
                       {'range': [threshold, 1], 'color': "green"}],
                   'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.7, 'value': prediction_proba}}))
        st.plotly_chart(fig)