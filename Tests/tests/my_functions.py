# -*- coding: utf-8 -*-
"""Projet7: EDA et feature engineering.ipynb

**HOME CREDIT DEFAULT RISK COMPETITION**: Le projet s'inscrit dans le cadre d'une compétition sur la prédiction du risque de défaut de crédit. Home Credit est une institution financière qui propose des prêts à des clients n'ayant pas ou peu d'historique de crédit.

**Objectif**: Construire un modèle capable de prédire la probabilité qu'un client ne rembourse pas son prêt (défaut de crédit) en fonction de diverses informations.
"""

"""#Feature engineering"""
from contextlib import contextmanager
import numpy as np
import pandas as pd
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
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
from sklearn.impute import SimpleImputer
import time
from contextlib import contextmanager
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import re
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score,adjusted_rand_score,precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import time
from catboost import CatBoostClassifier
import joblib
import mlflow
import os
from mlflow.models.signature import infer_signature

#***************************#Prétraitement des données et création de nouvelles variables*******************************************
###########################################################################################################################################
#***************************#Prétraitement des données et création de nouvelles variables*******************************************
###########################################################################################################################################
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('Tests/application_train_test.csv', nrows= num_rows)
    test_df = pd.read_csv('Tests/MyDrive/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = pd.concat([df,test_df])
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('Tests/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('Tests/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg.reset_index()

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('Tests/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg.reset_index()

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('Tests/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg.reset_index()

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('Tests/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg.reset_index()

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('Tests/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg.reset_index()


#Cout total
cost_matrix = {
    'FN': 10,  # Coût d'un faux négatif
    'FP': 1   # Coût d'un faux positif
}
def calculate_total_cost(y_true, y_pred, threshold, cost_matrix,greater_is_better=False):
    """
    Calcule le coût total des erreurs de prédiction.

    Args:
        y_true: Les vraies étiquettes (0 ou 1).
        y_pred: Les probabilités prédites (valeurs entre 0 et 1).
        threshold: Le seuil de classification (valeur entre 0 et 1).
            Si la probabilité prédite est supérieure ou égale au seuil,
            l'observation est classée comme positive (1), sinon négative (0).
        cost_matrix: Un dictionnaire contenant les coûts associés aux faux négatifs (FN) et aux faux positifs (FP).
            Exemple : {'FN': 10, 'FP': 1} signifie qu'un faux négatif coûte 10 et un faux positif coûte 1.

    Returns:
        Le coût total, calculé comme la somme des coûts des faux négatifs et des faux positifs.
    """

    # Convertir les probabilités prédites en classes (0 ou 1) en utilisant le seuil.
    y_pred_class = (y_pred >= threshold).astype(int)

    # Calculer le nombre de faux négatifs (FN):
    # observations qui sont positives en réalité (y_true == 1)
    # mais classées comme négatives par le modèle (y_pred_class == 0).
    FN = np.sum((y_true == 1) & (y_pred_class == 0))

    # Calculer le nombre de faux positifs (FP):
    # observations qui sont négatives en réalité (y_true == 0)
    # mais classées comme positives par le modèle (y_pred_class == 1).
    FP = np.sum((y_true == 0) & (y_pred_class == 1))

    # Calculer le coût total en utilisant la matrice de coût.
    total_cost = int(FN) * cost_matrix['FN'] + int(FP) * cost_matrix['FP']

    return total_cost

def optimize_threshold_and_metrics(y_true, y_pred_proba, cost_matrix):
    """
    Optimise le seuil de classification et calcule des métriques de performance
    importantes, notamment le score métier, l'AUC et l'accuracy.

    Args:
        y_true: Les vraies étiquettes (valeurs réelles) des données.
        y_pred_proba: Les probabilités prédites par le modèle pour chaque donnée.
        cost_matrix: La matrice de coût définissant les coûts associés
                     aux différentes erreurs de classification
                     (faux positifs, faux négatifs).

    Returns:
        Un dictionnaire contenant les informations suivantes :
            - optimal_threshold : le seuil optimal qui minimise le coût total.
            - business_score : le score métier, qui représente le coût total minimal.
            - auc_score : l'aire sous la courbe ROC (AUC), une mesure de la performance globale du modèle.
            - accuracy : l'accuracy (exactitude), le pourcentage de prédictions correctes.
    """

    # Générer une liste de seuils possibles entre 0 et 1
    thresholds = np.linspace(0, 1, 101)

    # Calculer le coût total pour chaque seuil
    costs = [calculate_total_cost(y_true, y_pred_proba, t, cost_matrix) for t in thresholds]

    # Trouver le seuil qui minimise le coût total
    optimal_threshold = thresholds[np.argmin(costs)]

    # Le score métier est le coût total minimal
    best_model_score = min(costs)

    # Calculer l'AUC
    auc_score = roc_auc_score(y_true, y_pred_proba)

    # Calculer l'accuracy en utilisant le seuil optimal
    accuracy = accuracy_score(y_true, (y_pred_proba >= optimal_threshold).astype(int))

    # Retourner les résultats dans un dictionnaire
    return {
        'optimal_threshold': optimal_threshold,
        'business_score': best_model_score,
        'auc_score': auc_score,
        'accuracy': accuracy
    }


#Standardization et imputation
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

def train_and_evaluate_models(X_scaled, y, cost_matrix):
    """Entraîne et évalue plusieurs modèles avec BayesianOptimization."""
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.3, random_state=101
    )
    
    # Define models and their hyperparameter spaces using skopt.space objects
    models = {
        "LogisticRegression": {
            "model": LogisticRegression( verbose=1,max_iter=1000),
            "params": {
                "C": Real(0.01, 10, prior="log-uniform"),
                "class_weight":Categorical(["balanced",None]),
                }
        },
        "RandomForestClassifier": {
            "model": RandomForestClassifier( verbose=1),
            "params": {
                "n_estimators": Integer(100, 200),  # Use Integer for discrete values
                "max_depth": Integer(5, 10),
                "class_weight":Categorical(["balanced",None]),
            },
        },

        "Balanced Random Forest":{
            "model":BalancedRandomForestClassifier(verbose=1,class_weight="balanced"),
            "params":{
                "n_estimators": Integer(100, 1000),
                "max_depth": Integer(5, 20),
               "min_samples_split": Integer(2, 20),
               "min_samples_leaf": Integer(1, 20),
                "class_weight":Categorical(["balanced",None]),

            },
         },
        "CatBoostClassifier": {
            "model": CatBoostClassifier(verbose=1),
            "params": {
                "iterations": Integer(100, 1000),
                "learning_rate": Real(0.01, 0.1, prior="log-uniform"),
                "depth": Integer(4, 10),
                "l2_leaf_reg": Real(1, 10, prior="log-uniform"),
                "border_count": Integer(32, 255),
                "auto_class_weights": Categorical(["Balanced", 'None']),  # Équilibre des classes
            },
        }

    }


    # Loop over the models and perform BayesianOptimization
    best_model = None
    best_score = float("inf")
    results = []
    for model_name, model_data in models.items():
        start_time = time.time()

        # Define a target function that includes the threshold optimization
        def target_function(params, model_name=model_name):  # Add model_name here
            model = models[model_name]["model"]  # Access model using model_name
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            # Optimize threshold and calculate business score within target_function
            optimized_metrics = optimize_threshold_and_metrics(
                y_val, y_pred_proba, cost_matrix
            )
            return optimized_metrics["business_score",]

        # Include model_name in the target function arguments
        # Use search_spaces and estimator instead of f
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=101)

        bayes_search = BayesSearchCV(
            model_data["model"],
            model_data["params"],
            n_iter=3,  # Number of iterations
            # Remove business_scorer here: scoring=business_scorer,
            cv=cv,
            n_jobs=-1,  # Use all available cores
            random_state=101,  # Set random state for reproducibility
            )
        # Modify the scoring parameter to use 'roc_auc'
        bayes_search.scoring = 'roc_auc'

            # Fit other models without sample weights
        bayes_search.fit(X_train, y_train)

        # Best model
        best_model_params = bayes_search.best_params_
        best_model = bayes_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Predictions using the best model
        y_pred_proba = best_model.predict_proba(X_val)[:, 1]
        y_pred_train = best_model.predict(X_train)
        y_pred_val = best_model.predict(X_val)

        # Évaluer les performances sur l'ensemble de test
        optimized_metrics = optimize_threshold_and_metrics(y_val, y_pred_proba, cost_matrix)
        end_time = time.time()  # Fin du chronométrage
        execution_time = end_time - start_time

        results.append(
            { "Modèle": model_name,
                "Score métier": optimized_metrics["business_score"],
                "Accuracy Train": accuracy_score(y_train, y_pred_train),
                "Accuracy Val": accuracy_score(y_val, y_pred_val),
                "precision": precision_score(y_val, y_pred_val),
                "recall": recall_score(y_val, y_pred_val),
                "f1": f1_score(y_val, y_pred_val),
                "Temps d'exécution": execution_time,
                "AUC": optimized_metrics["auc_score"], } )
        df_results = pd.DataFrame(results)
        if optimized_metrics["business_score"] < best_score:
            best_score = optimized_metrics["business_score"]
            best_model = best_model
    return best_model, df_results, best_model_params


def visualiser_resultats(df_results):
  """
  Génère et affiche trois graphiques pour visualiser les résultats des modèles :

  1. Un graphique à barres comparant les scores métiers des modèles.
  2. Un graphique de courbes ROC et AUC pour chaque modèle.
  3. Un graphique en boîte comparant les temps d’exécution des modèles.

  Args :
    df_results : Un DataFrame contenant les résultats des modèles,
                 y compris les colonnes « Modèle », « Score métier »,
                 « AUC » et « Temps d’exécution ».

  Returns :
    None. Les graphiques sont affichés directement.
  """
  # Graphique à barres pour les scores métiers
  fig1 = px.bar(df_results, x='Modèle', y='Score métier', color='Modèle',
               title='Comparaison des scores métiers des modèles')
  fig1.show()

  # Graphique de courbes ROC et AUC
  fig2 = go.Figure()
  for modele, auc in df_results[['Modèle', 'AUC']].values:
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             line=dict(dash='dash'),
                             name='Aléatoire',
                             showlegend=False))  # Ligne de base pour un classificateur aléatoire

    fig2.add_trace(go.Scatter(x=[0, 0, 1, 1],  # Points de la courbe ROC simplifiée
                             y=[0, auc, auc, 1],
                             mode='lines',
                             name=modele))

  fig2.update_layout(title='Courbes ROC et AUC',
                    xaxis_title='Taux de faux positifs (FPR)',
                    yaxis_title='Taux de vrais positifs (TPR)',
                    legend_title='Modèles')
  fig2.show()

  # Graphique en boîte pour les temps d’exécution
  fig3 = px.box(df_results, x='Modèle', y="Temps d'exécution", color='Modèle',
              title="Comparaison du temps d'exécution des modèles")
  fig3.show()

  return fig1, fig2, fig3  # Retourner les trois figures



def afficher_shap_values( my_best_model, X_val, X):
  """Affiche les valeurs SHAP pour un modèle donné.

  Args:
    my_best_model: Le modèle à expliquer.
    X_val: Les données de validation.
    X: Les données d'entraînement.

  Returns:
    Tuple: The paths to the saved figures.
  """

  # feature_names = imputer.get_feature_names_out(X.columns)
  # ou
  # feature_names = scaler.get_feature_names_out(X.columns)

  # En supposant que votre modèle a un attribut `feature_names_in_`
  feature_names = X.columns

  X_val = pd.DataFrame(X_val, columns=feature_names)
  explainer = shap.TreeExplainer( my_best_model)
  shap_values = explainer(X_val)

  # Graphique en barres avec titre
  feature_global=shap.plots.bar(shap_values, show=False)
  plt.title("Importance globale des variables")
  plt.show()

  # Graphique en cascade avec titre
  feature_local=shap.plots.waterfall(shap_values[0], show=False)
  plt.title("Impact des variables sur la prédiction pour la première observation")
  # Save the figure and get the path
  plt.show()

  # Return the paths to the saved figures
  return feature_global,feature_local
