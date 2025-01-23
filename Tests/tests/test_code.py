from contextlib import contextmanager
import numpy as np
import gc
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
import pytest

from my_functions import (
   application_train_test, bureau_and_balance,
    previous_applications, pos_cash, credit_card_balance,
    installments_payments
)


# Mock pour drive.mount
@pytest.fixture
def mock_drive_mount(monkeypatch):
    def mock_mount(*args, **kwargs):
        print("Mocking drive.mount")
    monkeypatch.setattr("google.colab.drive.mount", mock_mount)

import unittest
import pandas as pd
import numpy as np

# Importer les fonctions de modélisation et de prétraitement

class TestPreprocessingFunctions(unittest.TestCase):
    def setUp(self):
        """Initialiser les données de test."""
        # Vous pouvez utiliser des exemples de données ou des fichiers CSV plus petits pour les tests
        self.application_train_path = 'Tests/application_trains.csv'
        self.application_test_path = 'Tests/application_tests.csv'
        self.bureau_path = 'Tests/bureaux.csv'
        self.bureau_balance_path = 'Tests/bureau_balances.csv'
        self.credit_card_balance_path = 'Tests/credit_card_balances.csv'
        self.installments_payments_path = 'Tests/installments_payment.csv'
        self.pos_cash_balance_path = 'Tests/POSH_CASH_BALANCE.csv'
        self.previous_application_path = 'Tests/previous_application.csv'


    def test_application_train_test(self):
        """Tester la fonction application_train_test."""
        df = application_train_test(num_rows=1000)  # Lire un sous-ensemble de données pour les tests
        # Vérifier les dimensions du DataFrame résultant
        self.assertEqual(df.shape[0], 1000)  # Vérifier le nombre de lignes
        # Vérifier si certaines colonnes sont présentes et ont le type de données attendu
        self.assertIn('CODE_GENDER', df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df['CODE_GENDER']))
        self.assertIn('AMT_INCOME_TOTAL', df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df['AMT_INCOME_TOTAL']))

        self.assertIn('TARGET', df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(df['TARGET']))
        assert not df.empty
    # Vérifier le type de données de certaines colonnes
        assert df['AMT_INCOME_TOTAL'].dtype == np.float64
        assert df['CODE_GENDER'].dtype == np.int64

    def test_bureau_and_balance(self):
        """Tester la fonction bureau_and_balance."""
        bureau_agg = bureau_and_balance(num_rows=1000)
        # Vérifier les dimensions du DataFrame résultant
        self.assertEqual(bureau_agg.shape[0], 1000)  # Vérifier le nombre de lignes
        # Vérifier si certaines colonnes sont présentes et ont le type de données attendu
        self.assertIn('SK_ID_CURR', bureau_agg.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(bureau_agg['SK_ID_CURR']))
        # Vérifier le type de données de certaines colonnes
        assert bureau_agg['BURO_DAYS_CREDIT_MIN'].dtype == np.float64

    def test_previous_applications(self):
        """Tester la fonction previous_applications."""
        prev_agg = previous_applications(num_rows=1000)
        # Vérifier si certaines colonnes sont présentes et ont le type de données attendu
        self.assertIn('SK_ID_CURR', prev_agg.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(prev_agg['SK_ID_CURR']))
    def test_credit_card_balance(self):
        """Tester la fonction credit_card_balance."""
        cc_agg = credit_card_balance(num_rows=500)
        # Vérifier les dimensions du DataFrame résultant
        self.assertEqual(cc_agg.shape[0], 500)  # Vérifier le nombre de lignes
        # Vérifier si certaines colonnes sont présentes et ont le type de données attendu
        self.assertIn('SK_ID_CURR', cc_agg.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(cc_agg['SK_ID_CURR']))
    def test_installments_payments(self):
        """Tester la fonction installments_payments."""
        ins_agg = installments_payments(num_rows=1000)
        # Vérifier si certaines colonnes sont présentes et ont le type de données attendu
        self.assertIn('SK_ID_CURR', ins_agg.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(ins_agg['SK_ID_CURR']))
        assert not ins_agg.empty
        assert ins_agg['INSTAL_DPD_MAX'].dtype == np.float64
    def test_pos_cash(self):
        """Tester la fonction pos_cash."""
        pos_agg = pos_cash(num_rows=1000)
        # Vérifier que le DataFrame renvoyé n'est pas vide
        assert not pos_agg.empty
    # Vérifier le type de données de certaines colonnes
        assert pos_agg['POS_MONTHS_BALANCE_MAX'].dtype == np.int64

        # Vérifier si certaines colonnes sont présentes et ont le type de données attendu
        self.assertIn('SK_ID_CURR', pos_agg.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(pos_agg['SK_ID_CURR']))
    def test_bureau_and_balance(self):
        """Tester la fonction bureau_and_balance."""
        bureau_agg = bureau_and_balance(num_rows=1000)
        assert not bureau_agg.empty
        self.assertTrue(pd.api.types.is_numeric_dtype(bureau_agg['SK_ID_CURR']))
      


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
