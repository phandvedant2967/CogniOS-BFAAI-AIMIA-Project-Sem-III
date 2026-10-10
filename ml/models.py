"""Baseline Model Zoo encapsulating Logistic Regression, Random Forest, and XGBoost."""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.utils import MODELS_DIR, set_seed
from ml.data import TARGET_CLASSES

CLASS_MAP = {name: i for i, name in enumerate(TARGET_CLASSES)}
REVERSE_CLASS_MAP = {i: name for i, name in enumerate(TARGET_CLASSES)}


def get_model_a_logistic(seed=42):
    """Model A: Logistic Regression (linear interpretable baseline)."""
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )


def get_model_b_random_forest(seed=42):
    """Model B: Random Forest (nonlinear baseline, feature interactions)."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1
    )


def get_model_c_xgboost(seed=42):
    """Model C: XGBoost Classifier (gradient boosted decision trees)."""
    return XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss"
    )


class ModelWrapper:
    """Wrapper providing consistent train, predict, evaluate, and save/load interface."""

    def __init__(self, model_type="random_forest", seed=42):
        self.model_type = model_type
        self.seed = seed
        set_seed(seed)

        if model_type == "logistic_regression":
            self.model = get_model_a_logistic(seed)
        elif model_type == "random_forest":
            self.model = get_model_b_random_forest(seed)
        elif model_type == "xgboost":
            self.model = get_model_c_xgboost(seed)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X_train, y_train):
        """Fit model on transformed X_train and y_train string labels."""
        y_encoded = np.array([CLASS_MAP[y] for y in y_train])
        if self.model_type == "xgboost":
            self.model.fit(X_train, y_encoded)
        else:
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """Predict class string labels."""
        preds = self.model.predict(X)
        if isinstance(preds[0], (int, np.integer)):
            return np.array([REVERSE_CLASS_MAP[i] for i in preds])
        return preds

    def predict_proba(self, X):
        """Predict probability distributions across 4 target classes."""
        return self.model.predict_proba(X)

    def save(self, filepath=None):
        """Serialize trained model artifact to disk."""
        filename = f"baseline_{self.model_type}.pkl"
        path = filepath or os.path.join(MODELS_DIR, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved trained {self.model_type} model -> {path}")

    @classmethod
    def load(cls, filepath):
        """Load serialized model artifact."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        return joblib.load(filepath)
