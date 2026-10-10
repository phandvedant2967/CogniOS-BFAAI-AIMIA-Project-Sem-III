"""Preprocessing pipeline for numeric scaling and categorical encoding (fit-on-train only)."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.utils import MODELS_DIR


class PreprocessorPipeline:
    """Wrapper class for scikit-learn preprocessing pipeline fitted strictly on training data."""

    def __init__(self):
        self.numeric_cols = []
        self.categorical_cols = []
        self.transformer = None
        self.feature_names_out = []

    def fit(self, X_train):
        """Identify column types and fit preprocessor on training data only."""
        X_df = pd.DataFrame(X_train).copy()
        self.numeric_cols = [c for c in X_df.columns if pd.api.types.is_numeric_dtype(X_df[c])]
        self.categorical_cols = [c for c in X_df.columns if c not in self.numeric_cols]

        numeric_pipeline = ColumnTransformer(
            transformers=[
                ("imputer", SimpleImputer(strategy="median"), self.numeric_cols)
            ]
        )

        transformers = []
        if self.numeric_cols:
            transformers.append(
                ("num", ColumnTransformer([
                    ("impute_scale", ColumnTransformer([
                        ("imp", SimpleImputer(strategy="median"), self.numeric_cols)
                    ]), self.numeric_cols)
                ]), self.numeric_cols)
            )

        # Build clean sklearn ColumnTransformer
        ct_list = []
        if self.numeric_cols:
            from sklearn.pipeline import Pipeline
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler())
            ])
            ct_list.append(("num", num_pipe, self.numeric_cols))

        if self.categorical_cols:
            from sklearn.pipeline import Pipeline
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            ct_list.append(("cat", cat_pipe, self.categorical_cols))

        self.transformer = ColumnTransformer(transformers=ct_list)
        self.transformer.fit(X_df)

        self.feature_names_out = list(self.transformer.get_feature_names_out())

        print(f"Preprocessor fitted on {len(X_train)} training rows ({len(self.numeric_cols)} numeric, {len(self.categorical_cols)} categorical -> {len(self.feature_names_out)} output features).")
        return self

    def transform(self, X):
        """Transform dataset using fitted pipeline."""
        if self.transformer is None:
            raise RuntimeError("PreprocessorPipeline is not fitted yet!")
        X_df = pd.DataFrame(X).copy()
        transformed_arr = self.transformer.transform(X_df)
        return pd.DataFrame(transformed_arr, columns=self.feature_names_out, index=X_df.index)

    def fit_transform(self, X_train):
        """Fit on train and return transformed train DataFrame."""
        return self.fit(X_train).transform(X_train)

    def save(self, filepath=None):
        """Serialize fitted preprocessor pipeline to disk."""
        path = filepath or os.path.join(MODELS_DIR, "preprocessing.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved preprocessor pipeline -> {path}")

    @classmethod
    def load(cls, filepath=None):
        """Load serialized preprocessor pipeline."""
        path = filepath or os.path.join(MODELS_DIR, "preprocessing.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor file not found at: {path}")
        return joblib.load(path)
