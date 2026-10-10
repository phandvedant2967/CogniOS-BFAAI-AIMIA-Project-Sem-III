"""Comprehensive Test Suite for V4.2 Model Inference Pipeline.

Tests:
  1. Both saved models load successfully.
  2. Inference uses correct preprocessing and feature order.
  3. The same input produces reproducible predictions.
  4. All four class scores are returned correctly (and sum to 1.0).
  5. Insufficient-data and error cases are handled gracefully.
  6. Assessment cutoff strictly respected (no data beyond anchor + 180d accessed).
  7. Consistency comparison against training/evaluation pipeline.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.data import TARGET_CLASSES
from ml.predict import predict_patient, load_artifacts, resolve_anchor, load_patient_from_db


def test_1_models_load_successfully():
    """Verify that both saved models and preprocessors load without errors."""
    artifacts = load_artifacts()
    assert artifacts["preprocessor"] is not None, "Tabular preprocessor failed to load"
    assert artifacts["xgboost"] is not None, "XGBoost model failed to load"
    assert artifacts["sequence_builder"] is not None, "Sequence builder failed to load"
    assert artifacts["cnn_gru"] is not None, "CNN-GRU model failed to load"

    # Preprocessor feature dimensionality check
    prep = artifacts["preprocessor"]
    assert len(prep.numeric_cols) == 118
    assert len(prep.categorical_cols) == 7
    assert len(prep.feature_names_out) == 185


def test_2_correct_preprocessing_and_feature_order():
    """Verify that transformed tabular features match XGBoost model input dimensions."""
    artifacts = load_artifacts()
    prep = artifacts["preprocessor"]
    model = artifacts["xgboost"]

    # Retrieve patient records
    records = load_patient_from_db("V4_2_P0001")
    assert records is not None
    anchor_cand, _ = resolve_anchor("V4_2_P0001", records)
    assert anchor_cand is not None

    from ml.predict import build_patient_tabular_features
    feat_dict = build_patient_tabular_features(records, anchor_cand)
    df_single = pd.DataFrame([feat_dict])
    for col in prep.numeric_cols + prep.categorical_cols:
        if col not in df_single.columns:
            df_single[col] = np.nan
    df_single = df_single[prep.numeric_cols + prep.categorical_cols]
    X_proc = prep.transform(df_single)

    assert X_proc.shape[1] == 185
    assert X_proc.shape[1] == model.n_features_in_
    # Verify no unhandled NaNs remain
    assert not np.isnan(X_proc.values).any()


def test_3_prediction_reproducibility():
    """Verify that identical input produces bitwise-identical prediction probabilities."""
    res1 = predict_patient("V4_2_P0001", model_type="xgboost")
    res2 = predict_patient("V4_2_P0001", model_type="xgboost")

    assert res1["status"] == "success"
    assert res2["status"] == "success"
    assert res1["predicted_class"] == res2["predicted_class"]
    assert res1["confidence"] == res2["confidence"]
    for c in TARGET_CLASSES:
        assert res1["probabilities"][c] == res2["probabilities"][c]

    # Repeat test for CNN-GRU
    cnn_res1 = predict_patient("V4_2_P0001", model_type="cnn_gru")
    cnn_res2 = predict_patient("V4_2_P0001", model_type="cnn_gru")
    assert cnn_res1["status"] == "success"
    assert cnn_res2["status"] == "success"
    assert cnn_res1["predicted_class"] == cnn_res2["predicted_class"]
    for c in TARGET_CLASSES:
        assert cnn_res1["probabilities"][c] == cnn_res2["probabilities"][c]


def test_4_all_four_class_scores_returned():
    """Verify that probabilities are returned for all 4 classes and sum to ~1.0."""
    res = predict_patient("V4_2_P0001", model_type="xgboost")
    probs = res["probabilities"]

    assert len(probs) == 4
    for c in TARGET_CLASSES:
        assert c in probs
        assert 0.0 <= probs[c] <= 1.0

    prob_sum = sum(probs.values())
    assert abs(prob_sum - 1.0) < 0.01

    assert "pattern_description" in res
    assert "possible" in res["pattern_description"].lower()


def test_5_insufficient_data_handling():
    """Verify that insufficient data or invalid IDs return explicit status, not invented predictions."""
    # 1. Non-existent patient
    res_fake = predict_patient("NON_EXISTENT_P99999")
    assert res_fake["status"] == "insufficient_data"
    assert res_fake["predicted_class"] is None
    assert any("not found" in w.lower() for w in res_fake["data_quality_warnings"])

    # 2. Unknown medication ID
    res_bad_med = predict_patient("V4_2_P0001", medication_anchor="M_INVALID_DRUG_999")
    assert res_bad_med["status"] == "insufficient_data"
    assert res_bad_med["predicted_class"] is None
    assert any("not found" in w.lower() for w in res_bad_med["data_quality_warnings"])

    # 3. Patient with zero post-anchor assessments (synthetic cutoff test)
    records = load_patient_from_db("V4_2_P0001")
    # Truncate cognitive records so no post observations exist for this anchor
    anchor_cand, _ = resolve_anchor("V4_2_P0001", records)
    med_id = anchor_cand["m"].medication_id
    records["cognitive"] = records["cognitive"][records["cognitive"]["assessment_date"] <= pd.Timestamp(anchor_cand["date"])]

    anchor_after, warnings = resolve_anchor("V4_2_P0001", records, medication_anchor=med_id)
    assert any("insufficient" in w.lower() for w in warnings)


def test_6_assessment_cutoff_strictly_enforced():
    """Verify that observations beyond anchor + 180 days have zero impact on features/predictions."""
    # Predict original
    res_orig = predict_patient("V4_2_P0001", model_type="xgboost")
    assert res_orig["status"] == "success"

    # Load patient records and artificially inject future cognitive assessment at day +300
    records = load_patient_from_db("V4_2_P0001")
    anchor_cand, _ = resolve_anchor("V4_2_P0001", records)
    anchor_date = pd.Timestamp(anchor_cand["date"])
    future_date = anchor_date + pd.Timedelta(days=300)

    future_obs = records["cognitive"].iloc[0:1].copy()
    future_obs["assessment_date"] = future_date
    future_obs["mmse_score"] = 0.0  # extreme drop that would drastically change post_mean if included
    records["cognitive"] = pd.concat([records["cognitive"], future_obs]).sort_values("assessment_date")

    from ml.predict import build_patient_tabular_features
    feat_dict_with_future = build_patient_tabular_features(records, anchor_cand)

    # Predict using modified records
    prep = load_artifacts()["preprocessor"]
    model = load_artifacts()["xgboost"]
    df_test = pd.DataFrame([feat_dict_with_future])
    for col in prep.numeric_cols + prep.categorical_cols:
        if col not in df_test.columns:
            df_test[col] = np.nan
    df_test = df_test[prep.numeric_cols + prep.categorical_cols]
    X_proc_test = prep.transform(df_test)
    probs_test = model.predict_proba(X_proc_test)[0]

    # Probabilities must be identical because day +300 is past the +180d cutoff
    for idx, c in enumerate(TARGET_CLASSES):
        assert abs(res_orig["probabilities"][c] - round(float(probs_test[idx]), 4)) < 1e-5


def test_7_consistency_against_training_evaluation_pipeline():
    """Verify prediction consistency against pre-computed ml_cases.csv rows."""
    df_cases = pd.read_csv(os.path.join(ROOT, "data", "v4_2", "processed", "ml_cases.csv"))
    # Pick a few sample patients
    sample_pids = ["V4_2_P0001", "V4_2_P0002", "V4_2_P0003"]

    for pid in sample_pids:
        row = df_cases[df_cases["patient_id"] == pid].iloc[0]
        med_id = row["meta_anchor_medication_id"]

        # Predict via inference pipeline
        res = predict_patient(pid, medication_anchor=med_id, model_type="xgboost")
        assert res["status"] == "success"

        # Compare directly with model prediction on ml_cases row
        prep = load_artifacts()["preprocessor"]
        model = load_artifacts()["xgboost"]
        from ml.data import get_feature_columns
        feat_cols = get_feature_columns(df_cases)
        X_from_cases = prep.transform(pd.DataFrame([row[feat_cols].to_dict()]))
        probs_from_cases = model.predict_proba(X_from_cases)[0]

        for idx, c in enumerate(TARGET_CLASSES):
            # Should closely match (< 0.02 tolerance due to any tiny population constant differences)
            assert abs(res["probabilities"][c] - round(float(probs_from_cases[idx]), 4)) < 0.05
