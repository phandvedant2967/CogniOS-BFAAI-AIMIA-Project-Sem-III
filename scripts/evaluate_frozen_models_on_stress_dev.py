"""Evaluate frozen CNN-GRU and XGBoost models on the stress-test development cohort.

Reads:
  - data/stress_test/processed/ml_cases.csv (dev split)
  - data/stress_test/processed/patient_observations.csv
  - data/stress_test/processed/medication_episodes.csv
  - models/longitudinal_cnn_gru.pt
  - models/baseline_xgboost.pkl & models/preprocessing.pkl

Reports:
  - Overall accuracy, macro/weighted F1, ROC-AUC
  - Per-class metrics (precision, recall, F1)
  - Confusion matrices
  - Performance delta vs original V2 test set
"""

import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.data import TARGET_CLASSES, get_feature_columns
from ml.sequence_interface import SequenceFeatureBuilder, SequenceDataset
from ml.sequence_model import LongitudinalCNNGRU
from ml.evaluation import evaluate_predictions
from ml.utils import MODELS_DIR, RESULTS_DIR, save_json

STRESS_PROC = os.path.join(ROOT, "data", "stress_test", "processed")


def evaluate_frozen_xgboost(split_name="dev"):
    """Evaluate frozen Stage 3A XGBoost model on stress test split."""
    cases_path = os.path.join(STRESS_PROC, "ml_cases.csv")
    df = pd.read_csv(cases_path)
    sub = df[(df.split == split_name) & (df.meta_usable_case == 1)].copy()

    feat_cols = get_feature_columns(sub)
    X = sub[feat_cols]
    y_true = sub["target"].tolist()

    import joblib
    prep_path = os.path.join(MODELS_DIR, "preprocessing.pkl")
    preprocessor = joblib.load(prep_path)

    xgb_path = os.path.join(MODELS_DIR, "baseline_xgboost.pkl")
    xgb_wrapper = joblib.load(xgb_path)

    # Align features to preprocessor expected feature columns
    expected_cols = getattr(preprocessor, "feature_cols", None)
    if expected_cols is None and hasattr(preprocessor.transformer, "feature_names_in_"):
        expected_cols = preprocessor.transformer.feature_names_in_

    if expected_cols is not None:
        X_aligned = pd.DataFrame(0.0, index=sub.index, columns=expected_cols)
        for col in expected_cols:
            if col in sub.columns:
                X_aligned[col] = sub[col]
        X = X_aligned

    # Transform using frozen pipeline
    X_proc = preprocessor.transform(X)

    # Predict
    y_pred = xgb_wrapper.predict(X_proc).tolist()
    y_prob = xgb_wrapper.predict_proba(X_proc)

    metrics = evaluate_predictions(y_true, y_pred, y_prob=y_prob, model_name=f"frozen_xgboost_{split_name}")
    return metrics


def evaluate_frozen_cnn_gru(split_name="dev"):
    """Evaluate frozen Stage 4 CNN-GRU model on stress test split using saved normalizers."""
    ckpt_path = os.path.join(MODELS_DIR, "longitudinal_cnn_gru.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Build raw records for stress test
    builder = SequenceFeatureBuilder(proc_dir=STRESS_PROC)
    raw_records = builder.build_raw_patient_sequences()

    # Apply frozen normalizers and medians from checkpoint
    feature_names = ckpt["feature_names"]
    scaler_params = ckpt["scaler_params"]
    train_medians = ckpt["train_medians"]
    input_dim = ckpt["input_dim"]

    filtered_records = [r for r in raw_records if r["split"] == split_name]

    transformed_seqs = []
    lengths = []
    targets = []
    patient_ids = []

    for r in filtered_records:
        steps_arr = np.array(r["steps"], dtype=np.float32)

        # Impute and standardize using frozen V2 training parameters
        for col_idx, col_name in enumerate(feature_names):
            col_vals = steps_arr[:, col_idx]
            nan_mask = np.isnan(col_vals)
            if np.any(nan_mask):
                col_vals[nan_mask] = train_medians.get(col_name, 0.0)

            if col_name in scaler_params:
                mean_val, std_val = scaler_params[col_name]
                col_vals = (col_vals - mean_val) / std_val

            steps_arr[:, col_idx] = col_vals

        actual_len = min(len(steps_arr), builder.max_seq_len)
        padded = np.zeros((builder.max_seq_len, len(feature_names)), dtype=np.float32)
        padded[:actual_len, :] = steps_arr[:actual_len, :]

        transformed_seqs.append(padded)
        lengths.append(actual_len)
        targets.append(r["target"])
        patient_ids.append(r["patient_id"])

    seqs_tensor = torch.tensor(np.stack(transformed_seqs), dtype=torch.float32)
    lens_tensor = torch.tensor(lengths, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(seqs_tensor, lens_tensor, targets_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Initialize model with checkpoint weights
    model = LongitudinalCNNGRU(
        input_dim=input_dim,
        conv_dim=64,
        gru_dim=64,
        num_classes=4,
        dropout=0.2,
        kernel_size=3
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    with torch.no_grad():
        for x_b, len_b, y_b in loader:
            x_b = x_b.to(device)
            len_b = len_b.to(device)
            logits = model(x_b, len_b)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            y_true_all.extend(y_b.numpy().tolist())
            y_pred_all.extend(preds.cpu().numpy().tolist())
            y_prob_all.extend(probs.cpu().numpy().tolist())

    y_true_named = [TARGET_CLASSES[i] for i in y_true_all]
    y_pred_named = [TARGET_CLASSES[i] for i in y_pred_all]

    metrics = evaluate_predictions(y_true_named, y_pred_named, y_prob=y_prob_all, model_name=f"frozen_cnn_gru_{split_name}")
    return metrics


def main():
    print("=== Evaluating Frozen Models on Stress-Test Development Cohort (split='dev') ===")

    # 1. XGBoost
    print("\n--- 1. Evaluating Frozen XGBoost Baseline ---")
    xgb_metrics = evaluate_frozen_xgboost(split_name="dev")
    print(f"XGBoost Dev Accuracy: {xgb_metrics['accuracy']:.4f}")
    print(f"XGBoost Dev Macro F1: {xgb_metrics['macro_f1']:.4f}")
    print("Per-class F1:")
    for cls in TARGET_CLASSES:
        print(f"  {cls}: {xgb_metrics['per_class'][cls]['f1']:.4f}")

    # 2. CNN-GRU
    print("\n--- 2. Evaluating Frozen 1D CNN + GRU ---")
    cnn_metrics = evaluate_frozen_cnn_gru(split_name="dev")
    print(f"CNN-GRU Dev Accuracy: {cnn_metrics['accuracy']:.4f}")
    print(f"CNN-GRU Dev Macro F1: {cnn_metrics['macro_f1']:.4f}")
    print("Per-class F1:")
    for cls in TARGET_CLASSES:
        print(f"  {cls}: {cnn_metrics['per_class'][cls]['f1']:.4f}")

    # 3. Load original V2 test metrics for direct delta calculation
    v2_metrics_path = os.path.join(RESULTS_DIR, "sequence_metrics.json")
    with open(v2_metrics_path, "r", encoding="utf-8") as f:
        v2_bundle = json.load(f)

    v2_cnn_test = v2_bundle["sequence_model"]
    v2_xgb_test = v2_bundle["baseline_xgboost"]

    summary = {
        "evaluation_split": "dev",
        "cohort_size": 600,
        "frozen_xgboost_dev": xgb_metrics,
        "frozen_cnn_gru_dev": cnn_metrics,
        "v2_test_baseline": {
            "xgboost": v2_xgb_test,
            "cnn_gru": v2_cnn_test,
        },
        "deltas": {
            "xgboost_macro_f1_drop": round(xgb_metrics["macro_f1"] - v2_xgb_test["macro_f1"], 4),
            "cnn_gru_macro_f1_drop": round(cnn_metrics["macro_f1"] - v2_cnn_test["macro_f1"], 4),
        }
    }

    out_json = os.path.join(RESULTS_DIR, "stress_test_dev_evaluation.json")
    save_json(summary, out_json)
    print(f"\nSaved stress-test development evaluation -> {out_json}")
    print(f"CNN-GRU Macro-F1 Delta: {summary['deltas']['cnn_gru_macro_f1_drop']:+.4f} (from {v2_cnn_test['macro_f1']:.4f} to {cnn_metrics['macro_f1']:.4f})")
    print(f"XGBoost Macro-F1 Delta: {summary['deltas']['xgboost_macro_f1_drop']:+.4f} (from {v2_xgb_test['macro_f1']:.4f} to {xgb_metrics['macro_f1']:.4f})")


if __name__ == "__main__":
    main()
