"""Case-level misclassification analyzer and confusion pattern diagnosis."""

import os
import numpy as np
import pandas as pd
from ml.data import TARGET_CLASSES
from ml.utils import RESULTS_DIR, save_json


def analyze_misclassifications(y_true, y_pred, y_prob, meta_df, X_raw, out_dir=RESULTS_DIR):
    """Analyze misclassified cases and extract representative difficult case IDs."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    mis_mask = (y_true != y_pred)
    n_mis = int(np.sum(mis_mask))
    n_total = len(y_true)

    print(f"\n=======================================================")
    print(f"=== CASE-LEVEL ERROR ANALYSIS ({n_mis}/{n_total} misclassified = {n_mis/n_total:.1%}) ===")
    print(f"=======================================================")

    df_err = meta_df[mis_mask].copy()
    df_err["true_class"] = y_true[mis_mask]
    df_err["pred_class"] = y_pred[mis_mask]

    # Probabilities per class
    for idx, cls_name in enumerate(TARGET_CLASSES):
        df_err[f"prob_{cls_name}"] = y_prob[mis_mask, idx]

    # Extract key cognitive trajectory features for error inspection
    key_features = ["feat_mmse_delta", "feat_mmse_post_mean", "feat_knowledge_concordant_decline", "feat_ev_post_max_severity"]
    for kf in key_features:
        if kf in X_raw.columns:
            df_err[kf] = X_raw.loc[mis_mask, kf].values

    # Specific confusion breakdowns
    med_vs_prog = df_err[
        ((df_err.true_class == "medication_associated") & (df_err.pred_class == "disease_progression")) |
        ((df_err.true_class == "disease_progression") & (df_err.pred_class == "medication_associated"))
    ]

    unc_errors = df_err[df_err.true_class == "uncertain"]
    other_errors = df_err[df_err.true_class == "other_cause"]

    summary = {
        "total_test_cases": n_total,
        "total_misclassified": n_mis,
        "error_rate": round(n_mis / n_total, 4),
        "med_vs_prog_confusion_count": len(med_vs_prog),
        "uncertain_error_count": len(unc_errors),
        "other_cause_error_count": len(other_errors),
        "representative_misclassified_case_ids": df_err["case_id"].head(20).tolist()
    }

    save_json(summary, os.path.join(out_dir, "error_analysis_summary.json"))

    err_csv_path = os.path.join(out_dir, "misclassified_cases.csv")
    df_err.to_csv(err_csv_path, index=False)
    print(f"Saved misclassified cases inspection file -> {err_csv_path}")

    print(f"Confusion Breakdown:")
    print(f"  - medication_associated <-> disease_progression confusion: {len(med_vs_prog)} cases")
    print(f"  - uncertain cases misclassified: {len(unc_errors)} cases")
    print(f"  - other_cause cases misclassified: {len(other_errors)} cases")

    return summary, df_err
