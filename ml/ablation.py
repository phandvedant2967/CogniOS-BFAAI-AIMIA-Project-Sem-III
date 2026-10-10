"""Controlled Ablation Experiment Suite (Experiments A through E)."""

import os
import pandas as pd
from ml.data import categorize_feature_groups
from ml.evaluation import evaluate_predictions
from ml.models import ModelWrapper
from ml.preprocessing import PreprocessorPipeline
from ml.utils import RESULTS_DIR, save_json


def run_ablation_experiments(X_train, y_train, X_val, y_val, model_type="random_forest", seed=42):
    """Run 5 controlled ablation experiments on the exact same patient split."""
    feature_cols = list(X_train.columns)
    groups = categorize_feature_groups(feature_cols)

    # Define feature subsets for Experiments A through E
    ablation_specs = {
        "Exp_A_All_Features": feature_cols,
        "Exp_B_Cognitive_Trajectory_Only": groups["cognitive_trajectory"] + groups["symptom"] + groups["timing"],
        "Exp_C_Remove_Drug_Knowledge": [c for c in feature_cols if c not in groups["drug_knowledge"]],
        "Exp_D_Remove_Clinical_Events": [c for c in feature_cols if c not in groups["clinical_event"]],
        "Exp_E_Cognitive_Plus_Medication": groups["cognitive_trajectory"] + groups["symptom"] + groups["timing"] + groups["medication"] + groups["co_medication"],
    }

    results = {}
    print("\n=======================================================")
    print("=== RUNNING CONTROLLED ABLATION EXPERIMENTS (A-E) ===")
    print("=======================================================")

    for exp_name, cols in ablation_specs.items():
        if not cols:
            continue
        X_tr_sub = X_train[cols]
        X_va_sub = X_val[cols]

        prep = PreprocessorPipeline()
        X_tr_proc = prep.fit_transform(X_tr_sub)
        X_va_proc = prep.transform(X_va_sub)

        model = ModelWrapper(model_type=model_type, seed=seed)
        model.fit(X_tr_proc, y_train)

        preds = model.predict(X_va_proc)
        probs = model.predict_proba(X_va_proc)

        m = evaluate_predictions(y_val, preds, probs, model_name=exp_name)
        results[exp_name] = {
            "num_features": len(cols),
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "per_class_f1": {c: m["per_class"][c]["f1"] for c in m["per_class"]}
        }
        print(f"  - {exp_name:<35}: {len(cols):>3} features | Macro F1 = {m['macro_f1']:.4f} | Accuracy = {m['accuracy']:.4f}")

    out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    save_json(results, out_path)
    print(f"\nSaved ablation results -> {out_path}")
    return results
