"""Evaluation metrics calculation, confusion matrix, ROC-AUC, PR-AUC, and report generation."""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from ml.data import TARGET_CLASSES
from ml.utils import RESULTS_DIR, save_json


def evaluate_predictions(y_true, y_pred, y_prob=None, model_name="model"):
    """Compute complete classification metric dictionary."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float(accuracy_score(y_true, y_pred))

    # Macro & Weighted metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_CLASSES, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_CLASSES, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(
        y_true, y_pred, labels=TARGET_CLASSES, average=None, zero_division=0
    )

    per_class = {}
    for idx, cls_name in enumerate(TARGET_CLASSES):
        per_class[cls_name] = {
            "precision": float(p_class[idx]),
            "recall": float(r_class[idx]),
            "f1": float(f1_class[idx]),
            "support": int(support_class[idx]),
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    cm_dict = {
        cls_row: {
            cls_col: int(cm[i, j])
            for j, cls_col in enumerate(TARGET_CLASSES)
        }
        for i, cls_row in enumerate(TARGET_CLASSES)
    }

    metrics = {
        "model_name": model_name,
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "roc_auc_ovr": {},
        "pr_auc": {},
    }

    # ROC-AUC & PR-AUC if probabilities provided
    if y_prob is not None:
        y_prob = np.array(y_prob)
        # One-hot encode ground truth for multi-class OvR
        y_true_ohe = np.zeros((len(y_true), len(TARGET_CLASSES)))
        for i, label in enumerate(y_true):
            y_true_ohe[i, TARGET_CLASSES.index(label)] = 1.0

        for idx, cls_name in enumerate(TARGET_CLASSES):
            try:
                auc_val = float(roc_auc_score(y_true_ohe[:, idx], y_prob[:, idx]))
                pr_val = float(average_precision_score(y_true_ohe[:, idx], y_prob[:, idx]))
            except Exception:
                auc_val, pr_val = 0.5, 0.0
            metrics["roc_auc_ovr"][cls_name] = round(auc_val, 4)
            metrics["pr_auc"][cls_name] = round(pr_val, 4)

        try:
            metrics["macro_roc_auc"] = round(float(roc_auc_score(y_true_ohe, y_prob, multi_class="ovr", average="macro")), 4)
        except Exception:
            metrics["macro_roc_auc"] = 0.5

    return metrics


def generate_baseline_report(all_metrics, filepath=None):
    """Generate Markdown comparison report of all baseline models."""
    path = filepath or os.path.join(RESULTS_DIR, "BASELINE_REPORT.md")

    md = []
    md.append("# Baseline Models Evaluation Report\n")
    md.append("Comparing Model A (Logistic Regression), Model B (Random Forest), and Model C (XGBoost).\n")

    md.append("## 1. Overall Performance Comparison (Validation Split)\n")
    md.append("| Model Name | Accuracy | Macro F1 | Weighted F1 | Macro Precision | Macro Recall | Macro ROC-AUC |")
    md.append("|---|---|---|---|---|---|---|")

    for mname, m in all_metrics.items():
        roc_str = f"{m.get('macro_roc_auc', 0.0):.4f}"
        md.append(
            f"| **{mname}** | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['weighted_f1']:.4f} | "
            f"{m['macro_precision']:.4f} | {m['macro_recall']:.4f} | {roc_str} |"
        )

    md.append("\n## 2. Per-Class F1 Scores\n")
    md.append("| Model Name | medication_associated | disease_progression | other_cause | uncertain |")
    md.append("|---|---|---|---|---|")

    for mname, m in all_metrics.items():
        pc = m["per_class"]
        md.append(
            f"| **{mname}** | {pc['medication_associated']['f1']:.4f} | "
            f"{pc['disease_progression']['f1']:.4f} | {pc['other_cause']['f1']:.4f} | "
            f"{pc['uncertain']['f1']:.4f} |"
        )

    md.append("\n## 3. Confusion Matrix (Best Model)\n")
    best_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]["macro_f1"])
    best_cm = all_metrics[best_name]["confusion_matrix"]
    md.append(f"**Best Model:** `{best_name}` (Macro F1 = {all_metrics[best_name]['macro_f1']:.4f})\n")

    md.append("| True \\ Pred | medication_associated | disease_progression | other_cause | uncertain |")
    md.append("|---|---|---|---|---|")
    for r in TARGET_CLASSES:
        row_str = f"| **{r}** | " + " | ".join(str(best_cm[r][c]) for c in TARGET_CLASSES) + " |"
        md.append(row_str)

    content = "\n".join(md)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved baseline report -> {path}")
    return content
