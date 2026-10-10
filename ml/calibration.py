"""Probability calibration analysis, Brier score, and reliability curves."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ml.data import TARGET_CLASSES
from ml.utils import RESULTS_DIR


def evaluate_calibration(y_true, y_prob, out_dir=RESULTS_DIR):
    """Compute multi-class Brier score and reliability curves."""
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    y_true_ohe = np.zeros((len(y_true), len(TARGET_CLASSES)))
    for i, label in enumerate(y_true):
        y_true_ohe[i, TARGET_CLASSES.index(label)] = 1.0

    brier_scores = {}
    plt.figure(figsize=(8, 6))

    for idx, cls_name in enumerate(TARGET_CLASSES):
        prob_cls = y_prob[:, idx]
        true_cls = y_true_ohe[:, idx]

        bs = brier_score_loss(true_cls, prob_cls)
        brier_scores[cls_name] = round(float(bs), 4)

        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_cls, prob_cls, n_bins=8, strategy="uniform"
        )
        plt.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"{cls_name} (Brier={bs:.3f})"
        )

    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted probability")
    plt.title("Probability Calibration Curves (Validation Set)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(out_dir, "calibration_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    mean_brier = round(float(np.mean(list(brier_scores.values()))), 4)
    print(f"Saved calibration plot -> {plot_path}")
    print(f"Multi-class Mean Brier Score: {mean_brier:.4f}")

    return {
        "mean_brier_score": mean_brier,
        "per_class_brier": brier_scores
    }
