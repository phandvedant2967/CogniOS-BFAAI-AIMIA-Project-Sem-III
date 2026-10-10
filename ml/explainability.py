"""Feature importance extraction, domain group aggregation, and visualization."""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.data import categorize_feature_groups
from ml.utils import RESULTS_DIR


def extract_feature_importance(model_wrapper, feature_names):
    """Extract raw feature importances from trained model (LR coefficients or RF/XGB importances)."""
    raw_model = model_wrapper.model

    if hasattr(raw_model, "feature_importances_"):
        importances = raw_model.feature_importances_
    elif hasattr(raw_model, "coef_"):
        # For multi-class Logistic Regression: average absolute coefficient across classes
        importances = np.mean(np.abs(raw_model.coef_), axis=0)
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    # Normalize to sum to 1
    total = np.sum(importances)
    if total > 0:
        importances = importances / total

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return df_imp


def aggregate_by_feature_group(df_imp, feature_names):
    """Map features to their domain categories and sum importances per group."""
    groups = categorize_feature_groups(feature_names)
    feat_to_group = {}
    for gname, fcols in groups.items():
        for fc in fcols:
            feat_to_group[fc] = gname

    # Map output features (handles one-hot encoded prefixes)
    df_imp["group"] = df_imp["feature"].apply(
        lambda f: feat_to_group.get(f, feat_to_group.get(f.split("_")[0], "cognitive_trajectory"))
    )

    group_imp = df_imp.groupby("group")["importance"].sum().reset_index()
    group_imp = group_imp.sort_values("importance", ascending=False)
    return group_imp


def generate_explainability_artifacts(model_wrapper, feature_names, out_dir=RESULTS_DIR):
    """Generate feature_importance.csv and feature_importance.png."""
    os.makedirs(out_dir, exist_ok=True)

    df_imp = extract_feature_importance(model_wrapper, feature_names)
    csv_path = os.path.join(out_dir, "feature_importance.csv")
    df_imp.to_csv(csv_path, index=False)
    print(f"Saved feature importance -> {csv_path}")

    group_imp = aggregate_by_feature_group(df_imp, feature_names)

    # Create visual chart
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(data=df_imp.head(15), x="importance", y="feature", palette="viridis")
    plt.title("Top 15 Most Important Features")
    plt.xlabel("Normalized Importance")

    plt.subplot(1, 2, 2)
    sns.barplot(data=group_imp, x="importance", y="group", palette="magma")
    plt.title("Feature Importance Aggregated by Group")
    plt.xlabel("Total Share of Importance")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Saved feature importance plot -> {plot_path}")

    # Summary string answering strategic question
    top_groups = group_imp.to_dict(orient="records")
    print("\nFeature Importance by Category:")
    for g in top_groups:
        print(f"  - {g['group']:<22}: {g['importance']*100:.2f}%")

    return df_imp, group_imp
