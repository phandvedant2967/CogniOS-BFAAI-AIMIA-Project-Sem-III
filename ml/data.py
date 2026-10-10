"""Dataset loading, verification, feature manifests, and split isolation."""

import os
import pandas as pd
from ml.utils import DATA_V2_PROC, RESULTS_DIR, save_json

TARGET_COL = "target"
SPLIT_COL = "split"
PATIENT_ID_COL = "patient_id"
CASE_ID_COL = "case_id"

TARGET_CLASSES = ["medication_associated", "disease_progression", "other_cause", "uncertain"]

FORBIDDEN_KEYWORDS = [
    "trajectory_context", "scenario_type", "expected_temporal_pattern",
    "synthetic_truth_confidence", "doctor", "associated_medication",
    "alternative_clinical_cause", "gt_", "ground_truth"
]


def load_dataset(filepath=None):
    """Load ml_cases.csv dataset and run automatic verification checks."""
    path = filepath or os.path.join(DATA_V2_PROC, "ml_cases.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)
    verify_splits_and_patients(df)
    return df


def verify_splits_and_patients(df):
    """Verify patient-level isolation across splits and check class distributions."""
    train_pts = set(df[df[SPLIT_COL] == "train"][PATIENT_ID_COL])
    val_pts = set(df[df[SPLIT_COL] == "val"][PATIENT_ID_COL])
    test_pts = set(df[df[SPLIT_COL] == "test"][PATIENT_ID_COL])

    assert len(train_pts.intersection(val_pts)) == 0, "Leakage: Train and Validation share patients!"
    assert len(train_pts.intersection(test_pts)) == 0, "Leakage: Train and Test share patients!"
    assert len(val_pts.intersection(test_pts)) == 0, "Leakage: Validation and Test share patients!"

    # Filter to usable cases if usable flag exists
    if "meta_usable_case" in df.columns:
        df_usable = df[df.meta_usable_case == 1]
    else:
        df_usable = df

    print(f"Dataset verification passed: {len(df)} total cases, {len(df_usable)} usable cases.")
    print("Class distribution by split:")
    ct = pd.crosstab(df_usable[TARGET_COL], df_usable[SPLIT_COL]).reindex(TARGET_CLASSES)
    print(ct.to_string())


def get_feature_columns(df):
    """Identify legitimate model features (all feat_* columns, excluding forbidden)."""
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    bad_cols = [c for c in feat_cols if any(f in c.lower() for f in FORBIDDEN_KEYWORDS)]
    if bad_cols:
        raise ValueError(f"Forbidden feature(s) detected: {bad_cols}")
    return sorted(feat_cols)


def extract_split_data(df, split_name):
    """Extract X, y, case_ids, patient_ids for a given split (train, val, test)."""
    if "meta_usable_case" in df.columns:
        sub = df[(df[SPLIT_COL] == split_name) & (df["meta_usable_case"] == 1)].copy()
    else:
        sub = df[df[SPLIT_COL] == split_name].copy()

    feat_cols = get_feature_columns(sub)
    X = sub[feat_cols]
    y = sub[TARGET_COL]
    meta = sub[[CASE_ID_COL, PATIENT_ID_COL, SPLIT_COL]]
    return X, y, meta, feat_cols


def categorize_feature_groups(feature_cols):
    """Group feature columns into 8 defined domain categories."""
    groups = {
        "cognitive_trajectory": [],
        "symptom": [],
        "timing": [],
        "medication": [],
        "co_medication": [],
        "clinical_event": [],
        "patient_disease": [],
        "drug_knowledge": [],
    }

    for col in feature_cols:
        if any(s in col for s in ["mmse_", "memory_", "attention_", "executive_", "mood_"]) and not col.startswith("feat_drug_"):
            if "rate" in col:
                groups["symptom"].append(col)
            else:
                groups["cognitive_trajectory"].append(col)
        elif any(s in col for s in ["confusion_", "memory_loss_", "attention_prob_", "behavioral_chg_"]):
            groups["symptom"].append(col)
        elif col in ["feat_days_last_pre_to_anchor", "feat_days_anchor_to_first_post"]:
            groups["timing"].append(col)
        elif col.startswith("feat_anchor_") or col in [
            "feat_dose_change_ratio", "feat_days_on_drug_at_anchor",
            "feat_stopped_in_post_window", "feat_days_anchor_to_stop",
            "feat_mmse_change_after_stop", "feat_later_dose_change_in_post_window",
            "feat_days_anchor_to_later_dose_change"
        ]:
            groups["medication"].append(col)
        elif col.startswith("feat_other_") or col.startswith("feat_competing_"):
            groups["co_medication"].append(col)
        elif col.startswith("feat_ev_"):
            groups["clinical_event"].append(col)
        elif col.startswith("feat_age") or col.startswith("feat_sex") or col.startswith("feat_primary_") or col.startswith("feat_disease_") or col.startswith("feat_comorb_"):
            groups["patient_disease"].append(col)
        elif col.startswith("feat_drug_") or col == "feat_knowledge_concordant_decline":
            groups["drug_knowledge"].append(col)
        else:
            # Fallback based on prefix/suffix
            groups["cognitive_trajectory"].append(col)

    return groups


def save_feature_manifest(feature_cols, filepath=None):
    """Save feature manifest JSON file with feature list and group categorizations."""
    path = filepath or os.path.join(RESULTS_DIR, "feature_manifest.json")
    groups = categorize_feature_groups(feature_cols)
    manifest = {
        "total_features": len(feature_cols),
        "features": feature_cols,
        "feature_groups": groups,
        "group_counts": {k: len(v) for k, v in groups.items()}
    }
    save_json(manifest, path)
    print(f"Saved feature manifest ({len(feature_cols)} features across {len(groups)} groups) -> {path}")
    return manifest
