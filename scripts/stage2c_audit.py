"""Stage 2C audit of data/processed/ml_cases.csv. Read-only; no model is fitted.

Labels / ground-truth drug ids are used here ONLY to measure shortcuts, never to build features.
Univariate screens use the TRAIN split only. Usage: python scripts/stage2c_audit.py [ml_cases.csv]
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402

import argparse

parser = argparse.ArgumentParser(description="Stage 2C Audit")
parser.add_argument("--cases-path", default=os.path.join(ROOT, "data", "processed", "ml_cases.csv"), help="Path to ml_cases.csv")
parser.add_argument("--labels-path", default=os.path.join(ROOT, "data", "processed", "ground_truth_labels.csv"), help="Path to ground_truth_labels.csv")
parser.add_argument("--db-path", default=config.DB_PATH, help="Path to SQLite database")
args, _ = parser.parse_known_args()

cases_path = args.cases_path
labels_path = args.labels_path
db_path = args.db_path

ORDER = ["medication_associated", "disease_progression", "other_cause", "uncertain"]
pd.set_option("display.width", 220, "display.max_columns", 30)
df = pd.read_csv(cases_path)
u = df[df.meta_usable_case == 1].copy()
gtl = pd.read_csv(labels_path).set_index("patient_id")
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
mh = pd.read_sql("SELECT m.patient_id, m.medication_id, d.drug_name, d.drug_class, d.therapeutic_area "
                 "FROM medication_history m JOIN drugs d USING(medication_id)", conn)
feat = [c for c in df.columns if c.startswith("feat_")]


def cramers(ct):
    chi = chi2_contingency(ct)[0]
    return float(np.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1))))


def purity(x, y):
    ct = pd.crosstab(x, y)
    return ct.max(axis=1).sum() / ct.values.sum()


print("=== A. USABLE CASES ===")
print(pd.crosstab(df.target, df.meta_anchor_status, margins=True).reindex(ORDER + ["All"]))
print((df.groupby("target").meta_usable_case.mean().reindex(ORDER) * 100).round(1).to_string())
print(pd.crosstab(u.target, u.split).reindex(ORDER))

print("\n=== B. ANCHOR PROPERTIES BY CLASS (usable) ===")
print(u.groupby("target").agg(dose_change_anchor=("feat_anchor_is_dose_change", "mean"),
                              n_eligible=("meta_n_eligible_anchors", "mean"),
                              n_pre=("meta_n_pre_assessments", "mean"), n_post=("meta_n_post_assessments", "mean"),
                              days_enroll_to_anchor=("meta_days_enrollment_to_anchor", "mean"),
                              days_pre_to_anchor=("feat_days_last_pre_to_anchor", "mean"),
                              days_anchor_to_post=("feat_days_anchor_to_first_post", "mean")).reindex(ORDER).round(2))
u["gt_med"] = u.patient_id.map(gtl.gt_associated_medication_id)
m = u[u.target == "medication_associated"]
print(f"AUDIT anchor drug == ground-truth associated drug (medication cases, usable): "
      f"{(m.meta_anchor_medication_id == m.gt_med).mean():.3f}  ({int((m.meta_anchor_medication_id == m.gt_med).sum())}/{len(m)})")
allm = df[df.target == "medication_associated"]
has = mh.merge(gtl.gt_associated_medication_id.rename("gt"), left_on="patient_id", right_index=True)
print("all 1500 medication cases: GT drug is a candidate anchor in usable set -> see hit-rate above (ceiling ~ 49%)")

print("\n=== C. DRUG DISTRIBUTION ACROSS CLASSES ===")
rng = np.random.default_rng(0)
for col in ["meta_anchor_drug_name", "meta_anchor_drug_class", "meta_anchor_therapeutic_area"]:
    ct = pd.crosstab(u[col], u.target)
    null = np.mean([purity(u[col], rng.permutation(u.target.values)) for _ in range(100)])
    print(f"{col}: CramerV={cramers(ct):.3f}  lookup-purity={purity(u[col], u.target):.3f}  "
          f"permutation-null={null:.3f}  majority-class={u.target.value_counts(normalize=True).max():.3f}")
ct = pd.crosstab(u.meta_anchor_drug_name, u.target)[ORDER]
share = ct.div(ct.sum(axis=1), axis=0) * 100
share["n"] = ct.sum(axis=1)
print("\nAnchor drug: row % by class (overall: " + ", ".join(f"{k}={v*100:.0f}" for k, v in u.target.value_counts(normalize=True)[ORDER].items()) + ")")
print(share.round(0).astype({"n": int}).sort_values("medication_associated", ascending=False).to_string())
ta = pd.crosstab(u.meta_anchor_therapeutic_area, u.target, normalize="columns")[ORDER] * 100
print("\nAnchor therapeutic area, column %:\n", ta.round(1).to_string())
# patient-level drug presence (any record), independent of the anchor rule
mm = mh.merge(gtl.label.rename("target"), left_on="patient_id", right_index=True)
pres = mm.drop_duplicates(["patient_id", "therapeutic_area"])
pa = pd.crosstab(pres.therapeutic_area, pres.target)[ORDER] / df.target.value_counts()[ORDER].values * 100
print("\nShare of patients (all 5000) having >=1 drug of area, by class %:\n", pa.round(1).to_string())
# which drugs does the generator use as the "associated" drug?
g = gtl[gtl.label == "medication_associated"].gt_associated_medication_id.value_counts()
names = pd.read_sql("SELECT medication_id, drug_name, cognitive_risk_index cri FROM drugs", conn).set_index("medication_id")
prev = mh.drop_duplicates(["patient_id", "medication_id"]).medication_id.value_counts(normalize=True)
t = pd.DataFrame({"drug": names.drug_name, "cri": names.cri, "share_of_GT_assoc_%": (g / g.sum() * 100),
                  "share_of_all_med_records_%": prev * 100}).fillna(0)
t["lift"] = t["share_of_GT_assoc_%"] / t["share_of_all_med_records_%"]
print("\nGround-truth associated drug in the 1500 medication cases vs prevalence in all records:\n",
      t.sort_values("lift", ascending=False).round(1).to_string())
hi = names.index[names.cri >= 6]
print(f"AUDIT: share of GT-associated drugs with cognitive_risk_index>=6: {g[g.index.isin(hi)].sum()/g.sum()*100:.1f}% "
      f"(vs {prev[prev.index.isin(hi)].sum()*100:.1f}% of all medication records)")

print("\n=== D. UNIVARIATE SHORTCUT SCREEN (train split, usable; one-vs-rest AUC, no model fitted) ===")
tr = u[u.split == "train"]
rows = []
for c in feat:
    if not pd.api.types.is_numeric_dtype(tr[c]):
        ct = pd.crosstab(tr[c], tr.target)
        rows.append((c, "cat", cramers(ct), np.nan, None, np.nan))
        continue
    x = tr[c]
    best = (0.5, None)
    for k in ORDER:
        y = (tr.target == k).values
        ok = x.notna().values
        if y[ok].sum() < 20 or (~y[ok]).sum() < 20 or x[ok].nunique() < 2:
            continue
        r = rankdata(x[ok])
        n1, n0 = y[ok].sum(), (~y[ok]).sum()
        auc = (r[y[ok]].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        if abs(auc - .5) > abs(best[0] - .5):
            best = (auc, k)
    miss = tr.groupby("target")[c].apply(lambda s: s.isna().mean()).reindex(ORDER)
    rows.append((c, "num", np.nan, best[0], best[1], miss.max() - miss.min()))
R = pd.DataFrame(rows, columns=["feature", "type", "cramersV", "best_auc", "class", "missing_gap"])
print("numeric/binary features, top 20 by |AUC-0.5|:")
print(R[R.type == "num"].assign(dev=lambda d: (d.best_auc - .5).abs()).sort_values("dev", ascending=False)
      .head(20)[["feature", "best_auc", "class", "missing_gap"]].round(3).to_string(index=False))
print("categorical features:")
print(R[R.type == "cat"][["feature", "cramersV"]].round(3).to_string(index=False))

print("\n=== E. EVENT-BASED SHORTCUT CHECK (usable) ===")
evf = ["feat_ev_post_max_severity", "feat_ev_pre_max_severity", "feat_ev_post_any_cog_impact", "feat_ev_in_assessment_gap",
       "feat_ev_has_disease_milestone", "feat_ev_has_delirium"]
print(u.assign(any_window_event=(u.feat_ev_pre_max_severity + u.feat_ev_post_max_severity > 0) |
               u.feat_ev_nearest_abs_days.notna()).groupby("target")[evf + ["any_window_event"]].mean().reindex(ORDER).round(3))
print("\nKey cognitive change features by class (mean):")
kc = ["feat_mmse_delta", "feat_mmse_delta_early", "feat_mmse_delta_late", "feat_mmse_excess_decline",
      "feat_memory_delta", "feat_knowledge_concordant_decline", "feat_mmse_pre_slope30", "feat_mmse_post_slope30"]
print(u.groupby("target")[kc].mean().reindex(ORDER).round(3))
print("\nFeature missingness by class (%), selected:")
ms = ["feat_mmse_pre_slope30", "feat_mmse_excess_decline", "feat_mmse_days_to_first_decline", "feat_mmse_delta_late"]
print((u.groupby("target")[ms].agg(lambda s: s.isna().mean()).reindex(ORDER) * 100).round(1))
