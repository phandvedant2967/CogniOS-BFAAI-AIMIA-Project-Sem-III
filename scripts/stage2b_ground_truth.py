"""Stage 2B: attach ground-truth cases, write labels CSV, and audit for leakage.

Read-only on the database and on data/raw/*. No model is trained, nothing is fixed.
Usage: python scripts/stage2b_ground_truth.py
Writes: data/processed/ground_truth_labels.csv
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402

import argparse

parser = argparse.ArgumentParser(description="Stage 2B Ground Truth Labels")
parser.add_argument("--db-path", default=config.DB_PATH, help="Path to SQLite database")
parser.add_argument("--gt-path", default=os.path.join(ROOT, "data", "raw", "dataset6_Ground_Truth_Cases.xlsx"), help="Path to Dataset 6 Excel")
parser.add_argument("--out-path", default=os.path.join(ROOT, "data", "processed", "ground_truth_labels.csv"), help="Path to output labels CSV")
args, _ = parser.parse_known_args()

GT_PATH = args.gt_path
OUT = args.out_path
CLASS_MAP = {"Medication-associated": "medication_associated", "Disease progression": "disease_progression",
             "Other clinical cause": "other_cause", "Uncertain": "uncertain"}
ORDER = list(CLASS_MAP.values())

conn = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
q = lambda s, **k: pd.read_sql(s, conn, **k)  # noqa: E731
pat = q("SELECT * FROM patients")
mh = q("SELECT m.*, d.drug_name, d.therapeutic_area FROM medication_history m "
       "LEFT JOIN drugs d USING(medication_id)", parse_dates=["start_date"])
cog = q("SELECT * FROM cognitive_behavioral", parse_dates=["assessment_date"])
ev = q("SELECT * FROM clinical_events")
gt = pd.read_excel(GT_PATH, sheet_name="ground_truth_cases")
gt["label"] = gt.ground_truth_category.map(CLASS_MAP)
# trajectory_context stores the text "None" for missing
cog["trajectory_context"] = cog.trajectory_context.replace("None", np.nan)

print("=== 1. MATCHING ===")
print("cases:", len(gt), "| unique patients:", gt.patient_id.nunique())
gt["patient_found"] = gt.patient_id.isin(pat.patient_id)
gt["has_cognitive_data"] = gt.patient_id.isin(cog.patient_id)
gt["has_medication_data"] = gt.patient_id.isin(mh.patient_id)
key = mh[["patient_id", "medication_id", "drug_name"]]
m = gt.merge(key, left_on=["patient_id", "associated_medication_id"],
             right_on=["patient_id", "medication_id"], how="left")
m = m.drop_duplicates("ground_truth_case_id")
gt["assoc_med_in_patient_history"] = np.where(gt.associated_medication_id.isna(), np.nan,
                                              gt.ground_truth_case_id.isin(m[m.medication_id.notna()].ground_truth_case_id))
gt["assoc_drug_name_matches"] = np.where(gt.associated_medication_id.isna(), np.nan,
                                         (m.set_index("ground_truth_case_id").drug_name
                                          .reindex(gt.ground_truth_case_id).values == gt.associated_drug.values))
evt = ev.groupby("patient_id").event_type.apply(set)
gt["alt_cause_in_patient_events"] = [np.nan if pd.isna(c) else (c in evt.get(p, set()))
                                     for p, c in zip(gt.patient_id, gt.alternative_clinical_cause)]
for c in ["patient_found", "has_cognitive_data", "has_medication_data", "assoc_med_in_patient_history",
          "assoc_drug_name_matches", "alt_cause_in_patient_events"]:
    print(f"{c}: {gt[c].dropna().astype(bool).sum()} of {gt[c].notna().sum()}")

# labels file: id, label, split, match flags. Label-describing text columns are deliberately left out;
# the associated medication id is kept only for later medication-level work (NOT a feature).
out = gt[["ground_truth_case_id", "patient_id", "label", "evaluation_split",
          "associated_medication_id", "patient_found", "has_cognitive_data", "has_medication_data",
          "assoc_med_in_patient_history", "assoc_drug_name_matches", "alt_cause_in_patient_events"]]
out = out.rename(columns={"evaluation_split": "split", "associated_medication_id": "gt_associated_medication_id"})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
out.to_csv(OUT, index=False)
print("written", OUT)

print("\n=== 2. CLASS COUNTS ===")
print(gt.label.value_counts().reindex(ORDER).to_string())
print(pd.crosstab(gt.label, gt.evaluation_split).reindex(ORDER).to_string())

print("\n=== 3. LEAKAGE: ground-truth file columns ===")
for c in ["scenario_type", "expected_temporal_pattern", "associated_medication_id",
          "alternative_clinical_cause"]:
    x = gt[c].fillna("<missing>")
    pure = (pd.crosstab(x, gt.label).max(axis=1).sum()) / len(gt)
    print(f"{c}: predicts class with {pure:.1%} accuracy by lookup alone")
print(gt.groupby("label").synthetic_truth_confidence.agg(["min", "max", "mean"]).reindex(ORDER).round(3).to_string())

print("\n=== 3b. LEAKAGE: trajectory_context (row level) ===")
t = cog.merge(gt[["patient_id", "label"]], on="patient_id")
tc = pd.crosstab(t.trajectory_context.fillna("<none>"), t.label)[ORDER]
print(tc.to_string())
print("share of rows with a context value:\n", (t.trajectory_context.notna().groupby(t.label).mean()).reindex(ORDER).round(3).to_string())
anyctx = t.groupby("patient_id").agg(label=("label", "first"), any_ctx=("trajectory_context", lambda x: x.notna().any()))
print("share of patients with >=1 context value:\n", anyctx.groupby("label").any_ctx.mean().reindex(ORDER).round(3).to_string())

print("\n=== 3c. LEAKAGE: EHR-side shortcuts ===")
per = pd.DataFrame({"label": gt.set_index("patient_id").label})
per["n_cog"] = cog.groupby("patient_id").size()
per["n_events"] = ev.groupby("patient_id").size()
per["n_events"] = per.n_events.fillna(0)
per["n_meds"] = mh.groupby("patient_id").size()
per["span_days"] = cog.groupby("patient_id").assessment_date.agg(lambda x: (x.max() - x.min()).days)
print(per.groupby("label").agg(n_cog=("n_cog", "mean"), n_events=("n_events", "mean"), zero_events=("n_events", lambda x: (x == 0).mean()),
                               n_meds=("n_meds", "mean"), span_days=("span_days", "mean")).reindex(ORDER).round(2).to_string())
print("events per patient by class:\n", pd.crosstab(per.label, per.n_events.astype(int)).reindex(ORDER).to_string())
e = ev.merge(gt[["patient_id", "label"]], on="patient_id")
print("clinical_note x class (rows):")
print(pd.crosstab(e.clinical_note, e.label).reindex(columns=ORDER).to_string())
print("therapeutic area of medication records, column %:")
mm = mh.merge(gt[["patient_id", "label"]], on="patient_id")
print((pd.crosstab(mm.therapeutic_area, mm.label, normalize="columns")[ORDER] * 100).round(1).to_string())

print("\n=== 3d. LEAKAGE: pre/post observations around an anchor medication ===")
print("anchor = associated medication for medication cases; one random medication (seed 0) for other classes")
rng = np.random.default_rng(0)
rows = []
cg = {p: g for p, g in cog.groupby("patient_id")}
assoc = gt.set_index("patient_id").associated_medication_id
for p, g in mh.dropna(subset=["start_date"]).groupby("patient_id"):
    a = assoc.get(p)
    if pd.notna(a):
        r = g[g.medication_id == a].iloc[0]
    else:
        r = g.iloc[rng.integers(len(g))]
    x = cg.get(p)
    if x is None:
        continue
    pre, post = x[x.assessment_date < r.start_date], x[x.assessment_date >= r.start_date]
    rows.append(dict(label=per.label[p], n_pre=len(pre), n_post=len(post), mmse_pre=pre.mmse_score.mean(),
                     mmse_post=post.mmse_score.mean(), first_obs_before_anchor=len(pre) > 0,
                     days_anchor_to_first_obs=(x.assessment_date.min() - r.start_date).days))
pp = pd.DataFrame(rows)
pp["mmse_change"] = pp.mmse_post - pp.mmse_pre
print(pp.groupby("label").agg(n=("n_pre", "size"), n_pre=("n_pre", "mean"), n_post=("n_post", "mean"),
                              has_pre=("first_obs_before_anchor", "mean"), mmse_pre=("mmse_pre", "mean"),
                              mmse_post=("mmse_post", "mean"), mmse_change=("mmse_change", "mean"),
                              days_anchor_to_first_obs=("days_anchor_to_first_obs", "median")).reindex(ORDER).round(2).to_string())
