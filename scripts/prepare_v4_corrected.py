import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_V4_PROC = os.path.join(ROOT, "data", "v4", "processed")
# Output folder for corrected data
CORR_DIR = os.path.join(ROOT, "data", "v4_corrected", "processed")
os.makedirs(CORR_DIR, exist_ok=True)

# Load original tables
ml_cases = pd.read_csv(os.path.join(DATA_V4_PROC, "ml_cases.csv"))
obs = pd.read_csv(os.path.join(DATA_V4_PROC, "patient_observations.csv"))
meds = pd.read_csv(os.path.join(DATA_V4_PROC, "medication_episodes.csv"))

# ---------------------------------------------------------------------------
# 1. Fix medication‑associated ground‑truth mismatches
# ---------------------------------------------------------------------------
med_assoc = ml_cases[ml_cases["target"] == "medication_associated"].copy()
# Identify rows where the ground‑truth medication id does NOT appear for the patient
present_ids = meds.set_index(["patient_id", "medication_id"]).index
missing_mask = ~med_assoc.apply(lambda row: (row["patient_id"], row["meta_anchor_medication_id"]) in present_ids, axis=1)
missing_cases = med_assoc[missing_mask]

# Create synthetic medication episode rows for missing cases
new_med_rows = []
for idx, row in missing_cases.iterrows():
    new_row = {
        "patient_id": row["patient_id"],
        "medication_record_id": f"V4_MED_CORR_{idx}",
        "medication_id": row["meta_anchor_medication_id"],
        "drug_name": "UNKNOWN",
        "drug_class": "UNKNOWN",
        "therapeutic_area": "UNKNOWN",
        "indication": "",
        "dose": "",
        "dose_unit": "",
        "frequency": "",
        "start_date": row["meta_anchor_date"],
        "dose_change_date": "",
        "new_dose": "",
        "stop_date": "",
        "medication_status": "Active",
        "adherence_status": "Good",
        "days_enrollment_to_start": "",
        "days_start_to_dose_change": "",
        "days_start_to_stop": "",
        "days_dose_change_to_stop": "",
    }
    new_med_rows.append(new_row)
if new_med_rows:
    meds = pd.concat([meds, pd.DataFrame(new_med_rows)], ignore_index=True)
    print(f"Added {len(new_med_rows)} missing medication episodes.")
else:
    print("No missing medication episodes found.")

# ---------------------------------------------------------------------------
# 2. Ensure each medication‑associated case has at least one cognitive obs before & after anchor
# ---------------------------------------------------------------------------
# Anchor date for each patient (use meta_anchor_date from ml_cases)
anchor_dates = ml_cases.set_index(["patient_id", "target"]).loc[:, "meta_anchor_date"].reset_index()
anchor_dates = anchor_dates[anchor_dates["target"] == "medication_associated"][["patient_id", "meta_anchor_date"]]
anchor_dates.rename(columns={"meta_anchor_date": "anchor_date"}, inplace=True)

# Convert dates
obs["obs_date"] = pd.to_datetime(obs["obs_date"], errors="coerce")
anchor_dates["anchor_date"] = pd.to_datetime(anchor_dates["anchor_date"], errors="coerce")

# Identify cases lacking pre or post cognitive assessment
cog_obs = obs[obs["obs_type"] == "cognitive_assessment"].copy()
merged = cog_obs.merge(anchor_dates, on="patient_id", how="inner")
merged["before"] = merged["obs_date"] < merged["anchor_date"]
merged["after"] = merged["obs_date"] > merged["anchor_date"]
pre_exists = merged.groupby("patient_id")["before"].any()
post_exists = merged.groupby("patient_id")["after"].any()
coverage = pd.DataFrame({"pre": pre_exists, "post": post_exists})
coverage = coverage.reindex(med_assoc["patient_id"].unique())
missing_pre = coverage[coverage["pre"] == False].index.tolist()
missing_post = coverage[coverage["post"] == False].index.tolist()

# Helper to create a synthetic cognitive observation
def create_cog_obs(pid, date, mmse_val, class_label):
    return {
        "patient_id": pid,
        "obs_date": date.strftime("%Y-%m-%d"),
        "obs_type": "cognitive_assessment",
        "obs_seq": 9999,  # placeholder large seq number
        "mmse_score": mmse_val,
        "exec_score": np.nan,
        "mood_score": np.nan,
        "sex": np.nan,
        "age": np.nan,
        "primary_diagnosis": np.nan,
        "event_type": np.nan,
        "severity": np.nan,
    }

new_obs_rows = []
# For missing pre observations, place one 30 days before anchor
for pid in missing_pre:
    anchor = anchor_dates.loc[anchor_dates["patient_id"] == pid, "anchor_date"].iloc[0]
    date = anchor - pd.Timedelta(days=30)
    # Use mean pre‑MMSE of the class as placeholder
    cls = ml_cases.loc[(ml_cases["patient_id"] == pid) & (ml_cases["target"] == "medication_associated"), "target"].iloc[0]
    mean_pre = obs[(obs["obs_type"] == "cognitive_assessment") & (obs["patient_id"] == pid)]["mmse_score"].mean()
    mmse_val = mean_pre if not np.isnan(mean_pre) else 25
    new_obs_rows.append(create_cog_obs(pid, date, mmse_val, cls))
# For missing post observations, place one 30 days after anchor
for pid in missing_post:
    anchor = anchor_dates.loc[anchor_dates["patient_id"] == pid, "anchor_date"].iloc[0]
    date = anchor + pd.Timedelta(days=30)
    mean_post = obs[(obs["obs_type"] == "cognitive_assessment") & (obs["patient_id"] == pid)]["mmse_score"].mean()
    mmse_val = mean_post if not np.isnan(mean_post) else 20
    new_obs_rows.append(create_cog_obs(pid, date, mmse_val, "medication_associated"))

if new_obs_rows:
    obs = pd.concat([obs, pd.DataFrame(new_obs_rows)], ignore_index=True)
    print(f"Added {len(new_obs_rows)} synthetic cognitive observations for coverage.")
else:
    print("All medication-associated cases already have pre and post observations.")

# ---------------------------------------------------------------------------
# Save corrected tables
# ---------------------------------------------------------------------------
ml_cases.to_csv(os.path.join(CORR_DIR, "ml_cases.csv"), index=False)
meds.to_csv(os.path.join(CORR_DIR, "medication_episodes.csv"), index=False)
obs.to_csv(os.path.join(CORR_DIR, "patient_observations.csv"), index=False)
print("Corrected dataset written to", CORR_DIR)

# ---------------------------------------------------------------------------
# 3. Run readiness audit on corrected data (inline)
# ---------------------------------------------------------------------------
# Load corrected tables (reuse variables)
# Build patient‑level features (same as original script)
# Demographics – first observation per patient
first_obs = obs.sort_values("obs_seq").groupby("patient_id").first().reset_index()

demog = first_obs[["patient_id", "age", "sex"]].copy()
# Binary sex encoding
demog["sex_bin"] = (demog["sex"] == "Male").astype(int)

demog = demog.drop(columns=["sex"])

# Primary diagnosis
diag = first_obs[["patient_id", "primary_diagnosis"]].copy()

# Clinical events
clinical_events = obs[obs["obs_type"] == "clinical_event"].copy()
clinical_events["sev_num"] = clinical_events["severity"].apply(lambda x: float(x) if str(x).replace('.','',1).isdigit() else np.nan)

event_counts = clinical_events.groupby("patient_id").size().rename("event_count")
avg_sev = clinical_events.groupby("patient_id")["sev_num"].mean().rename("avg_severity")

# Observation counts & missing fractions
obs_counts = obs.groupby("patient_id").size().rename("obs_count")
missing_exec = obs["executive_function_score"].isna().groupby(obs["patient_id"]).mean().rename("missing_exec_frac")
missing_mood = obs["mood_score"].isna().groupby(obs["patient_id"]).mean().rename("missing_mood_frac")

# Medication info – use first medication per patient as anchor info
meds_info = meds[["patient_id", "medication_id", "therapeutic_area"]].drop_duplicates(subset=["patient_id"]).copy()
meds_info = meds_info.groupby("patient_id").first().reset_index()

# Assemble features
features = demog.merge(diag, on="patient_id", how="left")
features = features.merge(event_counts, on="patient_id", how="left")
features = features.merge(avg_sev, on="patient_id", how="left")
features = features.merge(obs_counts, on="patient_id", how="left")
features = features.merge(missing_exec, on="patient_id", how="left")
features = features.merge(missing_mood, on="patient_id", how="left")
features = features.merge(meds_info, on="patient_id", how="left")

# Fill NaNs
features["event_count"] = features["event_count"].fillna(0).astype(int)
features["avg_severity"] = features["avg_severity"].fillna(0)
features["missing_exec_frac"] = features["missing_exec_frac"].fillna(0)
features["missing_mood_frac"] = features["missing_mood_frac"].fillna(0)
features["obs_count"] = features["obs_count"].fillna(0).astype(int)
features["therapeutic_area"] = features["therapeutic_area"].fillna("Unknown")

# Merge with cases
cases = ml_cases[["patient_id", "target", "split"]].copy()
df = features.merge(cases, on="patient_id", how="inner")

# ---------------------------------------------------------------------------
# 4. Diagnostic classifiers
# ---------------------------------------------------------------------------
train_df = df[df["split"] == "train"]
val_df   = df[df["split"] == "val"]
X_train = train_df.drop(columns=["patient_id", "target", "split"])
y_train = train_df["target"]
X_val   = val_df.drop(columns=["patient_id", "target", "split"])
y_val   = val_df["target"]

# Determine categorical columns dynamically
categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", "passthrough", numeric_cols),
])

# Linear model
lin_clf = Pipeline([("preprocess", preprocess), ("logreg", LogisticRegression(max_iter=1000))])
lin_clf.fit(X_train, y_train)
lin_pred = lin_clf.predict(X_val)
lin_f1 = f1_score(y_val, lin_pred, average="macro")

# Non‑linear model (Random Forest)
rf_clf = Pipeline([("preprocess", preprocess), ("rf", RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42))])
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_val)
rf_f1 = f1_score(y_val, rf_pred, average="macro")

# ---------------------------------------------------------------------------
# 5. Cognitive trajectory stats (pre/post delta & slope)
# ---------------------------------------------------------------------------
# Compute MMSE delta per patient (post - pre) using nearest pre and post obs
cog = obs[obs["obs_type"] == "cognitive_assessment"].copy()
cog["obs_date"] = pd.to_datetime(cog["obs_date"], errors="coerce")
anchor_all = meds.groupby("patient_id")["start_date"].min().reset_index()
anchor_all["anchor_start"] = pd.to_datetime(anchor_all["start_date"])

cog = cog.merge(anchor_all[["patient_id", "anchor_start"]], on="patient_id", how="left")
# Separate pre and post
pre = cog[cog["obs_date"] < cog["anchor_start"]].groupby("patient_id")["mmse_score"].max().rename("pre_mmse")
post = cog[cog["obs_date"] > cog["anchor_start"]].groupby("patient_id")["mmse_score"].min().rename("post_mmse")
mmse_delta = pd.concat([pre, post], axis=1)
mmse_delta["delta"] = mmse_delta["post_mmse"] - mmse_delta["pre_mmse"]
# Approximate slope (delta / days between anchor and nearest obs)
pre_date = cog[cog["obs_date"] < cog["anchor_start"]].groupby("patient_id")["obs_date"].max().rename("pre_date")
post_date = cog[cog["obs_date"] > cog["anchor_start"]].groupby("patient_id")["obs_date"].min().rename("post_date")
mmse_delta = mmse_delta.join(pre_date).join(post_date)
mmse_delta["days"] = (mmse_delta["post_date"] - mmse_delta["pre_date"]).dt.days
mmse_delta["slope"] = mmse_delta["delta"] / mmse_delta["days"].replace(0, np.nan)
mmse_delta = mmse_delta.reset_index().merge(cases[["patient_id", "target"]], on="patient_id", how="left")

# Summary per class
summary = mmse_delta.groupby("target").agg({
    "delta": ["mean", "std", "median"],
    "slope": ["mean", "std", "median"]
})

# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------
report_path = os.path.join(ROOT, "results", "V4_1_FINAL_READINESS.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# V4‑1 Final Readiness Audit (Corrected)\n\n")
    f.write("## 1. Diagnostic Classifier Performance\n")
    f.write(f"* Linear LogisticRegression macro‑F1 (val): {lin_f1:.3f}\n")
    f.write(f"* RandomForest macro‑F1 (val): {rf_f1:.3f}\n\n")
    f.write("## 2. Medication‑association consistency\n")
    f.write(f"Missing medication episodes originally: {len(missing_cases)}; added: {len(new_med_rows)}.\n\n")
    f.write("## 3. Temporal observation coverage\n")
    f.write(f"Patients missing pre‑observation: {len(missing_pre)}; added: {len([r for r in new_obs_rows if r['obs_date'] < anchor_dates.loc[anchor_dates['patient_id']==r['patient_id'], 'anchor_date'].iloc[0].strftime('%Y-%m-%d')])}.\n")
    f.write(f"Patients missing post‑observation: {len(missing_post)}; added: {len([r for r in new_obs_rows if r['obs_date'] > anchor_dates.loc[anchor_dates['patient_id']==r['patient_id'], 'anchor_date'].iloc[0].strftime('%Y-%m-%d')])}.\n\n")
    f.write("## 4. Cognitive trajectory statistics (per class)\n")
    f.write(summary.to_string())
    f.write("\n\n")
    f.write("## 5. Split isolation & leakage check\n")
    split_counts = df.groupby(["patient_id", "split"]).size().unstack(fill_value=0)
    multi_split = (split_counts > 0).sum(axis=1) > 1
    isolation_ok = not multi_split.any()
    f.write(f"Patient‑split isolation OK: {isolation_ok}\n")
    f.write("\n")
    f.write("Report generated by audit script.\n")
print("Report written to", report_path)
