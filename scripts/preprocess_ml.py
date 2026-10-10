"""Stage 2: ML data preparation (no modelling, no labels).

Reads the existing SQLite database (read-only) through app.queries /
app.timeline and writes two tidy CSVs, keyed by patient_id:

    data/processed/patient_observations.csv   one row per cognitive assessment / clinical event
    data/processed/medication_episodes.csv    one row per medication record

Usage:  python scripts/preprocess_ml.py
Safe to re-run whenever the database changes; outputs are overwritten.

Rules followed
- Dates -> ISO YYYY-MM-DD; unparseable dates become missing (and are reported).
- Missing values stay missing (empty cell in CSV). Nothing is imputed or dropped.
- Only simple day-offsets are derived. No outcome, label, or causal field is created.
- Source labels (`trajectory_context`) are NOT carried over; see README note below.
- The database is opened read-only.
"""
import os
import sqlite3
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from app import queries  # noqa: E402
from app.timeline import build_patient_timeline  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "processed")
OBS_PATH = os.path.join(OUT_DIR, "patient_observations.csv")
EPI_PATH = os.path.join(OUT_DIR, "medication_episodes.csv")

COG_COLS = ["mmse_score", "memory_score", "attention_score", "executive_function_score",
            "mood_score", "confusion_present", "memory_loss_present",
            "attention_problem_present", "behavioral_change_present"]
EVENT_COLS = ["event_type", "event_category", "severity", "duration_days", "resolution_date",
              "cognitive_impact_possible", "functional_impact", "hospitalization_required"]
PATIENT_COLS = ["age", "sex", "primary_diagnosis", "disease_stage"]


# ---------------------------------------------------------------- helpers
def to_date(value):
    """ISO string / datetime -> pandas Timestamp, or NaT if missing/unparseable."""
    return pd.to_datetime(value, errors="coerce")


def day_diff(later, earlier):
    """Whole days between two Timestamps; None if either is missing."""
    if pd.isna(later) or pd.isna(earlier):
        return None
    return (later - earlier).days


def connect_readonly(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------- per-patient build
def last_on_or_before(dates, day):
    """Most recent Timestamp in sorted `dates` that is <= day (else None)."""
    best = None
    for d in dates:
        if d <= day:
            best = d
        else:
            break
    return best


def build_observations(patient, timeline):
    """Rows for one patient: cognitive assessments and clinical events, in date order.

    Medication offsets are patient-level ("days since the most recent medication
    start / dose change / stop of ANY drug, on or before this date"). They are
    time distances only; no drug is linked to any observation.
    """
    enroll = to_date(patient["enrollment_date"])
    diag = to_date(patient["diagnosis_date"])

    starts = sorted(to_date(e["date"]) for e in timeline if e["entry_type"] == "medication_start")
    changes = sorted(to_date(e["date"]) for e in timeline if e["entry_type"] == "medication_dose_change")
    stops = sorted(to_date(e["date"]) for e in timeline if e["entry_type"] == "medication_stop")

    rows = []
    for e in timeline:
        if e["entry_type"] not in ("cognitive_assessment", "clinical_event"):
            continue
        d = to_date(e["date"])
        det = e["details"]
        row = {"patient_id": patient["patient_id"], "obs_date": d.strftime("%Y-%m-%d") if pd.notna(d) else None,
               "obs_type": e["entry_type"], "source_id": e["source_id"]}
        row.update({k: patient[k] for k in PATIENT_COLS})
        row["days_since_diagnosis"] = day_diff(d, diag)
        row["days_since_enrollment"] = day_diff(d, enroll)
        row["days_since_last_med_start"] = day_diff(d, last_on_or_before(starts, d))
        row["days_since_last_dose_change"] = day_diff(d, last_on_or_before(changes, d))
        row["days_since_last_med_stop"] = day_diff(d, last_on_or_before(stops, d))
        is_cog = e["entry_type"] == "cognitive_assessment"
        for c in COG_COLS:
            row[c] = det[c] if is_cog else None
        for c in EVENT_COLS:
            row[c] = det[c] if not is_cog else None
        rows.append(row)

    for i, r in enumerate(rows, start=1):
        r["obs_seq"] = i
    return rows


def build_episodes(patient, med_rows):
    """One row per medication record with within-episode day offsets."""
    enroll = to_date(patient["enrollment_date"])
    rows = []
    for m in med_rows:
        start, change, stop = (to_date(m[k]) for k in ("start_date", "dose_change_date", "stop_date"))
        iso = lambda t: t.strftime("%Y-%m-%d") if pd.notna(t) else None  # noqa: E731
        rows.append({
            "patient_id": m["patient_id"],
            "medication_record_id": m["medication_record_id"],
            "medication_id": m["medication_id"],
            "drug_name": m["drug_name"],
            "drug_class": m["drug_class"],
            "therapeutic_area": m["therapeutic_area"],
            "indication": m["indication"],
            "dose": m["dose"],
            "dose_unit": m["dose_unit"],
            "frequency": m["frequency"],
            "start_date": iso(start),
            "dose_change_date": iso(change),
            "new_dose": m["new_dose"],
            "stop_date": iso(stop),
            "medication_status": m["medication_status"],
            "adherence_status": m["adherence_status"],
            "days_enrollment_to_start": day_diff(start, enroll),
            "days_start_to_dose_change": day_diff(change, start),
            "days_start_to_stop": day_diff(stop, start),
            "days_dose_change_to_stop": day_diff(stop, change),
        })
    return rows


# -------------------------------------------------------------------- driver
def run(db_path=config.DB_PATH, out_dir=OUT_DIR):
    conn = connect_readonly(db_path)
    patient_ids = [r["patient_id"] for r in conn.execute("SELECT patient_id FROM patients ORDER BY patient_id")]

    obs, epi = [], []
    for pid in patient_ids:
        patient = queries.get_patient(conn, pid)
        timeline = build_patient_timeline(conn, pid)
        obs.extend(build_observations(patient, timeline))
        epi.extend(build_episodes(patient, queries.get_medication_history(conn, pid)))
    conn.close()

    # Column order: ids/dates first, then features. Nullable ints stay ints with blanks.
    obs_df = pd.DataFrame(obs)
    obs_df = obs_df[["patient_id", "obs_seq", "obs_date", "obs_type", "source_id"]
                    + [c for c in obs_df.columns if c not in
                       ("patient_id", "obs_seq", "obs_date", "obs_type", "source_id")]]
    epi_df = pd.DataFrame(epi)

    for df in (obs_df, epi_df):
        for c in df.columns:
            if c.startswith("days_") or c in ("duration_days", "age"):
                df[c] = df[c].astype("Int64")

    os.makedirs(out_dir, exist_ok=True)
    obs_path = os.path.join(out_dir, "patient_observations.csv")
    epi_path = os.path.join(out_dir, "medication_episodes.csv")
    obs_df.to_csv(obs_path, index=False)
    epi_df.to_csv(epi_path, index=False)
    return patient_ids, obs_df, epi_df, obs_path, epi_path


def check_problems(obs_df, epi_df, n_patients):
    """Descriptive checks only. Nothing is fixed or removed."""
    out = []

    def add(msg, n):
        if n:
            out.append(f"{msg}: {n}")

    # bad / missing dates
    add("medication records with no start_date", epi_df.start_date.isna().sum())
    add("observations with no date", obs_df.obs_date.isna().sum())
    # medication logic
    add("stop_date before start_date", (epi_df.days_start_to_stop < 0).sum())
    add("dose_change_date before start_date", (epi_df.days_start_to_dose_change < 0).sum())
    add("dose_change_date after stop_date", (epi_df.days_dose_change_to_stop < 0).sum())
    add("status 'Active' but stop_date present",
        ((epi_df.medication_status == "Active") & epi_df.stop_date.notna()).sum())
    add("status 'Stopped' but no stop_date",
        ((epi_df.medication_status == "Stopped") & epi_df.stop_date.isna()).sum())
    add("dose_change_date without new_dose (or vice versa)",
        (epi_df.dose_change_date.isna() != epi_df.new_dose.isna()).sum())
    add("medication starts before patient enrollment", (epi_df.days_enrollment_to_start < 0).sum())
    add("missing adherence_status", epi_df.adherence_status.isna().sum())
    add("adherence_status = 'Unknown'", (epi_df.adherence_status == "Unknown").sum())
    # observations
    add("observations dated before diagnosis", (obs_df.days_since_diagnosis < 0).sum())
    add("observations dated before enrollment", (obs_df.days_since_enrollment < 0).sum())
    cog = obs_df[obs_df.obs_type == "cognitive_assessment"]
    ev = obs_df[obs_df.obs_type == "clinical_event"]
    add("MMSE outside 0-30", ((cog.mmse_score < 0) | (cog.mmse_score > 30)).sum())
    for c in ("memory_score", "attention_score", "executive_function_score"):
        add(f"{c} outside 0-100", ((cog[c] < 0) | (cog[c] > 100)).sum())
    add("cognitive rows with all 5 scores missing",
        cog[["mmse_score", "memory_score", "attention_score",
             "executive_function_score", "mood_score"]].isna().all(axis=1).sum())
    add("duplicate (patient, date) cognitive assessments",
        cog.duplicated(["patient_id", "obs_date"]).sum())
    add("event resolution_date before event date",
        (pd.to_datetime(ev.resolution_date) < pd.to_datetime(ev.obs_date)).sum())
    add("patients with no cognitive assessment", n_patients - cog.patient_id.nunique())
    add("patients with only one cognitive assessment", (cog.groupby("patient_id").size() == 1).sum())
    add("patients with no medication records", n_patients - epi_df.patient_id.nunique())
    # missingness of the main measurements
    miss = cog[["mmse_score", "memory_score", "attention_score", "executive_function_score",
                "mood_score"]].isna().mean().round(3)
    out.append("missing share in cognitive scores: " + ", ".join(f"{k}={v:.1%}" for k, v in miss.items()))
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2 ML data preprocessing")
    parser.add_argument("--db-path", default=config.DB_PATH, help="Path to SQLite database")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Output directory for processed CSVs")
    args = parser.parse_args()

    print(f"Reading (read-only): {args.db_path}")
    patient_ids, obs_df, epi_df, obs_path, epi_path = run(db_path=args.db_path, out_dir=args.out_dir)
    print(f"\nPatients processed:       {len(patient_ids)}")
    print(f"Observation rows:         {len(obs_df)}  "
          f"({(obs_df.obs_type == 'cognitive_assessment').sum()} cognitive, "
          f"{(obs_df.obs_type == 'clinical_event').sum()} clinical events)")
    print(f"Medication episodes:      {len(epi_df)}")
    print(f"\nWritten:\n  {obs_path}\n  {epi_path}")
    problems = check_problems(obs_df, epi_df, len(patient_ids))
    print("\nData checks (reported only, nothing altered):")
    for p in problems:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
