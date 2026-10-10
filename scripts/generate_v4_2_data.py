import argparse
import datetime
import os
import random
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PYTHON = sys.executable

# Paths for V4.2
RAW_V4_2 = os.path.join(ROOT, "data", "v4_2", "raw")
PROC_V4_2 = os.path.join(ROOT, "data", "v4_2", "processed")
DB_V4_2 = os.path.join(ROOT, "data", "v4_2", "ehr.db")

# ---------------------------------------------------------------------------
# Generation configuration – identical to V4 but output locations changed
# ---------------------------------------------------------------------------
CLASSES = ["medication_associated", "disease_progression", "other_cause", "uncertain"]
CLASS_COUNTS = {"medication_associated": 1250, "disease_progression": 1250, "other_cause": 1250, "uncertain": 1250}
CLASS_MAP_REVERSE = {
    "medication_associated": "Medication-associated",
    "disease_progression": "Disease progression",
    "other_cause": "Other clinical cause",
    "uncertain": "Uncertain",
}

# Event definitions – same as V4
EVENT_TYPES_BY_CAT = {
    "Vascular": ["Stroke", "Transient ischemic attack", "Cardiovascular event", "Hypotension episode"],
    "Infection": ["Infection", "Urinary tract infection", "Respiratory infection"],
    "Metabolic": ["Dehydration", "Electrolyte abnormality", "Hyperglycemia", "Hypoglycemia"],
    "Delirium": ["Delirium episode"],
    "Trauma": ["Fall", "Head injury"],
    "Hospitalization": ["Surgery", "Anesthesia exposure", "Hospitalization", "Emergency department visit"],
    "Psychiatric": ["Depressive episode", "Sleep disruption", "Pain exacerbation"],
    "Milestone": ["Disease progression milestone"],
}
ALL_EVENT_TYPES = [etype for types in EVENT_TYPES_BY_CAT.values() for etype in types]
EVENT_TYPE_TO_CAT = {t: cat for cat, types in EVENT_TYPES_BY_CAT.items() for t in types}

# ---------------------------------------------------------------------------
# Helper functions – identical to V4
# ---------------------------------------------------------------------------
def sample_baseline_slope():
    """Sample a baseline cognitive decline slope (points per year). Shared across classes."""
    return np.random.normal(loc=-1.2, scale=0.8)

def sample_med_effect(cls):
    """Sample medication‑associated cognitive impact. Overlaps across classes."""
    if cls == "medication_associated":
        return np.random.normal(loc=-3.0, scale=1.0)
    else:
        return np.random.normal(loc=-0.5, scale=1.0)

def sample_event_drop():
    """Sample magnitude of an acute clinical event impact (negative drop)."""
    return np.random.normal(loc=-2.5, scale=1.0)

def sample_disease_acceleration():
    """Return list of (start_day, end_day, extra_slope) acceleration periods."""
    periods = []
    n_acc = random.randint(0, 2)
    for _ in range(n_acc):
        start = random.randint(100, 800)
        duration = random.randint(30, 120)
        extra = np.random.normal(loc=-1.5, scale=0.5)
        periods.append((start, start + duration, extra))
    return periods

def extra_slope_for_day(day, periods):
    extra = 0.0
    for start, end, slope in periods:
        if start <= day <= end:
            extra += slope
    return extra

# ---------------------------------------------------------------------------
# Main generation routine – produces raw files under data/v4_2/raw
# ---------------------------------------------------------------------------
def generate_v4_2_raw(seed=2026, out_dir=RAW_V4_2):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Load drug knowledge base (shared with V2/V3)
    v2_drug_path = os.path.join(ROOT, "data", "v2", "raw", "dataset5_Drug_Knowledge_Base.xlsx")
    drug_kb = pd.read_excel(v2_drug_path, sheet_name="drug_knowledge")
    drugs_by_id = drug_kb.set_index("medication_id").to_dict(orient="index")
    drug_ids = drug_kb["medication_id"].tolist()

    patients = []
    med_history = []
    cog_records = []
    clinical_events = []
    gt_cases = []

    patient_idx = 1
    med_rec_idx = 1
    cog_rec_idx = 1
    event_rec_idx = 1

    for target_cls in CLASSES:
        count = CLASS_COUNTS[target_cls]
        n_train = int(count * 0.70)
        n_val = int(count * 0.15)
        n_test = count - n_train - n_val
        splits = ["train"] * n_train + ["val"] * n_val + ["final_test"] * n_test
        random.shuffle(splits)

        for split in splits:
            pid = f"V4_2_P{patient_idx:04d}"
            cid = f"V4_2_C{patient_idx:04d}"

            # Demographics
            age = int(np.clip(random.gauss(75.5, 7.5), 55, 94))
            sex = random.choice(["Female", "Male"])
            primary_diag = random.choice([
                "Alzheimer's Disease", "Vascular Dementia", "Lewy Body Dementia",
                "Frontotemporal Dementia", "Mild Cognitive Impairment"
            ])
            disease_stage = random.choice(["Mild", "Moderate", "Severe"])
            enroll_date = datetime.date(2021, 1, 1) + datetime.timedelta(days=random.randint(0, 300))
            diag_date = enroll_date - datetime.timedelta(days=random.randint(180, 1800))
            comorb_choices = ["Hypertension", "Dyslipidemia", "Osteoarthritis", "Type 2 Diabetes", "Depression", "CKD", "Atrial Fibrillation"]
            comorbs = "; ".join(random.sample(comorb_choices, k=random.randint(1, 3))) if random.random() < 0.75 else None

            patients.append({
                "patient_id": pid,
                "age": age,
                "sex": sex,
                "primary_diagnosis": primary_diag,
                "diagnosis_date": diag_date.strftime("%Y-%m-%d"),
                "disease_stage": disease_stage,
                "comorbidities": comorbs,
                "enrollment_date": enroll_date.strftime("%Y-%m-%d"),
            })

            # Baseline cognitive scores
            base_mmse = 25.0 if disease_stage == "Mild" else (20.5 if disease_stage == "Moderate" else 15.0)
            base_mmse = np.clip(base_mmse + random.uniform(-1.8, 1.8), 10.0, 29.5)
            base_memory = base_mmse * 3.2 + random.uniform(-3.5, 3.5)
            base_attention = base_mmse * 3.0 + random.uniform(-3.5, 3.5)
            base_exec = base_mmse * 3.0 + random.uniform(-3.5, 3.5)
            base_mood = 70.0 + random.uniform(-10, 10)

            # Irregular visit schedule – guarantee at least one pre and one post observation
            n_assessments = random.randint(8, 12)
            cur_day = random.randint(15, 50)
            assess_days = [cur_day]
            for _ in range(n_assessments - 1):
                cur_day += random.randint(30, 90)
                assess_days.append(cur_day)

            # Choose an anchor point that leaves at least one assessment before and after
            anchor_idx = random.randint(1, len(assess_days) - 2)
            anchor_day = assess_days[anchor_idx] - random.randint(5, 20)
            anchor_date = enroll_date + datetime.timedelta(days=anchor_day)

            # Primary medication – always created (even if class is not medication_associated)
            assoc_drug_id = random.choice(drug_ids)
            assoc_drug = drugs_by_id[assoc_drug_id]
            med_rec_id = f"V4_2_MED{med_rec_idx:05d}"; med_rec_idx += 1
            has_dose_change = random.random() < 0.35
            dose_chg_offset = random.randint(45, 120) if has_dose_change else None
            dose_chg_date = (anchor_date + datetime.timedelta(days=dose_chg_offset)).strftime("%Y-%m-%d") if has_dose_change else None
            init_dose = random.choice([10, 25, 50])
            new_dose = init_dose * 2 if has_dose_change else None
            has_stop = random.random() < 0.28
            stop_offset = (dose_chg_offset + random.randint(30, 90)) if (has_stop and has_dose_change) else (random.randint(90, 210) if has_stop else None)
            stop_date = (anchor_date + datetime.timedelta(days=stop_offset)).strftime("%Y-%m-%d") if has_stop else None

            med_history.append({
                "medication_record_id": med_rec_id,
                "patient_id": pid,
                "medication_id": assoc_drug_id,
                "drug_name": assoc_drug["drug_name"],
                "drug_class": assoc_drug["drug_class"],
                "therapeutic_area": assoc_drug["therapeutic_area"],
                "indication": "Treatment",
                "dose": init_dose,
                "dose_unit": "mg",
                "frequency": "Once daily",
                "start_date": anchor_date.strftime("%Y-%m-%d"),
                "dose_change_date": dose_chg_date,
                "new_dose": new_dose,
                "stop_date": stop_date,
                "medication_status": "Discontinued" if has_stop else "Active",
                "adherence_status": "Good",
            })

            # Co‑medications (present for all classes)
            n_co = random.randint(1, 4)
            for _ in range(n_co):
                co_id = random.choice([d for d in drug_ids if d != assoc_drug_id])
                co_drug = drugs_by_id[co_id]
                co_offset = random.randint(10, 380)
                co_start = enroll_date + datetime.timedelta(days=co_offset)
                co_rec_id = f"V4_2_MED{med_rec_idx:05d}"; med_rec_idx += 1
                med_history.append({
                    "medication_record_id": co_rec_id,
                    "patient_id": pid,
                    "medication_id": co_id,
                    "drug_name": co_drug["drug_name"],
                    "drug_class": co_drug["drug_class"],
                    "therapeutic_area": co_drug["therapeutic_area"],
                    "indication": "Management",
                    "dose": random.choice([10, 20, 50]),
                    "dose_unit": "mg",
                    "frequency": "Once daily",
                    "start_date": co_start.strftime("%Y-%m-%d"),
                    "dose_change_date": None,
                    "new_dose": None,
                    "stop_date": None,
                    "medication_status": "Active",
                    "adherence_status": "Good",
                })

            # Clinical events – class‑agnostic probability
            has_event = random.random() < 0.55
            competing_event = None
            competing_event_day = None
            if has_event:
                n_events = random.randint(1, 3)
                for _ in range(n_events):
                    ev_rec_id = f"V4_2_EV{event_rec_idx:05d}"; event_rec_idx += 1
                    etype = random.choice(ALL_EVENT_TYPES)
                    ev_day = random.randint(20, max(assess_days) + 30)
                    severity = random.choice(["Mild", "Mild", "Moderate", "Severe"])
                    cog_impact = random.choice(["Yes", "No", "No"])
                    hosp = random.choice(["Yes", "No", "No"])
                    ev_date = enroll_date + datetime.timedelta(days=ev_day)
                    dur = random.randint(2, 21)
                    res_date = ev_date + datetime.timedelta(days=dur)
                    clinical_events.append({
                        "clinical_event_id": ev_rec_id,
                        "patient_id": pid,
                        "event_date": ev_date.strftime("%Y-%m-%d"),
                        "event_type": etype,
                        "event_category": EVENT_TYPE_TO_CAT.get(etype, "General"),
                        "severity": severity,
                        "duration_days": dur,
                        "resolution_date": res_date.strftime("%Y-%m-%d"),
                        "cognitive_impact_possible": cog_impact,
                        "functional_impact": severity,
                        "hospitalization_required": hosp,
                        "clinical_note": f"Clinical note for {etype.lower()}.",
                    })
                    if competing_event is None:
                        competing_event = etype
                        competing_event_day = ev_day

            # Trajectory modelling
            pre_slope = sample_baseline_slope()
            med_effect = sample_med_effect(target_cls)
            latency = random.randint(5, 60)
            event_drop_mag = sample_event_drop() if competing_event_day is not None else 0.0
            accel_periods = sample_disease_acceleration()

            for a_day in assess_days:
                adate = enroll_date + datetime.timedelta(days=a_day)
                rel_anchor = a_day - anchor_day
                t_year = a_day / 365.25
                score_change = pre_slope * t_year
                score_change += extra_slope_for_day(a_day, accel_periods) * t_year
                if rel_anchor > latency:
                    post_t = (rel_anchor - latency) / 180.0
                    med_eff = min(med_effect, post_t * med_effect)
                    score_change += med_eff
                    if has_stop and stop_offset is not None and rel_anchor > (stop_offset + 30):
                        days_after_stop = rel_anchor - (stop_offset + 30)
                        recovery = min(abs(med_eff) * 0.5, (days_after_stop / 90.0) * (abs(med_eff) * 0.5))
                        score_change -= recovery
                if competing_event_day is not None and a_day >= competing_event_day:
                    days_since_ev = a_day - competing_event_day
                    dip = event_drop_mag * np.exp(-days_since_ev / 60.0)
                    score_change += dip
                # Biological noise
                noise_mmse = np.random.normal(0, 0.70)
                mmse_val = float(np.clip(base_mmse + score_change + noise_mmse, 6.0, 30.0))
                mem_val = float(np.clip(base_memory + score_change * 3.4 + np.random.normal(0, 2.0), 10.0, 100.0))
                att_val = float(np.clip(base_attention + score_change * 3.1 + np.random.normal(0, 2.0), 10.0, 100.0))
                exec_val = float(np.clip(base_exec + score_change * 3.1 + np.random.normal(0, 2.0), 10.0, 100.0))
                mood_val = float(np.clip(base_mood + score_change * 1.5 + np.random.normal(0, 2.5), 20.0, 100.0))
                # Missingness (15% each for exec and mood)
                if random.random() < 0.15:
                    exec_val = np.nan
                if random.random() < 0.15:
                    mood_val = np.nan
                cog_rec_id = f"V4_2_COG{cog_rec_idx:05d}"; cog_rec_idx += 1
                cog_records.append({
                    "cognitive_record_id": cog_rec_id,
                    "patient_id": pid,
                    "assessment_date": adate.strftime("%Y-%m-%d"),
                    "mmse_score": round(mmse_val, 1),
                    "memory_score": round(mem_val, 1),
                    "attention_score": round(att_val, 1),
                    "executive_function_score": round(exec_val, 1) if pd.notna(exec_val) else np.nan,
                    "mood_score": round(mood_val, 1) if pd.notna(mood_val) else np.nan,
                    "confusion_present": int((base_mmse - mmse_val) > 2.5 and random.random() < 0.85),
                    "memory_loss_present": int((base_mmse - mmse_val) > 2.0 and random.random() < 0.85),
                    "attention_problem_present": int((base_mmse - mmse_val) > 2.5 and random.random() < 0.80),
                    "behavioral_change_present": int((base_mmse - mmse_val) > 3.5 and random.random() < 0.75),
                    "trajectory_context": "None",
                })

            # Ground‑truth case record
            conf_val = round(random.uniform(0.70, 0.90), 2) if target_cls != "uncertain" else round(random.uniform(0.40, 0.60), 2)
            gt_cases.append({
                "ground_truth_case_id": cid,
                "patient_id": pid,
                "ground_truth_category": CLASS_MAP_REVERSE[target_cls],
                "scenario_type": f"Scenario_{target_cls}",
                "associated_medication_id": assoc_drug_id if target_cls == "medication_associated" else None,
                "associated_drug": assoc_drug["drug_name"] if target_cls == "medication_associated" else None,
                "alternative_clinical_cause": competing_event if target_cls == "other_cause" else None,
                "expected_temporal_pattern": "V4_2_overlapping_trajectories",
                "synthetic_truth_confidence": conf_val,
                "evaluation_split": split,
            })

            patient_idx += 1

    # -----------------------------------------------------------------------
    # Write raw output files
    # -----------------------------------------------------------------------
    print(f"Writing raw V4.2 dataset to: {out_dir}")
    pd.DataFrame(patients).to_csv(os.path.join(out_dir, "dataset1_Patients.csv"), index=False)

    with pd.ExcelWriter(os.path.join(out_dir, "dataset2_Medication_History.xlsx"), engine="openpyxl") as writer:
        pd.DataFrame(med_history).to_excel(writer, sheet_name="medication_history", index=False)
        drug_kb[["medication_id", "drug_name", "drug_class", "therapeutic_area"]].to_excel(
            writer, sheet_name="medication_reference", index=False
        )

    with pd.ExcelWriter(os.path.join(out_dir, "dataset3_Cognitive_Behavioral.xlsx"), engine="openpyxl") as writer:
        pd.DataFrame(cog_records).to_excel(writer, sheet_name="cognitive_behavioral", index=False)

    with pd.ExcelWriter(os.path.join(out_dir, "dataset4_Clinical_Events.xlsx"), engine="openpyxl") as writer:
        pd.DataFrame(clinical_events).to_excel(writer, sheet_name="clinical_events", index=False)

    with pd.ExcelWriter(os.path.join(out_dir, "dataset5_Drug_Knowledge_Base.xlsx"), engine="openpyxl") as writer:
        drug_kb.to_excel(writer, sheet_name="drug_knowledge", index=False)

    with pd.ExcelWriter(os.path.join(out_dir, "dataset6_Ground_Truth_Cases.xlsx"), engine="openpyxl") as writer:
        pd.DataFrame(gt_cases).to_excel(writer, sheet_name="ground_truth_cases", index=False)

    print("V4.2 raw cohort generation complete!")

# ---------------------------------------------------------------------------
# Pipeline runner – mirrors V4 but points to V4.2 directories
# ---------------------------------------------------------------------------
def run_v4_2_pipeline():
    print("\n--- Running V4.2 Import and Processing Pipeline ---")
    os.makedirs(PROC_V4_2, exist_ok=True)

    cmd_import = [PYTHON, "scripts/import_data.py", "--raw-dir", RAW_V4_2, "--db-path", DB_V4_2]
    print("Executing:", " ".join(cmd_import))
    subprocess.run(cmd_import, check=True)

    cmd_prep = [PYTHON, "scripts/preprocess_ml.py", "--db-path", DB_V4_2, "--out-dir", PROC_V4_2]
    print("Executing:", " ".join(cmd_prep))
    subprocess.run(cmd_prep, check=True)

    print("V4.2 import & preprocessing complete!")

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate V4.2 synthetic EHR cohort and run pipeline.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for reproducibility")
    args = parser.parse_args()
    generate_v4_2_raw(seed=args.seed)
    run_v4_2_pipeline()
