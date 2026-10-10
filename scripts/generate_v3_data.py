"""Stage 4D: Version 3 Realistic Synthetic Cohort Generator & Pipeline Runner.

Generates a diverse, realistic 5,000-patient longitudinal EHR dataset under data/v3/
avoiding fixed mathematical templates:
  - Diverse, non-linear trajectories with realistic biological noise
  - Pre-existing decline in medication-associated patients
  - Stochastic multi-medication changes across all classes
  - Persistent deficits and incomplete recovery in acute medical events
  - Overlapping competing causes for genuine clinical ambiguity
  - Irregular visit intervals and realistic missing subscores

Splits (patient-isolated):
  - train (70%, 3,500 patients): for model training
  - val (15%, 750 patients): for model selection and early stopping
  - final_test (15%, 750 patients): strictly held out for final evaluation
"""

import argparse
import datetime
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PYTHON = sys.executable

RAW_V3 = os.path.join(ROOT, "data", "v3", "raw")
PROC_V3 = os.path.join(ROOT, "data", "v3", "processed")
DB_V3 = os.path.join(ROOT, "data", "v3", "ehr.db")

CLASSES = ["medication_associated", "disease_progression", "other_cause", "uncertain"]
CLASS_COUNTS = {
    "medication_associated": 1250,
    "disease_progression": 1250,
    "other_cause": 1250,
    "uncertain": 1250,
}

CLASS_MAP_REVERSE = {
    "medication_associated": "Medication-associated",
    "disease_progression": "Disease progression",
    "other_cause": "Other clinical cause",
    "uncertain": "Uncertain",
}

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


def generate_v3_raw(seed=2026, out_dir=RAW_V3):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Load drug knowledge base
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

    # 5000 patients total across 4 classes
    for target_cls in CLASSES:
        count = CLASS_COUNTS[target_cls]
        n_train = int(count * 0.70)
        n_val = int(count * 0.15)
        n_test = count - n_train - n_val
        splits = ["train"] * n_train + ["val"] * n_val + ["final_test"] * n_test
        random.shuffle(splits)

        for sp_idx in range(count):
            split = splits[sp_idx]
            pid = f"V3_P{patient_idx:04d}"
            cid = f"V3_C{patient_idx:04d}"

            # Demographics & disease stage
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

            # Base cognitive scores
            base_mmse = 25.0 if disease_stage == "Mild" else (20.5 if disease_stage == "Moderate" else 15.0)
            base_mmse = np.clip(base_mmse + random.uniform(-1.8, 1.8), 10.0, 29.5)
            base_memory = base_mmse * 3.2 + random.uniform(-3.5, 3.5)
            base_attention = base_mmse * 3.0 + random.uniform(-3.5, 3.5)
            base_exec = base_mmse * 3.0 + random.uniform(-3.5, 3.5)
            base_mood = 70.0 + random.uniform(-10, 10)

            # Irregular visit intervals: 8 to 13 visits with non-uniform gaps (25 to 105 days)
            n_assessments = random.randint(8, 12)
            cur_day = random.randint(15, 50)
            assess_days = [cur_day]
            for _ in range(n_assessments - 1):
                cur_day += random.randint(30, 90)
                assess_days.append(cur_day)

            # Anchor medication placement
            anchor_idx = random.randint(2, len(assess_days) - 3)
            anchor_day = assess_days[anchor_idx] - random.randint(5, 20)
            anchor_date = enroll_date + datetime.timedelta(days=anchor_day)

            # Select primary medication
            assoc_drug_id = random.choice(drug_ids)
            assoc_drug = drugs_by_id[assoc_drug_id]

            med_rec_id = f"V3_MED{med_rec_idx:05d}"
            med_rec_idx += 1

            has_dose_change = (random.random() < 0.35)
            dose_chg_offset = random.randint(45, 120) if has_dose_change else None
            dose_chg_date = (anchor_date + datetime.timedelta(days=dose_chg_offset)).strftime("%Y-%m-%d") if has_dose_change else None
            init_dose = random.choice([10, 25, 50])
            new_dose = init_dose * 2 if has_dose_change else None

            # 25% of medication cases stop the drug with potential dechallenge recovery
            has_stop = (random.random() < 0.28)
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

            # Co-medications across all classes (1 to 4 concurrent meds)
            n_co = random.randint(1, 4)
            for _ in range(n_co):
                co_id = random.choice([d for d in drug_ids if d != assoc_drug_id])
                co_drug = drugs_by_id[co_id]
                co_offset = random.randint(10, 380)
                co_start = enroll_date + datetime.timedelta(days=co_offset)
                co_rec_id = f"V3_MED{med_rec_idx:05d}"
                med_rec_idx += 1
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

            # Clinical events: realistic distribution across all classes
            has_events = (random.random() < 0.72)
            competing_event = None
            competing_event_day = None

            if has_events:
                n_events = random.randint(1, 3)
                for ev_i in range(n_events):
                    ev_rec_id = f"V3_EV{event_rec_idx:05d}"
                    event_rec_idx += 1

                    if target_cls == "other_cause" and ev_i == 0:
                        etype = random.choice([
                            "Delirium episode", "Urinary tract infection", "Stroke",
                            "Dehydration", "Fall", "Respiratory infection", "Hypotension episode"
                        ])
                        competing_event = etype
                        ev_day = anchor_day + random.randint(15, 70)
                        competing_event_day = ev_day
                        severity = random.choice(["Moderate", "Severe"])
                        cog_impact = "Yes"
                        hosp = "Yes" if severity == "Severe" or etype in ["Stroke", "Delirium episode"] else "No"
                    elif target_cls == "uncertain" and ev_i == 0 and random.random() < 0.55:
                        etype = random.choice(["Urinary tract infection", "Fall", "Dehydration", "Delirium episode"])
                        competing_event = etype
                        ev_day = anchor_day + random.randint(-15, 30)
                        competing_event_day = ev_day
                        severity = "Moderate"
                        cog_impact = "Yes"
                        hosp = "No"
                    else:
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

            # Complex, Overlapping Trajectory Modeling
            if target_cls == "medication_associated":
                has_pre_decline = (random.random() < 0.48)
                pre_slope = random.uniform(-1.6, -0.5) if has_pre_decline else random.uniform(-0.3, 0.1)
                latency = random.randint(10, 50)
                med_drop_mag = random.uniform(2.2, 4.5)
            elif target_cls == "disease_progression":
                pre_slope = random.uniform(-3.0, -1.6)
            elif target_cls == "other_cause":
                pre_slope = random.uniform(-1.2, 0.1)
                event_drop_mag = random.uniform(2.8, 5.0)
                recovers = (random.random() < 0.35)
            else:  # uncertain
                pre_slope = random.uniform(-1.8, -0.6)
                med_fluc = random.uniform(1.0, 2.2)

            for a_day in assess_days:
                adate = enroll_date + datetime.timedelta(days=a_day)
                rel_anchor = a_day - anchor_day
                t_year = a_day / 365.25

                score_change = pre_slope * t_year

                if target_cls == "medication_associated":
                    if rel_anchor > latency:
                        post_t = (rel_anchor - latency) / 180.0
                        med_eff = min(med_drop_mag, post_t * med_drop_mag)
                        score_change -= med_eff
                        # Dechallenge recovery if drug stopped
                        if has_stop and stop_offset is not None and rel_anchor > (stop_offset + 30):
                            days_after_stop = rel_anchor - (stop_offset + 30)
                            recovery = min(med_eff * 0.5, (days_after_stop / 90.0) * (med_eff * 0.5))
                            score_change += recovery

                elif target_cls == "disease_progression":
                    # Non-linear progression with intermittent plateaus or acceleration
                    if a_day > anchor_day:
                        fluctuation = random.uniform(-0.6, 0.6)
                        score_change += fluctuation

                elif target_cls == "other_cause":
                    if competing_event_day is not None and a_day >= competing_event_day:
                        days_since_ev = a_day - competing_event_day
                        if recovers:
                            dip = event_drop_mag * np.exp(-days_since_ev / 60.0)
                        else:
                            dip = event_drop_mag * (0.65 + 0.35 * np.exp(-days_since_ev / 90.0))
                        score_change -= dip

                else:  # uncertain
                    if rel_anchor > 15:
                        score_change -= (rel_anchor / 200.0) * med_fluc
                    if competing_event_day is not None and a_day >= competing_event_day:
                        score_change -= 1.2

                # Biological measurement noise
                noise_mmse = np.random.normal(0, 0.70)
                mmse_val = float(np.clip(base_mmse + score_change + noise_mmse, 6.0, 30.0))

                mem_val = float(np.clip(base_memory + score_change * 3.4 + np.random.normal(0, 2.0), 10.0, 100.0))
                att_val = float(np.clip(base_attention + score_change * 3.1 + np.random.normal(0, 2.0), 10.0, 100.0))
                exec_val = float(np.clip(base_exec + score_change * 3.1 + np.random.normal(0, 2.0), 10.0, 100.0))
                mood_val = float(np.clip(base_mood + score_change * 1.5 + np.random.normal(0, 2.5), 20.0, 100.0))

                # Realistic missingness (~15% missingness for secondary subscores)
                if random.random() < 0.15:
                    exec_val = np.nan
                if random.random() < 0.15:
                    mood_val = np.nan

                total_drop = max(0.0, base_mmse - mmse_val)
                conf = 1 if (total_drop > 2.5 and random.random() < 0.85) else 0
                mem_loss = 1 if (total_drop > 2.0 and random.random() < 0.85) else 0
                att_prob = 1 if (total_drop > 2.5 and random.random() < 0.80) else 0
                beh_chg = 1 if (total_drop > 3.5 and random.random() < 0.75) else 0

                cog_rec_id = f"V3_COG{cog_rec_idx:05d}"
                cog_rec_idx += 1

                cog_records.append({
                    "cognitive_record_id": cog_rec_id,
                    "patient_id": pid,
                    "assessment_date": adate.strftime("%Y-%m-%d"),
                    "mmse_score": round(mmse_val, 1),
                    "memory_score": round(mem_val, 1),
                    "attention_score": round(att_val, 1),
                    "executive_function_score": round(exec_val, 1) if pd.notna(exec_val) else np.nan,
                    "mood_score": round(mood_val, 1) if pd.notna(mood_val) else np.nan,
                    "confusion_present": conf,
                    "memory_loss_present": mem_loss,
                    "attention_problem_present": att_prob,
                    "behavioral_change_present": beh_chg,
                    "trajectory_context": "None",
                })

            conf_val = round(random.uniform(0.70, 0.90), 2) if target_cls != "uncertain" else round(random.uniform(0.40, 0.60), 2)
            gt_cases.append({
                "ground_truth_case_id": cid,
                "patient_id": pid,
                "ground_truth_category": CLASS_MAP_REVERSE[target_cls],
                "scenario_type": f"Scenario_{target_cls}",
                "associated_medication_id": assoc_drug_id if target_cls == "medication_associated" else None,
                "associated_drug": assoc_drug["drug_name"] if target_cls == "medication_associated" else None,
                "alternative_clinical_cause": competing_event if target_cls == "other_cause" else None,
                "expected_temporal_pattern": "Realistic_V3_dynamics",
                "synthetic_truth_confidence": conf_val,
                "evaluation_split": split,
            })

            patient_idx += 1

    print(f"Writing raw V3 dataset to: {out_dir}")
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

    print("V3 raw cohort generation complete!")


def run_v3_pipeline():
    print("\n--- Running V3 Import and Processing Pipeline ---")
    os.makedirs(PROC_V3, exist_ok=True)

    # 1. Import to SQLite
    cmd_import = [
        PYTHON, "scripts/import_data.py",
        "--raw-dir", RAW_V3,
        "--db-path", DB_V3
    ]
    print("Executing:", " ".join(cmd_import))
    subprocess.run(cmd_import, check=True)

    # 2. Preprocess ML tables
    cmd_prep = [
        PYTHON, "scripts/preprocess_ml.py",
        "--db-path", DB_V3,
        "--out-dir", PROC_V3
    ]
    print("Executing:", " ".join(cmd_prep))
    subprocess.run(cmd_prep, check=True)

    # 3. Ground truth labels
    gt_excel = os.path.join(RAW_V3, "dataset6_Ground_Truth_Cases.xlsx")
    labels_csv = os.path.join(PROC_V3, "ground_truth_labels.csv")
    cmd_gt = [
        PYTHON, "scripts/stage2b_ground_truth.py",
        "--db-path", DB_V3,
        "--gt-path", gt_excel,
        "--out-path", labels_csv
    ]
    print("Executing:", " ".join(cmd_gt))
    subprocess.run(cmd_gt, check=True)

    # 4. ML Cases building
    cases_csv = os.path.join(PROC_V3, "ml_cases.csv")
    cmd_cases = [
        PYTHON, "scripts/stage2c_build_ml_cases.py",
        "--db-path", DB_V3,
        "--labels-path", labels_csv,
        "--out", cases_csv
    ]
    print("Executing:", " ".join(cmd_cases))
    subprocess.run(cmd_cases, check=True)

    print("\nV3 Pipeline Completed Successfully!")
    df_cases = pd.read_csv(cases_csv)
    print(f"Total ML Cases in V3: {len(df_cases)}")
    print(f"Usable Cases: {(df_cases.meta_usable_case == 1).sum()}")
    print("Split breakdown (usable cases):")
    print(pd.crosstab(df_cases[df_cases.meta_usable_case == 1].target, df_cases[df_cases.meta_usable_case == 1].split))


if __name__ == "__main__":
    generate_v3_raw()
    run_v3_pipeline()
