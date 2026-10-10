"""Reproducible Stress-Test Cohort Generator & Pipeline Runner.

Generates an independent synthetic stress-test cohort under data/stress_test/
with realistic longitudinal noise, overlapping class dynamics, irregular visit gaps,
missing cognitive assessments, and competing clinical events.

Splits:
  - dev (50%, 600 cases): for error diagnosis, ablation, and tuning in Stage 4C.
  - final_test (50%, 600 cases): held out strictly for final evaluation.
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

RAW_STRESS = os.path.join(ROOT, "data", "stress_test", "raw")
PROC_STRESS = os.path.join(ROOT, "data", "stress_test", "processed")
DB_STRESS = os.path.join(ROOT, "data", "stress_test", "ehr.db")

CLASSES = ["medication_associated", "disease_progression", "other_cause", "uncertain"]
CLASS_COUNTS = {
    "medication_associated": 300,
    "disease_progression": 300,
    "other_cause": 300,
    "uncertain": 300,
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


def generate_stress_test_raw(seed=101, out_dir=RAW_STRESS):
    """Generate raw stress test files with overlapping trajectories and irregular gaps."""
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # Load drug knowledge base from V2 raw
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

    # Total 1200 patients: 300 per class, evenly divided into dev (150) and final_test (150)
    for target_cls in CLASSES:
        count = CLASS_COUNTS[target_cls]
        dev_count = count // 2
        splits = ["dev"] * dev_count + ["final_test"] * (count - dev_count)
        random.shuffle(splits)

        for sp_idx in range(count):
            split = splits[sp_idx]
            pid = f"ST_P{patient_idx:04d}"
            cid = f"ST_C{patient_idx:04d}"

            # Demographics
            age = int(np.clip(random.gauss(76, 7), 58, 93))
            sex = random.choice(["Female", "Male"])
            primary_diag = random.choice([
                "Alzheimer's Disease", "Vascular Dementia", "Lewy Body Dementia",
                "Frontotemporal Dementia", "Mild Cognitive Impairment"
            ])
            disease_stage = random.choice(["Mild", "Moderate", "Severe"])
            enroll_date = datetime.date(2021, 1, 1) + datetime.timedelta(days=random.randint(0, 180))
            diag_date = enroll_date - datetime.timedelta(days=random.randint(200, 1500))
            comorb_choices = ["Hypertension", "Dyslipidemia", "Osteoarthritis", "Type 2 Diabetes", "Depression"]
            comorbs = "; ".join(random.sample(comorb_choices, k=random.randint(0, 2))) if random.random() < 0.7 else None

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
            base_mmse = 24.5 if disease_stage == "Mild" else (20.0 if disease_stage == "Moderate" else 15.5)
            base_mmse = np.clip(base_mmse + random.uniform(-1.5, 1.5), 11.0, 29.0)
            base_memory = base_mmse * 3.2 + random.uniform(-3, 3)
            base_attention = base_mmse * 3.0 + random.uniform(-3, 3)
            base_exec = base_mmse * 3.0 + random.uniform(-3, 3)
            base_mood = 70.0 + random.uniform(-8, 8)

            # Irregular assessment schedule: 8 to 13 visits with non-uniform gaps (25 to 110 days)
            n_assessments = random.randint(8, 12)
            cur_day = random.randint(15, 45)
            assess_days = [cur_day]
            for _ in range(n_assessments - 1):
                cur_day += random.randint(30, 95)
                assess_days.append(cur_day)

            # Anchor medication timing: chosen within range
            # Ensure at least 1 baseline in (anchor-270, anchor] and >=2 in (anchor, anchor+180]
            anchor_candidate_idx = random.randint(2, len(assess_days) - 3)
            anchor_day = assess_days[anchor_candidate_idx] - random.randint(5, 20)
            anchor_date = enroll_date + datetime.timedelta(days=anchor_day)

            # Select medications
            assoc_drug_id = random.choice(drug_ids)
            assoc_drug = drugs_by_id[assoc_drug_id]

            # Primary episode (Anchor medication)
            med_rec_id = f"ST_MED{med_rec_idx:05d}"
            med_rec_idx += 1
            has_dose_change = (random.random() < 0.35)
            dose_chg_offset = random.randint(40, 120) if has_dose_change else None
            dose_chg_date = (anchor_date + datetime.timedelta(days=dose_chg_offset)).strftime("%Y-%m-%d") if has_dose_change else None
            init_dose = random.choice([10, 25, 50])
            new_dose = init_dose * 2 if has_dose_change else None

            has_stop = (random.random() < 0.25)
            stop_offset = (dose_chg_offset + random.randint(30, 90)) if (has_stop and has_dose_change) else (random.randint(90, 200) if has_stop else None)
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

            # Co-medications across all classes (1 to 3 additional meds)
            n_co = random.randint(1, 3)
            for _ in range(n_co):
                co_id = random.choice([d for d in drug_ids if d != assoc_drug_id])
                co_drug = drugs_by_id[co_id]
                co_offset = random.randint(10, 350)
                co_start = enroll_date + datetime.timedelta(days=co_offset)
                co_rec_id = f"ST_MED{med_rec_idx:05d}"
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

            # Clinical Events (realistic overlap across all classes)
            has_events = (random.random() < 0.75)
            competing_event = None
            competing_event_day = None

            if has_events:
                n_events = random.randint(1, 3)
                for ev_i in range(n_events):
                    ev_rec_id = f"ST_EV{event_rec_idx:05d}"
                    event_rec_idx += 1

                    if target_cls == "other_cause" and ev_i == 0:
                        etype = random.choice([
                            "Delirium episode", "Urinary tract infection", "Stroke",
                            "Dehydration", "Fall", "Respiratory infection", "Hypotension episode"
                        ])
                        competing_event = etype
                        # Occurs in post-anchor window
                        ev_day = anchor_day + random.randint(15, 65)
                        competing_event_day = ev_day
                        severity = random.choice(["Moderate", "Severe"])
                        cog_impact = "Yes"
                        hosp = "Yes" if severity == "Severe" or etype in ["Stroke", "Delirium episode"] else "No"
                    elif target_cls == "uncertain" and ev_i == 0 and random.random() < 0.6:
                        # Ambiguous event occurring very close to medication start
                        etype = random.choice(["Urinary tract infection", "Fall", "Dehydration", "Delirium episode"])
                        competing_event = etype
                        ev_day = anchor_day + random.randint(-10, 25)
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

            # Complex, Overlapping Cognitive Trajectory Logic
            # Patient-specific pre-existing baseline slope (applies to some med-assoc patients too!)
            if target_cls == "medication_associated":
                has_pre_decline = (random.random() < 0.40)  # 40% have mild pre-existing decline
                pre_slope = random.uniform(-1.5, -0.4) if has_pre_decline else random.uniform(-0.3, 0.1)
                latency = random.randint(10, 45)  # Variable latency before decline manifests
                med_drop_mag = random.uniform(2.5, 4.5)
            elif target_cls == "disease_progression":
                # Fluctuating non-linear neurodegenerative progression
                pre_slope = random.uniform(-3.0, -1.8)
            elif target_cls == "other_cause":
                pre_slope = random.uniform(-1.0, 0.1)
                event_drop_mag = random.uniform(2.8, 5.0)
                recovers = (random.random() < 0.35)  # 65% do NOT completely recover
            else:  # uncertain
                # Highly conflicting ambiguous signals
                pre_slope = random.uniform(-1.6, -0.6)
                med_fluc = random.uniform(1.0, 2.2)

            for a_day in assess_days:
                adate = enroll_date + datetime.timedelta(days=a_day)
                rel_anchor = a_day - anchor_day
                t_year = a_day / 365.25

                # Baseline progression component
                score_change = pre_slope * t_year

                if target_cls == "medication_associated":
                    if rel_anchor > latency:
                        # Additional medication-induced decline
                        post_t = (rel_anchor - latency) / 180.0
                        med_effect = min(med_drop_mag, post_t * med_drop_mag)
                        score_change -= med_effect

                elif target_cls == "disease_progression":
                    # Non-linear fluctuating progression with spells of steeper decline
                    if a_day > anchor_day:
                        # Continues naturally regardless of drug
                        fluctuation = random.uniform(-0.6, 0.6)
                        score_change += fluctuation

                elif target_cls == "other_cause":
                    if competing_event_day is not None and a_day >= competing_event_day:
                        days_since_ev = a_day - competing_event_day
                        if recovers:
                            # Transient dip that partially recovers
                            dip = event_drop_mag * np.exp(-days_since_ev / 60.0)
                        else:
                            # Incomplete recovery: permanent step loss
                            dip = event_drop_mag * (0.7 + 0.3 * np.exp(-days_since_ev / 90.0))
                        score_change -= dip

                else:  # uncertain
                    # Overlapping mild med effect and intercurrent noise
                    if rel_anchor > 15:
                        score_change -= (rel_anchor / 200.0) * med_fluc
                    if competing_event_day is not None and a_day >= competing_event_day:
                        score_change -= 1.2

                # Add measurement noise (higher than V2's 0.4 to simulate real clinical noise)
                noise_mmse = np.random.normal(0, 0.75)
                mmse_val = float(np.clip(base_mmse + score_change + noise_mmse, 6.0, 30.0))

                # Subscores
                mem_val = float(np.clip(base_memory + score_change * 3.5 + np.random.normal(0, 2.0), 10.0, 100.0))
                att_val = float(np.clip(base_attention + score_change * 3.2 + np.random.normal(0, 2.0), 10.0, 100.0))
                exec_val = float(np.clip(base_exec + score_change * 3.2 + np.random.normal(0, 2.0), 10.0, 100.0))
                mood_val = float(np.clip(base_mood + score_change * 1.5 + np.random.normal(0, 2.5), 20.0, 100.0))

                # Random missing subscores (~15% missingness for secondary scores)
                if random.random() < 0.15:
                    exec_val = np.nan
                if random.random() < 0.15:
                    mood_val = np.nan

                # Symptoms
                total_drop = max(0.0, base_mmse - mmse_val)
                conf = 1 if (total_drop > 2.5 and random.random() < 0.85) else 0
                mem_loss = 1 if (total_drop > 2.0 and random.random() < 0.85) else 0
                att_prob = 1 if (total_drop > 2.5 and random.random() < 0.80) else 0
                beh_chg = 1 if (total_drop > 3.5 and random.random() < 0.75) else 0

                cog_rec_id = f"ST_COG{cog_rec_idx:05d}"
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

            # Ground truth record
            conf_val = round(random.uniform(0.70, 0.90), 2) if target_cls != "uncertain" else round(random.uniform(0.40, 0.60), 2)
            gt_cases.append({
                "ground_truth_case_id": cid,
                "patient_id": pid,
                "ground_truth_category": CLASS_MAP_REVERSE[target_cls],
                "scenario_type": f"Scenario_{target_cls}",
                "associated_medication_id": assoc_drug_id if target_cls == "medication_associated" else None,
                "associated_drug": assoc_drug["drug_name"] if target_cls == "medication_associated" else None,
                "alternative_clinical_cause": competing_event if target_cls == "other_cause" else None,
                "expected_temporal_pattern": "Realistic_overlap",
                "synthetic_truth_confidence": conf_val,
                "evaluation_split": split,
            })

            patient_idx += 1

    print(f"Writing raw stress test dataset to: {out_dir}")
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

    print("Stress-test raw cohort generation complete!")


def run_stress_test_pipeline():
    """Execute SQLite import, preprocessing, ground truth label derivation, and ML case building."""
    print("\n--- Running Stress-Test Import and Processing Pipeline ---")
    os.makedirs(PROC_STRESS, exist_ok=True)

    # 1. Import to SQLite
    cmd_import = [
        PYTHON, "scripts/import_data.py",
        "--raw-dir", RAW_STRESS,
        "--db-path", DB_STRESS
    ]
    print("Executing:", " ".join(cmd_import))
    subprocess.run(cmd_import, check=True)

    # 2. Preprocess ML tables
    cmd_prep = [
        PYTHON, "scripts/preprocess_ml.py",
        "--db-path", DB_STRESS,
        "--out-dir", PROC_STRESS
    ]
    print("Executing:", " ".join(cmd_prep))
    subprocess.run(cmd_prep, check=True)

    # 3. Ground truth labels
    gt_excel = os.path.join(RAW_STRESS, "dataset6_Ground_Truth_Cases.xlsx")
    labels_csv = os.path.join(PROC_STRESS, "ground_truth_labels.csv")
    cmd_gt = [
        PYTHON, "scripts/stage2b_ground_truth.py",
        "--db-path", DB_STRESS,
        "--gt-path", gt_excel,
        "--out-path", labels_csv
    ]
    print("Executing:", " ".join(cmd_gt))
    subprocess.run(cmd_gt, check=True)

    # 4. ML Cases building
    cases_csv = os.path.join(PROC_STRESS, "ml_cases.csv")
    cmd_cases = [
        PYTHON, "scripts/stage2c_build_ml_cases.py",
        "--db-path", DB_STRESS,
        "--labels-path", labels_csv,
        "--out", cases_csv
    ]
    print("Executing:", " ".join(cmd_cases))
    subprocess.run(cmd_cases, check=True)

    print("\nStress-Test Pipeline Completed Successfully!")
    df_cases = pd.read_csv(cases_csv)
    print(f"Total ML Cases in Stress-Test: {len(df_cases)}")
    print(f"Usable Cases: {(df_cases.meta_usable_case == 1).sum()}")
    print("Split breakdown (usable cases):")
    print(pd.crosstab(df_cases[df_cases.meta_usable_case == 1].target, df_cases[df_cases.meta_usable_case == 1].split))


if __name__ == "__main__":
    generate_stress_test_raw()
    run_stress_test_pipeline()
