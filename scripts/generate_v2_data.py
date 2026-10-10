"""Reproducible Version 2 Synthetic Data Generator.

Generates corrected V2 raw datasets under data/v2/raw/ (or custom directory):
  - dataset1_Patients.csv
  - dataset2_Medication_History.xlsx
  - dataset3_Cognitive_Behavioral.xlsx
  - dataset4_Clinical_Events.xlsx
  - dataset5_Drug_Knowledge_Base.xlsx (copied/standardized 25 drugs)
  - dataset6_Ground_Truth_Cases.xlsx

Phase 1 fixes:
  - Drug distribution: Uniform/balanced drug sampling across all 4 classes. High-risk drugs appear at ~30-35% in all classes.
  - Clinical events: Balanced event presence (~20% zero events across all classes; realistic mix of minor, serious, and unrelated events).

Phase 2 fixes:
  - Medication-associated: Baseline assessments in 270d pre-window, decline follows drug exposure. Anchor selection hits GT associated drug in >85% of cases.
  - Disease progression: Pre-existing decline continues steadily across medication start/changes.
  - Other cause: Cognitive drop aligned with competing clinical event.
  - Uncertain: Truly ambiguous overlapping scenarios.
"""

import argparse
import datetime
import os
import random
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RAW_V1_DIR = os.path.join(ROOT, "data", "raw")
OUT_DEFAULT = os.path.join(ROOT, "data", "v2", "raw")

CLASSES = ["medication_associated", "disease_progression", "other_cause", "uncertain"]
CLASS_COUNTS = {
    "medication_associated": 1500,
    "disease_progression": 1500,
    "other_cause": 1250,
    "uncertain": 750,
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

EVENT_TYPE_TO_CAT = {}
for cat, types in EVENT_TYPES_BY_CAT.items():
    for t in types:
        EVENT_TYPE_TO_CAT[t] = cat


def load_v1_drug_kb(raw_v1_dir=RAW_V1_DIR):
    path = os.path.join(raw_v1_dir, "dataset5_Drug_Knowledge_Base.xlsx")
    return pd.read_excel(path, sheet_name="drug_knowledge")


def generate_v2(seed=42, out_dir=OUT_DEFAULT, raw_v1_dir=RAW_V1_DIR):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    drug_kb = load_v1_drug_kb(raw_v1_dir)
    drug_ids = drug_kb["medication_id"].tolist()
    drugs_by_id = drug_kb.set_index("medication_id").to_dict(orient="index")

    patients = []
    med_history = []
    cog_records = []
    clinical_events = []
    gt_cases = []

    patient_idx = 1
    med_rec_idx = 1
    cog_rec_idx = 1
    event_rec_idx = 1

    # Class assignment for 5000 patients
    patient_targets = []
    for cls, count in CLASS_COUNTS.items():
        patient_targets.extend([cls] * count)

    # Shuffle patients while preserving class count
    patient_indices = list(range(len(patient_targets)))
    random.shuffle(patient_indices)

    # Determine splits per class
    splits_per_class = {}
    for cls, count in CLASS_COUNTS.items():
        n_train = int(count * 0.70)
        n_val = int(count * 0.15)
        n_test = count - n_train - n_val
        sp = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        random.shuffle(sp)
        splits_per_class[cls] = sp

    split_counters = {cls: 0 for cls in CLASSES}

    start_base_date = datetime.date(2020, 1, 1)

    for i in range(len(patient_targets)):
        pid = f"P{patient_idx:04d}"
        cid = f"C{patient_idx:04d}"
        target_cls = patient_targets[i]

        split = splits_per_class[target_cls][split_counters[target_cls]]
        split_counters[target_cls] += 1

        # Patient demographics
        age = random.randint(60, 88)
        sex = random.choice(["Female", "Male"])
        primary_diag = random.choice([
            "Alzheimer's Disease", "Vascular Dementia", "Lewy Body Dementia",
            "Frontotemporal Dementia", "Mild Cognitive Impairment"
        ])
        disease_stage = random.choice(["Mild", "Moderate", "Severe"])

        diag_offset_days = random.randint(365, 1800)
        diag_date = start_base_date - datetime.timedelta(days=diag_offset_days)
        enroll_date = start_base_date - datetime.timedelta(days=random.randint(30, 300))

        comorbidities_list = random.sample([
            "Hypertension", "Type 2 Diabetes", "Dyslipidemia", "Osteoarthritis",
            "Depression", "Atrial Fibrillation", "CKD"
        ], k=random.randint(0, 3))
        comorbidities_str = "; ".join(comorbidities_list) if comorbidities_list else "None"

        patients.append({
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "primary_diagnosis": primary_diag,
            "diagnosis_date": diag_date.strftime("%Y-%m-%d"),
            "disease_stage": disease_stage,
            "comorbidities": comorbidities_str,
            "enrollment_date": enroll_date.strftime("%Y-%m-%d"),
        })

        # Timeline setup: timeline spans day 0 to day 730 (2 years) from base_date
        base_date = enroll_date

        # Generate cognitive assessment schedule: 8 to 11 assessments over 2 years (~every 60-90 days)
        n_assessments = random.randint(8, 11)
        assess_offsets = sorted(random.sample(range(0, 700), n_assessments))
        # Ensure assessments cover early and late periods
        if 30 not in assess_offsets and assess_offsets[0] > 60:
            assess_offsets[0] = 30
        assess_offsets = sorted(list(set(assess_offsets)))

        # Baseline cognitive scores
        base_mmse = float(random.randint(22, 28))
        base_memory = float(random.randint(55, 85))
        base_attention = float(random.randint(55, 85))
        base_exec = float(random.randint(55, 85))
        base_mood = float(random.randint(60, 90))

        # Select associated drug & co-medications
        # CRITICAL FIX: Draw associated drug uniformly across ALL 25 drugs regardless of class!
        assoc_drug_id = random.choice(drug_ids)
        assoc_drug_info = drugs_by_id[assoc_drug_id]

        # Anchor date selection (day offset from base_date)
        # For medication_associated: place anchor around day 240, ensuring ~3 pre assessments (days 0..240) and ~4 post assessments (days 240..420)
        if target_cls == "medication_associated":
            anchor_day = random.randint(210, 270)
        else:
            anchor_day = random.randint(180, 360)

        assoc_start_date = base_date + datetime.timedelta(days=anchor_day)

        # Build medication history for patient (2 to 4 medications)
        # The associated medication record
        assoc_dose = random.choice([5, 10, 25, 50, 100])
        assoc_med_rec_id = f"MREC{med_rec_idx:05d}"
        med_rec_idx += 1

        is_dose_change_anchor = (random.random() < 0.15)
        if is_dose_change_anchor:
            start_d = assoc_start_date - datetime.timedelta(days=random.randint(60, 180))
            dose_change_d = assoc_start_date
            dose_val = assoc_dose
            new_dose_val = assoc_dose * 2
        else:
            start_d = assoc_start_date
            dose_change_d = None
            dose_val = assoc_dose
            new_dose_val = None

        # Stop date (some drugs stop in post window)
        if random.random() < 0.10:
            stop_d = assoc_start_date + datetime.timedelta(days=random.randint(60, 150))
            status = "Stopped"
        else:
            stop_d = None
            status = "Active"

        med_history.append({
            "medication_record_id": assoc_med_rec_id,
            "patient_id": pid,
            "medication_id": assoc_drug_id,
            "drug_name": assoc_drug_info["drug_name"],
            "drug_class": assoc_drug_info["drug_class"],
            "therapeutic_area": assoc_drug_info["therapeutic_area"],
            "indication": "Treatment",
            "dose": dose_val,
            "dose_unit": "mg",
            "frequency": "Once daily",
            "start_date": start_d.strftime("%Y-%m-%d"),
            "dose_change_date": dose_change_d.strftime("%Y-%m-%d") if dose_change_d else None,
            "new_dose": new_dose_val if dose_change_d else None,
            "stop_date": stop_d.strftime("%Y-%m-%d") if stop_d else None,
            "medication_status": status,
            "adherence_status": random.choice(["Good", "Good", "Moderate", "Good"]),
        })

        # Add 1-3 co-medications for all patients (placed to not overshadow the anchor in coverage)
        other_drug_pool = [d for d in drug_ids if d != assoc_drug_id]
        n_co_meds = random.randint(1, 3)
        co_drugs = random.sample(other_drug_pool, n_co_meds)

        for co_id in co_drugs:
            co_info = drugs_by_id[co_id]
            # Place start dates offset from anchor
            co_offset = random.choice([-150, -100, 180, 240, 300])
            co_start = base_date + datetime.timedelta(days=max(10, anchor_day + co_offset))
            co_rec_id = f"MREC{med_rec_idx:05d}"
            med_rec_idx += 1
            med_history.append({
                "medication_record_id": co_rec_id,
                "patient_id": pid,
                "medication_id": co_id,
                "drug_name": co_info["drug_name"],
                "drug_class": co_info["drug_class"],
                "therapeutic_area": co_info["therapeutic_area"],
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

        # CRITICAL FIX: Clinical events presence balance across all classes!
        # ~20% of patients in EVERY class have 0 clinical events.
        has_events = (random.random() >= 0.20)
        alt_cause_event = None

        if has_events:
            n_events = random.randint(1, 3)
            # Pick event types
            for ev_i in range(n_events):
                ev_rec_id = f"EV{event_rec_idx:05d}"
                event_rec_idx += 1

                if target_cls == "other_cause" and ev_i == 0:
                    # Competing clinical event for other_cause
                    etype = random.choice([
                        "Delirium episode", "Urinary tract infection", "Stroke",
                        "Dehydration", "Fall", "Respiratory infection", "Hypotension episode"
                    ])
                    alt_cause_event = etype
                    # Event happens near anchor (e.g., anchor_day + 15 to 45 days)
                    ev_offset = anchor_day + random.randint(10, 50)
                    severity = random.choice(["Moderate", "Severe"])
                    cog_impact = "Yes"
                    hosp = "Yes" if severity == "Severe" or etype in ["Stroke", "Delirium episode"] else "No"
                else:
                    etype = random.choice(ALL_EVENT_TYPES)
                    ev_offset = random.randint(30, 650)
                    severity = random.choice(["Mild", "Mild", "Moderate", "Severe"])
                    cog_impact = random.choice(["Yes", "No", "No"])
                    hosp = random.choice(["Yes", "No", "No"])

                ev_date = base_date + datetime.timedelta(days=ev_offset)
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
                    "clinical_note": f"Patient evaluated for {etype.lower()}.",
                })

        # Generate cognitive assessments trajectory based on class longitudinal logic
        for offset in assess_offsets:
            adate = base_date + datetime.timedelta(days=offset)
            days_from_anchor = offset - anchor_day

            # Baseline slopes & trajectory logic
            if target_cls == "medication_associated":
                # Flat baseline before anchor, sharp decline after anchor
                if days_from_anchor <= 0:
                    mmse = base_mmse + np.random.normal(0, 0.4)
                    mem = base_memory + np.random.normal(0, 1.0)
                    att = base_attention + np.random.normal(0, 1.0)
                    exec_f = base_exec + np.random.normal(0, 1.0)
                    mood = base_mood + np.random.normal(0, 1.0)
                    conf = 0
                    mem_loss = 0
                    att_prob = 0
                    beh_chg = 0
                else:
                    # Decline proportional to days post anchor
                    drop = min(5.0, (days_from_anchor / 180.0) * 3.5 + random.uniform(0.5, 1.5))
                    mmse = max(10.0, base_mmse - drop + np.random.normal(0, 0.4))
                    mem = max(20.0, base_memory - drop * 4.0 + np.random.normal(0, 1.0))
                    att = max(20.0, base_attention - drop * 3.5 + np.random.normal(0, 1.0))
                    exec_f = max(20.0, base_exec - drop * 3.5 + np.random.normal(0, 1.0))
                    mood = base_mood - drop * 2.0 + np.random.normal(0, 1.0)
                    conf = 1 if days_from_anchor > 30 else 0
                    mem_loss = 1 if drop > 2.0 else 0
                    att_prob = 1 if drop > 2.0 else 0
                    beh_chg = 1 if drop > 3.0 else 0

            elif target_cls == "disease_progression":
                # Pre-existing steady decline across entire timeline regardless of anchor
                overall_slope = (offset / 365.0) * 2.5 # ~2.5 MMSE drop per year
                mmse = max(10.0, base_mmse - overall_slope + np.random.normal(0, 0.5))
                mem = max(20.0, base_memory - overall_slope * 4.0 + np.random.normal(0, 1.2))
                att = max(20.0, base_attention - overall_slope * 3.5 + np.random.normal(0, 1.2))
                exec_f = max(20.0, base_exec - overall_slope * 3.5 + np.random.normal(0, 1.2))
                mood = base_mood - overall_slope * 1.5 + np.random.normal(0, 1.2)
                conf = 1 if overall_slope > 3.0 else 0
                mem_loss = 1 if overall_slope > 2.0 else 0
                att_prob = 1 if overall_slope > 2.5 else 0
                beh_chg = 1 if overall_slope > 4.0 else 0

            elif target_cls == "other_cause":
                # Stable baseline, drop coincides with competing clinical event window
                # Event window is around anchor_day + 20..120
                if days_from_anchor > 10 and days_from_anchor < 150:
                    event_drop = 3.5 + np.random.normal(0, 0.5)
                else:
                    event_drop = 0.5 if days_from_anchor >= 150 else 0.0

                mmse = max(10.0, base_mmse - event_drop + np.random.normal(0, 0.4))
                mem = max(20.0, base_memory - event_drop * 3.5 + np.random.normal(0, 1.0))
                att = max(20.0, base_attention - event_drop * 4.0 + np.random.normal(0, 1.0))
                exec_f = max(20.0, base_exec - event_drop * 3.0 + np.random.normal(0, 1.0))
                mood = base_mood - event_drop * 2.0 + np.random.normal(0, 1.0)
                conf = 1 if event_drop > 2.0 else 0
                mem_loss = 1 if event_drop > 2.5 else 0
                att_prob = 1 if event_drop > 2.0 else 0
                beh_chg = 1 if event_drop > 3.0 else 0

            else:  # uncertain
                # Ambiguous noise, slow decline + mild post-anchor fluctuation
                gradual = (offset / 365.0) * 1.0
                post_fluc = (days_from_anchor / 180.0) * 1.2 if days_from_anchor > 0 else 0.0
                tot_drop = gradual + post_fluc
                mmse = max(10.0, base_mmse - tot_drop + np.random.normal(0, 0.8))
                mem = max(20.0, base_memory - tot_drop * 3.0 + np.random.normal(0, 1.5))
                att = max(20.0, base_attention - tot_drop * 2.5 + np.random.normal(0, 1.5))
                exec_f = max(20.0, base_exec - tot_drop * 2.5 + np.random.normal(0, 1.5))
                mood = base_mood - tot_drop * 1.5 + np.random.normal(0, 1.5)
                conf = 1 if tot_drop > 2.5 else 0
                mem_loss = 1 if tot_drop > 2.0 else 0
                att_prob = 0
                beh_chg = 0

            cog_rec_id = f"COG{cog_rec_idx:05d}"
            cog_rec_idx += 1

            cog_records.append({
                "cognitive_record_id": cog_rec_id,
                "patient_id": pid,
                "assessment_date": adate.strftime("%Y-%m-%d"),
                "mmse_score": round(float(mmse), 1),
                "memory_score": round(float(mem), 1),
                "attention_score": round(float(att), 1),
                "executive_function_score": round(float(exec_f), 1),
                "mood_score": round(float(mood), 1),
                "confusion_present": int(conf),
                "memory_loss_present": int(mem_loss),
                "attention_problem_present": int(att_prob),
                "behavioral_change_present": int(beh_chg),
                "trajectory_context": "None",  # Fixed string "None" across all rows to eliminate context leakage
            })

        # Ground truth record
        conf_val = round(random.uniform(0.75, 0.95), 2) if target_cls != "uncertain" else round(random.uniform(0.45, 0.65), 2)
        gt_cases.append({
            "ground_truth_case_id": cid,
            "patient_id": pid,
            "ground_truth_category": CLASS_MAP_REVERSE[target_cls],
            "scenario_type": f"Scenario_{target_cls}",
            "associated_medication_id": assoc_drug_id if target_cls == "medication_associated" else (assoc_drug_id if random.random() < 0.3 else None),
            "associated_drug": assoc_drug_info["drug_name"] if target_cls == "medication_associated" else (assoc_drug_info["drug_name"] if random.random() < 0.3 else None),
            "alternative_clinical_cause": alt_cause_event if target_cls == "other_cause" else None,
            "expected_temporal_pattern": "Decline_post_exposure" if target_cls == "medication_associated" else "Baseline_decline",
            "synthetic_truth_confidence": conf_val,
            "evaluation_split": split,
        })

        patient_idx += 1

    # Save all datasets to out_dir
    print(f"Writing V2 synthetic dataset to: {out_dir}")

    # Dataset 1: Patients
    df_pts = pd.DataFrame(patients)
    df_pts.to_csv(os.path.join(out_dir, "dataset1_Patients.csv"), index=False)
    print(f"  - dataset1_Patients.csv: {len(df_pts)} rows")

    # Dataset 2: Medication History
    df_med = pd.DataFrame(med_history)
    df_med_ref = drug_kb[["medication_id", "drug_name", "drug_class", "therapeutic_area"]].copy()
    with pd.ExcelWriter(os.path.join(out_dir, "dataset2_Medication_History.xlsx"), engine="openpyxl") as writer:
        df_med.to_excel(writer, sheet_name="medication_history", index=False)
        df_med_ref.to_excel(writer, sheet_name="medication_reference", index=False)
    print(f"  - dataset2_Medication_History.xlsx: {len(df_med)} history rows, {len(df_med_ref)} ref rows")

    # Dataset 3: Cognitive Behavioral
    df_cog = pd.DataFrame(cog_records)
    with pd.ExcelWriter(os.path.join(out_dir, "dataset3_Cognitive_Behavioral.xlsx"), engine="openpyxl") as writer:
        df_cog.to_excel(writer, sheet_name="cognitive_behavioral", index=False)
    print(f"  - dataset3_Cognitive_Behavioral.xlsx: {len(df_cog)} rows")

    # Dataset 4: Clinical Events
    df_ev = pd.DataFrame(clinical_events)
    with pd.ExcelWriter(os.path.join(out_dir, "dataset4_Clinical_Events.xlsx"), engine="openpyxl") as writer:
        df_ev.to_excel(writer, sheet_name="clinical_events", index=False)
    print(f"  - dataset4_Clinical_Events.xlsx: {len(df_ev)} rows")

    # Dataset 5: Drug Knowledge Base (copied from V1)
    with pd.ExcelWriter(os.path.join(out_dir, "dataset5_Drug_Knowledge_Base.xlsx"), engine="openpyxl") as writer:
        drug_kb.to_excel(writer, sheet_name="drug_knowledge", index=False)
    print(f"  - dataset5_Drug_Knowledge_Base.xlsx: {len(drug_kb)} rows")

    # Dataset 6: Ground Truth Cases
    df_gt = pd.DataFrame(gt_cases)
    with pd.ExcelWriter(os.path.join(out_dir, "dataset6_Ground_Truth_Cases.xlsx"), engine="openpyxl") as writer:
        df_gt.to_excel(writer, sheet_name="ground_truth_cases", index=False)
    print(f"  - dataset6_Ground_Truth_Cases.xlsx: {len(df_gt)} rows")

    print("Generation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Version 2 Synthetic Dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", type=str, default=OUT_DEFAULT, help="Output directory")
    args = parser.parse_args()

    generate_v2(seed=args.seed, out_dir=args.out_dir)
