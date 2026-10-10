"""Sequence Data Interface and Dataset Builder for Longitudinal 1D CNN + GRU Model.

Maintains original time ordering of cognitive assessments, clinical events,
and medication exposure without flattening or destroying temporal structures.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from ml.data import TARGET_CLASSES, SPLIT_COL, PATIENT_ID_COL
from ml.utils import DATA_V2_PROC

EVENT_CATEGORIES = [
    "Vascular", "Metabolic", "Hospitalization", "Infection",
    "Psychiatric", "Trauma", "Delirium", "Milestone"
]

DIAGNOSIS_LIST = [
    "Alzheimer's Disease", "Vascular Dementia", "Lewy Body Dementia",
    "Frontotemporal Dementia", "Mild Cognitive Impairment"
]

CONTINUOUS_COLS = [
    "delta_t_prev", "days_rel_anchor", "days_since_last_assessment",
    "mmse_score", "memory_score", "attention_score",
    "executive_function_score", "mood_score", "event_duration_days",
    "anchor_current_dose", "anchor_days_since_start",
    "n_other_active_meds", "age"
]

MAX_SEQ_LEN = 14


class SequenceDataset(Dataset):
    """PyTorch Dataset for longitudinal EHR sequences."""

    def __init__(self, sequences, lengths, targets, patient_ids=None, case_ids=None):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.patient_ids = patient_ids or []
        self.case_ids = case_ids or []

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "length": self.lengths[idx],
            "target": self.targets[idx],
            "patient_id": self.patient_ids[idx] if idx < len(self.patient_ids) else "",
            "case_id": self.case_ids[idx] if idx < len(self.case_ids) else ""
        }


class SequenceFeatureBuilder:
    """Extracts, standardizes, and formats longitudinal sequence data for patients."""

    def __init__(self, proc_dir=DATA_V2_PROC, max_seq_len=MAX_SEQ_LEN):
        self.proc_dir = proc_dir
        self.max_seq_len = max_seq_len
        self.feature_names = []
        self.scaler_params = {}
        self.train_medians = {}
        self.is_fitted = False

    def _extract_patient_episodes(self, df_epi):
        """Index medication episodes by patient_id for fast temporal lookup."""
        patient_episodes = {}
        for pid, group in df_epi.groupby("patient_id"):
            patient_episodes[pid] = group.to_dict(orient="records")
        return patient_episodes

    def build_raw_patient_sequences(self):
        """Build chronological multi-feature sequence matrices for all usable cases."""
        cases_path = os.path.join(self.proc_dir, "ml_cases.csv")
        obs_path = os.path.join(self.proc_dir, "patient_observations.csv")
        epi_path = os.path.join(self.proc_dir, "medication_episodes.csv")

        cases_df = pd.read_csv(cases_path)
        obs_df = pd.read_csv(obs_path)
        epi_df = pd.read_csv(epi_path)

        usable_cases = cases_df[cases_df.meta_usable_case == 1].copy()
        patient_epi_map = self._extract_patient_episodes(epi_df)

        # Index observations by patient_id using sorted dict records for speed
        obs_records = obs_df.sort_values(["patient_id", "obs_seq"]).to_dict(orient="records")
        patient_obs_map = {}
        for r in obs_records:
            patient_obs_map.setdefault(r["patient_id"], []).append(r)

        feature_cols_defined = False
        patient_records = []

        for case_row in usable_cases.to_dict(orient="records"):
            pid = case_row["patient_id"]
            case_id = case_row["case_id"]
            split = case_row["split"]
            target_str = case_row["target"]
            target_idx = TARGET_CLASSES.index(target_str)

            anchor_day = float(case_row["meta_days_enrollment_to_anchor"])
            anchor_med_id = case_row["meta_anchor_medication_id"]

            if pid not in patient_obs_map:
                continue

            p_obs_list = patient_obs_map[pid]
            p_epis = patient_epi_map.get(pid, [])

            # Locate anchor episode specifics
            anchor_epi = next((e for e in p_epis if e.get("medication_id") == anchor_med_id), None)
            anchor_start = anchor_day
            anchor_init_dose = float(anchor_epi["dose"]) if anchor_epi and pd.notna(anchor_epi.get("dose")) else 0.0
            anchor_new_dose = float(anchor_epi["new_dose"]) if anchor_epi and pd.notna(anchor_epi.get("new_dose")) else anchor_init_dose
            
            anchor_dose_chg_day = None
            if anchor_epi and pd.notna(anchor_epi.get("days_start_to_dose_change")):
                anchor_dose_chg_day = anchor_start + float(anchor_epi["days_start_to_dose_change"])

            anchor_stop_day = None
            if anchor_epi and pd.notna(anchor_epi.get("days_start_to_stop")):
                anchor_stop_day = anchor_start + float(anchor_epi["days_start_to_stop"])

            # Step-by-step state tracking
            last_day = 0.0
            last_assessment_day = None
            last_mmse = np.nan
            last_memory = np.nan
            last_attention = np.nan
            last_executive = np.nan
            last_mood = np.nan
            last_confusion = 0.0
            last_memory_loss = 0.0
            last_attention_prob = 0.0
            last_behavioral_chg = 0.0

            step_feature_rows = []

            for step_idx, obs_row in enumerate(p_obs_list):
                obs_day = float(obs_row["days_since_enrollment"])
                days_rel_anchor = obs_day - anchor_day
                # Strictly enforce 180-day follow-up prediction horizon
                if days_rel_anchor > 180.0:
                    continue
                delta_t = 0.0 if len(step_feature_rows) == 0 else max(0.0, obs_day - last_day)
                is_post_anchor = 1.0 if obs_day >= anchor_day else 0.0
                last_day = obs_day

                is_assessment = 1.0 if obs_row["obs_type"] == "cognitive_assessment" else 0.0
                is_event = 1.0 if obs_row["obs_type"] == "clinical_event" else 0.0

                if is_assessment == 1.0:
                    last_assessment_day = obs_day
                    last_mmse = float(obs_row["mmse_score"]) if pd.notna(obs_row.get("mmse_score")) else last_mmse
                    last_memory = float(obs_row["memory_score"]) if pd.notna(obs_row.get("memory_score")) else last_memory
                    last_attention = float(obs_row["attention_score"]) if pd.notna(obs_row.get("attention_score")) else last_attention
                    last_executive = float(obs_row["executive_function_score"]) if pd.notna(obs_row.get("executive_function_score")) else last_executive
                    last_mood = float(obs_row["mood_score"]) if pd.notna(obs_row.get("mood_score")) else last_mood
                    last_confusion = 1.0 if obs_row.get("confusion_present") == 1 else 0.0
                    last_memory_loss = 1.0 if obs_row.get("memory_loss_present") == 1 else 0.0
                    last_attention_prob = 1.0 if obs_row.get("attention_problem_present") == 1 else 0.0
                    last_behavioral_chg = 1.0 if obs_row.get("behavioral_change_present") == 1 else 0.0

                days_since_last_ass = 0.0 if last_assessment_day is None else max(0.0, obs_day - last_assessment_day)

                # Clinical event features
                ev_cat_ohe = [1.0 if obs_row.get("event_category") == cat else 0.0 for cat in EVENT_CATEGORIES]
                severity_map = {"Mild": 1.0, "Moderate": 2.0, "Severe": 3.0}
                ev_severity = severity_map.get(obs_row.get("severity"), 0.0) if is_event == 1.0 else 0.0
                ev_duration = float(obs_row.get("duration_days", 0.0)) if is_event == 1.0 and pd.notna(obs_row.get("duration_days")) else 0.0
                ev_cog_impact = 1.0 if is_event == 1.0 and obs_row.get("cognitive_impact_possible") == 1 else 0.0
                ev_hosp = 1.0 if is_event == 1.0 and obs_row.get("hospitalization_required") == 1 else 0.0

                # Medication status at obs_day
                if obs_day < anchor_start:
                    anchor_is_active = 0.0
                    anchor_cur_dose = 0.0
                    anchor_dose_chgd = 0.0
                    anchor_stopped = 0.0
                    anchor_time_since = 0.0
                else:
                    if anchor_stop_day is not None and obs_day >= anchor_stop_day:
                        anchor_is_active = 0.0
                        anchor_cur_dose = 0.0
                        anchor_stopped = 1.0
                    else:
                        anchor_is_active = 1.0
                        anchor_stopped = 0.0
                        if anchor_dose_chg_day is not None and obs_day >= anchor_dose_chg_day:
                            anchor_cur_dose = anchor_new_dose
                        else:
                            anchor_cur_dose = anchor_init_dose

                    anchor_dose_chgd = 1.0 if (anchor_dose_chg_day is not None and obs_day >= anchor_dose_chg_day) else 0.0
                    anchor_time_since = max(0.0, obs_day - anchor_start)

                # Concurrent other medications at obs_day
                n_other_active = 0.0
                for ep in p_epis:
                    if ep.get("medication_id") == anchor_med_id:
                        continue
                    ep_start = float(ep.get("days_enrollment_to_start", 0.0))
                    ep_stop = None
                    if pd.notna(ep.get("days_start_to_stop")):
                        ep_stop = ep_start + float(ep["days_start_to_stop"])
                    if ep_start <= obs_day:
                        if ep_stop is None or obs_day < ep_stop:
                            n_other_active += 1.0

                # Demographics & baseline
                age = float(obs_row.get("age", 75.0))
                sex_is_female = 1.0 if obs_row.get("sex") == "Female" else 0.0
                p_diag = obs_row.get("primary_diagnosis")
                diag_ohe = [1.0 if p_diag == d else 0.0 for d in DIAGNOSIS_LIST]
                stage_map = {"Mild": 1.0, "Moderate": 2.0, "Severe": 3.0}
                disease_stage = stage_map.get(obs_row.get("disease_stage"), 1.0)
                diag_dict = {
                    f"diag_{d.replace(' ', '_').replace(chr(39), '')}": diag_ohe[di]
                    for di, d in enumerate(DIAGNOSIS_LIST)
                }

                feat_dict = {
                    # Time features (3)
                    "delta_t_prev": delta_t,
                    "days_rel_anchor": days_rel_anchor,
                    "is_post_anchor": is_post_anchor,
                    # Cognitive trajectory (11)
                    "is_cognitive_assessment": is_assessment,
                    "days_since_last_assessment": days_since_last_ass,
                    "mmse_score": last_mmse,
                    "memory_score": last_memory,
                    "attention_score": last_attention,
                    "executive_function_score": last_executive,
                    "mood_score": last_mood,
                    "confusion_present": last_confusion,
                    "memory_loss_present": last_memory_loss,
                    "attention_problem_present": last_attention_prob,
                    "behavioral_change_present": last_behavioral_chg,
                    # Clinical events (14)
                    "is_clinical_event": is_event,
                    **{f"event_{cat}": ev_cat_ohe[ci] for ci, cat in enumerate(EVENT_CATEGORIES)},
                    "event_severity": ev_severity,
                    "event_duration_days": ev_duration,
                    "event_cognitive_impact": ev_cog_impact,
                    "event_hospitalization": ev_hosp,
                    # Medication exposure (6)
                    "anchor_is_active": anchor_is_active,
                    "anchor_current_dose": anchor_cur_dose,
                    "anchor_dose_changed": anchor_dose_chgd,
                    "anchor_stopped": anchor_stopped,
                    "anchor_days_since_start": anchor_time_since,
                    "n_other_active_meds": n_other_active,
                    # Baseline/Demographics (8)
                    "age": age,
                    "sex_is_female": sex_is_female,
                    **diag_dict,
                    "disease_stage": disease_stage,
                }

                if not feature_cols_defined:
                    self.feature_names = list(feat_dict.keys())
                    feature_cols_defined = True

                step_feature_rows.append(list(feat_dict.values()))

            patient_records.append({
                "patient_id": pid,
                "case_id": case_id,
                "split": split,
                "target": target_idx,
                "target_name": target_str,
                "seq_len": len(step_feature_rows),
                "steps": step_feature_rows
            })

        return patient_records

    def fit_and_transform(self, patient_records):
        """Fit normalization parameters strictly on train split, then transform all splits."""
        train_records = [r for r in patient_records if r["split"] == "train"]

        # Collect train step values for continuous columns to compute median, mean, std
        train_steps_all = []
        for r in train_records:
            train_steps_all.extend(r["steps"])
        train_df = pd.DataFrame(train_steps_all, columns=self.feature_names)

        # Compute medians and scalers on train split only
        self.scaler_params = {}
        self.train_medians = {}

        for col in self.feature_names:
            col_series = train_df[col].dropna()
            self.train_medians[col] = float(col_series.median()) if len(col_series) > 0 else 0.0

            if col in CONTINUOUS_COLS:
                mean_val = float(col_series.mean())
                std_val = float(col_series.std())
                if std_val < 1e-6:
                    std_val = 1.0
                self.scaler_params[col] = (mean_val, std_val)

        self.is_fitted = True

        # Transform and package tensors by split dynamically
        splits_data = {}
        for r in patient_records:
            sp = r["split"]
            if sp not in splits_data:
                splits_data[sp] = []

            steps_arr = np.array(r["steps"], dtype=np.float32)

            # Impute and standardize
            for col_idx, col_name in enumerate(self.feature_names):
                col_vals = steps_arr[:, col_idx]
                nan_mask = np.isnan(col_vals)
                if np.any(nan_mask):
                    col_vals[nan_mask] = self.train_medians[col_name]

                if col_name in self.scaler_params:
                    mean_val, std_val = self.scaler_params[col_name]
                    col_vals = (col_vals - mean_val) / std_val

                steps_arr[:, col_idx] = col_vals

            # Pad or truncate to max_seq_len
            actual_len = min(len(steps_arr), self.max_seq_len)
            padded = np.zeros((self.max_seq_len, len(self.feature_names)), dtype=np.float32)
            padded[:actual_len, :] = steps_arr[:actual_len, :]

            splits_data[sp].append({
                "patient_id": r["patient_id"],
                "case_id": r["case_id"],
                "sequence": padded,
                "length": actual_len,
                "target": r["target"],
                "target_name": r["target_name"]
            })

        # Convert to SequenceDataset
        datasets = {}
        for sp_name, items in splits_data.items():
            if not items:
                continue
            seqs = np.stack([it["sequence"] for it in items])
            lens = np.array([it["length"] for it in items])
            targs = np.array([it["target"] for it in items])
            pids = [it["patient_id"] for it in items]
            cids = [it["case_id"] for it in items]

            datasets[sp_name] = SequenceDataset(seqs, lens, targs, pids, cids)

        return datasets


def get_feature_group_masks(feature_names):
    """Generate index masks for trajectory-only ablation experiments."""
    trajectory_indices = []
    medication_indices = []
    event_indices = []

    for idx, name in enumerate(feature_names):
        if any(name.startswith(p) for p in ["event_", "is_clinical_event"]):
            event_indices.append(idx)
        elif any(name.startswith(p) for p in ["anchor_", "n_other_active_meds"]):
            medication_indices.append(idx)
        else:
            trajectory_indices.append(idx)

    return {
        "trajectory": trajectory_indices,
        "medication": medication_indices,
        "event": event_indices
    }
