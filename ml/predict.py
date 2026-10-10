"""Reusable Model Inference Pipeline for Patient EHR Longitudinal Analysis.

Supports:
  - Default: Tabular XGBoost Baseline (selected for superior validation macro-F1: 0.3793)
  - Optional: Longitudinal 1D CNN-GRU (for deep sequential pattern comparison)

Constraints:
  - Assessment cutoff strictly enforced at anchor_date + 180 days.
  - No data after cutoff or ground-truth metadata accessed.
  - Reuses pre-fitted preprocessing pipelines; never refits during inference.
  - Safe error and insufficient-data handling without invented predictions.
  - Standardized clinical terminology ("possible medication-associated pattern").
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.data import TARGET_CLASSES, get_feature_columns
from ml.utils import MODELS_DIR

# Constants matching Stage 2C and training pipeline
PRE_DAYS = 270
POST_DAYS = 180
EARLY_DAYS = 90
EV_PRE_DAYS = 90
EV_POST_DAYS = 180
CO_PRE_DAYS = 90
MIN_PRE = 1
MIN_POST = 2
MIN_SLOPE_SPAN = 21
DECLINE_THRESH = 1.5
D = np.timedelta64(1, "D")

SCORES = ["mmse_score", "memory_score", "attention_score", "executive_function_score", "mood_score"]
SHORT = {
    "mmse_score": "mmse",
    "memory_score": "memory",
    "attention_score": "attention",
    "executive_function_score": "executive",
    "mood_score": "mood"
}
FLAGS = ["confusion_present", "memory_loss_present", "attention_problem_present", "behavioral_change_present"]
POT = {"Low": 0, "Moderate": 1, "High": 2, "None": 0}
ACB = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
SEV = {"Mild": 1, "Moderate": 2, "Severe": 3}
STAGE = {"Mild": 1, "Moderate": 2, "Severe": 3}
EVENT_GROUPS = {
    "vascular_cardio": ["Stroke", "Transient ischemic attack", "Cardiovascular event", "Hypotension episode"],
    "infection": ["Infection", "Urinary tract infection", "Respiratory infection"],
    "metabolic": ["Dehydration", "Electrolyte abnormality", "Hyperglycemia", "Hypoglycemia"],
    "delirium": ["Delirium episode"],
    "trauma": ["Fall", "Head injury"],
    "procedure_hospital": ["Surgery", "Anesthesia exposure", "Hospitalization", "Emergency department visit"],
    "psych_sleep_pain": ["Depressive episode", "Sleep disruption", "Pain exacerbation"],
    "disease_milestone": ["Disease progression milestone"],
}

# Population reference constants matching Stage 2C V4.2 training cohort
POPULATION_MED_REF_DOSE = {f"M{i:03d}": 20.0 for i in range(1, 26)}
POPULATION_SCORE_SD = {
    "mmse_score": 4.788065,
    "memory_score": 15.764081,
    "attention_score": 14.715941,
    "executive_function_score": 14.708020,
    "mood_score": 7.136856
}

PATTERN_DESCRIPTIONS = {
    "medication_associated": "Possible medication-associated pattern",
    "disease_progression": "Possible disease progression pattern",
    "other_cause": "Possible other clinical cause pattern",
    "uncertain": "Uncertain clinical trajectory pattern"
}

# In-memory artifact cache
_LOADED_ARTIFACTS = {
    "preprocessor": None,
    "xgboost": None,
    "sequence_builder": None,
    "cnn_gru": None
}


class V42LongitudinalCNNGRU(nn.Module):
    """Longitudinal 1D CNN + GRU model for V4.2 sequence inference."""

    def __init__(
        self,
        input_dim=41,
        conv_dim=64,
        gru_dim=64,
        num_classes=4,
        dropout=0.30,
        kernel_size=3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.conv_dim = conv_dim
        self.gru_dim = gru_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(input_dim, conv_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(conv_dim)
        self.conv2 = nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(conv_dim)
        self.conv_drop = nn.Dropout(p=dropout)

        self.gru = nn.GRU(
            input_size=conv_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        pooling_dim = gru_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(pooling_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths):
        seq_len = x.shape[1]
        x_trans = x.transpose(1, 2)

        c1 = F.relu(self.bn1(self.conv1(x_trans)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c_out = self.conv_drop(c2).transpose(1, 2)

        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        c_out = c_out * mask.unsqueeze(-1).float()

        gru_out, _ = self.gru(c_out)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, self.gru_dim)
        idx = torch.clamp(idx, min=0)
        last_hidden = gru_out.gather(1, idx).squeeze(1)

        mask_expanded = mask.unsqueeze(-1).float()
        sum_hidden = (gru_out * mask_expanded).sum(dim=1)
        lengths_expanded = torch.clamp(lengths.unsqueeze(1).float(), min=1.0)
        mean_hidden = sum_hidden / lengths_expanded

        pooled = torch.cat([last_hidden, mean_hidden], dim=-1)
        return self.classifier(pooled)


def load_artifacts():
    """Load or retrieve cached models and preprocessors."""
    global _LOADED_ARTIFACTS

    # Tabular preprocessor
    if _LOADED_ARTIFACTS["preprocessor"] is None:
        prep_path = os.path.join(MODELS_DIR, "v4_2_preprocessor.joblib")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"V4.2 Preprocessor not found at {prep_path}")
        _LOADED_ARTIFACTS["preprocessor"] = joblib.load(prep_path)

    # XGBoost model
    if _LOADED_ARTIFACTS["xgboost"] is None:
        xgb_path = os.path.join(MODELS_DIR, "v4_2_xgboost.joblib")
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"V4.2 XGBoost model not found at {xgb_path}")
        _LOADED_ARTIFACTS["xgboost"] = joblib.load(xgb_path)

    # Sequence builder
    if _LOADED_ARTIFACTS["sequence_builder"] is None:
        seq_path = os.path.join(MODELS_DIR, "v4_2_sequence_builder.joblib")
        if os.path.exists(seq_path):
            _LOADED_ARTIFACTS["sequence_builder"] = joblib.load(seq_path)

    # CNN-GRU model
    if _LOADED_ARTIFACTS["cnn_gru"] is None:
        cnn_path = os.path.join(MODELS_DIR, "v4_2_cnn_gru.pt")
        if os.path.exists(cnn_path):
            ckpt = torch.load(cnn_path, weights_only=False)
            model = V42LongitudinalCNNGRU(
                input_dim=ckpt.get("input_dim", 41),
                conv_dim=64,
                gru_dim=64,
                num_classes=4,
                dropout=0.30
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _LOADED_ARTIFACTS["cnn_gru"] = model

    return _LOADED_ARTIFACTS


def slope30(days, vals):
    """OLS slope in score units per 30 days; NaN with <2 valid points or no time spread."""
    m = ~np.isnan(vals)
    if m.sum() < 2:
        return np.nan
    x, y = days[m].astype(float), vals[m]
    if np.ptp(x) < MIN_SLOPE_SPAN:
        return np.nan
    return float(np.polyfit(x, y, 1)[0] * 30)


def nanmean(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if a.size and not np.all(np.isnan(a)) else np.nan


def load_patient_from_db(patient_id, db_path=None):
    """Retrieve raw patient records from SQLite database."""
    default_db = os.path.join(ROOT, "data", "v4_2", "ehr.db")
    path = db_path or (default_db if os.path.exists(default_db) else os.path.join(ROOT, "data", "ehr.db"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Database not found at {path}")

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    q = lambda s, **k: pd.read_sql(s, conn, **k)

    pat_df = q("SELECT * FROM patients WHERE patient_id = ?", params=(patient_id,),
               parse_dates=["diagnosis_date", "enrollment_date"])
    if pat_df.empty:
        conn.close()
        return None

    com_df = q("SELECT * FROM patient_comorbidities WHERE patient_id = ?", params=(patient_id,))
    drugs_df = q("SELECT * FROM drugs").set_index("medication_id")
    mh_df = q("SELECT * FROM medication_history WHERE patient_id = ?", params=(patient_id,),
              parse_dates=["start_date", "dose_change_date", "stop_date"])
    cog_df = q("SELECT * FROM cognitive_behavioral WHERE patient_id = ?", params=(patient_id,),
               parse_dates=["assessment_date"]).sort_values("assessment_date")
    ev_df = q("SELECT * FROM clinical_events WHERE patient_id = ?", params=(patient_id,),
              parse_dates=["event_date"]).sort_values("event_date")

    # Drop any generator labels if present
    if "trajectory_context" in cog_df.columns:
        cog_df = cog_df.drop(columns=["trajectory_context"])

    conn.close()
    return {
        "patient": pat_df.iloc[0],
        "comorbidities": com_df,
        "drugs": drugs_df,
        "medication_history": mh_df,
        "cognitive": cog_df,
        "events": ev_df
    }


def resolve_anchor(patient_id, records, medication_anchor=None):
    """Identify the anchor medication episode and evaluate longitudinal eligibility."""
    mh = records["medication_history"]
    cog = records["cognitive"]
    drugs = records["drugs"]

    warnings = []
    adate = cog["assessment_date"].values

    if mh.empty:
        return None, ["Patient has no recorded medication episodes in history."]

    # 1. Candidate anchor identification
    cands = []
    for m in mh.itertuples():
        for typ, d in (("start", m.start_date), ("dose_change", m.dose_change_date)):
            if pd.isna(d):
                continue
            dd = np.datetime64(d)
            npre = int(((adate <= dd) & (adate > dd - PRE_DAYS * D)).sum())
            npost = int(((adate > dd) & (adate <= dd + POST_DAYS * D)).sum())
            cri = drugs.loc[m.medication_id, "cognitive_risk_index"] if m.medication_id in drugs.index else 0
            cands.append(dict(m=m, typ=typ, date=dd, npre=npre, npost=npost, bal=min(npre, npost),
                              cri=cri, trank=0 if typ == "start" else 1))

    if not cands:
        return None, ["No valid medication start or dose-change dates found in records."]

    # 2. Filter by user-requested anchor if provided
    selected_cand = None
    if medication_anchor is not None:
        med_id_filter = None
        if isinstance(medication_anchor, str):
            med_id_filter = medication_anchor.strip()
        elif isinstance(medication_anchor, dict):
            med_id_filter = medication_anchor.get("medication_id")

        if med_id_filter:
            matched = [c for c in cands if c["m"].medication_id.lower() == med_id_filter.lower() or
                       getattr(c["m"], "drug_name", "").lower() == med_id_filter.lower()]
            if not matched:
                return None, [f"Requested medication '{med_id_filter}' not found in patient history."]
            # Best within matched
            matched_elig = [c for c in matched if c["npre"] >= MIN_PRE and c["npost"] >= MIN_POST]
            if matched_elig:
                selected_cand = min(matched_elig, key=lambda x: (-x["bal"], x["date"], x["m"].medication_record_id, x["trank"]))
            else:
                selected_cand = matched[0]
                warnings.append(f"Requested medication '{med_id_filter}' lacks sufficient surrounding cognitive assessments.")

    # 3. Label-free default anchor selection
    if selected_cand is None:
        elig = [x for x in cands if x["npre"] >= MIN_PRE and x["npost"] >= MIN_POST]
        if elig:
            selected_cand = min(elig, key=lambda x: (-x["bal"], x["date"], x["m"].medication_record_id, x["trank"]))
        else:
            selected_cand = cands[0]
            warnings.append("No medication event satisfies the full pre/post observation criteria.")

    # Check eligibility thresholds
    if selected_cand["npre"] < MIN_PRE:
        warnings.append(f"Insufficient baseline cognitive assessments: observed {selected_cand['npre']} in [-270d, 0d] window (minimum required: {MIN_PRE}).")
    if selected_cand["npost"] < MIN_POST:
        warnings.append(f"Insufficient follow-up cognitive assessments: observed {selected_cand['npost']} in (0d, +180d] window (minimum required: {MIN_POST}).")

    return selected_cand, warnings


def dose_details(anchor_m, anchor_type):
    """Recorded dose values for the anchor episode, taken straight from medication_history.

    For a dose-change anchor, ``previous_dose`` is the episode's recorded starting
    dose and ``new_dose`` the recorded post-change dose. For a start anchor there is
    no dose change, so ``new_dose`` is None. Missing values stay None (never 0).
    """
    def _num(v):
        return float(v) if pd.notna(v) else None

    unit = getattr(anchor_m, "dose_unit", None)
    return {
        "previous_dose": _num(anchor_m.dose),
        "new_dose": _num(anchor_m.new_dose) if anchor_type == "dose_change" else None,
        "dose_unit": unit if isinstance(unit, str) and unit else None,
    }


def build_patient_tabular_features(records, anchor_cand):
    """Construct the exact 125 tabular input features using only data on or before anchor + 180 days."""
    pat = records["patient"]
    com = records["comorbidities"]
    drugs = records["drugs"]
    mh = records["medication_history"]
    cog = records["cognitive"]
    ev = records["events"]

    a = anchor_cand
    m, typ, ad = a["m"], a["typ"], a["date"]
    kd = drugs.loc[m.medication_id] if m.medication_id in drugs.index else drugs.iloc[0]

    # Pre/Post window masks
    adate = cog["assessment_date"].values
    pre_m = (adate <= ad) & (adate > ad - PRE_DAYS * D)
    post_m = (adate > ad) & (adate <= ad + POST_DAYS * D)
    early_m = (adate > ad) & (adate <= ad + EARLY_DAYS * D)
    late_m = (adate > ad + EARLY_DAYS * D) & (adate <= ad + POST_DAYS * D)
    rel = ((adate - ad) / D).astype(float)

    last_pre_idx = np.where(pre_m)[0].max() if pre_m.any() else None
    first_post_idx = np.where(post_m)[0].min() if post_m.any() else None

    r = {}
    r["feat_days_last_pre_to_anchor"] = abs(rel[last_pre_idx]) if last_pre_idx is not None else np.nan
    r["feat_days_anchor_to_first_post"] = rel[first_post_idx] if first_post_idx is not None else np.nan

    deltas = {}
    for s in SCORES:
        v = cog[s].values.astype(float)
        k = SHORT[s]
        pv = v[pre_m] if pre_m.any() else np.array([])
        pl = pv[~np.isnan(pv)][-1] if (~np.isnan(pv)).any() else np.nan
        ps = slope30(rel[pre_m], pv) if pre_m.any() else np.nan
        qs = slope30(rel[post_m], v[post_m]) if post_m.any() else np.nan
        pm = nanmean(v[post_m]) if post_m.any() else np.nan
        r[f"feat_{k}_pre_last"] = pl
        r[f"feat_{k}_pre_slope30"] = ps
        r[f"feat_{k}_post_mean"] = pm
        r[f"feat_{k}_post_slope30"] = qs
        r[f"feat_{k}_delta"] = pm - pl if not (np.isnan(pm) or np.isnan(pl)) else np.nan
        r[f"feat_{k}_slope_change"] = qs - ps if not (np.isnan(qs) or np.isnan(ps)) else np.nan
        r[f"feat_{k}_delta_early"] = nanmean(v[early_m]) - pl if (early_m.any() and not np.isnan(pl)) else np.nan
        r[f"feat_{k}_delta_late"] = nanmean(v[late_m]) - pl if (late_m.any() and not np.isnan(pl)) else np.nan
        deltas[s] = pm - pl if not (np.isnan(pm) or np.isnan(pl)) else np.nan

        if s == "mmse_score":
            if (~np.isnan(pv)).sum() >= 3 and not np.isnan(ps) and post_m.any() and (~np.isnan(v[post_m])).any():
                exp = pl + ps / 30 * np.nanmean(rel[post_m][~np.isnan(v[post_m])])
                r["feat_mmse_excess_decline"] = pm - exp
            else:
                r["feat_mmse_excess_decline"] = np.nan
            dec = np.where(post_m & (v <= pl - DECLINE_THRESH))[0] if (post_m.any() and not np.isnan(pl)) else []
            r["feat_mmse_days_to_first_decline"] = rel[dec.min()] if len(dec) else np.nan

    for f in FLAGS:
        v = cog[f].values.astype(float) if f in cog.columns else np.zeros(len(cog))
        k = f.replace("_present", "")
        a_, b_ = (nanmean(v[pre_m]) if pre_m.any() else np.nan), (nanmean(v[post_m]) if post_m.any() else np.nan)
        r[f"feat_{k}_pre_rate"] = a_
        r[f"feat_{k}_post_rate"] = b_
        r[f"feat_{k}_rate_delta"] = b_ - a_ if not (np.isnan(b_) or np.isnan(a_)) else np.nan

    # Medication dose & changes
    dose_after = m.new_dose if typ == "dose_change" and pd.notna(m.new_dose) else m.dose
    r["feat_anchor_is_dose_change"] = int(typ == "dose_change")
    r["feat_anchor_dose_mg"] = dose_after if pd.notna(dose_after) else np.nan
    ref_d = POPULATION_MED_REF_DOSE.get(m.medication_id, 20.0)
    r["feat_anchor_dose_rel_to_drug_median"] = (dose_after / ref_d) if (pd.notna(dose_after) and ref_d) else 1.0
    r["feat_dose_change_ratio"] = (m.new_dose / m.dose) if (typ == "dose_change" and pd.notna(m.dose) and m.dose and pd.notna(m.new_dose)) else np.nan
    r["feat_anchor_frequency"] = m.frequency if pd.notna(m.frequency) else "Missing"
    r["feat_anchor_adherence"] = m.adherence_status if isinstance(m.adherence_status, str) else "Missing"
    r["feat_days_on_drug_at_anchor"] = (ad - np.datetime64(m.start_date)) / D if pd.notna(m.start_date) else 0.0

    stop = np.datetime64(m.stop_date) if pd.notna(m.stop_date) else None
    stopped = stop is not None and ad < stop <= ad + POST_DAYS * D
    r["feat_stopped_in_post_window"] = int(stopped)
    r["feat_days_anchor_to_stop"] = (stop - ad) / D if stopped else np.nan

    dech = np.nan
    if stopped:
        v = cog["mmse_score"].values.astype(float)
        before = np.where((adate <= stop) & (adate >= ad) & ~np.isnan(v))[0]
        after = (adate > stop) & (adate <= min(stop + 90 * D, ad + POST_DAYS * D))
        if len(before) and after.any():
            dech = nanmean(v[after]) - v[before[-1]]
    r["feat_mmse_change_after_stop"] = dech

    if typ == "start" and pd.notna(m.dose_change_date):
        dc = np.datetime64(m.dose_change_date)
        later = ad < dc <= ad + POST_DAYS * D
    else:
        later, dc = False, None
    r["feat_later_dose_change_in_post_window"] = int(later)
    r["feat_days_anchor_to_later_dose_change"] = (dc - ad) / D if later else np.nan

    # Other active medications
    oth = mh[mh.medication_record_id != m.medication_record_id]
    act = oth[(oth.start_date.values <= ad) & (oth.stop_date.isna() | (oth.stop_date.values > ad))]
    ak = drugs.loc[act.medication_id] if len(act) else drugs.iloc[0:0]

    for col, nm in [("anticholinergic_index", "anticholinergic"), ("sedation_index", "sedation"),
                    ("cognitive_risk_index", "cogrisk"), ("interaction_risk_index", "interaction")]:
        r[f"feat_other_active_{nm}_sum"] = float(ak[col].sum()) if len(ak) else 0.0
    r["feat_other_active_cogrisk_max"] = float(ak.cognitive_risk_index.max()) if len(ak) else 0.0
    r["feat_other_active_any_high_anticholinergic"] = int((ak.anticholinergic_burden == "High").any()) if len(ak) else 0
    r["feat_other_active_any_high_sedation"] = int((ak.sedation_potential == "High").any()) if len(ak) else 0
    r["feat_other_active_any_cognitive_therapy"] = int((ak.therapeutic_area == "Cognitive").any()) if len(ak) else 0

    chg = []
    for o in oth.itertuples():
        for kind, d_ in (("start", o.start_date), ("dose_change", o.dose_change_date), ("stop", o.stop_date)):
            if pd.notna(d_):
                chg.append((np.datetime64(d_), o.medication_id, kind))
    before_c = [(ad - d_) / D for d_, _, k_ in chg if k_ != "stop" and ad - CO_PRE_DAYS * D <= d_ <= ad]
    after_c = [(d_ - ad) / D for d_, _, k_ in chg if k_ != "stop" and ad < d_ <= ad + POST_DAYS * D]
    stops_c = [(d_ - ad) / D for d_, _, k_ in chg if k_ == "stop" and ad < d_ <= ad + POST_DAYS * D]
    r["feat_other_med_change_days_before"] = min(before_c) if before_c else np.nan
    r["feat_other_med_change_days_after"] = min(after_c) if after_c else np.nan
    r["feat_other_med_stop_days_after"] = min(stops_c) if stops_c else np.nan
    win_c = [drugs.loc[mid, "cognitive_risk_index"] for d_, mid, k_ in chg
             if k_ != "stop" and ad - CO_PRE_DAYS * D <= d_ <= ad + POST_DAYS * D and mid in drugs.index]
    r["feat_competing_med_change_max_cogrisk"] = float(max(win_c)) if win_c else 0.0

    # Clinical events
    edt = ev["event_date"].values if not ev.empty else np.array([])
    pre_e = ev[(edt <= ad) & (edt > ad - EV_PRE_DAYS * D)] if len(edt) else ev.iloc[0:0]
    post_e = ev[(edt > ad) & (edt <= ad + EV_POST_DAYS * D)] if len(edt) else ev.iloc[0:0]
    win_e = pd.concat([pre_e, post_e])

    sev_fn = lambda x: float(x.severity.map(SEV).max()) if len(x) and x.severity.notna().any() else 0.0
    r["feat_ev_pre_max_severity"] = sev_fn(pre_e)
    r["feat_ev_pre_any_cog_impact"] = int((pre_e.cognitive_impact_possible == "Yes").any()) if len(pre_e) else 0
    r["feat_ev_pre_days_since_last"] = (ad - pre_e.event_date.values.max()) / D if len(pre_e) else np.nan
    r["feat_ev_post_max_severity"] = sev_fn(post_e)
    r["feat_ev_post_any_cog_impact"] = int((post_e.cognitive_impact_possible == "Yes").any()) if len(post_e) else 0
    r["feat_ev_post_any_hospitalization"] = int((post_e.hospitalization_required == "Yes").any()) if len(post_e) else 0
    r["feat_ev_post_days_to_first"] = (post_e.event_date.values.min() - ad) / D if len(post_e) else np.nan
    gap_e = ev[(edt > adate[last_pre_idx]) & (edt <= adate[first_post_idx])] if (last_pre_idx is not None and first_post_idx is not None and len(edt)) else ev.iloc[0:0]
    r["feat_ev_in_assessment_gap"] = int(len(gap_e) > 0)
    r["feat_ev_window_max_duration_days"] = float(win_e.duration_days.max()) if (len(win_e) and win_e.duration_days.notna().any()) else 0.0

    for g, types in EVENT_GROUPS.items():
        r[f"feat_ev_has_{g}"] = int(win_e.event_type.isin(types).any()) if len(win_e) else 0
    if len(win_e):
        dist = ((win_e.event_date.values - ad) / D).astype(float)
        j = int(np.argmin(np.abs(dist)))
        r["feat_ev_nearest_abs_days"] = abs(dist[j])
        r["feat_ev_nearest_category"] = win_e.event_category.values[j]
    else:
        r["feat_ev_nearest_abs_days"] = np.nan
        r["feat_ev_nearest_category"] = "None"

    # Patient demographics & baseline
    r["feat_age"] = pat.age if pd.notna(pat.age) else 75.0
    r["feat_sex"] = pat.sex if pd.notna(pat.sex) else "Female"
    r["feat_primary_diagnosis"] = pat.primary_diagnosis if pd.notna(pat.primary_diagnosis) else "Alzheimer's Disease"
    r["feat_disease_stage"] = STAGE.get(pat.disease_stage, 1.0)
    diag_d = np.datetime64(pat.diagnosis_date) if pd.notna(pat.diagnosis_date) else ad
    r["feat_disease_duration_years"] = max(0.0, (ad - diag_d) / D / 365.25)

    comorb_types = ['atrial_fibrillation', 'ckd', 'depression', 'dyslipidemia', 'hypertension', 'osteoarthritis', 'type_2_diabetes']
    user_comorbs = set(com["comorbidity"].str.lower().str.replace(" ", "_").tolist()) if not com.empty else set()
    for t in comorb_types:
        r[f"feat_comorb_{t}"] = int(any(t in uc for uc in user_comorbs))

    # Drug Knowledge Base Concordance
    r["feat_drug_cognitive_risk_index"] = float(kd.cognitive_risk_index) if pd.notna(kd.cognitive_risk_index) else 0.0
    r["feat_drug_sedation_index"] = float(kd.sedation_index) if pd.notna(kd.sedation_index) else 0.0
    r["feat_drug_anticholinergic_index"] = float(kd.anticholinergic_index) if pd.notna(kd.anticholinergic_index) else 0.0
    r["feat_drug_interaction_risk_index"] = float(kd.interaction_risk_index) if pd.notna(kd.interaction_risk_index) else 0.0

    for col, nm in [("memory_effect_potential", "memory"), ("confusion_effect_potential", "confusion"),
                    ("attention_effect_potential", "attention"), ("mood_effect_potential", "mood"),
                    ("sedation_potential", "sedation")]:
        r[f"feat_drug_{nm}_potential"] = POT.get(kd.get(col), 0)
    r["feat_drug_anticholinergic_burden"] = ACB.get(kd.get("anticholinergic_burden"), 0)
    r["feat_drug_interaction_risk"] = POT.get(kd.get("interaction_risk"), 0)
    r["feat_drug_evidence_category"] = kd.get("evidence_category", "Class D")
    r["feat_drug_cognitive_relevance"] = kd.get("cognitive_relevance", "None")

    conc = 0.0
    sd_map = POPULATION_SCORE_SD
    for s, col in [("memory_score", "memory_effect_potential"), ("attention_score", "attention_effect_potential"),
                   ("mood_score", "mood_effect_potential")]:
        dv = deltas.get(s, np.nan)
        conc += POT.get(kd.get(col), 0) * max(0.0, -dv / sd_map[s]) if not np.isnan(dv) else 0.0
    cd = r.get("feat_confusion_rate_delta", np.nan)
    conc += POT.get(kd.get("confusion_effect_potential"), 0) * max(0.0, cd) if not np.isnan(cd) else 0.0
    r["feat_knowledge_concordant_decline"] = conc

    return r


def build_patient_sequence(records, anchor_cand, sequence_builder):
    """Construct normalized multi-step sequence tensor for CNN-GRU."""
    pat = records["patient"]
    drugs = records["drugs"]
    cog = records["cognitive"].copy()
    ev = records["events"].copy()
    mh = records["medication_history"].copy()

    a = anchor_cand
    anchor_day_ts = pd.Timestamp(a["date"])
    cutoff_ts = anchor_day_ts + pd.Timedelta(days=180)
    enroll_ts = pd.Timestamp(pat.enrollment_date) if pd.notna(pat.enrollment_date) else anchor_day_ts

    anchor_med_id = a["m"].medication_id
    anchor_start_day = (anchor_day_ts - enroll_ts).days
    anchor_init_dose = float(a["m"].dose) if pd.notna(a["m"].dose) else 0.0
    anchor_new_dose = float(a["m"].new_dose) if pd.notna(a["m"].new_dose) else anchor_init_dose

    anchor_dose_chg_day = None
    if pd.notna(a["m"].dose_change_date):
        anchor_dose_chg_day = (pd.Timestamp(a["m"].dose_change_date) - enroll_ts).days

    anchor_stop_day = None
    if pd.notna(a["m"].stop_date):
        anchor_stop_day = (pd.Timestamp(a["m"].stop_date) - enroll_ts).days

    # Merge observations and events chronologically on or before cutoff
    steps_list = []
    for _, row in cog[cog["assessment_date"] <= cutoff_ts].iterrows():
        steps_list.append({
            "date": pd.Timestamp(row["assessment_date"]),
            "type": "cognitive_assessment",
            "row": row
        })
    for _, row in ev[ev["event_date"] <= cutoff_ts].iterrows():
        steps_list.append({
            "date": pd.Timestamp(row["event_date"]),
            "type": "clinical_event",
            "row": row
        })
    steps_list.sort(key=lambda x: x["date"])

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
    for s_item in steps_list:
        obs_day = max(0.0, float((s_item["date"] - enroll_ts).days))
        days_rel_anchor = float((s_item["date"] - anchor_day_ts).days)
        if days_rel_anchor > 180.0:
            continue

        delta_t = 0.0 if len(step_feature_rows) == 0 else max(0.0, obs_day - last_day)
        is_post_anchor = 1.0 if days_rel_anchor >= 0 else 0.0
        last_day = obs_day

        is_assessment = 1.0 if s_item["type"] == "cognitive_assessment" else 0.0
        is_event = 1.0 if s_item["type"] == "clinical_event" else 0.0

        if is_assessment == 1.0:
            crow = s_item["row"]
            last_assessment_day = obs_day
            last_mmse = float(crow["mmse_score"]) if pd.notna(crow.get("mmse_score")) else last_mmse
            last_memory = float(crow["memory_score"]) if pd.notna(crow.get("memory_score")) else last_memory
            last_attention = float(crow["attention_score"]) if pd.notna(crow.get("attention_score")) else last_attention
            last_executive = float(crow["executive_function_score"]) if pd.notna(crow.get("executive_function_score")) else last_executive
            last_mood = float(crow["mood_score"]) if pd.notna(crow.get("mood_score")) else last_mood
            last_confusion = 1.0 if crow.get("confusion_present") == 1 else 0.0
            last_memory_loss = 1.0 if crow.get("memory_loss_present") == 1 else 0.0
            last_attention_prob = 1.0 if crow.get("attention_problem_present") == 1 else 0.0
            last_behavioral_chg = 1.0 if crow.get("behavioral_change_present") == 1 else 0.0

        days_since_last_ass = 0.0 if last_assessment_day is None else max(0.0, obs_day - last_assessment_day)

        # Clinical event details
        erow = s_item["row"] if is_event == 1.0 else {}
        event_categories = ["Vascular", "Metabolic", "Hospitalization", "Infection", "Psychiatric", "Trauma", "Delirium", "Milestone"]
        ev_cat_ohe = [1.0 if erow.get("event_category") == cat else 0.0 for cat in event_categories]
        severity_map = {"Mild": 1.0, "Moderate": 2.0, "Severe": 3.0}
        ev_severity = severity_map.get(erow.get("severity"), 0.0) if is_event == 1.0 else 0.0
        ev_duration = float(erow.get("duration_days", 0.0)) if is_event == 1.0 and pd.notna(erow.get("duration_days")) else 0.0
        ev_cog_impact = 1.0 if is_event == 1.0 and erow.get("cognitive_impact_possible") == "Yes" else 0.0
        ev_hosp = 1.0 if is_event == 1.0 and erow.get("hospitalization_required") == "Yes" else 0.0

        # Medication status at obs_day
        if obs_day < anchor_start_day:
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
            anchor_time_since = max(0.0, obs_day - anchor_start_day)

        n_other_active = 0.0
        for ep in mh.itertuples():
            if ep.medication_id == anchor_med_id or pd.isna(ep.start_date):
                continue
            ep_start = (pd.Timestamp(ep.start_date) - enroll_ts).days
            ep_stop = (pd.Timestamp(ep.stop_date) - enroll_ts).days if pd.notna(ep.stop_date) else None
            if ep_start <= obs_day and (ep_stop is None or obs_day < ep_stop):
                n_other_active += 1.0

        age = float(pat.age) if pd.notna(pat.age) else 75.0
        sex_is_female = 1.0 if pat.sex == "Female" else 0.0
        diagnosis_list = ["Alzheimer's Disease", "Vascular Dementia", "Lewy Body Dementia", "Frontotemporal Dementia", "Mild Cognitive Impairment"]
        diag_dict = {
            f"diag_{d.replace(' ', '_').replace(chr(39), '')}": 1.0 if pat.primary_diagnosis == d else 0.0
            for d in diagnosis_list
        }
        disease_stage = STAGE.get(pat.disease_stage, 1.0)

        feat_dict = {
            "delta_t_prev": delta_t,
            "days_rel_anchor": days_rel_anchor,
            "is_post_anchor": is_post_anchor,
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
            "is_clinical_event": is_event,
            **{f"event_{cat}": ev_cat_ohe[ci] for ci, cat in enumerate(event_categories)},
            "event_severity": ev_severity,
            "event_duration_days": ev_duration,
            "event_cognitive_impact": ev_cog_impact,
            "event_hospitalization": ev_hosp,
            "anchor_is_active": anchor_is_active,
            "anchor_current_dose": anchor_cur_dose,
            "anchor_dose_changed": anchor_dose_chgd,
            "anchor_stopped": anchor_stopped,
            "anchor_days_since_start": anchor_time_since,
            "n_other_active_meds": n_other_active,
            "age": age,
            "sex_is_female": sex_is_female,
            **diag_dict,
            "disease_stage": disease_stage,
        }
        step_feature_rows.append([feat_dict[fn] for fn in sequence_builder.feature_names])

    if not step_feature_rows:
        return None, 0

    steps_arr = np.array(step_feature_rows, dtype=np.float32)
    for col_idx, col_name in enumerate(sequence_builder.feature_names):
        col_vals = steps_arr[:, col_idx]
        nan_m = np.isnan(col_vals)
        if np.any(nan_m):
            col_vals[nan_m] = sequence_builder.train_medians.get(col_name, 0.0)
        if col_name in sequence_builder.scaler_params:
            m_, s_ = sequence_builder.scaler_params[col_name]
            col_vals = (col_vals - m_) / s_
        steps_arr[:, col_idx] = col_vals

    actual_len = min(len(steps_arr), sequence_builder.max_seq_len)
    padded = np.zeros((1, sequence_builder.max_seq_len, len(sequence_builder.feature_names)), dtype=np.float32)
    padded[0, :actual_len, :] = steps_arr[:actual_len, :]
    return torch.tensor(padded), actual_len


def extract_supporting_evidence(feat_dict, anchor_details):
    """Generate concise factual timeline evidence for clinical interpretability."""
    evidence = []

    # 1. Baseline and drop
    pl = feat_dict.get("feat_mmse_pre_last")
    pm = feat_dict.get("feat_mmse_post_mean")
    delta = feat_dict.get("feat_mmse_delta")
    if pd.notna(pl) and pd.notna(pm):
        evidence.append(f"MMSE score changed by {delta:+.1f} points (last baseline: {pl:.1f}, 180-day post mean: {pm:.1f}).")

    # 2. Slope trajectory change
    ps = feat_dict.get("feat_mmse_pre_slope30")
    qs = feat_dict.get("feat_mmse_post_slope30")
    sc = feat_dict.get("feat_mmse_slope_change")
    if pd.notna(sc):
        direction = "accelerated decline" if sc < -0.1 else ("attenuated" if sc > 0.1 else "remained steady")
        evidence.append(f"Cognitive trajectory {direction} post-anchor (30-day slope: pre {ps:+.2f} vs post {qs:+.2f}).")

    # 3. Anchor drug properties
    drug_name = anchor_details.get("drug_name", "Medication")
    cri = feat_dict.get("feat_drug_cognitive_risk_index", 0.0)
    acb = feat_dict.get("feat_drug_anticholinergic_burden", 0)
    if cri > 1.5 or acb > 0:
        evidence.append(f"Anchor drug {drug_name} has cognitive risk index {cri:.1f} and anticholinergic burden level {acb}.")

    # 4. Clinical events in window
    ev_post = feat_dict.get("feat_ev_post_any_cog_impact", 0)
    ev_hosp = feat_dict.get("feat_ev_post_any_hospitalization", 0)
    ev_cat = feat_dict.get("feat_ev_nearest_category", "None")
    if ev_post == 1 or ev_hosp == 1:
        evidence.append(f"Concurrent acute event ({ev_cat}) observed in follow-up horizon with functional/hospitalization impact.")
    elif ev_cat != "None":
        evidence.append(f"Nearest acute event category in window: {ev_cat}.")

    # 5. Dose change or discontinuation
    stopped = feat_dict.get("feat_stopped_in_post_window", 0)
    rech = feat_dict.get("feat_mmse_change_after_stop")
    if stopped == 1:
        if pd.notna(rech) and rech > 0:
            evidence.append(f"Medication discontinued within follow-up window with subsequent MMSE rebound ({rech:+.1f} pts).")
        else:
            evidence.append("Medication discontinued within follow-up window without significant cognitive rebound.")

    return evidence


def predict_patient(
    patient_id,
    medication_anchor=None,
    model_type="xgboost",
    db_path=None
):
    """Analyze a patient's longitudinal EHR records and predict trajectory pattern.

    Args:
        patient_id (str): Unique patient identifier (e.g. 'V4_2_P0001').
        medication_anchor (str|dict, optional): Specific medication ID (e.g. 'M001') or episode details.
            If None, resolves using the standard label-free candidate anchor rule.
        model_type (str): 'xgboost' (default, higher validation macro-F1) or 'cnn_gru'.
        db_path (str, optional): Custom SQLite database path.

    Returns:
        dict: Standardized prediction result containing status, predicted_class, pattern_description,
              probabilities, confidence, model_name, anchor_details, observation dates, warnings,
              and supporting evidence.
    """
    artifacts = load_artifacts()
    prep = artifacts["preprocessor"]
    xgb_model = artifacts["xgboost"]
    cnn_model = artifacts["cnn_gru"]
    seq_builder = artifacts["sequence_builder"]

    model_display_name = "XGBoost (Tabular Baseline)" if model_type.lower() == "xgboost" else "Longitudinal 1D CNN-GRU"

    # 1. Retrieve raw patient records
    records = load_patient_from_db(patient_id, db_path=db_path)
    if records is None:
        return {
            "status": "insufficient_data",
            "predicted_class": None,
            "pattern_description": "Insufficient longitudinal data for reliable assessment",
            "probabilities": {c: 0.0 for c in TARGET_CLASSES},
            "confidence": 0.0,
            "model_name": model_display_name,
            "anchor_details": None,
            "usable_observation_dates": {"pre_anchor": [], "post_anchor": [], "cutoff_date": None},
            "data_quality_warnings": [f"Patient ID '{patient_id}' not found in database records."],
            "supporting_evidence": []
        }

    # 2. Resolve anchor episode and verify eligibility
    anchor_cand, warnings = resolve_anchor(patient_id, records, medication_anchor=medication_anchor)
    if anchor_cand is None:
        return {
            "status": "insufficient_data",
            "predicted_class": None,
            "pattern_description": "Insufficient longitudinal data for reliable assessment",
            "probabilities": {c: 0.0 for c in TARGET_CLASSES},
            "confidence": 0.0,
            "model_name": model_display_name,
            "anchor_details": None,
            "usable_observation_dates": {"pre_anchor": [], "post_anchor": [], "cutoff_date": None},
            "data_quality_warnings": warnings,
            "supporting_evidence": []
        }

    anchor_date = pd.Timestamp(anchor_cand["date"])
    cutoff_date = anchor_date + pd.Timedelta(days=180)
    anchor_m = anchor_cand["m"]
    drugs = records["drugs"]
    kd = drugs.loc[anchor_m.medication_id] if anchor_m.medication_id in drugs.index else None

    # Collect usable observation dates within [anchor - 270d, anchor + 180d]
    cog = records["cognitive"]
    pre_obs = cog[(cog["assessment_date"] <= anchor_date) & (cog["assessment_date"] >= anchor_date - pd.Timedelta(days=270))]
    post_obs = cog[(cog["assessment_date"] > anchor_date) & (cog["assessment_date"] <= cutoff_date)]

    usable_dates = {
        "pre_anchor": [d.strftime("%Y-%m-%d") for d in pre_obs["assessment_date"]],
        "post_anchor": [d.strftime("%Y-%m-%d") for d in post_obs["assessment_date"]],
        "cutoff_date": cutoff_date.strftime("%Y-%m-%d")
    }

    # If insufficient assessments, return explicit status
    if anchor_cand["npre"] < MIN_PRE or anchor_cand["npost"] < MIN_POST:
        return {
            "status": "insufficient_data",
            "predicted_class": None,
            "pattern_description": "Insufficient longitudinal data for reliable assessment",
            "probabilities": {c: 0.0 for c in TARGET_CLASSES},
            "confidence": 0.0,
            "model_name": model_display_name,
            "anchor_details": {
                "medication_id": anchor_m.medication_id,
                "drug_name": getattr(kd, "drug_name", "Unknown"),
                "drug_class": getattr(kd, "drug_class", "Unknown"),
                "anchor_date": anchor_date.strftime("%Y-%m-%d"),
                "anchor_type": anchor_cand["typ"],
                **dose_details(anchor_m, anchor_cand["typ"]),
                "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
                "horizon_days": 180
            },
            "usable_observation_dates": usable_dates,
            "data_quality_warnings": warnings,
            "supporting_evidence": []
        }

    # 3. Build features strictly up to cutoff
    feat_dict = build_patient_tabular_features(records, anchor_cand)

    anchor_info = {
        "medication_id": anchor_m.medication_id,
        "drug_name": getattr(kd, "drug_name", "Unknown"),
        "drug_class": getattr(kd, "drug_class", "Unknown"),
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "anchor_type": anchor_cand["typ"],
        "dose_mg": float(anchor_m.dose) if pd.notna(anchor_m.dose) else 0.0,
        **dose_details(anchor_m, anchor_cand["typ"]),
        "cutoff_date": cutoff_date.strftime("%Y-%m-%d"),
        "horizon_days": 180
    }

    # 4. Perform Model Inference
    if model_type.lower() == "cnn_gru":
        if cnn_model is None or seq_builder is None:
            raise RuntimeError("CNN-GRU model or sequence builder artifact is missing.")
        seq_tensor, seq_len = build_patient_sequence(records, anchor_cand, seq_builder)
        if seq_tensor is None or seq_len == 0:
            return {
                "status": "insufficient_data",
                "predicted_class": None,
                "pattern_description": "Insufficient longitudinal sequence points",
                "probabilities": {c: 0.0 for c in TARGET_CLASSES},
                "confidence": 0.0,
                "model_name": model_display_name,
                "anchor_details": anchor_info,
                "usable_observation_dates": usable_dates,
                "data_quality_warnings": warnings + ["Could not assemble valid sequence steps."],
                "supporting_evidence": []
            }
        with torch.no_grad():
            logits = cnn_model(seq_tensor, torch.tensor([seq_len]))
            probs_arr = F.softmax(logits, dim=1).numpy()[0]
    else:
        # Default: XGBoost Tabular
        df_single = pd.DataFrame([feat_dict])
        for col in prep.numeric_cols + prep.categorical_cols:
            if col not in df_single.columns:
                df_single[col] = np.nan
        df_single = df_single[prep.numeric_cols + prep.categorical_cols]
        X_proc = prep.transform(df_single)
        probs_arr = xgb_model.predict_proba(X_proc)[0]

    prob_dict = {cls_name: round(float(probs_arr[i]), 4) for i, cls_name in enumerate(TARGET_CLASSES)}
    pred_idx = int(np.argmax(probs_arr))
    pred_class = TARGET_CLASSES[pred_idx]
    confidence = float(np.max(probs_arr))

    # 5. Extract supporting evidence
    evidence = extract_supporting_evidence(feat_dict, anchor_info)

    return {
        "status": "success",
        "predicted_class": pred_class,
        "pattern_description": PATTERN_DESCRIPTIONS.get(pred_class, f"Possible {pred_class} pattern"),
        "probabilities": prob_dict,
        "confidence": round(confidence, 4),
        "model_name": model_display_name,
        "anchor_details": anchor_info,
        "usable_observation_dates": usable_dates,
        "data_quality_warnings": warnings,
        "supporting_evidence": evidence
    }


def predict_case(feature_dict_or_df, model_type="xgboost"):
    """Legacy helper for pre-computed feature dictionaries or DataFrames."""
    artifacts = load_artifacts()
    prep = artifacts["preprocessor"]
    model = artifacts["xgboost"]

    if isinstance(feature_dict_or_df, dict):
        df_in = pd.DataFrame([feature_dict_or_df])
    elif isinstance(feature_dict_or_df, pd.Series):
        df_in = pd.DataFrame([feature_dict_or_df.to_dict()])
    else:
        df_in = pd.DataFrame(feature_dict_or_df).copy()

    for col in prep.numeric_cols + prep.categorical_cols:
        if col not in df_in.columns:
            df_in[col] = np.nan

    X_proc = prep.transform(df_in)
    probs = model.predict_proba(X_proc)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = TARGET_CLASSES[pred_idx]

    return {
        "status": "success",
        "predicted_class": pred_class,
        "pattern_description": PATTERN_DESCRIPTIONS.get(pred_class, f"Possible {pred_class} pattern"),
        "probabilities": {cls_name: round(float(probs[i]), 4) for i, cls_name in enumerate(TARGET_CLASSES)},
        "confidence": round(float(np.max(probs)), 4),
        "model_used": "v4_2_xgboost"
    }
