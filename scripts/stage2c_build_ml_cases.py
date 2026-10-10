"""Stage 2C: build the case-level ML dataset (no model is trained).

Reads the SQLite DB read-only (+ Stage 2B labels only AFTER features are built).
Writes: data/processed/ml_cases.csv   (one row per ground-truth case)

Usage:  python scripts/stage2c_build_ml_cases.py [--anchor-rule balanced|max_cog_risk]

Column groups in ml_cases.csv
  case_id, patient_id, split, target   identifiers / label (target is NEVER a feature)
  meta_*                               audit/bookkeeping; must NOT be used as features
  feat_*                               model inputs (all derived only from data around the anchor)

Anchor (identical for all four classes, label-free):
  candidates = every medication START and every DOSE CHANGE of every medication record.
  eligible   = >=1 cognitive assessment in (anchor-270d, anchor]  (baseline; same-day counts as baseline)
               and >=2 assessments in (anchor, anchor+180d].
  choice     = 'balanced' (default): max min(n_pre, n_post); ties -> earliest date, then record id,
               start before dose change.  Uses no drug knowledge and no label.
               'max_cog_risk' (optional): highest drug cognitive_risk_index first, then as above.
"""
import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config  # noqa: E402

OUT = os.path.join(ROOT, "data", "processed", "ml_cases.csv")
LABELS = os.path.join(ROOT, "data", "processed", "ground_truth_labels.csv")

PRE_DAYS, POST_DAYS, EARLY_DAYS = 270, 180, 90
MIN_PRE, MIN_POST = 1, 2
EV_PRE_DAYS, EV_POST_DAYS = 90, 180
CO_PRE_DAYS = 90
MIN_SLOPE_SPAN = 21           # days of spread required before a slope is computed
DECLINE_THRESH = 1.5          # MMSE points below the last baseline value
SCORES = ["mmse_score", "memory_score", "attention_score", "executive_function_score", "mood_score"]
SHORT = {"mmse_score": "mmse", "memory_score": "memory", "attention_score": "attention",
         "executive_function_score": "executive", "mood_score": "mood"}
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
D = np.timedelta64(1, "D")


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


def load(conn):
    q = lambda s, **k: pd.read_sql(s, conn, **k)  # noqa: E731
    pat = q("SELECT * FROM patients", parse_dates=["diagnosis_date", "enrollment_date"]).set_index("patient_id")
    com = q("SELECT * FROM patient_comorbidities")
    drugs = q("SELECT * FROM drugs").set_index("medication_id")
    mh = q("SELECT * FROM medication_history", parse_dates=["start_date", "dose_change_date", "stop_date"])
    cog = q("SELECT * FROM cognitive_behavioral", parse_dates=["assessment_date"]).sort_values(["patient_id", "assessment_date"])
    ev = q("SELECT * FROM clinical_events", parse_dates=["event_date"]).sort_values(["patient_id", "event_date"])
    # trajectory_context is dropped immediately and never used
    cog = cog.drop(columns=["trajectory_context"])
    return pat, com, drugs, mh, cog, ev


def build(conn, rule):
    pat, com, drugs, mh, cog, ev = load(conn)
    comorb_types = sorted(com.comorbidity.unique())
    comorb = com.groupby("patient_id").comorbidity.apply(set)
    # label-free population constants
    sd = {s: float(cog[s].std()) for s in SCORES}
    med_ref_dose = mh.groupby("medication_id").dose.median()
    cg = {p: g for p, g in cog.groupby("patient_id")}
    eg = {p: g for p, g in ev.groupby("patient_id")}
    mg = {p: g for p, g in mh.groupby("patient_id")}
    rows = []
    for pid in pat.index:
        P = pat.loc[pid]
        c, meds = cg[pid], mg[pid]
        adate = c.assessment_date.values
        r = {"patient_id": pid, "meta_usable_case": 0}
        # ---------------- candidates
        cands = []
        for m in meds.itertuples():
            for typ, d in (("start", m.start_date), ("dose_change", m.dose_change_date)):
                if pd.isna(d):
                    continue
                dd = np.datetime64(d)
                npre = int(((adate <= dd) & (adate > dd - PRE_DAYS * D)).sum())
                npost = int(((adate > dd) & (adate <= dd + POST_DAYS * D)).sum())
                cri = drugs.loc[m.medication_id, "cognitive_risk_index"]
                cands.append(dict(m=m, typ=typ, date=dd, npre=npre, npost=npost, bal=min(npre, npost),
                                  cri=cri, trank=0 if typ == "start" else 1))
        elig = [x for x in cands if x["npre"] >= MIN_PRE and x["npost"] >= MIN_POST]
        r["meta_n_candidate_anchors"] = len(cands)
        r["meta_n_eligible_anchors"] = len(elig)
        if not elig:
            if not cands:
                r["meta_anchor_status"] = "no_medication_event"
            elif not any(x["npre"] >= MIN_PRE for x in cands):
                r["meta_anchor_status"] = "no_baseline_assessment"
            else:
                r["meta_anchor_status"] = "insufficient_post_assessments"
            rows.append(r)
            continue
        if rule == "max_cog_risk":
            key = lambda x: (-x["cri"], -x["bal"], x["date"], x["m"].medication_record_id, x["trank"])  # noqa: E731
        else:
            key = lambda x: (-x["bal"], x["date"], x["m"].medication_record_id, x["trank"])  # noqa: E731
        a = min(elig, key=key)
        m, typ, ad = a["m"], a["typ"], a["date"]
        kd = drugs.loc[m.medication_id]
        r.update(meta_usable_case=1, meta_anchor_status="ok", meta_anchor_type=typ,
                 meta_anchor_date=str(ad)[:10], meta_anchor_record_id=m.medication_record_id,
                 meta_anchor_medication_id=m.medication_id, meta_anchor_drug_name=kd.drug_name,
                 meta_anchor_drug_class=kd.drug_class, meta_anchor_therapeutic_area=kd.therapeutic_area,
                 meta_n_pre_assessments=a["npre"], meta_n_post_assessments=a["npost"],
                 meta_days_enrollment_to_anchor=(ad - np.datetime64(P.enrollment_date)) / D)

        # ---------------- cognitive trajectory
        pre_m = (adate <= ad) & (adate > ad - PRE_DAYS * D)
        post_m = (adate > ad) & (adate <= ad + POST_DAYS * D)
        early_m = (adate > ad) & (adate <= ad + EARLY_DAYS * D)
        late_m = (adate > ad + EARLY_DAYS * D) & (adate <= ad + POST_DAYS * D)
        rel = ((adate - ad) / D).astype(float)
        last_pre_idx = np.where(pre_m)[0].max()
        first_post_idx = np.where(post_m)[0].min()
        r["feat_days_last_pre_to_anchor"] = abs(rel[last_pre_idx])
        r["feat_days_anchor_to_first_post"] = rel[first_post_idx]
        deltas = {}
        for s in SCORES:
            v = c[s].values.astype(float)
            k = SHORT[s]
            # last non-missing baseline value in window
            pv = v[pre_m]
            pl = pv[~np.isnan(pv)][-1] if (~np.isnan(pv)).any() else np.nan
            ps = slope30(rel[pre_m], pv)
            qs = slope30(rel[post_m], v[post_m])
            pm = nanmean(v[post_m])
            r[f"feat_{k}_pre_last"] = pl
            r[f"feat_{k}_pre_slope30"] = ps
            r[f"feat_{k}_post_mean"] = pm
            r[f"feat_{k}_post_slope30"] = qs
            r[f"feat_{k}_delta"] = pm - pl
            r[f"feat_{k}_slope_change"] = qs - ps
            r[f"feat_{k}_delta_early"] = nanmean(v[early_m]) - pl
            r[f"feat_{k}_delta_late"] = nanmean(v[late_m]) - pl
            deltas[s] = pm - pl
            if s == "mmse_score":
                # decline beyond what the pre-anchor slope would predict (needs >=3 baseline points)
                if (~np.isnan(pv)).sum() >= 3 and not np.isnan(ps):
                    exp = pl + ps / 30 * np.nanmean(rel[post_m][~np.isnan(v[post_m])]) if (~np.isnan(v[post_m])).any() else np.nan
                    r["feat_mmse_excess_decline"] = pm - exp
                else:
                    r["feat_mmse_excess_decline"] = np.nan
                dec = np.where(post_m & (v <= pl - DECLINE_THRESH))[0] if not np.isnan(pl) else []
                r["feat_mmse_days_to_first_decline"] = rel[dec.min()] if len(dec) else np.nan
        for f in FLAGS:
            v = c[f].values.astype(float)
            k = f.replace("_present", "")
            a_, b_ = nanmean(v[pre_m]), nanmean(v[post_m])
            r[f"feat_{k}_pre_rate"], r[f"feat_{k}_post_rate"], r[f"feat_{k}_rate_delta"] = a_, b_, b_ - a_

        # ---------------- medication / dose
        dose_after = m.new_dose if typ == "dose_change" else m.dose
        r["feat_anchor_is_dose_change"] = int(typ == "dose_change")
        r["feat_anchor_dose_mg"] = dose_after
        r["feat_anchor_dose_rel_to_drug_median"] = dose_after / med_ref_dose[m.medication_id]
        r["feat_dose_change_ratio"] = m.new_dose / m.dose if typ == "dose_change" and m.dose else np.nan
        r["feat_anchor_frequency"] = m.frequency
        r["feat_anchor_adherence"] = m.adherence_status if isinstance(m.adherence_status, str) else "Missing"
        r["feat_days_on_drug_at_anchor"] = (ad - np.datetime64(m.start_date)) / D
        stop = np.datetime64(m.stop_date) if pd.notna(m.stop_date) else None
        stopped = stop is not None and ad < stop <= ad + POST_DAYS * D
        r["feat_stopped_in_post_window"] = int(stopped)
        r["feat_days_anchor_to_stop"] = (stop - ad) / D if stopped else np.nan
        dech = np.nan
        if stopped:
            v = c["mmse_score"].values.astype(float)
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

        # ---------------- other active medications + co-medication timing
        oth = meds[meds.medication_record_id != m.medication_record_id]
        act = oth[(oth.start_date.values <= ad) & (oth.stop_date.isna() | (oth.stop_date.values > ad))]
        ak = drugs.loc[act.medication_id] if len(act) else drugs.iloc[0:0]
        for col, nm in [("anticholinergic_index", "anticholinergic"), ("sedation_index", "sedation"),
                        ("cognitive_risk_index", "cogrisk"), ("interaction_risk_index", "interaction")]:
            r[f"feat_other_active_{nm}_sum"] = float(ak[col].sum()) if len(ak) else 0.0
        r["feat_other_active_cogrisk_max"] = float(ak.cognitive_risk_index.max()) if len(ak) else 0.0
        r["feat_other_active_any_high_anticholinergic"] = int((ak.anticholinergic_burden == "High").any()) if len(ak) else 0
        r["feat_other_active_any_high_sedation"] = int((ak.sedation_potential == "High").any()) if len(ak) else 0
        r["feat_other_active_any_cognitive_therapy"] = int((ak.therapeutic_area == "Cognitive").any()) if len(ak) else 0
        chg = []   # (date, medication_id, kind) of other-drug starts/dose changes/stops
        for o in oth.itertuples():
            for kind, d in (("start", o.start_date), ("dose_change", o.dose_change_date), ("stop", o.stop_date)):
                if pd.notna(d):
                    chg.append((np.datetime64(d), o.medication_id, kind))
        before = [(ad - d) / D for d, _, k in chg if k != "stop" and ad - CO_PRE_DAYS * D <= d <= ad]
        after = [(d - ad) / D for d, _, k in chg if k != "stop" and ad < d <= ad + POST_DAYS * D]
        stops = [(d - ad) / D for d, _, k in chg if k == "stop" and ad < d <= ad + POST_DAYS * D]
        r["feat_other_med_change_days_before"] = min(before) if before else np.nan
        r["feat_other_med_change_days_after"] = min(after) if after else np.nan
        r["feat_other_med_stop_days_after"] = min(stops) if stops else np.nan
        win = [drugs.loc[mid, "cognitive_risk_index"] for d, mid, k in chg
               if k != "stop" and ad - CO_PRE_DAYS * D <= d <= ad + POST_DAYS * D]
        r["feat_competing_med_change_max_cogrisk"] = float(max(win)) if win else 0.0

        # ---------------- clinical events (windowed, no counts)
        e = eg.get(pid)
        e = e if e is not None else ev.iloc[0:0]
        edt = e.event_date.values
        pre_e = e[(edt <= ad) & (edt > ad - EV_PRE_DAYS * D)]
        post_e = e[(edt > ad) & (edt <= ad + EV_POST_DAYS * D)]
        win_e = pd.concat([pre_e, post_e])
        sev = lambda x: float(x.severity.map(SEV).max()) if len(x) and x.severity.notna().any() else 0.0  # noqa: E731
        r["feat_ev_pre_max_severity"] = sev(pre_e)
        r["feat_ev_pre_any_cog_impact"] = int((pre_e.cognitive_impact_possible == "Yes").any()) if len(pre_e) else 0
        r["feat_ev_pre_days_since_last"] = (ad - pre_e.event_date.values.max()) / D if len(pre_e) else np.nan
        r["feat_ev_post_max_severity"] = sev(post_e)
        r["feat_ev_post_any_cog_impact"] = int((post_e.cognitive_impact_possible == "Yes").any()) if len(post_e) else 0
        r["feat_ev_post_any_hospitalization"] = int((post_e.hospitalization_required == "Yes").any()) if len(post_e) else 0
        r["feat_ev_post_days_to_first"] = (post_e.event_date.values.min() - ad) / D if len(post_e) else np.nan
        gap = e[(edt > adate[last_pre_idx]) & (edt <= adate[first_post_idx])]
        r["feat_ev_in_assessment_gap"] = int(len(gap) > 0)
        r["feat_ev_window_max_duration_days"] = float(win_e.duration_days.max()) if len(win_e) else 0.0
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

        # ---------------- patient / disease
        r["feat_age"] = P.age
        r["feat_sex"] = P.sex
        r["feat_primary_diagnosis"] = P.primary_diagnosis
        r["feat_disease_stage"] = STAGE.get(P.disease_stage, np.nan)
        r["feat_disease_duration_years"] = (ad - np.datetime64(P.diagnosis_date)) / D / 365.25
        cs = comorb.get(pid, set())
        for t in comorb_types:
            r["feat_comorb_" + t.lower().replace(" ", "_")] = int(t in cs)

        # ---------------- drug knowledge of the anchor drug
        r["feat_drug_cognitive_risk_index"] = kd.cognitive_risk_index
        r["feat_drug_sedation_index"] = kd.sedation_index
        r["feat_drug_anticholinergic_index"] = kd.anticholinergic_index
        r["feat_drug_interaction_risk_index"] = kd.interaction_risk_index
        for col, nm in [("memory_effect_potential", "memory"), ("confusion_effect_potential", "confusion"),
                        ("attention_effect_potential", "attention"), ("mood_effect_potential", "mood"),
                        ("sedation_potential", "sedation")]:
            r[f"feat_drug_{nm}_potential"] = POT.get(kd[col], 0)
        r["feat_drug_anticholinergic_burden"] = ACB.get(kd.anticholinergic_burden, 0)
        r["feat_drug_interaction_risk"] = POT.get(kd.interaction_risk, 0)
        r["feat_drug_evidence_category"] = kd.evidence_category
        r["feat_drug_cognitive_relevance"] = kd.cognitive_relevance
        # observed decline in the domains the drug is KNOWN to affect (population SD scaling, label-free)
        conc = 0.0
        for s, col in [("memory_score", "memory_effect_potential"), ("attention_score", "attention_effect_potential"),
                       ("mood_score", "mood_effect_potential")]:
            dv = deltas[s]
            conc += POT.get(kd[col], 0) * max(0.0, -dv / sd[s]) if not np.isnan(dv) else 0.0
        cd = r["feat_confusion_rate_delta"]
        conc += POT.get(kd.confusion_effect_potential, 0) * max(0.0, cd) if not np.isnan(cd) else 0.0
        r["feat_knowledge_concordant_decline"] = conc
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-rule", default="balanced", choices=["balanced", "max_cog_risk"])
    ap.add_argument("--db-path", default=config.DB_PATH, help="Path to SQLite database")
    ap.add_argument("--labels-path", default=LABELS, help="Path to ground truth labels CSV")
    ap.add_argument("--out", default=OUT, help="Path to output ml_cases CSV")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True)
    X = build(conn, args.anchor_rule)
    # labels are attached only now, after all features exist
    lab = pd.read_csv(args.labels_path)[["ground_truth_case_id", "patient_id", "label", "split"]]
    lab = lab.rename(columns={"ground_truth_case_id": "case_id", "label": "target"})
    df = lab.merge(X, on="patient_id", how="left", validate="1:1")
    first = ["case_id", "patient_id", "split", "target"]
    meta = sorted(c for c in df.columns if c.startswith("meta_"))
    feat = [c for c in df.columns if c.startswith("feat_")]
    df = df[first + meta + feat]
    forbidden = ["trajectory_context", "scenario_type", "expected_temporal_pattern", "synthetic_truth_confidence",
                 "doctor", "associated_medication", "alternative_clinical_cause", "gt_", "ground_truth"]
    bad = [c for c in df.columns if any(f in c.lower() for f in forbidden)]
    assert not bad, bad
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"rule={args.anchor_rule} rows={len(df)} usable={int(df.meta_usable_case.sum())} features={len(feat)} -> {args.out}")


if __name__ == "__main__":
    main()
