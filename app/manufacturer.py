"""Manufacturer dashboard (Stage 9): page + de-identified aggregate APIs.

Data sources (both genuinely stored, nothing generated or imputed):
  ai_predictions   AI pattern flags written when a clinician runs an assessment
  doctor_reviews   assessments clinicians saved, linked to a stored prediction

What this module guarantees
  * Output is aggregate only. Patient IDs, hospital identifiers and record-level
    data are read server-side to count distinct patients, and are never serialised.
  * Minimum group size: every drug group and every displayed count needs at least
    MIN_GROUP (5) distinct patients. Smaller cells are returned as
    ``{"value": null, "suppressed": true}`` and never as a number. Drug groups
    below the threshold are dropped before anything else is computed, so headline
    totals cover only reportable groups.
  * Breakdowns that sum to a total use pooling (``pool_small``) so a hidden small
    cell cannot be recovered by subtraction.
  * AI predictions are pattern flags for clinician review. They are never
    reported as confirmed adverse effects; doctor assessments are reported
    separately and are likewise assessments, not confirmation.

Known residual risk (documented in the report): a user who runs many differently
filtered queries could difference results. There is no query-rate limiting or
authentication in this stage.
"""
import calendar
import re
from collections import defaultdict
from contextlib import closing
from datetime import date

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from . import store

bp = Blueprint("manufacturer", __name__)

MIN_GROUP = 5
CLASSES = ["medication_associated", "disease_progression", "other_cause", "uncertain"]
CLASS_LABELS = {
    "medication_associated": "Medication-associated pattern",
    "disease_progression": "Disease-progression pattern",
    "other_cause": "Other-cause pattern",
    "uncertain": "Uncertain pattern",
}
EFFECTS = {k: v[2] for k, v in store.EFFECT_COLUMNS.items()}
MODELS = {"xgboost", "cnn_gru"}
INTERVALS = {"month", "quarter"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")

LABELS = {
    "ai": "AI-generated pattern flags. Model output for clinician review; not confirmed adverse effects.",
    "doctor": "Doctor assessments. A clinician's judgement on a case; not a confirmed causal finding.",
    "effects": "Symptom flags newly recorded in assessments within 180 days after the medication event "
               "(and not recorded in the 270 days before). Recorded observations only; they do not show that a medication caused them.",
    "privacy": f"Aggregated and de-identified. Groups with fewer than {MIN_GROUP} patients are withheld and shown as \"<{MIN_GROUP}\".",
}


# ---------------------------------------------------------------------------
# Suppression helpers (pure; unit-tested)
# ---------------------------------------------------------------------------

def n_patients(rows):
    return len({r["patient_id"] for r in rows})


def visible(rows):
    return n_patients(rows) >= MIN_GROUP


def cell(rows):
    """Count of cases, or a suppressed marker when fewer than MIN_GROUP patients. Zero is reported as 0."""
    if not rows:
        return {"value": 0, "suppressed": False}
    if not visible(rows):
        return {"value": None, "suppressed": True}
    return {"value": len(rows), "suppressed": False}


def binary_split(part_rows, whole_rows):
    """Report ``part`` of ``whole`` only if both the part and the remainder are reportable (or empty)."""
    ids = {id(r) for r in part_rows}
    rest = [r for r in whole_rows if id(r) not in ids]
    ok = (not part_rows or visible(part_rows)) and (not rest or visible(rest))
    return cell(part_rows) if ok else {"value": None, "suppressed": True}


def pool_small(groups):
    """Breakdown over mutually exclusive groups {name: rows}.

    Groups below the threshold are merged into one ``other_combined`` pool; while that pool is
    non-empty but still below the threshold, the smallest reportable group is merged in too.
    The pool is therefore either empty or reportable, so no hidden count can be recovered by
    subtracting the shown cells from the total. Returns (shown {name: cell}, pooled cell|None, pooled_names).
    """
    shown = {k: v for k, v in groups.items() if v and visible(v)}
    pooled_names = [k for k, v in groups.items() if v and not visible(v)]
    pool = [r for k in pooled_names for r in groups[k]]
    while pool and not visible(pool) and shown:
        smallest = min(shown, key=lambda k: n_patients(shown[k]))
        pool = pool + shown.pop(smallest)
        pooled_names.append(smallest)
    if pool and not visible(pool):
        return {}, {"value": None, "suppressed": True}, pooled_names
    return ({k: cell(v) for k, v in shown.items()},
            cell(pool) if pool else None, pooled_names)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_filters(args):
    f = {"drug": None, "effect": None, "from": None, "to": None, "model": None, "interval": "quarter"}
    drug = args.get("drug")
    if drug:
        if not _ID_RE.match(drug.strip()):
            return None, "'drug' must be a medication ID such as M001."
        f["drug"] = drug.strip().upper()
    effect = args.get("effect")
    if effect:
        if effect not in EFFECTS:
            return None, f"'effect' must be one of {sorted(EFFECTS)}."
        f["effect"] = effect
    for key in ("from", "to"):
        v = args.get(key)
        if v:
            if not _DATE_RE.match(v):
                return None, f"'{key}' must be a date in YYYY-MM-DD format."
            try:
                date.fromisoformat(v)
            except ValueError:
                return None, f"'{key}' is not a valid calendar date."
            f[key] = v
    if f["from"] and f["to"] and f["from"] > f["to"]:
        return None, "'from' must not be later than 'to'."
    model = args.get("model")
    if model:
        if model not in MODELS:
            return None, f"'model' must be one of {sorted(MODELS)}."
        f["model"] = model
    interval = args.get("interval")
    if interval:
        if interval not in INTERVALS:
            return None, f"'interval' must be one of {sorted(INTERVALS)}."
        f["interval"] = interval
    return f, None


def load_cases(conn, f):
    """One row per case (patient, medication, anchor), newest prediction winning when both models ran.
    Each carries the latest linked doctor review category, if any. Rows are used server-side only."""
    where, args = [], []
    if f["model"]:
        where.append("model_type = ?"); args.append(f["model"])
    inner = "SELECT * FROM ai_predictions" + (" WHERE " + " AND ".join(where) if where else "")
    sql = f"""
    WITH pred AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY patient_id, medication_id, anchor_type, anchor_date
                                   ORDER BY predicted_at DESC, prediction_id DESC) AS rn
      FROM ({inner})
    ),
    rev AS (
      SELECT p.patient_id, p.medication_id, p.anchor_type, p.anchor_date, r.review_category,
             ROW_NUMBER() OVER (PARTITION BY p.patient_id, p.medication_id, p.anchor_type, p.anchor_date
                                ORDER BY r.reviewed_at DESC, r.review_id DESC) AS rn
      FROM doctor_reviews r JOIN ai_predictions p ON p.prediction_id = r.prediction_id
    )
    SELECT pred.patient_id, pred.medication_id, pred.drug_name, pred.drug_class, pred.predicted_class,
           pred.anchor_date, pred.effect_confusion, pred.effect_memory_loss, pred.effect_attention,
           pred.effect_behavioral, rev.review_category
    FROM pred
    LEFT JOIN rev ON rev.patient_id = pred.patient_id AND rev.medication_id = pred.medication_id
                 AND rev.anchor_type = pred.anchor_type AND rev.anchor_date = pred.anchor_date AND rev.rn = 1
    WHERE pred.rn = 1
    """
    rows = [dict(r) for r in conn.execute(sql, args)]
    if f["drug"]:
        rows = [r for r in rows if r["medication_id"] == f["drug"]]
    if f["from"]:
        rows = [r for r in rows if r["anchor_date"] >= f["from"]]
    if f["to"]:
        rows = [r for r in rows if r["anchor_date"] <= f["to"]]
    if f["effect"]:
        col = store.EFFECT_COLUMNS[f["effect"]][0]
        rows = [r for r in rows if r[col] == 1]
    return rows


def _period(anchor_date, interval):
    y, m = anchor_date[:4], int(anchor_date[5:7])
    return f"{y}-{m:02d}" if interval == "month" else f"{y}-Q{(m - 1) // 3 + 1}"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _class_breakdown(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    groups = {c: groups.get(c, []) for c in CLASSES}
    shown, pooled, pooled_names = pool_small(groups)
    return {
        "shown": {c: shown[c] for c in CLASSES if c in shown},
        "other_combined": pooled,
        "hidden_classes": sorted(pooled_names),
    }


def aggregate(rows, f):
    by_drug = defaultdict(list)
    for r in rows:
        by_drug[r["medication_id"]].append(r)

    reportable = {k: v for k, v in by_drug.items() if visible(v)}
    withheld = len(by_drug) - len(reportable)
    rep_rows = [r for v in reportable.values() for r in v]

    # --- drug-wise summaries -------------------------------------------------
    drugs = []
    for mid, v in sorted(reportable.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        flagged_rows = [r for r in v if r["predicted_class"] == "medication_associated"]
        reviewed = [r for r in v if r["review_category"]]
        likely_med = [r for r in reviewed if r["review_category"] == "medication_associated"]
        ai_bd = _class_breakdown(v, "predicted_class")
        if not flagged_rows:
            flagged_cell = {"value": 0, "suppressed": False}
        else:
            flagged_cell = ai_bd["shown"].get("medication_associated")
        total = len(v)
        drugs.append({
            "medication_id": mid,
            "drug_name": v[0]["drug_name"],
            "drug_class": v[0]["drug_class"],
            "ai": {
                "record_type": "ai_prediction",
                "cases": {"value": total, "suppressed": False},
                "flagged": flagged_cell or {"value": None, "suppressed": True},
                "flagged_share": (round(flagged_cell["value"] / total, 3)
                                  if flagged_cell and not flagged_cell["suppressed"] else None),
                "classes": ai_bd,
            },
            "doctor": {
                "record_type": "doctor_assessment",
                "reviewed_cases": cell(reviewed),
                "likely_medication_associated": binary_split(likely_med, reviewed) if visible(reviewed) else {"value": None, "suppressed": True},
                "categories": _class_breakdown(reviewed, "review_category") if visible(reviewed) else None,
                "review_coverage": (round(len(reviewed) / total, 3) if visible(reviewed) else None),
            },
        })

    # --- effects ------------------------------------------------------------
    effects = []
    for key, (col, _flag, label) in store.EFFECT_COLUMNS.items():
        with_eff = [r for r in rep_rows if r[col] == 1]
        known = [r for r in rep_rows if r[col] is not None]
        flagged = [r for r in with_eff if r["predicted_class"] == "medication_associated"]
        rev = [r for r in with_eff if r["review_category"]]
        rev_med = [r for r in rev if r["review_category"] == "medication_associated"]
        effects.append({
            "effect": key, "label": label,
            "cases_with_effect": cell(with_eff),
            "cases_assessable": cell(known),
            "rate": (round(len(with_eff) / len(known), 3) if visible(known) and visible(with_eff) else None),
            "ai_flagged_with_effect": binary_split(flagged, with_eff) if visible(with_eff) else {"value": None, "suppressed": True},
            "doctor_reviewed_with_effect": cell(rev),
            "doctor_likely_medication_associated": binary_split(rev_med, rev) if visible(rev) else {"value": None, "suppressed": True},
        })

    # --- trend over time (by anchor date of the assessed medication event) ----
    periods = defaultdict(list)
    for r in rep_rows:
        periods[_period(r["anchor_date"], f["interval"])].append(r)
    trend = []
    for p in sorted(periods):
        v = periods[p]
        fl = [r for r in v if r["predicted_class"] == "medication_associated"]
        rv = [r for r in v if r["review_category"]]
        trend.append({
            "period": p,
            "cases": cell(v),
            "ai_flagged": binary_split(fl, v) if visible(v) else {"value": None, "suppressed": True},
            "doctor_reviewed": cell(rv) if visible(v) else {"value": None, "suppressed": True},
        })

    reviewed_all = [r for r in rep_rows if r["review_category"]]
    flagged_all = [r for r in rep_rows if r["predicted_class"] == "medication_associated"]
    totals = {
        "cases": cell(rep_rows),
        "ai_flagged": binary_split(flagged_all, rep_rows) if rep_rows else {"value": 0, "suppressed": False},
        "doctor_reviewed": cell(reviewed_all),
        "drug_groups_shown": len(reportable),
        "drug_groups_withheld": withheld,
    }
    return totals, drugs, effects, trend


def _empty_state(total_stored, matched_rows, reportable_count):
    if total_stored == 0:
        return {"state": "no_data",
                "message": "No AI predictions have been stored yet. Signals appear here after clinicians run assessments in the clinician dashboard."}
    if not matched_rows:
        return {"state": "no_match", "message": "No stored cases match these filters. Try widening the date range or clearing a filter."}
    if reportable_count == 0:
        return {"state": "below_threshold",
                "message": f"Cases exist for these filters, but every drug group has fewer than {MIN_GROUP} patients, so nothing can be shown without risking re-identification."}
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _bad(msg):
    return jsonify({"error": "bad_request", "message": msg}), 400


@bp.get("/manufacturer")
def page():
    return send_from_directory(current_app.static_folder, "manufacturer.html")


@bp.get("/api/manufacturer/filters")
def filters():
    with closing(store.connect(current_app.config["APP_DB_PATH"])) as conn:
        rows = load_cases(conn, {"drug": None, "effect": None, "from": None, "to": None, "model": None})
    by_drug = defaultdict(list)
    for r in rows:
        by_drug[r["medication_id"]].append(r)
    drugs = [{"medication_id": k, "drug_name": v[0]["drug_name"]}
             for k, v in sorted(by_drug.items(), key=lambda kv: kv[1][0]["drug_name"] or "") if visible(v)]
    rep = [r for v in by_drug.values() if visible(v) for r in v]
    dates = sorted(r["anchor_date"] for r in rep)
    bounds = None
    if dates:  # month granularity only: an exact min/max would expose one record's date
        y, m = int(dates[-1][:4]), int(dates[-1][5:7])
        bounds = {"from": dates[0][:7] + "-01", "to": f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"}
    return jsonify({
        "drugs": drugs,
        "effects": [{"value": k, "label": v} for k, v in EFFECTS.items()],
        "models": [{"value": "xgboost", "label": "XGBoost"}, {"value": "cnn_gru", "label": "CNN-GRU"}],
        "date_bounds": bounds,
        "min_group_size": MIN_GROUP,
    })


@bp.get("/api/manufacturer/summary")
def summary():
    f, err = _parse_filters(request.args)
    if err:
        return _bad(err)
    with closing(store.connect(current_app.config["APP_DB_PATH"])) as conn:
        total_stored = conn.execute("SELECT COUNT(*) FROM ai_predictions").fetchone()[0]
        rows = load_cases(conn, f)
    totals, drugs, effects, trend = aggregate(rows, f)
    empty = _empty_state(total_stored, rows, totals["drug_groups_shown"])
    if empty:
        effects, trend, drugs = [], [], []
    return jsonify({
        "filters": f,
        "min_group_size": MIN_GROUP,
        "labels": LABELS,
        "class_labels": CLASS_LABELS,
        "empty": empty,
        "totals": totals,
        "drugs": drugs,
        "effects": effects,
        "trend": {"interval": f["interval"], "basis": "anchor_date", "points": trend},
    })
