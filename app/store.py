"""Application database: AI prediction log and doctor reviews (SQLite).

Separate from the EHR databases, which are only ever opened read-only.
Two tables with deliberately different names and ``record_type`` values so an
AI prediction can never be mistaken for a doctor assessment:

  ai_predictions   one row per (patient, medication, anchor, model); written by POST /api/predictions
  doctor_reviews   one row per review a clinician saves; append-only
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

REVIEW_CATEGORIES = {
    "medication_associated": "Likely medication-associated",
    "disease_progression": "Likely disease progression",
    "other_cause": "Likely other cause",
    "uncertain": "Uncertain",
}
NOTE_MAX = 500

# Effect domains derived from the four recorded symptom flags.
EFFECT_COLUMNS = {
    "confusion": ("effect_confusion", "confusion_present", "Confusion"),
    "memory_loss": ("effect_memory_loss", "memory_loss_present", "Memory loss"),
    "attention": ("effect_attention", "attention_problem_present", "Attention problems"),
    "behavioral": ("effect_behavioral", "behavioral_change_present", "Behavioral change"),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS ai_predictions (
    prediction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT NOT NULL,
    medication_id   TEXT NOT NULL,
    drug_name       TEXT,
    drug_class      TEXT,
    model_type      TEXT NOT NULL,
    model_name      TEXT,
    predicted_class TEXT NOT NULL,
    prob_medication_associated REAL,
    prob_disease_progression   REAL,
    prob_other_cause           REAL,
    prob_uncertain             REAL,
    anchor_type     TEXT NOT NULL,
    anchor_date     TEXT NOT NULL,
    effect_confusion   INTEGER,
    effect_memory_loss INTEGER,
    effect_attention   INTEGER,
    effect_behavioral  INTEGER,
    predicted_at    TEXT NOT NULL,
    UNIQUE (patient_id, medication_id, anchor_type, anchor_date, model_type)
);
CREATE TABLE IF NOT EXISTS doctor_reviews (
    review_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id      TEXT NOT NULL,
    review_category TEXT NOT NULL CHECK (review_category IN
        ('medication_associated','disease_progression','other_cause','uncertain')),
    clinical_note   TEXT NOT NULL DEFAULT '',
    medication_id   TEXT,
    drug_name       TEXT,
    drug_class      TEXT,
    prediction_id   INTEGER REFERENCES ai_predictions(prediction_id),
    ai_predicted_class_at_review TEXT,
    ai_model_name_at_review      TEXT,
    reviewed_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_patient ON doctor_reviews(patient_id, reviewed_at);
CREATE INDEX IF NOT EXISTS idx_reviews_pred ON doctor_reviews(prediction_id);
CREATE INDEX IF NOT EXISTS idx_pred_drug ON ai_predictions(medication_id, anchor_date);
"""


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path):
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with closing(connect(path)) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def open_readonly(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# AI prediction log
# ---------------------------------------------------------------------------

def compute_effects(ehr_path, patient_id, anchor_date):
    """Which recorded symptom flags are newly present after the anchor.

    For each flag: 1 if at least one assessment in (anchor, anchor+180d] has it
    present and none in the 270 days up to the anchor did; 0 if not; None when
    either window has no recorded value for that flag (cannot be judged).
    This describes recorded data only. It is not a causal statement.
    """
    out = {}
    with closing(open_readonly(ehr_path)) as conn:
        for key, (col_out, flag, _label) in EFFECT_COLUMNS.items():
            pre = [r[0] for r in conn.execute(
                f"SELECT {flag} FROM cognitive_behavioral WHERE patient_id=? AND {flag} IS NOT NULL "
                "AND assessment_date <= ? AND assessment_date > date(?, '-270 day')",
                (patient_id, anchor_date, anchor_date))]
            post = [r[0] for r in conn.execute(
                f"SELECT {flag} FROM cognitive_behavioral WHERE patient_id=? AND {flag} IS NOT NULL "
                "AND assessment_date > ? AND assessment_date <= date(?, '+180 day')",
                (patient_id, anchor_date, anchor_date))]
            out[col_out] = None if (not pre or not post) else int(max(post) == 1 and max(pre) == 0)
    return out


def log_prediction(app_db, ehr_path, patient_id, model_type, result):
    """Store a successful prediction; returns prediction_id. Re-running the same
    case with the same model updates that row (so counts are never inflated)."""
    a = result["anchor_details"]
    probs = result["probabilities"]
    eff = compute_effects(ehr_path, patient_id, a["anchor_date"])
    with closing(connect(app_db)) as conn:
        row = conn.execute(
            """
            INSERT INTO ai_predictions (patient_id, medication_id, drug_name, drug_class, model_type, model_name,
                predicted_class, prob_medication_associated, prob_disease_progression, prob_other_cause,
                prob_uncertain, anchor_type, anchor_date, effect_confusion, effect_memory_loss,
                effect_attention, effect_behavioral, predicted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT (patient_id, medication_id, anchor_type, anchor_date, model_type) DO UPDATE SET
                drug_name=excluded.drug_name, drug_class=excluded.drug_class, model_name=excluded.model_name,
                predicted_class=excluded.predicted_class,
                prob_medication_associated=excluded.prob_medication_associated,
                prob_disease_progression=excluded.prob_disease_progression,
                prob_other_cause=excluded.prob_other_cause, prob_uncertain=excluded.prob_uncertain,
                effect_confusion=excluded.effect_confusion, effect_memory_loss=excluded.effect_memory_loss,
                effect_attention=excluded.effect_attention, effect_behavioral=excluded.effect_behavioral,
                predicted_at=excluded.predicted_at
            RETURNING prediction_id
            """,
            (patient_id, a["medication_id"], a.get("drug_name"), a.get("drug_class"), model_type,
             result.get("model_name"), result["predicted_class"],
             probs.get("medication_associated"), probs.get("disease_progression"),
             probs.get("other_cause"), probs.get("uncertain"),
             a["anchor_type"], a["anchor_date"],
             eff["effect_confusion"], eff["effect_memory_loss"], eff["effect_attention"],
             eff["effect_behavioral"], utc_now()),
        ).fetchone()
        conn.commit()
        return row[0]


def get_prediction(app_db, prediction_id):
    with closing(connect(app_db)) as conn:
        r = conn.execute("SELECT * FROM ai_predictions WHERE prediction_id=?", (prediction_id,)).fetchone()
        return dict(r) if r else None


# ---------------------------------------------------------------------------
# Doctor reviews
# ---------------------------------------------------------------------------

def save_review(app_db, patient_id, category, note, medication, prediction):
    """Insert one review. ``medication`` is a dict (medication_id, drug_name, drug_class) or None;
    ``prediction`` is an ai_predictions row dict or None. The timestamp is set here, by the server."""
    with closing(connect(app_db)) as conn:
        cur = conn.execute(
            """INSERT INTO doctor_reviews (patient_id, review_category, clinical_note, medication_id, drug_name,
                   drug_class, prediction_id, ai_predicted_class_at_review, ai_model_name_at_review, reviewed_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (patient_id, category, note,
             medication["medication_id"] if medication else None,
             medication.get("drug_name") if medication else None,
             medication.get("drug_class") if medication else None,
             prediction["prediction_id"] if prediction else None,
             prediction["predicted_class"] if prediction else None,
             prediction["model_name"] if prediction else None,
             utc_now()))
        conn.commit()
        return cur.lastrowid


def review_to_dict(row, prediction=None):
    """Public JSON shape. Doctor assessment and AI prediction are separate, labelled objects."""
    r = dict(row)
    medication = None
    if r["medication_id"]:
        medication = {"medication_id": r["medication_id"], "drug_name": r["drug_name"], "drug_class": r["drug_class"]}
    pred_ref = None
    if r["prediction_id"] is not None:
        pred_ref = {
            "record_type": "ai_prediction",
            "prediction_id": r["prediction_id"],
            "predicted_class_at_review": r["ai_predicted_class_at_review"],
            "model_name": r["ai_model_name_at_review"],
            "anchor_type": (prediction or {}).get("anchor_type"),
            "anchor_date": (prediction or {}).get("anchor_date"),
        }
    return {
        "record_type": "doctor_assessment",
        "assessed_by": "doctor",
        "review_id": r["review_id"],
        "patient_id": r["patient_id"],
        "review_category": r["review_category"],
        "review_label": REVIEW_CATEGORIES[r["review_category"]],
        "clinical_note": r["clinical_note"],
        "medication": medication,
        "prediction_reference": pred_ref,
        "reviewed_at": r["reviewed_at"],
    }


def list_reviews(app_db, patient_id=None, medication_id=None, prediction_id=None, limit=100):
    where, args = [], []
    for col, val in (("patient_id", patient_id), ("medication_id", medication_id), ("prediction_id", prediction_id)):
        if val is not None:
            where.append(f"r.{col} = ?")
            args.append(val)
    sql = ("SELECT r.*, p.anchor_type AS p_anchor_type, p.anchor_date AS p_anchor_date "
           "FROM doctor_reviews r LEFT JOIN ai_predictions p ON p.prediction_id = r.prediction_id")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.reviewed_at DESC, r.review_id DESC LIMIT ?"
    args.append(limit)
    with closing(connect(app_db)) as conn:
        rows = conn.execute(sql, args).fetchall()
    return [review_to_dict(r, {"anchor_type": r["p_anchor_type"], "anchor_date": r["p_anchor_date"]}) for r in rows]


def get_review(app_db, review_id):
    with closing(connect(app_db)) as conn:
        r = conn.execute(
            "SELECT r.*, p.anchor_type AS p_anchor_type, p.anchor_date AS p_anchor_date "
            "FROM doctor_reviews r LEFT JOIN ai_predictions p ON p.prediction_id = r.prediction_id "
            "WHERE r.review_id = ?", (review_id,)).fetchone()
    if r is None:
        return None
    return review_to_dict(r, {"anchor_type": r["p_anchor_type"], "anchor_date": r["p_anchor_date"]})
