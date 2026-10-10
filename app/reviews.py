"""Doctor reviews: POST /api/reviews, GET /api/reviews, GET /api/reviews/<id>.

A review is the clinician's own assessment (``record_type: doctor_assessment``).
It may point at a stored AI prediction, which is returned as a separate,
labelled ``prediction_reference``; the two are never merged.
"""
import re
from contextlib import closing

from flask import Blueprint, current_app, jsonify, request

from . import store

bp = Blueprint("reviews", __name__, url_prefix="/api")

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")
_ALLOWED = {"patient_id", "review_category", "clinical_note", "medication_id", "prediction_id"}


def _err(status, code, message, field=None):
    body = {"error": code, "message": message}
    if field:
        body["field"] = field
    return jsonify(body), status


def _ehr_path():
    return current_app.config.get("ML_DB_PATH") or current_app.config["DB_PATH"]


@bp.post("/reviews")
def create_review():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return _err(400, "bad_request", "Request body must be a JSON object with Content-Type: application/json.")
    unknown = set(payload) - _ALLOWED
    if unknown:
        return _err(400, "bad_request", f"Unexpected field(s): {sorted(unknown)}. Allowed: {sorted(_ALLOWED)}.")

    pid = payload.get("patient_id")
    if not isinstance(pid, str) or not _ID_RE.match(pid.strip()):
        return _err(400, "bad_request", "'patient_id' is required (letters, numbers, underscores, hyphens).", "patient_id")
    pid = pid.strip().upper()

    category = payload.get("review_category")
    if category not in store.REVIEW_CATEGORIES:
        return _err(400, "bad_request", f"'review_category' must be one of {sorted(store.REVIEW_CATEGORIES)}.", "review_category")

    note = payload.get("clinical_note", "")
    if note is None:
        note = ""
    if not isinstance(note, str):
        return _err(400, "bad_request", "'clinical_note' must be a string.", "clinical_note")
    note = note.strip()
    if len(note) > store.NOTE_MAX:
        return _err(400, "bad_request", f"'clinical_note' must be {store.NOTE_MAX} characters or fewer.", "clinical_note")

    med_id = payload.get("medication_id")
    if med_id is not None and (not isinstance(med_id, str) or not _ID_RE.match(med_id.strip())):
        return _err(400, "bad_request", "'medication_id' must be a medication ID such as 'M001'.", "medication_id")
    med_id = med_id.strip().upper() if med_id else None

    pred_id = payload.get("prediction_id")
    if pred_id is not None and (isinstance(pred_id, bool) or not isinstance(pred_id, int)):
        return _err(400, "bad_request", "'prediction_id' must be an integer.", "prediction_id")

    # Patient and medication must exist in the EHR database the dashboard reads.
    with closing(store.open_readonly(_ehr_path())) as ehr:
        if ehr.execute("SELECT 1 FROM patients WHERE patient_id=?", (pid,)).fetchone() is None:
            return jsonify({"error": "not_found", "entity": "patient", "identifier": pid}), 404
        medication = None
        if med_id:
            medication = ehr.execute(
                "SELECT DISTINCT d.medication_id, d.drug_name, d.drug_class FROM medication_history m "
                "JOIN drugs d ON d.medication_id = m.medication_id WHERE m.patient_id=? AND m.medication_id=?",
                (pid, med_id)).fetchone()
            if medication is None:
                return jsonify({"error": "not_found", "entity": "medication", "identifier": med_id}), 404
            medication = dict(medication)

    app_db = current_app.config["APP_DB_PATH"]
    prediction = None
    if pred_id is not None:
        prediction = store.get_prediction(app_db, pred_id)
        if prediction is None:
            return jsonify({"error": "not_found", "entity": "prediction", "identifier": pred_id}), 404
        if prediction["patient_id"] != pid:
            return _err(400, "bad_request", "That prediction belongs to a different patient.", "prediction_id")
        if medication is None:
            medication = {"medication_id": prediction["medication_id"], "drug_name": prediction["drug_name"],
                          "drug_class": prediction["drug_class"]}
        elif medication["medication_id"] != prediction["medication_id"]:
            return _err(400, "bad_request", "'medication_id' does not match the referenced prediction.", "medication_id")

    review_id = store.save_review(app_db, pid, category, note, medication, prediction)
    return jsonify(store.get_review(app_db, review_id)), 201


@bp.get("/reviews")
def list_reviews():
    pid = request.args.get("patient_id")
    if pid is not None:
        if not _ID_RE.match(pid.strip()):
            return _err(400, "bad_request", "'patient_id' is not a valid ID.", "patient_id")
        pid = pid.strip().upper()
    med = request.args.get("medication_id")
    if med is not None:
        if not _ID_RE.match(med.strip()):
            return _err(400, "bad_request", "'medication_id' is not a valid ID.", "medication_id")
        med = med.strip().upper()
    pred = request.args.get("prediction_id")
    if pred is not None:
        if not pred.isdigit():
            return _err(400, "bad_request", "'prediction_id' must be an integer.", "prediction_id")
        pred = int(pred)
    limit = request.args.get("limit", "100")
    if not limit.isdigit() or not (1 <= int(limit) <= 500):
        return _err(400, "bad_request", "'limit' must be an integer from 1 to 500.", "limit")
    items = store.list_reviews(current_app.config["APP_DB_PATH"], pid, med, pred, int(limit))
    return jsonify({"record_type": "doctor_assessment", "count": len(items), "reviews": items})


@bp.get("/reviews/<int:review_id>")
def get_review(review_id):
    r = store.get_review(current_app.config["APP_DB_PATH"], review_id)
    if r is None:
        return jsonify({"error": "not_found", "entity": "review", "identifier": review_id}), 404
    return jsonify(r)
