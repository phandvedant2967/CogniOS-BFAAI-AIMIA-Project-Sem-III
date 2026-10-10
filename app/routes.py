"""API endpoints (read-only)."""
from flask import Blueprint, jsonify

from . import queries
from .db import get_db
from .timeline import build_patient_timeline

bp = Blueprint("api", __name__, url_prefix="/api")


def _not_found(what, ident):
    return jsonify({"error": f"{what} '{ident}' not found"}), 404


def _patient_or_404(patient_id):
    """Return (patient_dict, None) or (None, error_response)."""
    pid = patient_id.strip().upper()
    patient = queries.get_patient(get_db(), pid)
    if patient is None:
        return None, _not_found("Patient", patient_id)
    return patient, None


@bp.get("/patients/<patient_id>")
def patient(patient_id):
    p, err = _patient_or_404(patient_id)
    return err if err else jsonify(p)


@bp.get("/patients/<patient_id>/medical-history")
def medical_history(patient_id):
    """Baseline medical background from the patients table."""
    p, err = _patient_or_404(patient_id)
    if err:
        return err
    return jsonify({
        "patient_id": p["patient_id"],
        "primary_diagnosis": p["primary_diagnosis"],
        "diagnosis_date": p["diagnosis_date"],
        "disease_stage": p["disease_stage"],
        "comorbidities": queries.get_comorbidities(get_db(), p["patient_id"]),
        "enrollment_date": p["enrollment_date"],
    })


@bp.get("/patients/<patient_id>/medications")
def medications(patient_id):
    p, err = _patient_or_404(patient_id)
    if err:
        return err
    rows = queries.get_medication_history(get_db(), p["patient_id"])
    return jsonify({"patient_id": p["patient_id"], "count": len(rows), "medications": rows})


@bp.get("/patients/<patient_id>/cognitive")
def cognitive(patient_id):
    p, err = _patient_or_404(patient_id)
    if err:
        return err
    rows = queries.get_cognitive_history(get_db(), p["patient_id"])
    return jsonify({"patient_id": p["patient_id"], "count": len(rows), "assessments": rows})


@bp.get("/patients/<patient_id>/events")
def events(patient_id):
    p, err = _patient_or_404(patient_id)
    if err:
        return err
    rows = queries.get_clinical_events(get_db(), p["patient_id"])
    return jsonify({"patient_id": p["patient_id"], "count": len(rows), "events": rows})


@bp.get("/patients/<patient_id>/timeline")
def timeline(patient_id):
    p, err = _patient_or_404(patient_id)
    if err:
        return err
    entries = build_patient_timeline(get_db(), p["patient_id"])
    return jsonify({"patient_id": p["patient_id"], "count": len(entries), "timeline": entries})


@bp.get("/drugs")
def drugs():
    rows = queries.list_drugs(get_db())
    return jsonify({"count": len(rows), "drugs": rows})


@bp.get("/drugs/<identifier>")
def drug(identifier):
    """identifier = medication_id (M005) or drug name (Sertraline)."""
    d = queries.get_drug(get_db(), identifier.strip())
    return jsonify(d) if d else _not_found("Drug", identifier)
