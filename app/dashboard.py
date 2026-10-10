"""Doctor dashboard (Stage 7): serves the CogniOS UI and one read-only
patient-bundle endpoint.

Why a separate endpoint: the original REST routes read ``DB_PATH``
(patient IDs like ``P0001``), while POST /api/predictions reads ``ML_DB_PATH``
(patient IDs like ``V4_2_P0001``). The two databases have different patients,
so history shown next to an AI assessment must come from the ML database.
This module reuses ``app.queries`` and ``app.timeline`` unchanged and only
points them at ``ML_DB_PATH``. Nothing is written.
"""
import re
import sqlite3

from flask import Blueprint, current_app, g, jsonify, send_from_directory

from . import queries
from .timeline import build_patient_timeline

bp = Blueprint("dashboard", __name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,40}$")


def _ml_conn():
    """Read-only connection to the database the prediction API uses."""
    if "dashboard_db" not in g:
        path = current_app.config.get("ML_DB_PATH") or current_app.config["DB_PATH"]
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        g.dashboard_db = conn
    return g.dashboard_db


@bp.teardown_app_request
def _close(_exc=None):
    conn = g.pop("dashboard_db", None)
    if conn is not None:
        conn.close()


@bp.get("/")
def index():
    return send_from_directory(current_app.static_folder, "index.html")


@bp.get("/api/dashboard/patients/<patient_id>")
def patient_bundle(patient_id):
    """Everything the dashboard shows for one patient, in one response."""
    raw = patient_id.strip()
    if not _ID_RE.match(raw):
        return jsonify({
            "error": "invalid_patient_id",
            "message": "Patient IDs contain only letters, numbers, underscores and hyphens "
                       "(for example V4_2_P0001).",
        }), 400

    pid = raw.upper()
    conn = _ml_conn()
    patient = queries.get_patient(conn, pid)
    if patient is None:
        return jsonify({
            "error": "not_found",
            "entity": "patient",
            "identifier": raw,
            "message": f"No patient '{raw}' in the assessment database. "
                       "IDs in this database look like V4_2_P0001.",
        }), 404

    medications = queries.get_medication_history(conn, pid)
    cognitive = queries.get_cognitive_history(conn, pid)
    events = queries.get_clinical_events(conn, pid)
    timeline = build_patient_timeline(conn, pid)

    return jsonify({
        "patient": patient,
        "medical_history": {
            "primary_diagnosis": patient["primary_diagnosis"],
            "diagnosis_date": patient["diagnosis_date"],
            "disease_stage": patient["disease_stage"],
            "comorbidities": queries.get_comorbidities(conn, pid),
            "enrollment_date": patient["enrollment_date"],
        },
        "medications": medications,
        "cognitive": cognitive,
        "events": events,
        "timeline": timeline,
        "counts": {
            "medications": len(medications),
            "cognitive": len(cognitive),
            "events": len(events),
            "timeline": len(timeline),
        },
    })
