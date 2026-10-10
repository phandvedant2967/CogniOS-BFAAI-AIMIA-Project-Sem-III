"""Prediction endpoint: POST /api/predictions.

Connects Flask to the ml.predict inference pipeline without re-fitting
any preprocessing objects or accessing ground-truth labels.

Design principles:
  - Validates all request fields before touching the ML pipeline.
  - Returns an explicit ``insufficient_data`` JSON body instead of a
    prediction whenever the patient lacks the required longitudinal
    coverage (not an HTTP error).
  - A missing patient or an invalid medication ID yields a 404 or 400
    response respectively, so clients can distinguish those cases.
  - Model-loading failures are surfaced as 503 (service unavailable).
  - The 180-day assessment cutoff is enforced entirely within
    ``ml.predict.predict_patient``; this layer does not duplicate it.
"""

import logging

from flask import Blueprint, current_app, jsonify, request

from . import store

logger = logging.getLogger(__name__)

bp = Blueprint("predictions", __name__, url_prefix="/api")

# Accepted model-type tokens
_VALID_MODELS = {"xgboost", "cnn_gru"}
_DEFAULT_MODEL = "xgboost"

# Lazy-loaded inference function; imported once on first request so the
# heavy torch/joblib machinery is not loaded at application start-up.
_predict_patient = None


def _get_predict_fn():
    """Return the predict_patient callable, importing ml.predict lazily."""
    global _predict_patient
    if _predict_patient is None:
        from ml.predict import predict_patient  # noqa: PLC0415
        _predict_patient = predict_patient
    return _predict_patient


def _bad_request(message, field=None):
    body = {"error": "bad_request", "message": message}
    if field:
        body["field"] = field
    return jsonify(body), 400


def _not_found(entity, identifier):
    return jsonify({"error": "not_found", "entity": entity, "identifier": identifier}), 404


def _service_error(message):
    return jsonify({"error": "service_unavailable", "message": message}), 503


# ---------------------------------------------------------------------------
# POST /api/predictions
# ---------------------------------------------------------------------------

@bp.post("/predictions")
def create_prediction():
    """Run ML inference for a single patient.

    Request body (JSON):
    {
        "patient_id":        "V4_2_P0001",         # required
        "medication_id":     "M001",               # optional – auto-resolved if omitted
        "model_type":        "xgboost"             # optional – "xgboost" (default) | "cnn_gru"
    }

    Success response  (HTTP 200):
    {
        "status":               "success",
        "patient_id":           "V4_2_P0001",
        "predicted_class":      "medication_associated",
        "pattern_description":  "Possible medication-associated pattern",
        "probabilities": {
            "medication_associated": 0.4821,
            "disease_progression":   0.2103,
            "other_cause":           0.1934,
            "uncertain":             0.1142
        },
        "confidence":           0.4821,
        "model_name":           "XGBoost (Tabular Baseline)",
        "anchor_details": {
            "medication_id":  "M001",
            "drug_name":      "Donepezil",
            "drug_class":     "Cholinesterase Inhibitor",
            "anchor_date":    "2023-04-15",
            "anchor_type":    "start",
            "dose_mg":        10.0,
            "cutoff_date":    "2023-10-12",
            "horizon_days":   180
        },
        "usable_observation_dates": {
            "pre_anchor":  ["2023-01-10", "2023-03-05"],
            "post_anchor": ["2023-06-20", "2023-09-01"],
            "cutoff_date": "2023-10-12"
        },
        "data_quality_warnings": [],
        "supporting_evidence": [
            "MMSE score changed by -3.2 points (last baseline: 24.0, 180-day post mean: 20.8).",
            "Cognitive trajectory accelerated decline post-anchor (30-day slope: pre -0.30 vs post -0.82)."
        ]
    }

    Insufficient-data response  (HTTP 200):
    {
        "status":               "insufficient_data",
        "patient_id":           "V4_2_P0001",
        "predicted_class":      null,
        "pattern_description":  "Insufficient longitudinal data for reliable assessment",
        "probabilities":        {"medication_associated": 0.0, ...},
        "confidence":           0.0,
        "model_name":           "XGBoost (Tabular Baseline)",
        "anchor_details":       null | {...},
        "usable_observation_dates": {...},
        "data_quality_warnings": ["..."],
        "supporting_evidence":  []
    }

    Error responses:
      400  – missing/invalid request fields
      404  – patient or medication not found
      503  – model artifact missing or failed to load
    """
    # ── 1. Parse and validate request body ──────────────────────────────────
    payload = request.get_json(silent=True)
    if not payload:
        return _bad_request("Request body must be JSON with Content-Type: application/json.")

    # patient_id: required, non-empty string
    patient_id = payload.get("patient_id")
    if not patient_id or not isinstance(patient_id, str) or not patient_id.strip():
        return _bad_request("'patient_id' is required and must be a non-empty string.", field="patient_id")
    patient_id = patient_id.strip()

    # medication_id: optional, must be string if present
    medication_id = payload.get("medication_id")
    if medication_id is not None:
        if not isinstance(medication_id, str) or not medication_id.strip():
            return _bad_request(
                "'medication_id' must be a non-empty string when provided (e.g. 'M001').",
                field="medication_id",
            )
        medication_id = medication_id.strip()

    # model_type: optional, must be one of the allowed values
    model_type = payload.get("model_type", _DEFAULT_MODEL)
    if not isinstance(model_type, str) or model_type.strip().lower() not in _VALID_MODELS:
        return _bad_request(
            f"'model_type' must be one of: {sorted(_VALID_MODELS)}. "
            f"Received: {model_type!r}.",
            field="model_type",
        )
    model_type = model_type.strip().lower()

    # Reject any unexpected extra keys to prevent confusion
    allowed_keys = {"patient_id", "medication_id", "model_type"}
    unknown = set(payload.keys()) - allowed_keys
    if unknown:
        return _bad_request(
            f"Unexpected field(s) in request: {sorted(unknown)}. "
            f"Allowed fields: {sorted(allowed_keys)}."
        )

    # ── 2. Resolve the database path from Flask app config ──────────────────
    # Inference reads from the V4.2 database (read-only); the Flask app
    # config may point to a different db for the read-only REST endpoints.
    db_path = current_app.config.get("ML_DB_PATH") or current_app.config.get("DB_PATH")

    # ── 3. Invoke the inference pipeline ────────────────────────────────────
    try:
        predict_fn = _get_predict_fn()
    except Exception as exc:
        logger.exception("Failed to import ml.predict")
        return _service_error(f"Model inference module could not be loaded: {exc}")

    try:
        result = predict_fn(
            patient_id=patient_id,
            medication_anchor=medication_id,   # None means auto-resolve
            model_type=model_type,
            db_path=db_path,
        )
    except FileNotFoundError as exc:
        # Missing model artifact or database
        logger.error("FileNotFoundError during inference: %s", exc)
        return _service_error(str(exc))
    except RuntimeError as exc:
        # CNN-GRU artifacts missing, or architecture mismatch
        logger.error("RuntimeError during inference: %s", exc)
        return _service_error(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during inference for patient %s", patient_id)
        return _service_error(f"Unexpected inference error: {exc}")

    # ── 4. Map predict_patient() status to HTTP semantics ───────────────────
    status = result.get("status")
    warnings = result.get("data_quality_warnings", [])

    # Patient not found → 404
    if status == "insufficient_data" and any("not found" in w.lower() for w in warnings):
        # Check whether it's a patient-not-found or medication-not-found case
        if any("patient id" in w.lower() for w in warnings):
            return _not_found("patient", patient_id)
        if medication_id and any(medication_id.lower() in w.lower() for w in warnings):
            return _not_found("medication", medication_id)

    # ── 5. Attach patient_id to response for convenience ────────────────────
    result["patient_id"] = patient_id
    result["record_type"] = "ai_prediction"

    # Store successful predictions so reviews and the manufacturer dashboard can refer to them.
    # A logging failure must never block the clinician's assessment.
    result["prediction_id"] = None
    if result.get("status") == "success" and result.get("anchor_details"):
        try:
            result["prediction_id"] = store.log_prediction(
                current_app.config["APP_DB_PATH"], db_path, patient_id.upper(), model_type, result)
        except Exception:
            logger.exception("Could not store prediction for patient %s", patient_id)

    # Reorder keys for readability (patient_id first, then status)
    ordered = {"status": result["status"], "patient_id": patient_id}
    for k, v in result.items():
        if k not in ordered:
            ordered[k] = v

    return jsonify(ordered), 200
