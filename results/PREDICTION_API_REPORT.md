# Prediction API Report

## Overview

Stage 6 integrates the V4.2 ML inference pipeline into the existing Flask backend via a single new endpoint.
No models were retrained, no datasets were modified, and no preprocessing objects were re-fitted.

---

## Endpoint

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `/api/predictions` |
| **Content-Type** | `application/json` |
| **Default model** | XGBoost (Tabular Baseline) — higher validation macro-F1 (0.3793) |
| **Optional model** | Longitudinal 1D CNN-GRU |
| **Assessment cutoff** | Anchor date **+ 180 days** (strictly enforced in `ml/predict.py`) |

---

## Request

### Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `patient_id` | string | **Yes** | Patient identifier (e.g. `"V4_2_P0001"`). Case-sensitive, whitespace-stripped. |
| `medication_id` | string | No | Specific medication to use as anchor (e.g. `"M001"`). If omitted, the pipeline auto-resolves the best eligible anchor using the label-free candidate rule. |
| `model_type` | string | No | `"xgboost"` *(default)* or `"cnn_gru"`. |

### Example

```json
{
    "patient_id":   "V4_2_P0001",
    "medication_id": "M001",
    "model_type":   "xgboost"
}
```

### Validation Rules

- `patient_id` must be a non-empty string after stripping whitespace.
- `medication_id`, when provided, must be a non-empty string.
- `model_type` must be exactly `"xgboost"` or `"cnn_gru"`.
- Unknown fields in the request body are rejected.
- A plain-text or form-encoded body is rejected.

---

## Responses

### HTTP 200 — Success

```json
{
    "status":              "success",
    "patient_id":          "V4_2_P0001",
    "predicted_class":     "medication_associated",
    "pattern_description": "Possible medication-associated pattern",
    "probabilities": {
        "medication_associated": 0.4821,
        "disease_progression":   0.2103,
        "other_cause":           0.1934,
        "uncertain":             0.1142
    },
    "confidence":          0.4821,
    "model_name":          "XGBoost (Tabular Baseline)",
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
        "Cognitive trajectory accelerated decline post-anchor (30-day slope: pre -0.30 vs post -0.82).",
        "Anchor drug Donepezil has cognitive risk index 2.1 and anticholinergic burden level 1."
    ]
}
```

### HTTP 200 — Insufficient Data

Returned when a patient is found but lacks the required longitudinal coverage (≥ 1 pre-anchor and ≥ 2 post-anchor cognitive assessments within the window).  
**No prediction is invented.** `predicted_class` is `null` and all probabilities are `0.0`.

```json
{
    "status":              "insufficient_data",
    "patient_id":          "V4_2_P0042",
    "predicted_class":     null,
    "pattern_description": "Insufficient longitudinal data for reliable assessment",
    "probabilities": {
        "medication_associated": 0.0,
        "disease_progression":   0.0,
        "other_cause":           0.0,
        "uncertain":             0.0
    },
    "confidence":          0.0,
    "model_name":          "XGBoost (Tabular Baseline)",
    "anchor_details":      null,
    "usable_observation_dates": {
        "pre_anchor":  [],
        "post_anchor": [],
        "cutoff_date": null
    },
    "data_quality_warnings": [
        "Insufficient follow-up cognitive assessments: observed 0 in (0d, +180d] window (minimum required: 2)."
    ],
    "supporting_evidence": []
}
```

### HTTP 400 — Bad Request

Returned for validation failures: missing/blank `patient_id`, invalid `model_type`, unexpected fields, or a non-JSON body.

```json
{
    "error":   "bad_request",
    "message": "'model_type' must be one of: ['cnn_gru', 'xgboost']. Received: 'random_forest'.",
    "field":   "model_type"
}
```

### HTTP 404 — Not Found

Returned when the `patient_id` does not exist in the ML database, or when the specified `medication_id` is not in that patient's history.

```json
{
    "error":      "not_found",
    "entity":     "patient",
    "identifier": "V4_2_P99999"
}
```

```json
{
    "error":      "not_found",
    "entity":     "medication",
    "identifier": "M_INVALID_999"
}
```

### HTTP 503 — Service Unavailable

Returned when a model artifact file or the ML database cannot be found, or when the inference module fails to load.

```json
{
    "error":   "service_unavailable",
    "message": "V4.2 XGBoost model not found at models/v4_2_xgboost.joblib"
}
```

---

## Response Fields

| Field | Type | Always present | Description |
|---|---|---|---|
| `status` | string | ✅ | `"success"` or `"insufficient_data"` |
| `patient_id` | string | ✅ | Echoes the requested patient ID |
| `predicted_class` | string \| null | ✅ | One of the four classes, or `null` if insufficient data |
| `pattern_description` | string | ✅ | Human-readable pattern label (uses "Possible … pattern" phrasing) |
| `probabilities` | object | ✅ | Model predicted probabilities for all four classes (sum ≈ 1.0) |
| `confidence` | float | ✅ | Probability of the winning class |
| `model_name` | string | ✅ | Full display name of the model used |
| `anchor_details` | object \| null | ✅ | Medication anchor details and 180-day cutoff date |
| `usable_observation_dates` | object | ✅ | ISO dates of pre- and post-anchor cognitive assessments used |
| `data_quality_warnings` | array | ✅ | Non-empty when coverage is borderline or a specific concern exists |
| `supporting_evidence` | array | ✅ | Factual timeline observations used to contextualise the prediction |

### Predicted Classes

| Class | Meaning |
|---|---|
| `medication_associated` | Possible medication-associated cognitive pattern |
| `disease_progression` | Possible disease progression pattern |
| `other_cause` | Possible other clinical cause pattern |
| `uncertain` | Uncertain clinical trajectory pattern |

> [!IMPORTANT]
> Probabilities are **raw model predicted probabilities** (multi-class softmax over XGBoost tree margin outputs). No post-hoc calibrator (Platt scaling, isotonic regression) was fitted on validation data. These probabilities support ranking and triage, but should not be interpreted as exact empirical frequencies.

> [!NOTE]
> The `pattern_description` always uses the phrase "Possible … pattern". The API never makes causal claims (e.g. "this drug caused cognitive harm").

---

## Design Decisions

### 180-Day Assessment Cutoff
All observations, medication records, and clinical events dated after `anchor_date + 180 days` are excluded inside `ml/predict.py` before any feature is computed. The endpoint layer does not re-implement or duplicate this logic.

### Preprocessing Consistency
The endpoint calls `predict_patient()` which reuses the pre-fitted `v4_2_preprocessor.joblib` (via `prep.transform()`). No fitting occurs during inference. Population-level reference constants (`POPULATION_MED_REF_DOSE`, `POPULATION_SCORE_SD`) are baked into `ml/predict.py` and match the training pipeline exactly (max feature difference: ≤ 1.91 × 10⁻⁸).

### Auto-Anchor Resolution
When `medication_id` is omitted the pipeline applies the same label-free candidate anchor rule used during training: it selects the medication episode with the best balance of pre/post cognitive assessments, highest cognitive risk index, and earliest start date — without access to any ground-truth labels.

### Explicit Insufficient-Data Response
The endpoint returns HTTP 200 with `"status": "insufficient_data"` rather than a fabricated prediction whenever required longitudinal coverage is missing. A 404 is used only when the patient or medication does not exist at all in the database.

### Model Lazy-Loading
The heavy `ml.predict` module (torch, joblib) is imported on the first request, not at Flask application start-up, to keep startup time short.

---

## New Files

| File | Purpose |
|---|---|
| [`app/predictions.py`](../app/predictions.py) | Flask Blueprint implementing `POST /api/predictions` |
| [`tests/test_predictions_api.py`](../tests/test_predictions_api.py) | 15 automated tests (see below) |

### Modified Files

| File | Change |
|---|---|
| [`app/__init__.py`](../app/__init__.py) | Registers `predictions.bp`; adds `ML_DB_PATH` config |
| [`config.py`](../config.py) | Adds `ML_DB_PATH` constant (default: `data/v4_2/ehr.db`) |

---

## Test Results

```
tests/test_predictions_api.py::test_01_successful_xgboost_prediction   PASSED
tests/test_predictions_api.py::test_02_successful_cnn_gru_prediction   PASSED
tests/test_predictions_api.py::test_03_missing_patient_id_returns_400  PASSED
tests/test_predictions_api.py::test_04_blank_patient_id_returns_400    PASSED
tests/test_predictions_api.py::test_05_invalid_model_type_returns_400  PASSED
tests/test_predictions_api.py::test_06_extra_fields_return_400         PASSED
tests/test_predictions_api.py::test_07_non_json_body_returns_400       PASSED
tests/test_predictions_api.py::test_08_nonexistent_patient_returns_404 PASSED
tests/test_predictions_api.py::test_09_invalid_medication_id_returns_404 PASSED
tests/test_predictions_api.py::test_10_insufficient_data_returns_explicit_status PASSED
tests/test_predictions_api.py::test_11_response_schema_complete        PASSED
tests/test_predictions_api.py::test_12_cutoff_180d_enforced_in_api    PASSED
tests/test_predictions_api.py::test_13_prediction_is_reproducible     PASSED
tests/test_predictions_api.py::test_14_predicted_class_always_in_target_classes PASSED
tests/test_predictions_api.py::test_15_data_quality_warnings_always_list PASSED

============================= 15 passed in 15.46s ==============================
```

---

## Configuration

| Environment Variable | Default | Purpose |
|---|---|---|
| `EHR_DB_PATH` | `data/ehr.db` | Primary REST API database (original dataset) |
| `EHR_ML_DB_PATH` | `data/v4_2/ehr.db` | ML inference database (V4.2 dataset) |

---

## Constraints Respected

- ✅ No models retrained
- ✅ No datasets modified or regenerated
- ✅ No new preprocessing fitting during inference
- ✅ 180-day cutoff strictly enforced (test 12 verifies invariance)
- ✅ No causal harm claims in any response field
- ✅ Existing V4.2 training metrics untouched
- ✅ All prior Stage 5 inference tests still pass (`tests/test_inference_pipeline.py`: 7/7 green)
