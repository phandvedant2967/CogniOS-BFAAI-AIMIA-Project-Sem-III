# V4.2 Model Inference Pipeline Report

## 1. Overview and Core Function

The Stage 5 inference pipeline provides a reusable, production-grade interface in [`ml/predict.py`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/ml/predict.py) for analyzing a patient's longitudinal EHR records using our trained V4.2 models.

Primary API function:
```python
from ml.predict import predict_patient

result = predict_patient(
    patient_id="V4_2_P0001",
    medication_anchor=None,      # Specific medication ID (e.g. 'M015') or None for automatic anchor
    model_type="xgboost",        # 'xgboost' (default) or 'cnn_gru'
    db_path=None                 # Path to SQLite DB (defaults to data/v4_2/ehr.db)
)
```

### Model Selection
- **Default Model**: **XGBoost (Tabular Baseline)** is configured as the default model because it achieved superior validation Macro-F1 (**0.3793** vs 0.3711 for CNN-GRU) during model selection.
- **Optional Model**: **Longitudinal 1D CNN-GRU** is supported as an optional comparison model using chronological sequence tensors.
- In accordance with methodological standards, model selection was strictly determined using validation split performance, without consulting test split results.

---

## 2. Input Requirements & Eligibility Rules

### Input Parameters
1. `patient_id` *(str, required)*: The patient's identifier (e.g. `'V4_2_P0001'`).
2. `medication_anchor` *(str | dict, optional)*:
   - If provided as string: Target medication ID (e.g. `'M001'`, `'M015'`) or drug name.
   - If `None`: Automatically identifies candidates (all medication starts and dose changes) and resolves the most balanced anchor using the label-free candidate rule (`max min(n_pre, n_post)`).
3. `model_type` *(str, optional, default='xgboost')*: Model architecture to query (`'xgboost'` or `'cnn_gru'`).
4. `db_path` *(str, optional)*: SQLite database file path (defaults to [`data/v4_2/ehr.db`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/data/v4_2/ehr.db)).

### Longitudinal Eligibility Thresholds (Strictly Enforced)
To prevent invented predictions from sparse data, a case must meet:
- **Pre-Anchor Baseline Window**: $\ge 1$ cognitive assessment in `(anchor - 270 days, anchor]`.
- **Post-Anchor Follow-up Window**: $\ge 2$ cognitive assessments in `(anchor, anchor + 180 days]`.
- **Assessment Cutoff**: Exactly `anchor_date + 180 days`. No observation, event, or prescription after this date is ever accessed or incorporated.

---

## 3. Output Structure

The inference function returns a structured dictionary:

```json
{
  "status": "success",
  "predicted_class": "medication_associated",
  "pattern_description": "Possible medication-associated pattern",
  "probabilities": {
    "medication_associated": 0.8679,
    "disease_progression": 0.0580,
    "other_cause": 0.0396,
    "uncertain": 0.0344
  },
  "confidence": 0.8679,
  "model_name": "XGBoost (Tabular Baseline)",
  "anchor_details": {
    "medication_id": "M015",
    "drug_name": "Donepezil",
    "drug_class": "Cholinesterase Inhibitor",
    "anchor_date": "2022-08-28",
    "anchor_type": "start",
    "dose_mg": 50.0,
    "cutoff_date": "2023-02-24",
    "horizon_days": 180
  },
  "usable_observation_dates": {
    "pre_anchor": ["2022-03-12", "2022-06-16"],
    "post_anchor": ["2022-09-30", "2022-12-14", "2023-02-18"],
    "cutoff_date": "2023-02-24"
  },
  "data_quality_warnings": [],
  "supporting_evidence": [
    "MMSE score changed by -4.5 points (last baseline: 13.0, 180-day post mean: 8.5).",
    "Cognitive trajectory accelerated decline post-anchor (30-day slope: pre -0.15 vs post -0.82).",
    "Anchor drug Donepezil has cognitive risk index 2.0 and anticholinergic burden level 0.",
    "Nearest acute event category in window: None."
  ]
}
```

### Verification of Probability Calibration
- **Finding**: **No probability calibrator (e.g. `CalibratedClassifierCV`) was fitted or saved** for the trained XGBoost model.
- **Classification of Outputs**: The output probability scores are **raw model predicted probabilities** (computed via multi-class softmax over boosted tree logits during `predict_proba()`), **not post-hoc calibrated probabilities**.
- In the code and reports, these values are strictly identified as model predicted probabilities.

### Insufficient-Data Response
When a patient lacks sufficient records, an invalid medication ID is passed, or fewer than required observations exist:

```json
{
  "status": "insufficient_data",
  "predicted_class": null,
  "pattern_description": "Insufficient longitudinal data for reliable assessment",
  "probabilities": {
    "medication_associated": 0.0,
    "disease_progression": 0.0,
    "other_cause": 0.0,
    "uncertain": 0.0
  },
  "confidence": 0.0,
  "model_name": "XGBoost (Tabular Baseline)",
  "anchor_details": { ... },
  "usable_observation_dates": { ... },
  "data_quality_warnings": [
    "Insufficient follow-up cognitive assessments: observed 0 in (0d, +180d] window (minimum required: 2)."
  ],
  "supporting_evidence": []
}
```

### Standardized Clinical Phrasing
Per design guidelines, model scores are never described as causal certainty or probability of drug harm. Outputs consistently use:
- `"Possible medication-associated pattern"`
- `"Possible disease progression pattern"`
- `"Possible other clinical cause pattern"`
- `"Uncertain clinical trajectory pattern"`

---

## 4. Verification and Test Results

A full automated test suite was implemented in [`tests/test_inference_pipeline.py`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/tests/test_inference_pipeline.py). All 7 tests passed:

| Test Case | Objective | Result |
| :--- | :--- | :---: |
| `test_1_models_load_successfully` | Verified that both models, preprocessors, and builders load from disk with correct dimensionality. | **PASSED** |
| `test_2_correct_preprocessing_and_feature_order` | Checked that single-patient tabular features match the 185 transformed model features without NaNs. | **PASSED** |
| `test_3_prediction_reproducibility` | Confirmed deterministic, identical predictions across repeated calls for both XGBoost and CNN-GRU. | **PASSED** |
| `test_4_all_four_class_scores_returned` | Verified that probabilities are produced for all 4 classes and sum to $1.00 \pm 0.01$. | **PASSED** |
| `test_5_insufficient_data_handling` | Confirmed safe handling of missing patients, invalid drug IDs, and truncated observations without fabricated predictions. | **PASSED** |
| `test_6_assessment_cutoff_strictly_enforced` | Injected future observations at day $+300$; confirmed 100% invariance and strict cutoff filtering at $+180$ days. | **PASSED** |
| `test_7_consistency_against_training_evaluation_pipeline` | Compared inference predictions against training/eval cases, confirming numerical consistency. | **PASSED** |

### Feature Parity & Preprocessing Alignment
1. **Identical Feature Construction**: Single-patient feature extraction in `ml/predict.py` reuses the exact Stage 2C logic and population reference constants:
   - `POPULATION_MED_REF_DOSE`: median dose per drug across cohort (20.0 mg).
   - `POPULATION_SCORE_SD`: domain standard deviations from cohort (`mmse`: 4.79, `memory`: 15.76, `attention`: 14.72, `executive`: 14.71, `mood`: 7.14).
2. **Numeric Precision**: Comparing single-patient feature extraction on patient `V4_2_P0001` against the training row in `ml_cases.csv` yielded a maximum feature difference of **$1.91 \times 10^{-8}$** and a model prediction probability difference of **$0.0000$** (bitwise identical).
3. **Column Order & Preprocessing**: Preprocessor input columns are aligned with `prep.numeric_cols + prep.categorical_cols` prior to calling `prep.transform()`, ensuring 100% fidelity to training.
4. **180-Day Horizon Enforcement**: Any observation after `anchor_date + 180 days` is excluded at the database retrieval and window masking steps. Injected test observations at $+300$ days produced zero change in features or predictions.

---

## 5. Artifacts and Reusability

- **Inference Module**: [`ml/predict.py`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/ml/predict.py)
- **Sequence Preprocessing Artifact**: [`models/v4_2_sequence_builder.joblib`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/models/v4_2_sequence_builder.joblib) (fitted once on train split)
- **Tabular Preprocessor Artifact**: [`models/v4_2_preprocessor.joblib`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/models/v4_2_preprocessor.joblib) (fitted once on train split)
- **Model Checkpoints**:
  - [`models/v4_2_xgboost.joblib`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/models/v4_2_xgboost.joblib)
  - [`models/v4_2_cnn_gru.pt`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/models/v4_2_cnn_gru.pt)
- **Automated Tests**: [`tests/test_inference_pipeline.py`](file:///C:/Users/hp/Downloads/ehr_backend_stage2c/ehr_backend/tests/test_inference_pipeline.py)

---

## 6. Scope and Limitations

1. **Synthetic-Data Prototype**: Models are trained on synthetic EHR records designed to exhibit overlapping, realistic trajectory variations. Output patterns represent computational risk stratification prototypes, not clinical diagnoses.
2. **Horizon Dependency**: Predictions assess trajectory patterns within a 180-day horizon post-anchor. Late-onset medication effects or delayed clinical events beyond 180 days are outside this model's scope.
3. **No Retraining at Inference**: Preprocessing transformations are frozen to train-set parameters. Unknown categorical levels are gracefully mapped to `'Missing'` categories without crashing or refitting.
4. **Probability Interpretation**: Model scores reflect softmax decision margins from tree boosting, not calibrated post-hoc probabilities.
