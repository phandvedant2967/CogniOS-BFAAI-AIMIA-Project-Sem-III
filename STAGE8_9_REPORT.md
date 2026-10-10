# Stage 8 and 9 report: saved reviews, dose fix, model compatibility, manufacturer dashboard

The request had two parts that conflict on one point (part 1 said not to start the manufacturer dashboard, part 2 asked for it). I did them in order: Stage 8 (three fixes) first, then Stage 9 (dashboard) on top.

**Not touched:** models, datasets, `data/*/ehr.db`, `ml/` except one additive helper. No retraining.

## Stage 8: three fixes

**1. Doctor reviews are saved (SQLite)**
- New application database `data/cognios_app.db`, separate from the EHR databases (which stay read-only). Created on first use; not shipped.
- `doctor_reviews`: review category, note, patient ID, medication (ID, name, class), `prediction_id` reference, a snapshot of the AI class and model at review time, and a server-set UTC timestamp. Append-only: a patient can have several reviews, newest first.
- `ai_predictions`: successful predictions are now stored (one row per patient, medication, anchor and model; re-running updates the row, so counts are not inflated). Insufficient-data results are not stored.
- APIs: `POST /api/reviews`, `GET /api/reviews`, `GET /api/reviews/<id>`. Validation: category must be one of four, note at most 500 characters, patient and medication must exist, a prediction must belong to the same patient and medication, unknown fields (including a client-supplied timestamp) are rejected.
- Doctor vs AI: reviews carry `record_type: "doctor_assessment"`; predictions carry `"ai_prediction"`. A review shows the AI prediction only as a separate `prediction_reference`. In the UI each is labelled with its own chip, and the saved-review list shows "AI prediction at the time" on its own line.
- The UI now saves through the API and lists saved assessments. "Download as JSON" still works and downloads the saved record. The old "held in this page only" text is gone.

**2. Dose-change details**
- `anchor_details` gains `previous_dose`, `new_dose`, `dose_unit`, taken from the recorded medication record. `dose_mg` is unchanged. For a start anchor `new_dose` is null. Missing values stay null and show as "not recorded", never 0.
- UI: dose-change anchors show "Previous dose" and "New dose". Checked on real data: V4_2_P0002 / Escitalopram shows 50 mg → 100 mg, matching the database.

**3. Model compatibility**
- The artifacts were written with scikit-learn 1.9.1; the environment had 1.8.0 (version warnings). `requirements.txt` now pins `scikit-learn==1.9.1` and `xgboost==3.4.1` (the version used to verify). `torch` is `>=2.2` (verified with 2.14.1).
- Verified: all three joblib artifacts load with warnings treated as errors; the XGBoost and CNN-GRU models load, produce 4-class output, and run end to end through the API.

## Stage 9: manufacturer dashboard (`/manufacturer`)

Sections: KPI cards, drug-wise safety signals (AI and doctor columns side by side, with an AI pattern-mix bar), cognitive/behavioral effects chart, trend over time (quarter or month, with a table view), and filters for drug, effect, date range, model.

**Data used:** only stored `ai_predictions` and `doctor_reviews`. Nothing is generated. With no stored predictions the page shows "No signals yet". Other honest states: no match for the filters, and "not enough cases".

**Definitions**
- *Case* = one patient, medication and anchor. If both models ran on a case, the newest prediction counts once.
- *Effect* = a recorded symptom flag (confusion, memory loss, attention problems, behavioral change) that is present in assessments within 180 days after the anchor and was not present in the 270 days before. It is computed once when the prediction is stored. It describes recorded data and is not causal.
- *Date* (filters and trend) = the anchor date of the medication start or dose change.
- *Doctor-reviewed case* = a case with a saved review linked to its prediction (latest review counts). Reviews with no linked prediction are not included. The clinician page tells the doctor when a review is not linked.

**Wording:** AI results are "pattern flags for clinician review, not confirmed adverse effects". Doctor results are "assessments, not a confirmed causal finding". The two use different chips and separate columns.

**Privacy**
- Patient IDs and hospital identifiers are read server-side only to count distinct patients and never leave the server. No individual records are exposed.
- Minimum group size 5 (distinct patients). Drug groups below 5 are dropped before anything else is computed; headline totals cover only shown groups, and only the number of withheld groups is reported. Every displayed count below 5 is returned as `{value: null, suppressed: true}` and shown as "<5". Zero is shown as 0.
- Complementary suppression: a "flagged" count is hidden when the remainder would be a small cell; class and category breakdowns pool small cells so they cannot be recovered by subtraction. Filter options list only drugs with at least 5 patients; date bounds are month-level.
- Tests assert that no patient identifier appears in any response.

## Tests

| | Before | After |
|---|---|---|
| Python (pytest, includes Node wrappers) | 46 | **102 passed** |
| Node tests | 15 | 26 (17 clinician + 9 manufacturer) |

New: `tests/test_stage8_fixes.py` (29), `tests/test_manufacturer.py` (26), `tests/test_manufacturer_ui_logic.py` (1), `tests/js/manufacturer.test.js` (9), `tests/conftest.py` (keeps tests off the real application database). All 45 other original tests pass unchanged. Three clinician JS tests encoded the old behaviour (starting-dose note, reviews "not stored") and were updated or replaced for the intended change. A headless-browser run of both pages (filters, assessment, dose display, review save and list) had no console errors.

## Limitations

- **The manufacturer dashboard is empty in the shipped project.** It fills as clinicians run assessments. Screenshots and the browser check used a throwaway database built from real model outputs; it is not included. I did not bulk-run predictions to populate it.
- **No authentication or access control.** Anyone who can reach the server can open `/manufacturer`. Add login and roles before real use.
- **Residual re-identification risk:** someone could compare results across many slightly different filters. There is no query limiting. The thresholds protect a single view, not repeated querying.
- Reviews cannot be edited or deleted. Effects are limited to the four recorded symptom flags and need assessments both before and after the anchor. The trend uses the medication-event date, not the date the prediction was run.
- XGBoost 3.4.1 is pinned because it was verified to load the model; the original training version is not recorded in the artifact.
- This sandbox ran torch without GPU libraries (CPU only); CUDA was not tested.
