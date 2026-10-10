# EHR Data Foundation (Flask + SQLite)

Step 1 of the medication-safety / cognitive-risk project: store the 5 synthetic datasets as related SQLite tables and expose read-only retrieval endpoints. **No ML, no causal logic, no review logic** — data storage and retrieval only.

## Setup & run

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python scripts/import_data.py     # (re)builds data/ehr.db from data/raw/*
python run.py                     # http://127.0.0.1:5000
```

A pre-built `data/ehr.db` is included; re-running the import script rebuilds it from scratch.
Requires Python 3.10+.

Try it:

```bash
curl http://127.0.0.1:5000/api/patients/P0001
curl http://127.0.0.1:5000/api/patients/P0001/timeline
curl http://127.0.0.1:5000/api/drugs/Sertraline
```

## Project layout

```
config.py                  paths (DB, raw data); env overrides EHR_DB_PATH / EHR_RAW_DIR
run.py                     starts Flask
scripts/import_data.py     CLI: builds the database, prints a row-count check
app/__init__.py            app factory
app/schema.sql             table definitions, keys, indexes
app/importer.py            loading + cleaning logic
app/db.py                  per-request SQLite connection
app/queries.py             plain data-access functions (reusable by the ML step)
app/timeline.py            build_patient_timeline(conn, patient_id)
app/routes.py              API endpoints
data/raw/                  the 5 original files, untouched
data/ehr.db                SQLite database
```

## Tables

| Table | Primary key | Foreign keys | Source |
|---|---|---|---|
| `patients` | patient_id | – | dataset 1 |
| `patient_comorbidities` | (patient_id, comorbidity) | patient_id → patients | dataset 1 (comorbidities split) |
| `drugs` | medication_id | – | dataset 5 (+ dataset 2 `medication_reference`, identical) |
| `medication_history` | medication_record_id | patient_id → patients, medication_id → drugs | dataset 2 |
| `cognitive_behavioral` | cognitive_record_id | patient_id → patients | dataset 3 |
| `clinical_events` | clinical_event_id | patient_id → patients | dataset 4 |

Full column definitions: `app/schema.sql`.

## Endpoints (all GET)

| Endpoint | Returns |
|---|---|
| `/api/patients/<id>` | patient record |
| `/api/patients/<id>/medical-history` | diagnosis, stage, comorbidity list, dates |
| `/api/patients/<id>/medications` | medication episodes (joined with drug name/class) |
| `/api/patients/<id>/cognitive` | cognitive/behavioral assessments |
| `/api/patients/<id>/events` | clinical events |
| `/api/patients/<id>/timeline` | all dated records merged chronologically |
| `/api/drugs` | all drugs |
| `/api/drugs/<id or name>` | one drug, e.g. `M005` or `Sertraline` |

Patient IDs are case-insensitive. Unknown IDs return 404 with a JSON error.

## Data handling rules

- **Dates**: stored as `YYYY-MM-DD` text. Clinical-event dates arrived as `YYYY-MM-DD 00:00:00`; the time part (always midnight) is dropped.
- **Missing values**: empty cells become `NULL`. No rows are removed; the importer verifies source vs loaded row counts and foreign keys.
- **Literal `'None'`**: appears in the source as a real category (`drugs.anticholinergic_burden`, `cognitive_behavioral.trajectory_context`, `patients.comorbidities`). It is kept as the text `'None'`, not converted to NULL. For comorbidities, `'None'` is kept in `patients.comorbidities` but produces no row in `patient_comorbidities`.
- **Source labels kept as given**: `trajectory_context`, `adherence_status` (`Unknown` stays `Unknown`), `cognitive_impact_possible`, etc. are stored as-is and not interpreted.
- **Normalization**: `drug_name`, `drug_class`, `therapeutic_area` were repeated in dataset 2 and verified identical to dataset 5, so they are stored once in `drugs` and joined in queries.
- No values are derived, imputed or inferred.

## Timeline

`build_patient_timeline(conn, patient_id)` in `app/timeline.py` merges: diagnosis, enrollment, medication start / dose change / stop, cognitive assessments, clinical events and their resolution dates. Entries are sorted by date; same-day entries use a fixed display order, which is only a tie-break and implies nothing about sequence or cause. Each entry has `date`, `entry_type`, `source_table`, `source_id`, `summary`, `details` (the full source record).

## Stage 2: ML data preparation

`python scripts/preprocess_ml.py` reads `data/ehr.db` (read-only) via `app/queries.py` and `build_patient_timeline()`, and regenerates:

- `data/processed/patient_observations.csv` – one row per cognitive assessment / clinical event, date-ordered per patient, with day offsets (since diagnosis, enrollment, last medication start / dose change / stop of any drug).
- `data/processed/medication_episodes.csv` – one row per medication record, with day offsets between start, dose change and stop.

Dates are ISO; missing values stay blank; nothing imputed or dropped. No labels are used or created, and `trajectory_context` is deliberately not carried over. The script prints row counts and descriptive data checks on each run. No model training.

## Stage 2B (ground truth audit)
`python scripts/stage2b_ground_truth.py` writes `data/processed/ground_truth_labels.csv` and prints the leakage audit. See `STAGE2B_REPORT.md`. No model is trained and raw data is untouched.

## Stage 7: CogniOS doctor dashboard

```bash
pip install -r requirements.txt
python run.py            # open http://127.0.0.1:5000/
```

Look up a patient by ID (IDs in the assessment database look like `V4_2_P0001`), review history and timeline, pick a medication, run an XGBoost (default) or CNN-GRU assessment, and record a review.

- `app/dashboard.py`: serves the page and `GET /api/dashboard/patients/<id>`, a read-only bundle (profile, medications, assessments, events, timeline) built with the existing `app.queries` and `app.timeline` functions.
- It reads `ML_DB_PATH` (the V4.2 database), the same database `POST /api/predictions` uses. The original `/api/patients/...` routes read `data/ehr.db`, which holds different patients (`P0001`...), so they are not used by the dashboard.
- `app/static/`: `index.html`, `css/cognios.css` (design system, see `docs/DESIGN_SYSTEM.md`), `js/logic.js` (pure logic), `js/render.js` (pure renderers), `js/app.js` (controller). No build step.
- Doctor reviews are held in the page only. Nothing is written to the server; a JSON download is offered.
- Tests: `python -m pytest tests` (API and integration), `node --test tests/js/dashboard.test.js` (logic and renderers; also run by pytest when Node is installed).


---

## Stage 8–9 additions

**Install (versions matter):** the saved models were written with scikit-learn **1.9.1**, which is pinned in `requirements.txt`. `pip install -r requirements.txt`.

**Pages**

| URL | Audience |
|---|---|
| `/` | Clinician dashboard (patient review, AI assessment, doctor review) |
| `/manufacturer` | Manufacturer dashboard (aggregated, de-identified safety signals) |

**Application database** `data/cognios_app.db` (override with `COGNIOS_APP_DB_PATH`) is created on first run. It holds logged AI predictions and doctor reviews. The EHR databases and datasets are only ever opened read-only. The shipped project contains no stored predictions or reviews, so the manufacturer dashboard starts with an empty state until clinicians run assessments.

**Endpoints added**

| Endpoint | Purpose |
|---|---|
| `POST /api/reviews` | Save a doctor assessment (`patient_id`, `review_category`, optional `clinical_note`, `medication_id`, `prediction_id`). The server sets the timestamp. |
| `GET /api/reviews?patient_id=&medication_id=&prediction_id=&limit=` | List saved reviews, newest first |
| `GET /api/reviews/<id>` | One review |
| `POST /api/predictions` | Unchanged request; response now also has `record_type: "ai_prediction"`, `prediction_id`, and `anchor_details.previous_dose / new_dose / dose_unit` |
| `GET /api/manufacturer/filters` | Filter options (only drugs with at least 5 patients) |
| `GET /api/manufacturer/summary?drug=&effect=&from=&to=&model=&interval=` | Aggregated, de-identified signals; groups under 5 patients are suppressed |

Doctor reviews are returned as `record_type: "doctor_assessment"`; the AI prediction a review refers to is a separate `prediction_reference` object.

**Tests:** `python -m pytest` (Python and, via wrappers, the Node tests; Node.js is optional).
