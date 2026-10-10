> **Update:** the limitations about unsaved reviews, the dose-change display and the scikit-learn version were resolved in Stage 8; the manufacturer dashboard was added in Stage 9. See `STAGE8_9_REPORT.md`.

# Stage 7 report: CogniOS doctor dashboard

**Files added**: `app/dashboard.py`, `app/static/index.html`, `app/static/css/cognios.css`, `app/static/js/{logic,render,app}.js`, `tests/test_dashboard_api.py`, `tests/test_dashboard_ui_logic.py`, `tests/js/dashboard.test.js`, `docs/DESIGN_SYSTEM.md`, this report.
**Files changed**: `app/__init__.py` (registers the blueprint), `README.md`, `requirements.txt` (inference and test dependencies listed).
**Not touched**: models, datasets, `ml/`, `app/predictions.py`, existing routes and tests.

**Working**: patient lookup with validation; history, medications, assessments and events; chronological timeline with filters and an MMSE chart; honest "not recorded" display; medication selection (dropdown or table row); XGBoost / CNN-GRU assessment through `POST /api/predictions` showing class, four scores, anchor, evidence, warnings and insufficient-data status; four-option review with note.

**Tests**: 46 passed (22 original + 24 new: API/integration tests against real models, plus 15 JS logic/render tests). A headless-browser walkthrough (lookup, assessment, review) ran with no console errors.

**Limitations**
- Reviews are not stored (no storage layer exists); they live in the page and can be downloaded as JSON.
- The dashboard reads the V4.2 database; original-dataset IDs (`P0001`) are not found there by design.
- The API chooses the anchor episode for a medication; for a dose-change anchor, the API reports the episode's starting dose.
- No login, no multi-user state. Sidebar has one page.
- Saved model files were built with scikit-learn 1.9.1; this environment had 1.8.0 (loads with a version warning). Use matching versions in production.
- Manufacturer dashboard not implemented.
