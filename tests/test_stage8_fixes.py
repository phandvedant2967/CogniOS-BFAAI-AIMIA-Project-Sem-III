"""Stage 8: saved doctor reviews, dose-change details, model/scikit-learn compatibility.

Uses the real V4.2 database (opened read-only) and the real models. Reviews and logged
predictions go to a temporary application database, never to the repo.
"""
import hashlib
import os
import re
import sqlite3
import sys
import warnings

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

V42_DB = os.path.join(ROOT, "data", "v4_2", "ehr.db")
pytestmark = pytest.mark.skipif(not os.path.exists(V42_DB), reason="V4.2 database not found.")

DOSE_CASE = ("V4_2_P0002", "M006")   # recorded 50 mg -> 100 mg on 2021-06-19
START_CASE = ("V4_2_P0011", "M001")


def make_app(tmp_path, name="app.db"):
    from app import create_app
    app = create_app(db_path=V42_DB, ml_db_path=V42_DB, app_db_path=str(tmp_path / name))
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(tmp_path):
    return make_app(tmp_path).test_client()


def predict(client, pid, mid=None, model="xgboost"):
    body = {"patient_id": pid, "model_type": model}
    if mid:
        body["medication_id"] = mid
    r = client.post("/api/predictions", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- 3. model compatibility

def _pinned_sklearn():
    text = open(os.path.join(ROOT, "requirements.txt")).read()
    m = re.search(r"^scikit-learn==([0-9][^\s#]*)", text, re.M)
    assert m, "requirements.txt must pin scikit-learn with =="
    return m.group(1)


def test_scikit_learn_is_pinned_to_the_installed_and_artifact_version():
    import sklearn
    assert _pinned_sklearn() == sklearn.__version__


def test_artifacts_load_without_any_version_warning():
    import joblib
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        warnings.simplefilter("error", UserWarning)
        for f in ("v4_2_preprocessor.joblib", "v4_2_xgboost.joblib", "v4_2_sequence_builder.joblib"):
            joblib.load(os.path.join(ROOT, "models", f))


def test_both_models_load_and_produce_probabilities():
    import numpy as np
    import torch
    from ml.predict import load_artifacts
    a = load_artifacts()
    assert type(a["xgboost"]).__name__ == "XGBClassifier"
    assert type(a["cnn_gru"]).__name__ == "V42LongitudinalCNNGRU"
    assert len(a["xgboost"].classes_) == 4
    dim = a["cnn_gru"].input_dim
    with torch.no_grad():
        out = a["cnn_gru"](torch.zeros(1, 5, dim), torch.tensor([5]))
    assert out.shape == (1, 4) and np.isfinite(out.numpy()).all()


@pytest.mark.parametrize("model", ["xgboost", "cnn_gru"])
def test_both_models_run_end_to_end(client, model):
    r = predict(client, *START_CASE, model=model)
    assert r["status"] == "success"
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-3


# ---------------------------------------------------------------- 2. dose-change details

def test_dose_change_anchor_reports_recorded_previous_and_new_dose(client):
    r = predict(client, *DOSE_CASE)
    a = r["anchor_details"]
    assert a["anchor_type"] == "dose_change"
    conn = sqlite3.connect(V42_DB)
    dose, new, unit, change = conn.execute(
        "SELECT dose, new_dose, dose_unit, dose_change_date FROM medication_history "
        "WHERE patient_id=? AND medication_id=?", DOSE_CASE).fetchone()
    assert a["anchor_date"] == change
    assert (a["previous_dose"], a["new_dose"], a["dose_unit"]) == (dose, new, unit)
    assert a["previous_dose"] != a["new_dose"]
    assert a["dose_mg"] == dose   # existing field is unchanged


def test_start_anchor_has_no_new_dose(client):
    a = predict(client, *START_CASE)["anchor_details"]
    assert a["anchor_type"] == "start"
    assert a["new_dose"] is None and a["previous_dose"] is not None


def test_missing_dose_values_stay_null_not_zero(tmp_path):
    import shutil
    dst = tmp_path / "ehr.db"
    shutil.copy(V42_DB, dst)
    conn = sqlite3.connect(dst)
    conn.execute("UPDATE medication_history SET new_dose=NULL WHERE patient_id=? AND medication_id=?", DOSE_CASE)
    conn.commit(); conn.close()
    from app import create_app
    app = create_app(db_path=str(dst), ml_db_path=str(dst), app_db_path=str(tmp_path / "a.db"))
    r = app.test_client().post("/api/predictions", json={"patient_id": DOSE_CASE[0], "medication_id": DOSE_CASE[1]}).get_json()
    a = r["anchor_details"]
    assert a["anchor_type"] == "dose_change" and a["new_dose"] is None and a["previous_dose"] == 50.0


# ---------------------------------------------------------------- 1. reviews

def test_prediction_is_stored_and_rerun_does_not_duplicate(client, tmp_path):
    first = predict(client, *START_CASE)
    again = predict(client, *START_CASE)
    assert first["record_type"] == "ai_prediction"
    assert isinstance(first["prediction_id"], int) and first["prediction_id"] == again["prediction_id"]
    conn = sqlite3.connect(tmp_path / "app.db")
    assert conn.execute("SELECT COUNT(*) FROM ai_predictions").fetchone()[0] == 1
    row = conn.execute("SELECT predicted_class, medication_id, anchor_type FROM ai_predictions").fetchone()
    assert row == (first["predicted_class"], "M001", "start")


def test_insufficient_data_is_not_stored(client, tmp_path):
    import shutil
    dst = tmp_path / "sparse.db"
    shutil.copy(V42_DB, dst)
    c = sqlite3.connect(dst); c.execute("DELETE FROM cognitive_behavioral WHERE patient_id='V4_2_P0011'"); c.commit(); c.close()
    from app import create_app
    app = create_app(db_path=str(dst), ml_db_path=str(dst), app_db_path=str(tmp_path / "a2.db"))
    r = app.test_client().post("/api/predictions", json={"patient_id": "V4_2_P0011"}).get_json()
    assert r["status"] == "insufficient_data" and r["prediction_id"] is None
    assert sqlite3.connect(tmp_path / "a2.db").execute("SELECT COUNT(*) FROM ai_predictions").fetchone()[0] == 0


def test_save_and_retrieve_review_with_prediction_reference(client):
    pred = predict(client, *START_CASE)
    ts_before = None
    r = client.post("/api/reviews", json={
        "patient_id": START_CASE[0], "review_category": "other_cause", "clinical_note": "  Looks like progression. ",
        "medication_id": "M001", "prediction_id": pred["prediction_id"]})
    assert r.status_code == 201
    rec = r.get_json()
    # doctor assessment and AI prediction are distinct, labelled objects
    assert rec["record_type"] == "doctor_assessment" and rec["assessed_by"] == "doctor"
    assert rec["review_category"] == "other_cause" and rec["review_label"] == "Likely other cause"
    assert rec["clinical_note"] == "Looks like progression."
    assert rec["patient_id"] == START_CASE[0]
    assert rec["medication"]["medication_id"] == "M001" and rec["medication"]["drug_name"]
    ref = rec["prediction_reference"]
    assert ref["record_type"] == "ai_prediction" and ref["prediction_id"] == pred["prediction_id"]
    assert ref["predicted_class_at_review"] == pred["predicted_class"]
    assert ref["anchor_date"] == pred["anchor_details"]["anchor_date"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", rec["reviewed_at"])

    got = client.get(f"/api/reviews/{rec['review_id']}").get_json()
    assert got == rec
    lst = client.get(f"/api/reviews?patient_id={START_CASE[0]}").get_json()
    assert lst["count"] == 1 and lst["reviews"][0] == rec and lst["record_type"] == "doctor_assessment"


def test_review_without_assessment_has_no_prediction_reference(client):
    rec = client.post("/api/reviews", json={"patient_id": "V4_2_P0011", "review_category": "uncertain"}).get_json()
    assert rec["prediction_reference"] is None and rec["medication"] is None and rec["clinical_note"] == ""


def test_reviews_survive_a_restart_and_keep_history_newest_first(tmp_path):
    app1 = make_app(tmp_path)
    c1 = app1.test_client()
    a = c1.post("/api/reviews", json={"patient_id": "V4_2_P0011", "review_category": "uncertain"}).get_json()
    b = c1.post("/api/reviews", json={"patient_id": "V4_2_P0011", "review_category": "other_cause", "medication_id": "M001"}).get_json()
    c2 = make_app(tmp_path).test_client()   # new app instance, same database file
    items = c2.get("/api/reviews?patient_id=V4_2_P0011").get_json()["reviews"]
    assert [i["review_id"] for i in items] == [b["review_id"], a["review_id"]]
    assert [i["review_id"] for i in c2.get("/api/reviews?patient_id=V4_2_P0011&medication_id=M001").get_json()["reviews"]] == [b["review_id"]]
    assert c2.get("/api/reviews?patient_id=V4_2_P0099").get_json()["count"] == 0


@pytest.mark.parametrize("body,status", [
    ({}, 400),
    ({"patient_id": "V4_2_P0011"}, 400),
    ({"patient_id": "V4_2_P0011", "review_category": "caused_by_drug"}, 400),
    ({"patient_id": "bad id!", "review_category": "uncertain"}, 400),
    ({"patient_id": "V4_2_P0011", "review_category": "uncertain", "clinical_note": "x" * 501}, 400),
    ({"patient_id": "V4_2_P0011", "review_category": "uncertain", "clinical_note": 5}, 400),
    ({"patient_id": "V4_2_P0011", "review_category": "uncertain", "reviewed_at": "2020-01-01T00:00:00Z"}, 400),
    ({"patient_id": "V4_2_P0011", "review_category": "uncertain", "prediction_id": "7"}, 400),
    ({"patient_id": "V4_2_P0011", "review_category": "uncertain", "prediction_id": True}, 400),
    ({"patient_id": "V4_2_P9999", "review_category": "uncertain"}, 404),
    ({"patient_id": "V4_2_P0011", "review_category": "uncertain", "medication_id": "M025"}, 404),
    ({"patient_id": "V4_2_P0011", "review_category": "uncertain", "prediction_id": 12345}, 404),
])
def test_invalid_reviews_are_rejected_and_nothing_is_stored(client, tmp_path, body, status):
    r = client.post("/api/reviews", json=body)
    assert r.status_code == status
    assert sqlite3.connect(tmp_path / "app.db").execute("SELECT COUNT(*) FROM doctor_reviews").fetchone()[0] == 0


def test_review_cannot_reference_another_patients_or_medications_prediction(client):
    pred = predict(client, *START_CASE)
    r = client.post("/api/reviews", json={"patient_id": "V4_2_P0018", "review_category": "uncertain", "prediction_id": pred["prediction_id"]})
    assert r.status_code == 400 and r.get_json()["field"] == "prediction_id"
    # same patient, but a different (valid) medication than the prediction's
    conn = sqlite3.connect(V42_DB)
    other = conn.execute("SELECT medication_id FROM medication_history WHERE patient_id=? AND medication_id<>'M001' LIMIT 1", (START_CASE[0],)).fetchone()
    if other:
        r = client.post("/api/reviews", json={"patient_id": START_CASE[0], "review_category": "uncertain",
                                              "medication_id": other[0], "prediction_id": pred["prediction_id"]})
        assert r.status_code == 400 and r.get_json()["field"] == "medication_id"


def test_review_listing_validation_and_missing_review(client):
    assert client.get("/api/reviews?limit=0").status_code == 400
    assert client.get("/api/reviews?limit=501").status_code == 400
    assert client.get("/api/reviews?prediction_id=abc").status_code == 400
    assert client.get("/api/reviews?patient_id=a%20b").status_code == 400
    assert client.get("/api/reviews/999").status_code == 404


def test_saving_reviews_and_predictions_never_modifies_the_ehr_database(client):
    before = sha(V42_DB)
    pred = predict(client, *START_CASE)
    client.post("/api/reviews", json={"patient_id": START_CASE[0], "review_category": "uncertain", "prediction_id": pred["prediction_id"]})
    assert sha(V42_DB) == before


def test_clinician_page_serves_review_ui_without_the_old_session_only_wording():
    js = open(os.path.join(ROOT, "app", "static", "js", "app.js")).read()
    assert "/api/reviews" in js and "held in this page only" not in js
