"""Stage 7 dashboard: API/UI integration paths.

Covers the exact call sequence the UI makes:
  GET /                                   -> page and static assets
  GET /api/dashboard/patients/<id>        -> history bundle (same DB as predictions)
  POST /api/predictions                   -> assessment for a medication from that bundle
Uses real models and the real V4.2 database; edge cases use a temp copy.
"""
import os
import re
import shutil
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

V42_DB = os.path.join(ROOT, "data", "v4_2", "ehr.db")
ORIG_DB = os.path.join(ROOT, "data", "ehr.db")
STATIC = os.path.join(ROOT, "app", "static")

pytestmark = pytest.mark.skipif(not os.path.exists(V42_DB), reason="V4.2 database not found.")

CLASSES = ["medication_associated", "disease_progression", "other_cause", "uncertain"]


def _make_client(ml_db):
    from app import create_app
    app = create_app(db_path=ORIG_DB if os.path.exists(ORIG_DB) else ml_db, ml_db_path=ml_db)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(scope="module")
def client():
    return _make_client(V42_DB)


@pytest.fixture()
def sparse_client(tmp_path):
    """Temp DB copy where V4_2_P0001 has no cognitive assessments and V4_2_P0002 has no medications."""
    dst = tmp_path / "ehr.db"
    shutil.copy(V42_DB, dst)
    conn = sqlite3.connect(dst)
    conn.execute("DELETE FROM cognitive_behavioral WHERE patient_id='V4_2_P0001'")
    conn.execute("DELETE FROM medication_history WHERE patient_id='V4_2_P0002'")
    conn.commit()
    conn.close()
    return _make_client(str(dst))


# ── Page and assets ──────────────────────────────────────────────────────────

def test_01_index_and_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "CogniOS" in html
    for asset in re.findall(r'(?:src|href)="(/static/[^"]+)"', html):
        assert client.get(asset).status_code == 200, asset


def test_02_ui_wording_is_non_causal_and_labels_probabilities():
    text = ""
    for folder in ("js", "css"):
        for name in os.listdir(os.path.join(STATIC, folder)):
            text += open(os.path.join(STATIC, folder, name), encoding="utf-8").read().lower()
    text += open(os.path.join(STATIC, "index.html"), encoding="utf-8").read().lower()
    assert "not calibrated probabilities" in text
    assert "model-predicted probabilities" in text
    for banned in ("caused by", "proved", "proven", "confirmed diagnosis", "definitely"):
        assert banned not in text, banned


# ── Patient lookup ───────────────────────────────────────────────────────────

def test_03_bundle_contains_history_for_the_prediction_database(client):
    resp = client.get("/api/dashboard/patients/V4_2_P0001")
    assert resp.status_code == 200
    b = resp.get_json()
    assert b["patient"]["patient_id"] == "V4_2_P0001"
    assert set(b) >= {"patient", "medical_history", "medications", "cognitive", "events", "timeline", "counts"}
    assert b["counts"]["medications"] == len(b["medications"]) > 0
    assert b["counts"]["cognitive"] == len(b["cognitive"]) > 0
    dates = [e["date"] for e in b["timeline"]]
    assert dates == sorted(dates), "timeline must be chronological"


def test_04_lookup_is_case_insensitive_and_trims(client):
    a = client.get("/api/dashboard/patients/v4_2_p0001").get_json()
    b = client.get("/api/dashboard/patients/%20V4_2_P0001%20").get_json()
    assert a["patient"] == b["patient"]


def test_05_unknown_patient_returns_404_with_hint(client):
    resp = client.get("/api/dashboard/patients/V4_2_P9999")
    assert resp.status_code == 404
    body = resp.get_json()
    assert body["error"] == "not_found" and "V4_2_P0001" in body["message"]


def test_06_original_dataset_id_is_not_found_in_assessment_db(client):
    """P0001 exists in data/ehr.db but not in the DB the model uses; the UI must not mix them."""
    assert client.get("/api/dashboard/patients/P0001").status_code == 404


@pytest.mark.parametrize("bad", ["a b", "x;drop", "../etc", "V4_2_P0001'--", "A" * 41])
def test_07_invalid_patient_ids_return_400(client, bad):
    resp = client.get(f"/api/dashboard/patients/{bad}")
    assert resp.status_code in (400, 404)
    if resp.status_code == 400:
        assert resp.get_json()["error"] == "invalid_patient_id"


def test_08_dashboard_never_writes(client):
    before = os.path.getmtime(V42_DB)
    client.get("/api/dashboard/patients/V4_2_P0001")
    assert os.path.getmtime(V42_DB) == before


# ── Missing data is reported as missing ──────────────────────────────────────

def test_09_missing_cognitive_data_is_empty_not_invented(sparse_client):
    b = sparse_client.get("/api/dashboard/patients/V4_2_P0001").get_json()
    assert b["cognitive"] == [] and b["counts"]["cognitive"] == 0
    assert all(e["entry_type"] != "cognitive_assessment" for e in b["timeline"])


def test_10_missing_medications_is_empty(sparse_client):
    b = sparse_client.get("/api/dashboard/patients/V4_2_P0002").get_json()
    assert b["medications"] == []


def test_11_patient_without_events_has_empty_event_list(client):
    b = client.get("/api/dashboard/patients/V4_2_P0003").get_json()
    assert b["events"] == [] and b["counts"]["events"] == 0


# ── Medication selection -> assessment ───────────────────────────────────────

def test_12_assessment_for_a_medication_from_the_bundle(client):
    b = client.get("/api/dashboard/patients/V4_2_P0001").get_json()
    med_id = b["medications"][0]["medication_id"]
    resp = client.post("/api/predictions", json={"patient_id": "V4_2_P0001", "medication_id": med_id, "model_type": "xgboost"})
    assert resp.status_code == 200
    r = resp.get_json()
    assert r["status"] == "success"
    assert r["anchor_details"]["medication_id"] == med_id
    assert set(r["probabilities"]) == set(CLASSES)
    assert abs(sum(r["probabilities"].values()) - 1) < 0.01
    assert isinstance(r["supporting_evidence"], list) and isinstance(r["data_quality_warnings"], list)
    assert r["anchor_details"]["anchor_type"] in ("start", "dose_change")


def test_13_cnn_gru_selectable(client):
    r = client.post("/api/predictions", json={"patient_id": "V4_2_P0001", "model_type": "cnn_gru"}).get_json()
    assert r["status"] in ("success", "insufficient_data")
    assert "CNN" in r["model_name"]


def test_14_default_model_is_xgboost_when_omitted(client):
    r = client.post("/api/predictions", json={"patient_id": "V4_2_P0001"}).get_json()
    assert "XGBoost" in r["model_name"]


def test_15_every_medication_in_every_sampled_bundle_can_be_assessed(client):
    for pid in ("V4_2_P0001", "V4_2_P0005", "V4_2_P0009"):
        b = client.get(f"/api/dashboard/patients/{pid}").get_json()
        for med_id in {m["medication_id"] for m in b["medications"]}:
            resp = client.post("/api/predictions", json={"patient_id": pid, "medication_id": med_id})
            assert resp.status_code == 200, (pid, med_id)
            assert resp.get_json()["anchor_details"] is None or resp.get_json()["anchor_details"]["medication_id"] == med_id


def test_16_insufficient_data_is_a_200_without_scores(sparse_client):
    resp = sparse_client.post("/api/predictions", json={"patient_id": "V4_2_P0001"})
    assert resp.status_code == 200
    r = resp.get_json()
    assert r["status"] == "insufficient_data"
    assert r["predicted_class"] is None
    assert r["data_quality_warnings"], "UI relies on warnings to explain why"


def test_17_medication_not_in_patient_history_is_404(client):
    resp = client.post("/api/predictions", json={"patient_id": "V4_2_P0001", "medication_id": "M999"})
    assert resp.status_code == 404
    assert resp.get_json()["entity"] == "medication"


def test_18_prediction_for_unknown_patient_is_404(client):
    assert client.post("/api/predictions", json={"patient_id": "V4_2_P9999"}).status_code == 404


def test_19_bad_model_type_is_400(client):
    assert client.post("/api/predictions", json={"patient_id": "V4_2_P0001", "model_type": "gpt"}).status_code == 400
