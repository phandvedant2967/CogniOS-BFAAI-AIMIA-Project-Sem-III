"""Stage 9: manufacturer dashboard.

Three layers:
  1. pure suppression helpers
  2. API against a temporary application DB seeded with synthetic rows (exact, known counts)
  3. end-to-end: real predictions + real reviews through the public APIs, then the aggregate
"""
import json
import os
import re
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import manufacturer as mf  # noqa: E402
from app import store  # noqa: E402

V42_DB = os.path.join(ROOT, "data", "v4_2", "ehr.db")


# ------------------------------------------------------------------ 1. helpers

def rows(n, patients=None, **kw):
    """n synthetic case rows spread over `patients` distinct patients (default: one each)."""
    patients = patients or n
    return [dict(patient_id=f"T{i % patients}", **kw) for i in range(n)]


def test_cell_suppresses_small_groups_but_reports_zero():
    assert mf.cell([]) == {"value": 0, "suppressed": False}
    assert mf.cell(rows(4)) == {"value": None, "suppressed": True}
    assert mf.cell(rows(5)) == {"value": 5, "suppressed": False}


def test_threshold_counts_distinct_patients_not_cases():
    assert mf.cell(rows(9, patients=4)) == {"value": None, "suppressed": True}
    assert mf.cell(rows(9, patients=5)) == {"value": 9, "suppressed": False}


def test_binary_split_hides_part_when_the_remainder_is_small():
    whole = rows(12)
    assert mf.binary_split(whole[:7], whole)["value"] == 7          # 7 and 5
    assert mf.binary_split(whole[:8], whole)["suppressed"]           # remainder 4 would be recoverable
    assert mf.binary_split(whole[:3], whole)["suppressed"]
    assert mf.binary_split([], whole)["value"] == 0
    assert mf.binary_split(whole, whole)["value"] == 12


def test_pool_small_makes_hidden_cells_unrecoverable():
    g = {"a": rows(20), "b": rows(7, 7), "c": rows(2), "d": rows(1)}
    for i, rr in enumerate(g["c"] + g["d"]):
        rr["patient_id"] = f"X{i}"
    shown, pooled, names = mf.pool_small(g)
    # c + d = 3 (< 5): the smallest reportable group (b) is merged so the pool is reportable
    assert set(shown) == {"a"} and pooled == {"value": 10, "suppressed": False}
    assert set(names) == {"b", "c", "d"}
    # every number that is shown is >= 5
    assert all(c["value"] >= 5 for c in shown.values())


def test_pool_small_with_nothing_hidden_and_when_everything_is_tiny():
    shown, pooled, _ = mf.pool_small({"a": rows(6), "b": rows(9)})
    assert pooled is None and {k: v["value"] for k, v in shown.items()} == {"a": 6, "b": 9}
    shown, pooled, _ = mf.pool_small({"a": rows(2), "b": rows(1)})
    assert shown == {} and pooled["suppressed"]


# ------------------------------------------------------------------ 2. seeded API

def seed(path, specs):
    """specs: list of dicts: drug, n (patients), cls, anchor, effects(set), reviewed(int) ..."""
    store.init_db(path)
    conn = store.connect(path)
    pid = 0
    for sp in specs:
        for i in range(sp["n"]):
            pid += 1
            eff = {c: (1 if k in sp.get("effects", ()) else 0) for k, (c, _f, _l) in store.EFFECT_COLUMNS.items()}
            cur = conn.execute(
                """INSERT INTO ai_predictions (patient_id, medication_id, drug_name, drug_class, model_type, model_name,
                   predicted_class, anchor_type, anchor_date, effect_confusion, effect_memory_loss, effect_attention,
                   effect_behavioral, predicted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (f"SECRETPATIENT{pid:04d}", sp["drug"], sp["name"], "Class X", sp.get("model", "xgboost"), "M",
                 sp["cls"] if i < sp.get("n_cls", sp["n"]) else sp.get("other_cls", "uncertain"),
                 "start", sp["anchor"], eff["effect_confusion"], eff["effect_memory_loss"], eff["effect_attention"],
                 eff["effect_behavioral"], "2026-01-01T00:00:00Z"))
            if i < sp.get("reviewed", 0):
                conn.execute(
                    "INSERT INTO doctor_reviews (patient_id, review_category, prediction_id, medication_id, reviewed_at) VALUES (?,?,?,?,?)",
                    (f"SECRETPATIENT{pid:04d}", sp.get("review_cat", "medication_associated"), cur.lastrowid, sp["drug"], "2026-02-01T00:00:00Z"))
    conn.commit(); conn.close()


@pytest.fixture()
def seeded(tmp_path):
    path = str(tmp_path / "app.db")
    seed(path, [
        # M001: 14 patients, 9 AI-flagged (remainder 5), 7 reviewed, confusion in all
        dict(drug="M001", name="Donepezil", n=14, cls="medication_associated", n_cls=9, other_cls="other_cause",
             anchor="2022-02-10", effects={"confusion"}, reviewed=7),
        # M002: 6 patients, none flagged, later date, memory loss
        dict(drug="M002", name="Sertraline", n=6, cls="uncertain", anchor="2022-08-20", effects={"memory_loss"}),
        # M003: only 3 patients: must be withheld entirely
        dict(drug="M003", name="Zolpidem", n=3, cls="medication_associated", anchor="2022-03-01", effects={"confusion"}, reviewed=3),
    ])
    from app import create_app
    app = create_app(db_path=V42_DB, ml_db_path=V42_DB, app_db_path=path)
    app.config["TESTING"] = True
    return app.test_client(), path


def summary(client, qs=""):
    r = client.get("/api/manufacturer/summary" + qs)
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


def test_empty_database_gives_an_honest_empty_state(tmp_path):
    from app import create_app
    c = create_app(db_path=V42_DB, ml_db_path=V42_DB, app_db_path=str(tmp_path / "e.db")).test_client()
    s = summary(c)
    assert s["empty"]["state"] == "no_data" and "No AI predictions have been stored" in s["empty"]["message"]
    assert s["drugs"] == [] and s["effects"] == [] and s["trend"]["points"] == []
    f = c.get("/api/manufacturer/filters").get_json()
    assert f["drugs"] == [] and f["date_bounds"] is None


def test_summary_counts_are_exact_and_small_groups_are_withheld(seeded):
    c, _ = seeded
    s = summary(c)
    assert s["empty"] is None and s["min_group_size"] == 5
    t = s["totals"]
    assert t["cases"] == {"value": 20, "suppressed": False}            # 14 + 6, not the 3 withheld
    assert t["drug_groups_shown"] == 2 and t["drug_groups_withheld"] == 1
    assert [d["medication_id"] for d in s["drugs"]] == ["M001", "M002"]
    d1 = s["drugs"][0]
    assert d1["ai"]["cases"]["value"] == 14
    assert d1["ai"]["flagged"]["value"] == 9 and d1["ai"]["flagged_share"] == round(9 / 14, 3)
    assert d1["doctor"]["reviewed_cases"]["value"] == 7
    assert d1["doctor"]["review_coverage"] == round(7 / 14, 3)
    d2 = s["drugs"][1]
    assert d2["ai"]["flagged"] == {"value": 0, "suppressed": False}
    assert d2["doctor"]["reviewed_cases"] == {"value": 0, "suppressed": False}


def test_ai_and_doctor_data_are_separate_and_labelled(seeded):
    c, _ = seeded
    s = summary(c)
    d = s["drugs"][0]
    assert d["ai"]["record_type"] == "ai_prediction" and d["doctor"]["record_type"] == "doctor_assessment"
    assert "not confirmed adverse effects" in s["labels"]["ai"]
    assert "not a confirmed causal finding" in s["labels"]["doctor"]
    # the doctor's category breakdown uses the review categories, never AI classes by accident
    assert d["doctor"]["categories"]["shown"]["medication_associated"]["value"] == 7


def test_effects_and_trend_cover_only_reportable_groups(seeded):
    c, _ = seeded
    s = summary(c)
    eff = {e["effect"]: e for e in s["effects"]}
    assert eff["confusion"]["cases_with_effect"]["value"] == 14          # M003's 3 confusion cases are excluded
    assert eff["memory_loss"]["cases_with_effect"]["value"] == 6
    assert eff["attention"]["cases_with_effect"] == {"value": 0, "suppressed": False}
    assert eff["confusion"]["rate"] == round(14 / 20, 3)
    assert eff["confusion"]["ai_flagged_with_effect"]["value"] == 9
    # memory loss: 6 cases, none flagged -> 0 is reported as 0, not hidden
    assert eff["memory_loss"]["ai_flagged_with_effect"] == {"value": 0, "suppressed": False}
    q = {p["period"]: p for p in s["trend"]["points"]}
    assert q["2022-Q1"]["cases"]["value"] == 14 and q["2022-Q3"]["cases"]["value"] == 6
    m = summary(c, "?interval=month")["trend"]["points"]
    assert {p["period"] for p in m} == {"2022-02", "2022-08"}


def test_filters_drug_effect_date_and_model(seeded):
    c, _ = seeded
    assert [d["medication_id"] for d in summary(c, "?drug=M002")["drugs"]] == ["M002"]
    s = summary(c, "?effect=memory_loss")
    assert [d["medication_id"] for d in s["drugs"]] == ["M002"] and s["totals"]["cases"]["value"] == 6
    s = summary(c, "?from=2022-06-01&to=2022-12-31")
    assert [d["medication_id"] for d in s["drugs"]] == ["M002"]
    assert summary(c, "?from=2023-01-01")["empty"]["state"] == "no_match"
    assert summary(c, "?model=cnn_gru")["empty"]["state"] == "no_match"
    assert summary(c, "?model=xgboost")["totals"]["cases"]["value"] == 20


def test_flagged_count_is_hidden_when_the_remainder_would_be_a_small_cell(tmp_path):
    path = str(tmp_path / "p.db")
    # 12 cases, 8 flagged: 12 - 8 = 4 would be recoverable, so the flagged count must be withheld
    seed(path, [dict(drug="M001", name="Donepezil", n=12, cls="medication_associated", n_cls=8,
                     other_cls="other_cause", anchor="2022-02-10", effects={"confusion"}, reviewed=7,
                     review_cat="medication_associated")])
    from app import create_app
    c = create_app(db_path=V42_DB, ml_db_path=V42_DB, app_db_path=path).test_client()
    s = summary(c)
    d = s["drugs"][0]
    assert d["ai"]["cases"]["value"] == 12
    assert d["ai"]["flagged"] == {"value": None, "suppressed": True} and d["ai"]["flagged_share"] is None
    assert s["totals"]["ai_flagged"]["suppressed"]
    assert s["effects"][0]["ai_flagged_with_effect"]["suppressed"]
    assert s["trend"]["points"][0]["ai_flagged"]["suppressed"]
    # reviewed 7 (all one category) is reportable
    assert d["doctor"]["reviewed_cases"]["value"] == 7
    assert d["doctor"]["likely_medication_associated"]["value"] == 7


def test_filtering_down_below_five_never_shows_a_small_number(seeded):
    c, _ = seeded
    s = summary(c, "?drug=M003")
    assert s["empty"]["state"] == "below_threshold"
    assert s["drugs"] == [] and "fewer than 5" in s["empty"]["message"]
    text = json.dumps(s)
    assert "Zolpidem" not in text and '"value": 3' not in text


def test_no_identifiers_appear_in_any_response(seeded):
    c, path = seeded
    conn = sqlite3.connect(path)
    ids = [r[0] for r in conn.execute("SELECT DISTINCT patient_id FROM ai_predictions")]
    bodies = [c.get(u).get_data(as_text=True) for u in (
        "/api/manufacturer/summary", "/api/manufacturer/summary?interval=month", "/api/manufacturer/summary?drug=M001",
        "/api/manufacturer/filters", "/manufacturer")]
    for b in bodies:
        assert "SECRETPATIENT" not in b and not any(i in b for i in ids)
        assert not re.search(r'"(patient_id|review_id|prediction_id|hospital\w*|name_of_patient)"', b)
    assert "V4_2_P" not in "".join(bodies)


def test_filter_options_exclude_unreportable_drugs_and_use_month_bounds(seeded):
    c, _ = seeded
    f = c.get("/api/manufacturer/filters").get_json()
    assert [d["medication_id"] for d in f["drugs"]] == ["M001", "M002"]
    assert f["date_bounds"] == {"from": "2022-02-01", "to": "2022-08-31"}


@pytest.mark.parametrize("qs", ["?effect=cancer", "?from=2022-13-01", "?from=2022/01/01", "?from=2023-01-01&to=2022-01-01",
                                 "?model=gpt", "?interval=week", "?drug=M001;DROP", "?to=2022-02-30"])
def test_invalid_filters_return_400(seeded, qs):
    c, _ = seeded
    r = c.get("/api/manufacturer/summary" + qs)
    assert r.status_code == 400 and r.get_json()["error"] == "bad_request"


def test_two_models_on_one_case_count_once(tmp_path):
    path = str(tmp_path / "m.db")
    seed(path, [dict(drug="M001", name="Donepezil", n=6, cls="medication_associated", anchor="2022-02-10"),
                dict(drug="M001", name="Donepezil", n=0, cls="x", anchor="2022-02-10")])
    conn = sqlite3.connect(path)
    # same cases re-predicted by the other model
    conn.execute("""INSERT INTO ai_predictions (patient_id, medication_id, drug_name, drug_class, model_type, model_name,
        predicted_class, anchor_type, anchor_date, predicted_at)
        SELECT patient_id, medication_id, drug_name, drug_class, 'cnn_gru', 'C', 'uncertain', anchor_type, anchor_date, '2026-03-01T00:00:00Z'
        FROM ai_predictions""")
    conn.commit(); conn.close()
    from app import create_app
    c = create_app(db_path=V42_DB, ml_db_path=V42_DB, app_db_path=path).test_client()
    assert summary(c)["totals"]["cases"]["value"] == 6                   # not 12
    assert summary(c, "?model=xgboost")["drugs"][0]["ai"]["flagged"]["value"] == 6
    assert summary(c, "?model=cnn_gru")["drugs"][0]["ai"]["flagged"]["value"] == 0


def test_manufacturer_api_is_read_only(seeded):
    c, path = seeded
    before = open(path, "rb").read()
    c.get("/api/manufacturer/summary"); c.get("/api/manufacturer/filters")
    assert open(path, "rb").read() == before
    assert c.post("/api/manufacturer/summary").status_code == 405


def test_page_and_assets_are_served(seeded):
    c, _ = seeded
    html = c.get("/manufacturer").get_data(as_text=True)
    assert "MANUFACTURER" in html and "/static/js/mfr.js" in html and "/static/css/cognios.css" in html
    for p in ("/static/js/mfr.js", "/static/js/mfr-app.js"):
        assert c.get(p).status_code == 200


# ------------------------------------------------------------------ 3. end to end (real models)

@pytest.mark.skipif(not os.path.exists(V42_DB), reason="V4.2 database not found.")
def test_end_to_end_predictions_and_reviews_reach_the_dashboard(tmp_path):
    from app import create_app
    app = create_app(db_path=V42_DB, ml_db_path=V42_DB, app_db_path=str(tmp_path / "e2e.db"))
    app.config["TESTING"] = True
    c = app.test_client()
    pids = [r[0] for r in sqlite3.connect(V42_DB).execute(
        "SELECT patient_id FROM medication_history WHERE medication_id='M001' ORDER BY patient_id LIMIT 14")]
    results = []
    for pid in pids:
        r = c.post("/api/predictions", json={"patient_id": pid, "medication_id": "M001"}).get_json()
        if r["status"] == "success":
            results.append(r)
    assert len(results) >= 10
    # an honest empty/threshold state for a drug with no stored predictions
    assert summary(c, "?drug=M009")["empty"]["state"] in ("no_match", "below_threshold")

    reviewed = results[:8]
    for r in reviewed:
        assert c.post("/api/reviews", json={"patient_id": r["patient_id"], "review_category": "other_cause",
                                            "prediction_id": r["prediction_id"]}).status_code == 201
    # a review not linked to a prediction must not appear in aggregates
    c.post("/api/reviews", json={"patient_id": pids[0], "review_category": "other_cause", "medication_id": "M001"})

    s = summary(c)
    d = s["drugs"][0]
    n = len(results)
    flagged = sum(r["predicted_class"] == "medication_associated" for r in results)
    assert d["medication_id"] == "M001" and d["ai"]["cases"]["value"] == n
    assert d["doctor"]["reviewed_cases"]["value"] == 8
    cats = d["doctor"]["categories"]["shown"]
    assert cats["other_cause"]["value"] == 8 and "medication_associated" not in cats
    if 5 <= flagged and (n - flagged == 0 or n - flagged >= 5):
        assert d["ai"]["flagged"]["value"] == flagged
    else:
        assert d["ai"]["flagged"]["value"] in (None, flagged, 0)
    # privacy over the whole real response
    text = json.dumps(s)
    assert "V4_2_P" not in text and "prediction_id" not in text
