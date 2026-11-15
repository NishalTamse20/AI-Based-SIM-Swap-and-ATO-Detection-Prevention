from collections import Counter

import pytest
from fastapi.testclient import TestClient

from backend.app import db
from backend.app.main import _json_safe, create_app
from ml.models.reference_model import load_reference_model
from ml.pipeline import assess_case, load_events_by_user

EVENTS_PATH = "data/synthetic/ato_synthetic_events.csv"
EVALUATION_FIELDS = {"ato_label", "scenario_type", "paysim_isFraud", "paysim_isFlaggedFraud"}


@pytest.fixture(scope="module")
def reference():
    model, metadata = load_reference_model()
    return model, metadata, load_events_by_user(EVENTS_PATH)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    url = f"sqlite:///{(tmp_path_factory.mktemp('api') / 'api.db').as_posix()}"
    with TestClient(create_app(database_url=url)) as test_client:
        yield test_client


def test_health_reports_loaded_model_and_database(client, reference):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["database_backend"] == "sqlite"
    assert body["model_sha256"] == reference[1]["model_sha256"]


def test_cases_are_exactly_the_phase8_test_split(client, reference):
    cases = client.get("/api/cases").json()
    assert sorted(case["user_id"] for case in cases) == sorted(reference[1]["test_user_ids"])
    assert {case["trigger_type"] for case in cases} <= {"SIM_SWAP", "SIM_REPLACEMENT", "ESIM_CHANGE"}


def test_training_split_user_is_not_served(client, reference):
    train_user = reference[1]["train_user_ids"][0]
    assert client.get(f"/api/cases/{train_user}/events").status_code == 404
    assert client.post(f"/api/cases/{train_user}/assess").status_code == 404


def test_events_hide_evaluation_fields_and_mark_window(client, reference):
    user = reference[1]["test_user_ids"][0]
    body = client.get(f"/api/cases/{user}/events").json()
    assert len(body["events"]) == len(reference[2][user])
    for event in body["events"]:
        assert not EVALUATION_FIELDS & event.keys()
    assert [event["phase"] for event in body["events"]].count("TRIGGER") == 1


def test_api_response_equals_pipeline_for_every_demo_case(client, reference):
    model, metadata, events = reference
    outcomes = Counter()
    for user in metadata["test_user_ids"]:
        body = client.post(f"/api/cases/{user}/assess").json()
        expected = _json_safe(assess_case(events[user], model=model, model_metadata=metadata))
        assert body["response"] == expected["response"]
        assert body["feature_snapshot"] == expected["feature_snapshot"]
        assert body["risk_score_0_100"] is None and body["risk_level"] == "UNDETERMINED"
        outcomes[body["outcome"]] += 1
    # Documented outcome counts for the 200 test users (docs/requirements/assessment-pipeline.md).
    assert outcomes == {"AMBIGUOUS": 94, "CORROBORATED_RISK_INDICATOR": 54, "NO_ATO_RISK_INDICATOR": 52}


def test_assessments_are_stored_and_retrievable(client, reference):
    user = reference[1]["test_user_ids"][1]
    created = client.post(f"/api/cases/{user}/assess").json()
    stored = client.get(f"/api/assessments/{created['id']}").json()
    assert stored["response"] == created["response"]
    assert client.get("/api/assessments?limit=1").json()[0]["id"] == created["id"]
    latest = {case["user_id"]: case["latest_outcome"] for case in client.get("/api/cases").json()}
    assert latest[user] == created["outcome"]
    assert client.get("/api/assessments/999999").status_code == 404


def test_unreachable_database_url_falls_back_to_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "SQLITE_FALLBACK_URL", f"sqlite:///{(tmp_path / 'fallback.db').as_posix()}")
    engine, backend = db.create_db_engine("postgresql+psycopg://demo:demo@127.0.0.1:1/none")
    assert backend == "sqlite-fallback"
    engine.dispose()
