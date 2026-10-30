import copy
import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import pandas as pd
import pytest

from ml.features.feature_generator import FEATURE_COLUMNS, IDENTIFIER_COLUMNS
from ml.features.temporal_features import FEATURE_COLUMNS as TEMPORAL_FEATURE_COLUMNS
from ml.models import reference_model
from ml.pipeline import assess_case, load_events_by_user
from ml.rules.rule_baseline import generate_rule_baselines
from simulator.baseline.behavioural_baseline import BASELINE_FIELDS


EVENTS_PATH = "data/synthetic/ato_synthetic_events.csv"
OUTCOMES = {"AMBIGUOUS", "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE", "CORROBORATED_RISK_INDICATOR",
            "MODEL_ONLY_RISK_INDICATOR", "NO_ATO_RISK_INDICATOR"}


def _csv_rows(path):
    with open(path, newline="", encoding="utf-8") as source:
        return {row["user_id"]: row for row in csv.DictReader(source)}


def _text(value):
    return "" if value is None else str(value)


@pytest.fixture(scope="module")
def events_by_user():
    return load_events_by_user(EVENTS_PATH)


@pytest.fixture(scope="module")
def unmodelled(events_by_user):
    return {user: assess_case(rows) for user, rows in events_by_user.items()}


@pytest.fixture(scope="module")
def reference():
    if not (reference_model.DEFAULT_ARTIFACT_DIR / reference_model.METADATA_FILENAME).exists():
        pytest.skip("reference model artifact missing; run python -m ml.models.reference_model")
    return reference_model.load_reference_model()


@pytest.fixture(scope="module")
def modelled_test_users(events_by_user, reference):
    model, metadata = reference
    return {user: assess_case(events_by_user[user], model=model, model_metadata=metadata)
            for user in metadata["test_user_ids"]}


def test_profiles_match_saved_phase5_profiles_for_all_users(unmodelled):
    saved = _csv_rows("data/synthetic/behavioural_baseline_profiles.csv")
    assert set(saved) == set(unmodelled) and len(saved) == 1000
    for user, result in unmodelled.items():
        assert {f: _text(result["baseline_profile"][f]) for f in BASELINE_FIELDS} == {f: saved[user][f] for f in BASELINE_FIELDS}


def test_features_match_saved_phase6_features_for_all_users(unmodelled):
    saved = _csv_rows("data/processed/features.csv")
    columns = (*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS)
    for user, result in unmodelled.items():
        assert {c: _text(result["feature_snapshot"][c]) for c in columns} == {c: saved[user][c] for c in columns}


def test_temporal_evidence_matches_saved_phase9_features_for_all_users(unmodelled):
    saved = _csv_rows("data/processed/temporal_features.csv")
    for user, result in unmodelled.items():
        assert {c: _text(result["temporal_evidence"][c]) for c in TEMPORAL_FEATURE_COLUMNS} == \
               {c: saved[user][c] for c in TEMPORAL_FEATURE_COLUMNS}


def test_rule_outcomes_match_batch_phase7_for_all_users(unmodelled):
    batch = generate_rule_baselines(pd.read_csv("data/processed/features.csv").to_dict("records"))["multi_signal"]
    expected = {row["user_id"]: (row["prediction"], row["matched_rules"]) for row in batch}
    for user, result in unmodelled.items():
        rule = result["response"]["component_evidence"]["rule_evidence"]
        assert (rule["prediction"], rule["matched_rules"]) == expected[user]


def test_rule_and_model_match_saved_phase8_test_rows_exactly(modelled_test_users):
    saved = pd.read_csv("data/processed/random_forest/predictions.csv", float_precision="round_trip",
                        dtype={"user_id": str})
    assert len(saved) == 200
    for row in saved.itertuples():
        evidence = modelled_test_users[row.user_id]["response"]["component_evidence"]
        assert evidence["rule_evidence"]["prediction"] == row.phase7_multi_signal_prediction
        assert evidence["model_evidence"]["prediction"] == row.rf_prediction
        assert evidence["model_evidence"]["probability_ato"] == row.rf_probability_ato


def test_every_response_keeps_the_no_score_decision(unmodelled, modelled_test_users):
    for result in [*unmodelled.values(), *modelled_test_users.values()]:
        response = result["response"]
        assert response["risk_score_0_100"] is None and response["risk_level"] == "UNDETERMINED"
        assert response["outcome"] in OUTCOMES
        assert response["provenance"]["model_probability_calibrated"] is False


def test_missing_model_gives_insufficient_evidence_not_safety(unmodelled):
    for result in unmodelled.values():
        response = result["response"]
        assert response["component_evidence"]["model_evidence"] is None
        if response["evidence_status"] != "AMBIGUOUS":
            assert response["outcome"] == "INSUFFICIENT_EVIDENCE" and response["recommended_action"] is None


def test_labels_and_scenario_type_never_influence_the_assessment(events_by_user, reference):
    model, metadata = reference
    for user in metadata["test_user_ids"][:25]:
        original = assess_case(events_by_user[user], model=model, model_metadata=metadata)
        relabelled = [{**row, "ato_label": "1" if row["ato_label"] != "1" else "0", "scenario_type": "CHANGED"}
                      for row in events_by_user[user]]
        unlabelled = [{k: v for k, v in row.items() if k not in ("ato_label", "scenario_type")}
                      for row in events_by_user[user]]
        assert assess_case(relabelled, model=model, model_metadata=metadata) == original
        assert assess_case(unlabelled, model=model, model_metadata=metadata) == original


def test_baseline_uses_only_pre_trigger_history(events_by_user):
    rows = events_by_user["USR-000001"]
    trigger = next(r for r in rows if r["event_category"] == "TELECOM")
    start = datetime.fromisoformat(trigger["timestamp"])
    base = assess_case(rows)["baseline_profile"]

    def _event(offset, event_id):
        return {**trigger, "event_id": event_id, "event_category": "AUTHENTICATION", "event_type": "LOGIN",
                "status": "SUCCESS", "device_id": "DEV-EXTRA", "timestamp": (start + offset).isoformat(timespec="seconds")}

    later = [*rows, _event(timedelta(minutes=5), "EVT-X1"), _event(timedelta(days=2), "EVT-X2")]
    assert assess_case(later)["baseline_profile"] == base
    earlier = [*rows, _event(timedelta(days=-1), "EVT-X3")]
    changed = assess_case(earlier)["baseline_profile"]
    assert changed["baseline_event_count"] == base["baseline_event_count"] + 1
    assert "DEV-EXTRA" in changed["known_device_ids"]


def test_assessment_is_deterministic(events_by_user, reference):
    model, metadata = reference
    user = metadata["test_user_ids"][0]
    first = assess_case(events_by_user[user], model=model, model_metadata=metadata)
    second = assess_case(copy.deepcopy(events_by_user[user]), model=model, model_metadata=metadata)
    assert first == second


@pytest.mark.parametrize("mutation, message", [
    ("two_triggers", "exactly one SIM/eSIM trigger"),
    ("bad_event", "invalid event_type"),
    ("two_users", "exactly one user_id"),
    ("no_history", "no pre-trigger history"),
    ("missing_timestamp", "missing required field 'timestamp'"),
    ("blank_timestamp", "missing required field 'timestamp'"),
    ("missing_event_id", "missing required field 'event_id'"),
    ("duplicate_history_event", "duplicate event_id 'EVT-000001-01'"),
    ("duplicate_trigger_record", "duplicate event_id"),
])
def test_invalid_cases_are_rejected(events_by_user, mutation, message):
    rows = copy.deepcopy(events_by_user["USR-000001"])
    trigger = next(r for r in rows if r["event_category"] == "TELECOM")
    if mutation == "missing_timestamp":
        del rows[0]["timestamp"]
    elif mutation == "blank_timestamp":
        rows[0]["timestamp"] = "  "
    elif mutation == "missing_event_id":
        rows[0]["event_id"] = ""
    elif mutation == "duplicate_history_event":
        rows.append(dict(rows[0]))
    elif mutation == "duplicate_trigger_record":
        rows.append(dict(trigger))
    elif mutation == "two_triggers":
        rows.append({**trigger, "event_id": "EVT-DUP"})
    elif mutation == "bad_event":
        rows.append({**trigger, "event_id": "EVT-BAD", "event_category": "ACCOUNT", "event_type": "BENEFICIARY_ADDED"})
    elif mutation == "two_users":
        rows.append({**trigger, "event_id": "EVT-OTHER", "user_id": "USR-OTHER"})
    else:
        rows = [r for r in rows if r["timestamp"] >= trigger["timestamp"]]
    with pytest.raises(ValueError, match=message):
        assess_case(rows)


def test_distinct_events_are_still_counted_and_valid_cases_unchanged(events_by_user):
    rows = events_by_user["USR-000001"]
    base = assess_case(rows)
    assert assess_case(copy.deepcopy(rows)) == base
    history_row = rows[0]
    distinct = [*rows, {**history_row, "event_id": "EVT-000001-EXTRA"}]
    assert assess_case(distinct)["baseline_profile"]["baseline_event_count"] == \
        base["baseline_profile"]["baseline_event_count"] + 1


def test_current_behaviour_on_legacy_templates(events_by_user, modelled_test_users):
    """Characterises outcomes on the legacy synthetic templates (labels joined only after assessment).

    Outcomes are fully determined by scenario template, which reflects template dependence rather
    than generalisation. Suspicious SIM replacement cases are AMBIGUOUS (rule M1), not flagged.
    """
    scenario = {user: next(r for r in rows if r["event_category"] == "TELECOM")["scenario_type"]
                for user, rows in events_by_user.items()}
    table = defaultdict(Counter)
    for user, result in modelled_test_users.items():
        table[scenario[user]][result["response"]["outcome"]] += 1
    assert {s: set(c) for s, c in table.items()} == {
        "LEGITIMATE_ESIM_CHANGE": {"NO_ATO_RISK_INDICATOR"},
        "LEGITIMATE_SIM_REPLACEMENT": {"NO_ATO_RISK_INDICATOR"},
        "LEGITIMATE_ESIM_DEVICE_RECOVERY": {"AMBIGUOUS"},
        "LEGITIMATE_SIM_AUTH_RECOVERY": {"AMBIGUOUS"},
        "LEGITIMATE_SIM_DEVICE_RECOVERY": {"AMBIGUOUS"},
        "SIM_ESIM_AUTH_ANOMALY": {"CORROBORATED_RISK_INDICATOR"},
        "SUSPICIOUS_ESIM_CHANGE": {"CORROBORATED_RISK_INDICATOR"},
        "SUSPICIOUS_SIM_REPLACEMENT": {"AMBIGUOUS"},
    }
    legitimate = [r["response"] for u, r in modelled_test_users.items() if scenario[u].startswith("LEGITIMATE")]
    assert all(response["recommended_action"] is None for response in legitimate)
