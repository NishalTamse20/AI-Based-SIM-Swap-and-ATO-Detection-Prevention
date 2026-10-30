import csv
from copy import deepcopy
from datetime import datetime, timedelta

from ml.features.feature_generator import FEATURE_COLUMNS
from simulator.baseline.behavioural_baseline import build_baseline_profiles, validate_baseline_cutoffs
from simulator.generators.sim_replacement_baseline import (
    build_historical_event_input,
    prepare_controlled_sim_replacement,
)
from simulator.generators.sim_replacement_cohort import generate_sim_replacement_cohort


def test_profiles_match_each_case_and_phase6_features(tmp_path, paysim_fixture):
    cohort = generate_sim_replacement_cohort()
    result = prepare_controlled_sim_replacement(cohort, paysim_fixture, tmp_path / "controlled")
    triggers = {row["user_id"]: row for row in cohort["events"] if row["event_category"] == "TELECOM"}
    profiles = {row["user_id"]: row for row in result["baseline_profiles"]}
    assert set(profiles) == set(triggers)
    for user_id, trigger in triggers.items():
        profile = profiles[user_id]
        assert profile["account_id"] == trigger["account_id"]
        assert profile["trigger_event_id"] == trigger["event_id"]
        assert profile["trigger_cutoff"] == trigger["timestamp"]
    assert len(result["features"]) == len(cohort["cases"])
    assert all(set(FEATURE_COLUMNS) <= set(row) for row in result["features"])


def test_history_is_strictly_pretrigger_and_profiles_validate(tmp_path, paysim_fixture):
    cohort = generate_sim_replacement_cohort()
    history = build_historical_event_input(cohort["events"], paysim_fixture, 42)
    triggers = [row for row in cohort["events"] if row["event_category"] == "TELECOM"]
    validation_input = history + triggers
    profiles = build_baseline_profiles(validation_input)
    assert validate_baseline_cutoffs(validation_input, profiles) == []
    by_user = {}
    for event in history:
        by_user.setdefault(event["user_id"], []).append(event)
    for trigger in triggers:
        trigger_time = datetime.fromisoformat(trigger["timestamp"])
        user_history = by_user[trigger["user_id"]]
        assert len(user_history) == 10
        assert all(datetime.fromisoformat(row["timestamp"]) < trigger_time for row in user_history)
        assert {row["device_id"] for row in user_history} == {trigger["device_id"]}
        assert sum(row["event_type"] == "LOGIN" and row["status"] == "SUCCESS" for row in user_history) == 5
        assert sum(row["event_category"] == "TRANSACTION" for row in user_history) == 5
        assert all(row["scenario_type"] is None and row["ato_label"] is None for row in user_history)


def test_history_profiles_and_input_features_are_independent_of_labels_and_future_events(tmp_path, paysim_fixture):
    cohort = generate_sim_replacement_cohort()
    changed = deepcopy(cohort)
    for event in changed["events"]:
        event["scenario_type"] = "ALTERED"
        event["ato_label"] = 1 - int(event["ato_label"])
    trigger = next(row for row in changed["events"] if row["event_category"] == "TELECOM")
    future = deepcopy(trigger)
    future["event_id"] = "EVT-SIMREP-FUTURE-ONLY"
    future["event_category"] = "TRANSACTION"
    future["event_type"] = "TRANSACTION"
    future["timestamp"] = (datetime.fromisoformat(trigger["timestamp"]) + timedelta(minutes=16)).isoformat(timespec="seconds")
    changed["events"].append(future)
    first = prepare_controlled_sim_replacement(cohort, paysim_fixture, tmp_path / "first")
    second = prepare_controlled_sim_replacement(changed, paysim_fixture, tmp_path / "second")
    assert first["historical_events"] == second["historical_events"]
    assert first["baseline_profiles"] == second["baseline_profiles"]
    assert [tuple(row[column] for column in FEATURE_COLUMNS) for row in first["features"]] == [
        tuple(row[column] for column in FEATURE_COLUMNS) for row in second["features"]
    ]


def test_outputs_are_isolated_and_existing_artifact_sentinels_are_unchanged(tmp_path, paysim_fixture):
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    frozen_files = [frozen_dir / "features.csv", frozen_dir / "predictions.csv"]
    for path in frozen_files:
        path.write_bytes(b"frozen artifact sentinel\n")
    before = {path: path.read_bytes() for path in frozen_files}

    output_dir = tmp_path / "data" / "processed" / "controlled_sim_replacement"
    raw_before = paysim_fixture.read_bytes()
    prepare_controlled_sim_replacement(generate_sim_replacement_cohort(), paysim_fixture, output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "historical_event_input.csv", "baseline_profiles.csv", "features.csv"
    }
    assert all(path.read_bytes() == content for path, content in before.items())
    assert paysim_fixture.read_bytes() == raw_before
    with (output_dir / "historical_event_input.csv").open(newline="", encoding="utf-8") as source:
        assert len(list(csv.DictReader(source))) == 60

