import csv
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal

from ml.features.feature_generator import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    OUTPUT_COLUMNS,
    TARGET_COLUMN,
    build_feature_rows,
    generate_features,
)
from simulator.baseline.behavioural_baseline import build_baseline_profiles
from simulator.config.generator_config import GeneratorConfig
from simulator.generators.dataset_generator import generate_dataset


def _dataset(tmp_path, paysim_fixture, count=5):
    path = tmp_path / "events.csv"
    events = generate_dataset(GeneratorConfig(count, 42, paysim_fixture, path))
    profiles = build_baseline_profiles(events)
    return path, events, profiles


def test_expected_columns_and_row_count(tmp_path, paysim_fixture):
    event_path, _, profiles = _dataset(tmp_path, paysim_fixture)
    baseline_path = tmp_path / "baseline.csv"
    with baseline_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=profiles[0].keys())
        writer.writeheader()
        writer.writerows(profiles)
    output_path = tmp_path / "features.csv"
    rows = generate_features(event_path, baseline_path, output_path)
    assert len(rows) == 5
    with output_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames) == OUTPUT_COLUMNS
        assert len(list(reader)) == 5
    assert len(FEATURE_COLUMNS) == 25


def test_target_scenario_and_pay_sim_flags_are_not_input_features(tmp_path, paysim_fixture):
    _, events, profiles = _dataset(tmp_path, paysim_fixture)
    rows = build_feature_rows(events, profiles)
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert "scenario_type" not in FEATURE_COLUMNS
    assert "paysim_isFraud" not in FEATURE_COLUMNS
    assert "paysim_isFlaggedFraud" not in FEATURE_COLUMNS
    assert not ({"scenario_type", "paysim_isFraud", "paysim_isFlaggedFraud"} & set(OUTPUT_COLUMNS))
    assert not (set(IDENTIFIER_COLUMNS) & set(FEATURE_COLUMNS))
    assert all(TARGET_COLUMN in row for row in rows)


def test_label_and_pay_sim_flag_changes_do_not_change_features(tmp_path, paysim_fixture):
    _, events, profiles = _dataset(tmp_path, paysim_fixture)
    original = build_feature_rows(events, profiles)
    changed = deepcopy(events)
    for event in changed:
        event["scenario_type"] = "CHANGED_SCENARIO"
        event["ato_label"] = "0" if str(event["ato_label"]) == "1" else "1"
        event["paysim_isFraud"] = "1"
        event["paysim_isFlaggedFraud"] = "1"
    changed_rows = build_feature_rows(changed, profiles)
    assert [{key: row[key] for key in FEATURE_COLUMNS} for row in original] == [
        {key: row[key] for key in FEATURE_COLUMNS} for row in changed_rows
    ]


def test_post_cutoff_events_do_not_change_prediction_features(tmp_path, paysim_fixture):
    _, events, profiles = _dataset(tmp_path, paysim_fixture)
    target_user = next(event["user_id"] for event in events if event["scenario_type"] == "SUSPICIOUS_SIM_REPLACEMENT")
    expected = build_feature_rows(events, profiles)
    changed = deepcopy(events)
    later_event = next(event for event in changed if event["user_id"] == target_user
                       and event["scenario_type"] and event["event_category"] == "TRANSACTION")
    assert later_event["timestamp"].endswith("00:30:00")
    later_event["amount"] = "999999999"
    actual = build_feature_rows(changed, profiles)
    expected_row = next(row for row in expected if row["user_id"] == target_user)
    actual_row = next(row for row in actual if row["user_id"] == target_user)
    assert {key: expected_row[key] for key in FEATURE_COLUMNS} == {
        key: actual_row[key] for key in FEATURE_COLUMNS
    }
    assert expected_row["transaction_count"] == 0
    assert expected_row["transaction_amount"] is None
    assert expected_row["transaction_type"] == ""
    assert expected_row["transaction_deviation"] is None


def test_event_at_exactly_15_minutes_is_included(tmp_path, paysim_fixture):
    _, events, profiles = _dataset(tmp_path, paysim_fixture)
    target_user = next(event["user_id"] for event in events if event["scenario_type"] == "SUSPICIOUS_SIM_REPLACEMENT")
    trigger = next(event for event in events if event["user_id"] == target_user and event["event_category"] == "TELECOM")
    transaction = next(event for event in events if event["user_id"] == target_user
                       and event["scenario_type"] and event["event_category"] == "TRANSACTION")
    transaction["timestamp"] = (datetime.fromisoformat(trigger["timestamp"]) + timedelta(minutes=15)).isoformat(timespec="seconds")
    row = next(row for row in build_feature_rows(events, profiles) if row["user_id"] == target_user)
    assert row["transaction_count"] == 1
    assert row["transaction_amount"] == Decimal(transaction["amount"])
    assert row["transaction_deviation"] is not None


def test_fixed_cutoff_applies_uniformly_across_scenarios(tmp_path, paysim_fixture):
    _, events, profiles = _dataset(tmp_path, paysim_fixture)
    rows = build_feature_rows(events, profiles)
    by_type = {
        next(event["scenario_type"] for event in events
             if event["user_id"] == row["user_id"] and event["event_category"] == "TELECOM"): row
        for row in rows
    }
    assert all(row["transaction_count"] == 0 for row in rows)
    assert by_type["SUSPICIOUS_SIM_REPLACEMENT"]["password_reset"] == 1
    assert by_type["SUSPICIOUS_SIM_REPLACEMENT"]["new_beneficiary"] == 0
    assert by_type["SUSPICIOUS_ESIM_CHANGE"]["device_deviation"] == 1
    assert by_type["SIM_ESIM_AUTH_ANOMALY"]["failed_login_count"] == 2


def test_feature_generation_is_reproducible(tmp_path, paysim_fixture):
    event_path, _, profiles = _dataset(tmp_path, paysim_fixture)
    baseline_path = tmp_path / "baseline.csv"
    with baseline_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=profiles[0].keys())
        writer.writeheader()
        writer.writerows(profiles)
    first_path, second_path = tmp_path / "first.csv", tmp_path / "second.csv"
    first = generate_features(event_path, baseline_path, first_path)
    second = generate_features(event_path, baseline_path, second_path)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
