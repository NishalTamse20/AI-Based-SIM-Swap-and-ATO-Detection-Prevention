import csv
import json
from copy import deepcopy
from datetime import datetime, timedelta

from simulator.baseline.behavioural_baseline import (
    build_baseline_profiles,
    build_profiles_from_csv,
    validate_baseline_cutoffs,
)
from simulator.config.generator_config import GeneratorConfig
from simulator.generators.dataset_generator import generate_dataset


def _events(tmp_path, paysim_fixture):
    return generate_dataset(GeneratorConfig(5, 42, paysim_fixture, tmp_path / "events.csv"))


def test_baseline_uses_only_rows_before_trigger(tmp_path, paysim_fixture):
    events = _events(tmp_path, paysim_fixture)
    profiles = build_baseline_profiles(events)
    assert len(profiles) == 5
    for profile in profiles:
        assert profile["baseline_event_count"] == 10
        assert profile["successful_login_count"] == 5
        assert profile["transaction_count"] == 5
        known_devices = json.loads(profile["known_device_ids"])
        assert len(known_devices) == 1 and known_devices[0]
        assert profile["baseline_start"] < profile["trigger_cutoff"]
        assert profile["transaction_amount_sum"] > 0
        assert "ato_label" not in profile and "scenario_type" not in profile
    assert len({profile["account_id"] for profile in profiles}) == 5


def test_post_trigger_and_label_or_paysim_flag_changes_do_not_change_baseline(tmp_path, paysim_fixture):
    events = _events(tmp_path, paysim_fixture)
    expected = build_baseline_profiles(events)
    assert validate_baseline_cutoffs(events, expected) == []
    changed = deepcopy(events)
    for event in changed:
        if event["scenario_type"]:
            event["ato_label"] = "0" if event["ato_label"] == 1 else "1"
            if event["event_category"] == "TRANSACTION":
                event["amount"] = "999999999999"
        if event["event_category"] == "TRANSACTION":
            event["paysim_isFraud"] = "1"
            event["paysim_isFlaggedFraud"] = "1"
    assert build_baseline_profiles(changed) == expected

    future_only = deepcopy(events)
    future_row = next(event for event in future_only if event["scenario_type"] and event["event_category"] == "TRANSACTION")
    future_row["timestamp"] = (datetime.fromisoformat(future_row["timestamp"]) + timedelta(days=1)).isoformat(timespec="seconds")
    assert build_baseline_profiles(future_only) == expected


def test_baseline_output_is_reproducible(tmp_path, paysim_fixture):
    source = tmp_path / "events.csv"
    events = _events(tmp_path, paysim_fixture)
    with source.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=events[0].keys())
        writer.writeheader()
        writer.writerows(events)
    first_path, second_path = tmp_path / "first.csv", tmp_path / "second.csv"
    first = build_profiles_from_csv(source, first_path)
    second = build_profiles_from_csv(source, second_path)
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
