import csv
from collections import Counter

from simulator.config.generator_config import GeneratorConfig
from simulator.generators.dataset_generator import generate_dataset
from simulator.generators.event_generator import EVENT_FIELDS


def test_schema_validity_and_configurable_size(tmp_path, paysim_fixture):
    output = tmp_path / "events.csv"
    events = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, output))
    assert len({event["user_id"] for event in events}) == 10
    assert all(tuple(event) == EVENT_FIELDS for event in events)
    with output.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len(events)
    assert set(EVENT_FIELDS).issubset(rows[0])
    assert Counter(row["scenario_type"] for row in rows)


def test_same_seed_generates_identical_events(tmp_path, paysim_fixture):
    first = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, tmp_path / "first.csv", 5, 30))
    second = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, tmp_path / "second.csv", 5, 30))
    assert first == second


def test_scenario_count_must_support_exact_balance():
    import pytest

    with pytest.raises(ValueError, match="divisible by 5"):
        GeneratorConfig(11)


def test_history_length_and_window_are_configurable(tmp_path, paysim_fixture):
    events = generate_dataset(GeneratorConfig(5, 42, paysim_fixture, tmp_path / "events.csv", 3, 10))
    history_by_user = {}
    trigger_by_user = {}
    for event in events:
        if event["event_category"] == "TELECOM":
            trigger_by_user[event["user_id"]] = event["timestamp"]
        if not event["scenario_type"]:
            history_by_user.setdefault(event["user_id"], []).append(event)
    assert all(len(history) == 6 for history in history_by_user.values())
    assert len(history_by_user) == 5
    from datetime import datetime, timedelta

    for user_id, history in history_by_user.items():
        trigger_time = datetime.fromisoformat(trigger_by_user[user_id])
        assert all(trigger_time - datetime.fromisoformat(event["timestamp"]) <= timedelta(days=10)
                   for event in history)
