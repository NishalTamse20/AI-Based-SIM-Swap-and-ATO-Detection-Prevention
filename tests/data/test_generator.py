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
    first = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, tmp_path / "first.csv"))
    second = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, tmp_path / "second.csv"))
    assert first == second


def test_scenario_count_must_support_exact_balance():
    import pytest

    with pytest.raises(ValueError, match="divisible by 5"):
        GeneratorConfig(11)
