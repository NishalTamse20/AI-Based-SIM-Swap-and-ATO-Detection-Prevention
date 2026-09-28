from datetime import datetime

from simulator.generators.dataset_generator import generate_dataset
from simulator.config.generator_config import GeneratorConfig


def test_events_are_chronological_per_scenario(tmp_path, paysim_fixture):
    events = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, tmp_path / "events.csv"))
    timestamps_by_user = {}
    for event in events:
        timestamps_by_user.setdefault(event["user_id"], []).append(datetime.fromisoformat(event["timestamp"]))
    assert all(times == sorted(times) for times in timestamps_by_user.values())
