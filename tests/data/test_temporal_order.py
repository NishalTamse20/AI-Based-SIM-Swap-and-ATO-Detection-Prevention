from datetime import datetime, timedelta

from simulator.generators.dataset_generator import generate_dataset
from simulator.config.generator_config import GeneratorConfig


def test_events_are_chronological_per_scenario(tmp_path, paysim_fixture):
    events = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, tmp_path / "events.csv"))
    timestamps_by_user = {}
    for event in events:
        timestamps_by_user.setdefault(event["user_id"], []).append(datetime.fromisoformat(event["timestamp"]))
    assert all(times == sorted(times) for times in timestamps_by_user.values())


def test_pretrigger_history_precedes_trigger_and_has_required_activity(tmp_path, paysim_fixture):
    events = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, tmp_path / "events.csv"))
    by_user = {}
    for event in events:
        by_user.setdefault(event["user_id"], []).append(event)

    for scenario_events in by_user.values():
        trigger = next(event for event in scenario_events if event["event_category"] == "TELECOM")
        trigger_time = datetime.fromisoformat(trigger["timestamp"])
        history = [event for event in scenario_events if not event["scenario_type"]]
        assert len(history) == 10
        history_times = [datetime.fromisoformat(event["timestamp"]) for event in history]
        assert all(timestamp < trigger_time for timestamp in history_times)
        assert min(history_times) >= trigger_time - timedelta(days=30)
        assert sum(event["event_category"] == "AUTHENTICATION" and event["event_type"] == "LOGIN"
                   and event["status"] == "SUCCESS" for event in history) == 5
        assert sum(event["event_category"] == "TRANSACTION" and event["event_type"] == "TRANSACTION"
                   and event["status"] == "SUCCESS" for event in history) == 5
        assert len({event["device_id"] for event in history}) == 1
        assert {event["device_id"] for event in history} == {trigger["device_id"]}
