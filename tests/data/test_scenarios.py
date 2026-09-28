from collections import Counter, defaultdict
from datetime import datetime

from simulator.config.generator_config import GeneratorConfig
from simulator.generators.dataset_generator import generate_dataset
from simulator.scenarios.scenario_runner import SCENARIO_TYPES


def test_documented_scenario_labels_and_sim_is_not_label_source(tmp_path, paysim_fixture):
    expected = {
        "LEGITIMATE_SIM_REPLACEMENT": 0,
        "LEGITIMATE_ESIM_CHANGE": 0,
        "SUSPICIOUS_SIM_REPLACEMENT": 1,
        "SUSPICIOUS_ESIM_CHANGE": 1,
        "SIM_ESIM_AUTH_ANOMALY": 1,
    }
    assert set(SCENARIO_TYPES) == set(expected)
    events = generate_dataset(GeneratorConfig(5, 42, paysim_fixture, tmp_path / "events.csv"))
    by_user = defaultdict(list)
    for event in events:
        by_user[event["user_id"]].append(event)
    for scenario_events in by_user.values():
        trigger = next(event for event in scenario_events if event["event_category"] == "TELECOM")
        scenario_type = trigger["scenario_type"]
        label = expected[scenario_type]
        post_trigger = [event for event in scenario_events if event["scenario_type"]]
        history = [event for event in scenario_events if not event["scenario_type"]]
        assert {event["ato_label"] for event in post_trigger} == {label}
        assert all(event["ato_label"] is None and event["scenario_type"] is None for event in history)
        telecom_events = [event for event in post_trigger if event["event_category"] == "TELECOM"]
        assert len(telecom_events) == 1
        assert telecom_events[0]["ato_label"] == label


def test_supported_categories_and_specific_event_types(tmp_path, paysim_fixture):
    events = generate_dataset(GeneratorConfig(10, 42, paysim_fixture, tmp_path / "events.csv"))
    supported = {
        "TELECOM": {"SIM_SWAP", "SIM_REPLACEMENT", "ESIM_CHANGE"},
        "DEVICE": {"NEW_DEVICE", "DEVICE_CHANGE", "DEVICE_FINGERPRINT_CHANGE"},
        "AUTHENTICATION": {"LOGIN", "FAILED_LOGIN", "MFA_FAILURE", "MFA_SUCCESS", "PASSWORD_CHANGE"},
        "RECOVERY": {"PASSWORD_RESET", "ACCOUNT_RECOVERY", "RECOVERY_CONTACT_CHANGE"},
        "ACCOUNT": {"NEW_BENEFICIARY", "BENEFICIARY_CHANGE", "PROFILE_CHANGE"},
        "TRANSACTION": {"TRANSACTION", "TRANSFER", "WITHDRAWAL"},
        "BEHAVIOUR": {"LOGIN_TIME_DEVIATION", "LOCATION_DEVIATION", "DEVICE_DEVIATION", "TRANSACTION_PATTERN_DEVIATION"},
    }
    assert Counter(event["event_category"] for event in events)
    assert all(event["event_type"] in supported[event["event_category"]] for event in events)
