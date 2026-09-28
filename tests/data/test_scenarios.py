from collections import Counter

from simulator.generators.dataset_generator import generate_dataset
from simulator.scenarios.scenario_runner import SCENARIO_TYPES, run_scenario


def test_documented_scenario_labels_and_sim_is_not_label_source():
    expected = {
        "LEGITIMATE_SIM_REPLACEMENT": 0,
        "LEGITIMATE_ESIM_CHANGE": 0,
        "SUSPICIOUS_SIM_REPLACEMENT": 1,
        "SUSPICIOUS_ESIM_CHANGE": 1,
        "SIM_ESIM_AUTH_ANOMALY": 1,
    }
    assert set(SCENARIO_TYPES) == set(expected)
    for index, (scenario_type, label) in enumerate(expected.items(), 1):
        events = run_scenario(index, scenario_type)
        assert {event["ato_label"] for event in events} == {label}
        telecom_events = [event for event in events if event["event_category"] == "TELECOM"]
        assert len(telecom_events) == 1
        assert telecom_events[0]["ato_label"] == label
        assert len(events) > 1


def test_supported_categories_and_specific_event_types(tmp_path, paysim_fixture):
    events = generate_dataset(__import__("simulator.config.generator_config", fromlist=["GeneratorConfig"]).GeneratorConfig(
        10, 42, paysim_fixture, tmp_path / "events.csv"))
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
