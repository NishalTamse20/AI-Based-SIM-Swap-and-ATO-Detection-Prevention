from collections import Counter
from copy import deepcopy

import pandas as pd

from ml.rules.rule_baseline import (
    AMBIGUOUS,
    ATO_RISK_INDICATOR,
    NO_INDICATOR,
    generate_rule_baselines,
    multi_signal_prediction,
    telecom_context_prediction,
)


def _row(**updates):
    row = {
        "user_id": "USR-1",
        "account_id": "ACC-1",
        "event_id": "EVT-1",
        "telecom_event_present": 1,
        "sim_change": 1,
        "esim_change": 0,
        "new_device": 0,
        "authentication_anomaly": 0,
        "password_reset": 0,
        "failed_login_count": 0,
        "device_change": 0,
        "device_deviation": 0,
    }
    row.update(updates)
    return row


def test_t0_trigger_is_context_only_and_returns_explanation():
    result = telecom_context_prediction(_row())
    assert result["prediction"] == NO_INDICATOR
    assert result["matched_rules"] == ["T0_TRIGGER_CONTEXT_ONLY"]
    assert result["feature_evidence"] == {
        "telecom_event_present": 1,
        "sim_change": 1,
        "esim_change": 0,
    }
    assert "context" in result["explanation"]


def test_t1_no_trigger_context_returns_no_indicator():
    result = telecom_context_prediction(_row(telecom_event_present=0, sim_change=0, esim_change=0))
    assert result["prediction"] == NO_INDICATOR
    assert result["matched_rules"] == ["T1_NO_TRIGGER_CONTEXT"]


def test_m1_ambiguous_recovery_pattern_is_not_forced_to_a_binary_class():
    result = multi_signal_prediction(_row(
        new_device=1,
        authentication_anomaly=1,
        password_reset=1,
        failed_login_count=1,
        device_change=1,
        device_deviation=1,
    ))
    assert result["prediction"] == AMBIGUOUS
    assert result["matched_rules"] == ["M1_AMBIGUOUS_DEVICE_AUTH_RECOVERY"]
    assert result["feature_evidence"]["failed_login_count"] == 1


def test_m1_takes_precedence_over_device_auth_corroboration_signals():
    result = multi_signal_prediction(_row(
        new_device=1,
        authentication_anomaly=1,
        password_reset=1,
        device_change=1,
        device_deviation=1,
    ))
    assert result["matched_rules"] == ["M1_AMBIGUOUS_DEVICE_AUTH_RECOVERY"]
    assert result["prediction"] == AMBIGUOUS


def test_m2_device_authentication_corroboration_produces_risk_indicator():
    result = multi_signal_prediction(_row(
        new_device=1,
        authentication_anomaly=1,
        password_reset=0,
        device_deviation=1,
    ))
    assert result["prediction"] == ATO_RISK_INDICATOR
    assert result["matched_rules"] == ["M2_DEVICE_AUTH_CORROBORATION"]
    assert "not confirmed fraud" in result["explanation"]


def test_m2_accepts_device_change_alternative():
    result = multi_signal_prediction(_row(
        new_device=1,
        authentication_anomaly=1,
        device_change=1,
    ))
    assert result["prediction"] == ATO_RISK_INDICATOR
    assert result["matched_rules"] == ["M2_DEVICE_AUTH_CORROBORATION"]


def test_m3_no_configured_combination_is_not_asserted_as_legitimate():
    result = multi_signal_prediction(_row(new_device=1, authentication_anomaly=1))
    assert result["prediction"] == NO_INDICATOR
    assert result["matched_rules"] == ["M3_NO_CONFIGURED_MULTI_SIGNAL_MATCH"]
    assert "does not establish legitimate" in result["explanation"]


def test_rule_outputs_do_not_depend_on_target_scenario_fraud_or_transaction_fields():
    original = _row(new_device=1, authentication_anomaly=1, password_reset=1)
    changed = deepcopy(original)
    changed.update({
        "ato_label": 1,
        "scenario_type": "leaked-label-test",
        "paysim_isFraud": 1,
        "paysim_isFlaggedFraud": 1,
        "transaction_amount": 999999,
        "transaction_count": 100,
        "transaction_type": "CASH_OUT",
        "transaction_deviation": 999,
    })
    assert telecom_context_prediction(original) == telecom_context_prediction(changed)
    assert multi_signal_prediction(original) == multi_signal_prediction(changed)


def test_batch_baselines_are_reproducible_and_return_one_explained_result_per_row():
    rows = [
        _row(),
        _row(new_device=1, authentication_anomaly=1, password_reset=1),
        _row(new_device=1, authentication_anomaly=1, device_deviation=1),
    ]
    first = generate_rule_baselines(rows)
    second = generate_rule_baselines(rows)
    assert first == second
    for baseline_results in first.values():
        assert len(baseline_results) == len(rows)
        assert all({"prediction", "matched_rules", "feature_evidence", "explanation"} <= result.keys()
                   for result in baseline_results)


def test_current_feature_file_rule_output_counts():
    features = pd.read_csv("data/processed/features.csv").to_dict(orient="records")
    outputs = generate_rule_baselines(features)
    assert len(outputs["telecom_context"]) == 1000
    assert len(outputs["multi_signal"]) == 1000
    telecom_counts = Counter(result["prediction"] for result in outputs["telecom_context"])
    multi_counts = Counter(result["prediction"] for result in outputs["multi_signal"])
    assert telecom_counts == {NO_INDICATOR: 1000}
    assert multi_counts == {
        NO_INDICATOR: 250,
        AMBIGUOUS: 500,
        ATO_RISK_INDICATOR: 250,
    }
