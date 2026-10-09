from datetime import datetime, timedelta

import pandas as pd

from ml.features.temporal_features import (
    FEATURE_COLUMNS,
    build_temporal_feature_rows,
    generate_temporal_features,
)


T = datetime(2026, 1, 1, 12, 0, 0)


def _event(event_id, category, event_type, offset_seconds, **extra):
    return {
        "event_id": event_id,
        "user_id": "U1",
        "account_id": "A1",
        "event_category": category,
        "event_type": event_type,
        "timestamp": (T + timedelta(seconds=offset_seconds)).isoformat(),
        **extra,
    }


def _minimal_events():
    return [
        _event("trigger", "TELECOM", "SIM_SWAP", 0),
        _event("device", "DEVICE", "NEW_DEVICE", 30),
        _event("fail1", "AUTHENTICATION", "FAILED_LOGIN", 60),
        _event("fail2", "AUTHENTICATION", "FAILED_LOGIN", 120),
        _event("reset", "RECOVERY", "PASSWORD_RESET", 180),
        _event("recovery", "RECOVERY", "ACCOUNT_RECOVERY", 240),
    ]


def test_feature_schema_and_exact_inclusive_cutoff(tmp_path):
    rows = _minimal_events() + [
        _event("at-cutoff", "DEVICE", "DEVICE_CHANGE", 900),
        _event("after-cutoff", "AUTHENTICATION", "FAILED_LOGIN", 901),
    ]
    output = tmp_path / "temporal.csv"
    result = generate_temporal_features(_write_events(tmp_path, rows), output)
    frame = pd.read_csv(output)
    assert tuple(frame.columns[3:]) == FEATURE_COLUMNS
    assert result[0]["post_trigger_device_event_count"] == 2
    assert result[0]["post_trigger_failed_login_count"] == 2


def test_no_post_cutoff_event_type_can_change_any_feature():
    in_window = _minimal_events() + [_event("cutoff-device", "DEVICE", "DEVICE_CHANGE", 900)]
    future_events = [
        _event("future-device", "DEVICE", "NEW_DEVICE", 901),
        _event("future-login", "AUTHENTICATION", "FAILED_LOGIN", 902),
        _event("future-mfa", "AUTHENTICATION", "MFA_FAILURE", 903),
        _event("future-reset", "RECOVERY", "PASSWORD_RESET", 904),
        _event("future-recovery", "RECOVERY", "ACCOUNT_RECOVERY", 905),
    ]
    assert build_temporal_feature_rows(in_window) == build_temporal_feature_rows(in_window + future_events)


def test_event_order_and_exact_time_differences():
    result = build_temporal_feature_rows(_minimal_events())[0]
    assert result["time_to_new_device_seconds"] == 30
    assert result["time_to_failed_login_seconds"] == 60
    assert result["time_to_password_reset_seconds"] == 180
    assert result["time_to_recovery_seconds"] == 180
    assert result["time_to_auth_anomaly_seconds"] == 60
    assert result["telecom_to_device_sequence"] == 1
    assert result["telecom_to_auth_sequence"] == 1
    assert result["telecom_to_recovery_sequence"] == 1
    assert result["device_before_auth_sequence"] == 1
    assert result["auth_before_recovery_sequence"] == 1


def test_counts_and_same_timestamp_does_not_claim_order():
    rows = [
        _event("trigger", "TELECOM", "ESIM_CHANGE", 0),
        _event("same-device", "DEVICE", "NEW_DEVICE", 0),
        _event("same-auth", "AUTHENTICATION", "FAILED_LOGIN", 0),
        _event("same-recovery", "RECOVERY", "PASSWORD_RESET", 0),
        _event("deviation", "BEHAVIOUR", "DEVICE_DEVIATION", 1),
    ]
    result = build_temporal_feature_rows(rows)[0]
    assert result["time_to_new_device_seconds"] == 0
    assert result["time_to_auth_anomaly_seconds"] == 0
    assert result["time_to_recovery_seconds"] == 0
    assert result["post_trigger_failed_login_count"] == 1
    assert result["post_trigger_device_event_count"] == 2
    assert result["post_trigger_recovery_event_count"] == 1
    assert result["telecom_to_device_sequence"] == 0
    assert result["device_before_auth_sequence"] == 0
    assert result["auth_before_recovery_sequence"] == 0


def test_missing_event_fields_remain_missing_and_counts_zero():
    result = build_temporal_feature_rows([_event("trigger", "TELECOM", "SIM_REPLACEMENT", 0)])[0]
    for feature in FEATURE_COLUMNS[:5]:
        assert result[feature] is None
    for feature in FEATURE_COLUMNS[5:]:
        assert result[feature] == 0


def test_mfa_failure_uses_existing_auth_anomaly_semantics():
    result = build_temporal_feature_rows([
        _event("trigger", "TELECOM", "ESIM_CHANGE", 0),
        _event("mfa", "AUTHENTICATION", "MFA_FAILURE", 25),
    ])[0]
    assert result["time_to_auth_anomaly_seconds"] == 25
    assert result["telecom_to_auth_sequence"] == 1
    assert result["post_trigger_failed_login_count"] == 0


def test_labels_scenario_and_paysim_fields_never_enter_feature_output():
    rows = _minimal_events()
    for index, row in enumerate(rows):
        row.update({
            "ato_label": index % 2,
            "scenario_type": f"SCENARIO_{index}",
            "paysim_isFraud": 1,
            "paysim_isFlaggedFraud": 1,
        })
    output = build_temporal_feature_rows(rows)
    assert set(output[0]) == {"user_id", "account_id", "event_id", *FEATURE_COLUMNS}
    altered = [dict(row, ato_label=99, scenario_type="OTHER", paysim_isFraud=0, paysim_isFlaggedFraud=0) for row in rows]
    assert build_temporal_feature_rows(altered) == output


def _write_events(tmp_path, rows):
    import csv

    path = tmp_path / "events.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path
