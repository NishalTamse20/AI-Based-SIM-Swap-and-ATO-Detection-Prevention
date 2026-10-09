"""Build Phase 9 temporal features at the frozen trigger+15m snapshot."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "ato_synthetic_events.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_features.csv"
OBSERVATION_MINUTES = 15
IDENTIFIER_COLUMNS = ("user_id", "account_id", "event_id")
FEATURE_COLUMNS = (
    "time_to_new_device_seconds",
    "time_to_failed_login_seconds",
    "time_to_password_reset_seconds",
    "time_to_recovery_seconds",
    "time_to_auth_anomaly_seconds",
    "post_trigger_failed_login_count",
    "post_trigger_device_event_count",
    "post_trigger_recovery_event_count",
    "telecom_to_device_sequence",
    "telecom_to_auth_sequence",
    "telecom_to_recovery_sequence",
    "device_before_auth_sequence",
    "auth_before_recovery_sequence",
)
OUTPUT_COLUMNS = (*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS)
AUTH_ANOMALY_TYPES = frozenset({"FAILED_LOGIN", "MFA_FAILURE"})


def _timestamp(row: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(str(row["timestamp"]))


def _seconds(first: datetime, trigger: datetime) -> float:
    return (first - trigger).total_seconds()


def _first_time(rows: list[dict[str, Any]]) -> datetime | None:
    return min((_timestamp(row) for row in rows), default=None)


def build_temporal_feature_rows(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one label-free temporal feature row per telecom-trigger user.

    Only event category/type/timestamp and traceability identifiers are read.
    Rows at the trigger time are valid for time-to-event/count features, but
    sequence features require a strictly later event to establish ordering.
    """
    events_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in event_rows:
        if not event.get("user_id") or not event.get("timestamp"):
            raise ValueError("each event must contain user_id and timestamp")
        events_by_user[str(event["user_id"])].append(event)

    output: list[dict[str, Any]] = []
    for user_id in sorted(events_by_user):
        events = events_by_user[user_id]
        triggers = [event for event in events if event.get("event_category") == "TELECOM"]
        if len(triggers) != 1:
            raise ValueError(f"user {user_id} must have exactly one telecom trigger")
        trigger = triggers[0]
        trigger_time = _timestamp(trigger)
        cutoff = trigger_time + timedelta(minutes=OBSERVATION_MINUTES)
        observed = [
            event for event in events
            if trigger_time <= _timestamp(event) <= cutoff
        ]
        new_devices = [
            row for row in observed
            if row.get("event_category") == "DEVICE" and row.get("event_type") == "NEW_DEVICE"
        ]
        failed_logins = [
            row for row in observed
            if row.get("event_category") == "AUTHENTICATION" and row.get("event_type") == "FAILED_LOGIN"
        ]
        auth_anomalies = [
            row for row in observed
            if row.get("event_category") == "AUTHENTICATION"
            and row.get("event_type") in AUTH_ANOMALY_TYPES
        ]
        password_resets = [
            row for row in observed
            if row.get("event_category") == "RECOVERY" and row.get("event_type") == "PASSWORD_RESET"
        ]
        recovery_events = [row for row in observed if row.get("event_category") == "RECOVERY"]
        device_events = [
            row for row in observed
            if row.get("event_category") == "DEVICE"
            or (row.get("event_category") == "BEHAVIOUR" and row.get("event_type") == "DEVICE_DEVIATION")
        ]

        new_device_time = _first_time(new_devices)
        failed_login_time = _first_time(failed_logins)
        password_reset_time = _first_time(password_resets)
        recovery_time = _first_time(recovery_events)
        auth_time = _first_time(auth_anomalies)
        # All sequence decisions use the documented project event semantics.
        strict_new_device = [row for row in new_devices if _timestamp(row) > trigger_time]
        strict_auth = [row for row in auth_anomalies if _timestamp(row) > trigger_time]
        strict_recovery = [row for row in recovery_events if _timestamp(row) > trigger_time]
        strict_password_reset = [row for row in password_resets if _timestamp(row) > trigger_time]
        strict_device_time = _first_time(strict_new_device)
        strict_auth_time = _first_time(strict_auth)
        strict_recovery_time = _first_time(strict_recovery)
        strict_password_time = _first_time(strict_password_reset)

        account_ids = {str(row.get("account_id", "")) for row in events}
        if len(account_ids) != 1:
            raise ValueError(f"user {user_id} has inconsistent account identifiers")
        output.append({
            "user_id": user_id,
            "account_id": next(iter(account_ids)),
            "event_id": str(trigger.get("event_id", "")),
            "time_to_new_device_seconds": None if new_device_time is None else _seconds(new_device_time, trigger_time),
            "time_to_failed_login_seconds": None if failed_login_time is None else _seconds(failed_login_time, trigger_time),
            "time_to_password_reset_seconds": None if password_reset_time is None else _seconds(password_reset_time, trigger_time),
            "time_to_recovery_seconds": None if recovery_time is None else _seconds(recovery_time, trigger_time),
            "time_to_auth_anomaly_seconds": None if auth_time is None else _seconds(auth_time, trigger_time),
            "post_trigger_failed_login_count": len(failed_logins),
            "post_trigger_device_event_count": len(device_events),
            "post_trigger_recovery_event_count": len(recovery_events),
            "telecom_to_device_sequence": int(strict_device_time is not None),
            "telecom_to_auth_sequence": int(strict_auth_time is not None),
            "telecom_to_recovery_sequence": int(strict_recovery_time is not None),
            "device_before_auth_sequence": int(
                strict_device_time is not None and strict_auth_time is not None
                and strict_device_time < strict_auth_time
            ),
            "auth_before_recovery_sequence": int(
                strict_auth_time is not None and strict_recovery_time is not None
                and strict_auth_time < strict_recovery_time
            ),
        })
    return output


def generate_temporal_features(
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """Read the event artifact and write a label-free temporal feature CSV."""
    with Path(events_path).open("r", newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    features = build_temporal_feature_rows(rows)
    destination_path = Path(output_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(features)
    return features


if __name__ == "__main__":
    generated = generate_temporal_features()
    print(f"Generated {len(generated)} temporal feature rows at {DEFAULT_OUTPUT_PATH}")
