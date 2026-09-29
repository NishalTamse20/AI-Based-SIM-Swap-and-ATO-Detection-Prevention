"""Build scenario-level features from observed events and pre-trigger profiles."""

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "ato_synthetic_events.csv"
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "data" / "synthetic" / "behavioural_baseline_profiles.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
PREDICTION_WINDOW_MINUTES = 15

IDENTIFIER_COLUMNS = ("user_id", "account_id", "event_id")
FEATURE_COLUMNS = (
    "sim_change", "esim_change", "telecom_event_present",
    "new_device", "device_change", "device_deviation", "known_device",
    "failed_login_count", "authentication_anomaly", "password_reset",
    "account_recovery", "recovery_event_present",
    "new_beneficiary", "account_change",
    "transaction_amount", "transaction_count", "transaction_type", "transaction_deviation",
    "baseline_event_count", "baseline_event_frequency", "baseline_transaction_count",
    "baseline_transaction_amount_mean", "baseline_transaction_amount_min",
    "baseline_transaction_amount_max", "baseline_successful_login_count",
)
TARGET_COLUMN = "ato_label"
OUTPUT_COLUMNS = (*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN)


def _as_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))


def build_feature_rows(
    event_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return one feature row per user at the frozen trigger+15m snapshot.

    The trigger and events through the inclusive 15-minute cutoff are observed.
    Historical profile columns come from the Phase 5 artifact. Labels are
    carried only as the output target and never participate in feature values.
    """
    events_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profiles_by_user: dict[str, dict[str, Any]] = {}
    for event in event_rows:
        events_by_user[event["user_id"]].append(event)
    for profile in baseline_rows:
        user_id = profile["user_id"]
        if user_id in profiles_by_user:
            raise ValueError(f"duplicate baseline profile for user {user_id}")
        profiles_by_user[user_id] = profile
    if set(events_by_user) != set(profiles_by_user):
        raise ValueError("event users and baseline profile users do not match")

    output: list[dict[str, Any]] = []
    for user_id in sorted(events_by_user):
        events = events_by_user[user_id]
        profile = profiles_by_user[user_id]
        account_ids = {event["account_id"] for event in events}
        if len(account_ids) != 1 or profile["account_id"] not in account_ids:
            raise ValueError(f"user {user_id} has inconsistent account identifiers")
        triggers = [event for event in events if event["event_category"] == "TELECOM"]
        if len(triggers) != 1:
            raise ValueError(f"user {user_id} must have exactly one SIM/eSIM trigger")
        trigger = triggers[0]
        trigger_time = _as_datetime(trigger["timestamp"])
        cutoff = trigger_time + timedelta(minutes=PREDICTION_WINDOW_MINUTES)
        if _as_datetime(profile["trigger_cutoff"]) != trigger_time:
            raise ValueError(f"baseline profile cutoff does not match trigger for user {user_id}")

        observed = [
            event for event in events
            if trigger_time <= _as_datetime(event["timestamp"]) <= cutoff
        ]
        telecom_types = {
            event["event_type"] for event in observed if event["event_category"] == "TELECOM"
        }
        device_events = [
            event for event in observed
            if event["event_category"] != "TELECOM" and event.get("device_id")
        ]
        known_devices = set(json.loads(profile["known_device_ids"]))
        observed_device_ids = {event["device_id"] for event in device_events}
        post_transactions = [
            event for event in observed if event["event_category"] == "TRANSACTION"
        ]
        transaction_amounts = [Decimal(event["amount"]) for event in post_transactions]
        transaction_types = sorted({event.get("transaction_type", "") for event in post_transactions if event.get("transaction_type")})
        transaction_amount = sum(transaction_amounts, Decimal(0)) if transaction_amounts else None
        post_transaction_mean = (
            sum(transaction_amounts, Decimal(0)) / len(transaction_amounts)
            if transaction_amounts else None
        )
        baseline_transaction_mean = _decimal(profile.get("transaction_amount_mean"))
        transaction_deviation = (
            post_transaction_mean - baseline_transaction_mean
            if post_transaction_mean is not None and baseline_transaction_mean is not None
            else None
        )
        failed_login_count = sum(
            event["event_category"] == "AUTHENTICATION" and event["event_type"] == "FAILED_LOGIN"
            for event in observed
        )
        authentication_anomaly = any(
            event["event_category"] == "AUTHENTICATION"
            and event["event_type"] in {"FAILED_LOGIN", "MFA_FAILURE"}
            for event in observed
        )
        recovery_events = [event for event in observed if event["event_category"] == "RECOVERY"]
        account_events = [event for event in observed if event["event_category"] == "ACCOUNT"]

        output.append({
            "user_id": user_id,
            "account_id": profile["account_id"],
            "event_id": trigger["event_id"],
            "sim_change": int(bool(telecom_types & {"SIM_SWAP", "SIM_REPLACEMENT"})),
            "esim_change": int("ESIM_CHANGE" in telecom_types),
            "telecom_event_present": int(bool(telecom_types)),
            "new_device": int(any(event["event_category"] == "DEVICE" and event["event_type"] == "NEW_DEVICE" for event in observed)),
            "device_change": int(any(event["event_category"] == "DEVICE" and event["event_type"] == "DEVICE_CHANGE" for event in observed)),
            "device_deviation": int(any(
                (event["event_category"] == "BEHAVIOUR" and event["event_type"] == "DEVICE_DEVIATION")
                or (event["event_category"] == "DEVICE" and event["status"] == "DEVIATION")
                for event in observed
            )),
            "known_device": int(bool(observed_device_ids) and observed_device_ids.issubset(known_devices)),
            "failed_login_count": failed_login_count,
            "authentication_anomaly": int(authentication_anomaly),
            "password_reset": int(any(event["event_type"] == "PASSWORD_RESET" for event in recovery_events)),
            "account_recovery": int(any(event["event_type"] == "ACCOUNT_RECOVERY" for event in recovery_events)),
            "recovery_event_present": int(bool(recovery_events)),
            "new_beneficiary": int(any(event["event_type"] == "NEW_BENEFICIARY" for event in account_events)),
            "account_change": int(bool(account_events)),
            "transaction_amount": transaction_amount,
            "transaction_count": len(post_transactions),
            "transaction_type": "|".join(transaction_types),
            "transaction_deviation": transaction_deviation,
            "baseline_event_count": int(profile["baseline_event_count"]),
            "baseline_event_frequency": _decimal(profile["event_frequency_per_day"]),
            "baseline_transaction_count": int(profile["transaction_count"]),
            "baseline_transaction_amount_mean": _decimal(profile.get("transaction_amount_mean")),
            "baseline_transaction_amount_min": _decimal(profile.get("transaction_amount_min")),
            "baseline_transaction_amount_max": _decimal(profile.get("transaction_amount_max")),
            "baseline_successful_login_count": int(profile["successful_login_count"]),
            TARGET_COLUMN: int(trigger["ato_label"]),
        })
    return output


def generate_features(
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    baseline_path: str | Path = DEFAULT_BASELINE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> list[dict[str, Any]]:
    """Generate features CSV from the Phase 4 and Phase 5 CSV artifacts."""
    with Path(events_path).open("r", newline="", encoding="utf-8") as source:
        event_rows = list(csv.DictReader(source))
    from simulator.validation.dataset_validator import validate_dataset

    validation = validate_dataset(events_path)
    if not validation.is_valid:
        raise ValueError("event dataset validation failed: " + "; ".join(validation.errors))
    with Path(baseline_path).open("r", newline="", encoding="utf-8") as source:
        baseline_rows = list(csv.DictReader(source))
    rows = build_feature_rows(event_rows, baseline_rows)
    destination_path = Path(output_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    generated = generate_features()
    print(f"Generated {len(generated)} feature rows at {DEFAULT_OUTPUT_PATH}")
