"""Build descriptive profiles using only pre-trigger event information."""

import csv
import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_PATH = PROJECT_ROOT / "data" / "synthetic" / "behavioural_baseline_profiles.csv"
BASELINE_FIELDS = (
    "user_id", "account_id", "trigger_event_id", "baseline_start", "trigger_cutoff",
    "baseline_observation_days", "baseline_event_count", "event_frequency_per_day",
    "known_device_ids", "successful_login_count", "transaction_count",
    "transaction_amount_sum", "transaction_amount_mean", "transaction_amount_min",
    "transaction_amount_max",
)


def _parse_time(event: dict[str, str]) -> datetime:
    return datetime.fromisoformat(event["timestamp"])


def build_baseline_profiles(events: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build one profile per user/account from rows strictly before its trigger.

    Ground-truth labels and PaySim fraud flags are never read by this function.
    """
    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        by_user[event["user_id"]].append(event)

    profiles: list[dict[str, Any]] = []
    for user_id in sorted(by_user):
        user_events = by_user[user_id]
        accounts = {event["account_id"] for event in user_events}
        if len(accounts) != 1:
            raise ValueError(f"user {user_id} has inconsistent account_id values")
        triggers = [event for event in user_events if event["event_category"] == "TELECOM"]
        if len(triggers) != 1:
            raise ValueError(f"user {user_id} must have exactly one SIM/eSIM trigger")
        trigger = triggers[0]
        trigger_time = _parse_time(trigger)

        # The strict timestamp filter is the only route into any profile measure.
        history = [event for event in user_events if _parse_time(event) < trigger_time]
        if not history:
            raise ValueError(f"user {user_id} has no pre-trigger history")
        history.sort(key=_parse_time)

        login_count = sum(
            event["event_category"] == "AUTHENTICATION"
            and event["event_type"] == "LOGIN"
            and event["status"] == "SUCCESS"
            for event in history
        )
        transactions = [
            Decimal(event["amount"])
            for event in history
            if event["event_category"] == "TRANSACTION"
        ]
        devices = sorted({event["device_id"] for event in history if event.get("device_id")})
        start_time = _parse_time(history[0])
        observed_days = Decimal(str((trigger_time - start_time).total_seconds())) / Decimal(86400)
        event_count = len(history)
        frequency = Decimal(event_count) / observed_days if observed_days else Decimal(0)

        transaction_sum = sum(transactions, Decimal(0))
        transaction_mean = transaction_sum / len(transactions) if transactions else None
        profiles.append({
            "user_id": user_id,
            "account_id": next(iter(accounts)),
            "trigger_event_id": trigger["event_id"],
            "baseline_start": start_time.isoformat(timespec="seconds"),
            "trigger_cutoff": trigger_time.isoformat(timespec="seconds"),
            "baseline_observation_days": observed_days,
            "baseline_event_count": event_count,
            "event_frequency_per_day": frequency,
            "known_device_ids": json.dumps(devices, separators=(",", ":")),
            "successful_login_count": login_count,
            "transaction_count": len(transactions),
            "transaction_amount_sum": transaction_sum,
            "transaction_amount_mean": transaction_mean,
            "transaction_amount_min": min(transactions) if transactions else None,
            "transaction_amount_max": max(transactions) if transactions else None,
        })
    return profiles


def validate_baseline_cutoffs(events: list[dict[str, str]],
                              profiles: list[dict[str, Any]]) -> list[str]:
    """Confirm every profile window ends at its trigger and uses only prior rows."""
    by_user: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in events:
        by_user[event["user_id"]].append(event)
    profile_by_user = {profile["user_id"]: profile for profile in profiles}
    errors: list[str] = []
    if set(by_user) != set(profile_by_user):
        errors.append("baseline profile users do not match dataset users")
    for user_id, user_events in by_user.items():
        profile = profile_by_user.get(user_id)
        if profile is None:
            continue
        triggers = [event for event in user_events if event["event_category"] == "TELECOM"]
        if len(triggers) != 1:
            errors.append(f"user {user_id}: cannot verify baseline cutoff without one trigger")
            continue
        trigger = triggers[0]
        cutoff = _parse_time(trigger)
        eligible = [event for event in user_events if _parse_time(event) < cutoff]
        if profile["baseline_event_count"] != len(eligible):
            errors.append(f"user {user_id}: baseline includes missing or post-trigger events")
        if not eligible or profile["baseline_start"] != min(eligible, key=_parse_time)["timestamp"]:
            errors.append(f"user {user_id}: baseline start does not match pre-trigger data")
        if profile["trigger_cutoff"] != trigger["timestamp"]:
            errors.append(f"user {user_id}: baseline cutoff does not match trigger timestamp")
    return errors


def build_profiles_from_csv(input_path: str | Path,
                            output_path: str | Path = DEFAULT_BASELINE_PATH) -> list[dict[str, Any]]:
    """Read events, build pre-trigger profiles, and write profile CSV."""
    with Path(input_path).open("r", newline="", encoding="utf-8") as source:
        events = list(csv.DictReader(source))
    profiles = build_baseline_profiles(events)
    cutoff_errors = validate_baseline_cutoffs(events, profiles)
    if cutoff_errors:
        raise ValueError("baseline cutoff validation failed: " + "; ".join(cutoff_errors))
    destination_path = Path(output_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=BASELINE_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(profiles)
    return profiles
