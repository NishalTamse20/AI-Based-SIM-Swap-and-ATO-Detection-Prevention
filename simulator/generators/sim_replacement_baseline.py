"""Prepare isolated pre-trigger histories, profiles, and Phase 6 features."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ml.features.feature_generator import OUTPUT_COLUMNS as FEATURE_COLUMNS
from ml.features.feature_generator import build_feature_rows
from simulator.baseline.behavioural_baseline import (
    BASELINE_FIELDS,
    build_baseline_profiles,
    validate_baseline_cutoffs,
)
from simulator.config.generator_config import DEFAULT_PAYSIM_PATH
from simulator.generators.event_generator import EVENT_FIELDS, make_event
from simulator.generators.paysim_loader import sample_transactions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "controlled_sim_replacement"
HISTORY_LENGTH = 5
HISTORY_WINDOW_DAYS = 30


def build_historical_event_input(
    cohort_events: list[dict[str, Any]],
    paysim_path: str | Path = DEFAULT_PAYSIM_PATH,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """Create 5 LOGIN and 5 PaySim-backed TRANSACTION events per trigger.

    Only trigger identifiers, timestamp, and device are read from the cohort.
    Label, scenario, family metadata, and post-trigger rows do not contribute
    to any historical event or historical transaction assignment.
    """
    triggers_by_user: dict[str, dict[str, Any]] = {}
    for event in cohort_events:
        if event.get("event_category") != "TELECOM":
            continue
        user_id = str(event["user_id"])
        if user_id in triggers_by_user:
            raise ValueError(f"multiple telecom triggers for user {user_id}")
        if not event.get("account_id") or not event.get("event_id") or not event.get("device_id"):
            raise ValueError(f"trigger identifiers/device missing for user {user_id}")
        if event.get("event_type") != "SIM_REPLACEMENT":
            raise ValueError(f"unexpected trigger type for user {user_id}")
        triggers_by_user[user_id] = event
    if not triggers_by_user:
        raise ValueError("cohort contains no telecom triggers")

    transaction_rows = iter(sample_transactions(
        paysim_path, len(triggers_by_user) * HISTORY_LENGTH, random_seed
    ))
    history_events: list[dict[str, Any]] = []
    for case_index, user_id in enumerate(sorted(triggers_by_user), start=1):
        trigger = triggers_by_user[user_id]
        trigger_time = datetime.fromisoformat(str(trigger["timestamp"]))
        account_id = str(trigger["account_id"])
        known_device_id = str(trigger["device_id"])
        history_event_count = HISTORY_LENGTH * 2
        for event_index in range(history_event_count):
            fraction = event_index / history_event_count
            timestamp = trigger_time - timedelta(
                days=HISTORY_WINDOW_DAYS * (1 - fraction)
            )
            is_login = event_index % 2 == 0
            history_events.append(make_event(
                event_id=f"EVT-SIMREP-HIST-{case_index:04d}-{event_index + 1:02d}",
                user_id=user_id,
                account_id=account_id,
                event_category="AUTHENTICATION" if is_login else "TRANSACTION",
                event_type="LOGIN" if is_login else "TRANSACTION",
                timestamp=timestamp,
                device_id=known_device_id,
                scenario_type=None,
                ato_label=None,
                status="SUCCESS",
                transaction=None if is_login else next(transaction_rows),
            ))
    history_events.sort(key=lambda row: (row["user_id"], row["timestamp"], row["event_id"]))
    return history_events


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def prepare_controlled_sim_replacement(
    cohort: dict[str, Any],
    paysim_path: str | Path = DEFAULT_PAYSIM_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Write separate history/profile/feature artifacts for the controlled cohort."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort_events = cohort["events"]
    triggers = [row for row in cohort_events if row.get("event_category") == "TELECOM"]
    history_events = build_historical_event_input(cohort_events, paysim_path, random_seed)

    # The Phase 5 input contains history plus trigger rows only. Post-trigger
    # events and all case/family metadata are excluded from profile building.
    baseline_input = history_events + triggers
    profiles = build_baseline_profiles(baseline_input)
    cutoff_errors = validate_baseline_cutoffs(baseline_input, profiles)
    if cutoff_errors:
        raise ValueError("baseline cutoff validation failed: " + "; ".join(cutoff_errors))

    triggers_by_user = {str(row["user_id"]): row for row in triggers}
    profiles_by_user = {str(row["user_id"]): row for row in profiles}
    if set(triggers_by_user) != set(profiles_by_user):
        raise ValueError("profile users do not match controlled cohort trigger users")
    for user_id, trigger in triggers_by_user.items():
        profile = profiles_by_user[user_id]
        if profile["account_id"] != trigger["account_id"]:
            raise ValueError(f"profile account_id mismatch for user {user_id}")
        if profile["trigger_event_id"] != trigger["event_id"]:
            raise ValueError(f"profile trigger_event_id mismatch for user {user_id}")
        if profile["trigger_cutoff"] != trigger["timestamp"]:
            raise ValueError(f"profile trigger_cutoff mismatch for user {user_id}")

    # Use the reusable Phase 6 row builder directly. This cohort is not the
    # full Phase 4 dataset, so its general dataset validator is not invoked.
    feature_rows = build_feature_rows(cohort_events, profiles)
    _write_csv(output_dir / "historical_event_input.csv", history_events, EVENT_FIELDS)
    _write_csv(output_dir / "baseline_profiles.csv", profiles, BASELINE_FIELDS)
    _write_csv(output_dir / "features.csv", feature_rows, FEATURE_COLUMNS)
    return {
        "historical_events": history_events,
        "baseline_profiles": profiles,
        "features": feature_rows,
        "output_dir": output_dir,
    }

