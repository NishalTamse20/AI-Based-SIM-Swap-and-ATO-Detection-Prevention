"""Validate Phase 4 event data and its pre-trigger history."""

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from simulator.generators.event_generator import EVENT_FIELDS
from simulator.scenarios.scenario_runner import SCENARIOS


CATEGORY_EVENT_TYPES = {
    "TELECOM": {"SIM_SWAP", "SIM_REPLACEMENT", "ESIM_CHANGE"},
    "DEVICE": {"NEW_DEVICE", "DEVICE_CHANGE", "DEVICE_FINGERPRINT_CHANGE"},
    "AUTHENTICATION": {"LOGIN", "FAILED_LOGIN", "MFA_FAILURE", "MFA_SUCCESS", "PASSWORD_CHANGE"},
    "RECOVERY": {"PASSWORD_RESET", "ACCOUNT_RECOVERY", "RECOVERY_CONTACT_CHANGE"},
    "ACCOUNT": {"NEW_BENEFICIARY", "BENEFICIARY_CHANGE", "PROFILE_CHANGE"},
    "TRANSACTION": {"TRANSACTION", "TRANSFER", "WITHDRAWAL"},
    "BEHAVIOUR": {"LOGIN_TIME_DEVIATION", "LOCATION_DEVIATION", "DEVICE_DEVIATION", "TRANSACTION_PATTERN_DEVIATION"},
}
PAYSIM_TRANSACTION_TYPES = {"PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}
SCENARIO_LABELS = {name: label for name, label, _, _ in SCENARIOS}
SCENARIO_TELECOM_TYPES = {
    "LEGITIMATE_SIM_REPLACEMENT": {"SIM_REPLACEMENT"},
    "LEGITIMATE_ESIM_CHANGE": {"ESIM_CHANGE"},
    "SUSPICIOUS_SIM_REPLACEMENT": {"SIM_REPLACEMENT"},
    "SUSPICIOUS_ESIM_CHANGE": {"ESIM_CHANGE"},
    "SIM_ESIM_AUTH_ANOMALY": {"SIM_SWAP", "ESIM_CHANGE"},
}
REQUIRED_ROW_VALUES = (
    "event_id", "user_id", "account_id", "event_category", "event_type",
    "timestamp", "device_id", "status", "source",
)


@dataclass
class ValidationResult:
    """Summary of checks run against a generated event CSV."""

    event_count: int = 0
    user_count: int = 0
    scenario_count: int = 0
    history_event_count: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _timestamp(value: str, row_number: int, errors: list[str]) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        errors.append(f"row {row_number}: invalid timestamp {value!r}")
        return None


def validate_dataset(path: str | Path) -> ValidationResult:
    """Check the event schema, scenario labels, ordering, and trigger cutoff."""
    result = ValidationResult()
    errors = result.errors
    by_user: dict[str, list[tuple[int, dict[str, str], datetime]]] = defaultdict(list)
    seen_event_ids: set[str] = set()

    with Path(path).open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or ())
        missing_columns = set(EVENT_FIELDS) - columns
        if missing_columns:
            errors.append(f"missing required columns: {', '.join(sorted(missing_columns))}")
        if not reader.fieldnames:
            return result

        for row_number, row in enumerate(reader, start=2):
            result.event_count += 1
            missing = [field for field in REQUIRED_ROW_VALUES if not (row.get(field) or "").strip()]
            if missing:
                errors.append(f"row {row_number}: missing required values: {', '.join(missing)}")

            event_id = row.get("event_id", "")
            if event_id:
                if event_id in seen_event_ids:
                    errors.append(f"row {row_number}: duplicate event_id {event_id!r}")
                seen_event_ids.add(event_id)

            category, event_type = row.get("event_category", ""), row.get("event_type", "")
            if category not in CATEGORY_EVENT_TYPES:
                errors.append(f"row {row_number}: invalid event_category {category!r}")
            elif event_type not in CATEGORY_EVENT_TYPES[category]:
                errors.append(f"row {row_number}: invalid event_type {event_type!r} for {category}")

            timestamp = _timestamp(row.get("timestamp", ""), row_number, errors)
            if category == "TRANSACTION":
                try:
                    amount = Decimal(row.get("amount", ""))
                    if not amount.is_finite():
                        raise InvalidOperation
                except (InvalidOperation, ValueError):
                    errors.append(f"row {row_number}: transaction amount is missing or invalid")
                transaction_type = row.get("transaction_type", "")
                if transaction_type not in PAYSIM_TRANSACTION_TYPES:
                    errors.append(f"row {row_number}: invalid transaction_type {transaction_type!r}")

            user_id = row.get("user_id", "")
            if user_id and timestamp is not None:
                by_user[user_id].append((row_number, row, timestamp))

    if result.event_count == 0:
        errors.append("dataset has no event rows")
    result.user_count = len(by_user)
    for user_id, entries in by_user.items():
        times = [timestamp for _, _, timestamp in entries]
        if times != sorted(times):
            errors.append(f"user {user_id}: events are not chronologically ordered")

        account_ids = {row.get("account_id", "") for _, row, _ in entries}
        if len(account_ids) != 1:
            errors.append(f"user {user_id}: account_id is inconsistent")

        telecom = [(row, timestamp) for _, row, timestamp in entries if row.get("event_category") == "TELECOM"]
        if len(telecom) != 1:
            errors.append(f"user {user_id}: expected exactly one SIM/eSIM trigger, found {len(telecom)}")
            continue
        trigger, trigger_time = telecom[0]
        scenario_type = trigger.get("scenario_type", "")
        label = trigger.get("ato_label", "")
        result.scenario_count += 1
        if scenario_type not in SCENARIO_LABELS:
            errors.append(f"user {user_id}: invalid scenario_type {scenario_type!r}")
        elif label != str(SCENARIO_LABELS[scenario_type]):
            errors.append(f"user {user_id}: ato_label does not match scenario_type")
        if trigger.get("event_type") not in SCENARIO_TELECOM_TYPES.get(scenario_type, set()):
            errors.append(f"user {user_id}: telecom event does not match scenario_type")

        history = [(row, timestamp) for _, row, timestamp in entries if timestamp < trigger_time]
        post_trigger = [(row, timestamp) for _, row, timestamp in entries if timestamp >= trigger_time]
        result.history_event_count += len(history)
        if not history:
            errors.append(f"user {user_id}: no pre-trigger history")
        if any((row.get("scenario_type") or "") or (row.get("ato_label") or "") for row, _ in history):
            errors.append(f"user {user_id}: pre-trigger history contains scenario/ATO labels")
        if any(not (row.get("scenario_type") or "") or not (row.get("ato_label") or "") for row, _ in post_trigger):
            errors.append(f"user {user_id}: trigger/post-trigger row is missing scenario/ATO labels")
        for row, _ in post_trigger:
            row_scenario = row.get("scenario_type", "")
            row_label = row.get("ato_label", "")
            if row_scenario not in SCENARIO_LABELS:
                errors.append(f"user {user_id}: invalid scenario_type {row_scenario!r}")
            elif row_label != str(SCENARIO_LABELS[row_scenario]):
                errors.append(f"user {user_id}: post-trigger ato_label does not match scenario_type")

        login_count = sum(
            row.get("event_category") == "AUTHENTICATION" and row.get("event_type") == "LOGIN"
            and row.get("status") == "SUCCESS" for row, _ in history
        )
        transaction_count = sum(
            row.get("event_category") == "TRANSACTION" and row.get("event_type") == "TRANSACTION"
            and row.get("status") == "SUCCESS" for row, _ in history
        )
        if login_count != 5 or transaction_count != 5:
            errors.append(f"user {user_id}: expected five successful historical logins and transactions")
        trigger_device = trigger.get("device_id")
        if any(row.get("device_id") != trigger_device for row, _ in history):
            errors.append(f"user {user_id}: pre-trigger activity does not use the trigger's known device")

    return result
