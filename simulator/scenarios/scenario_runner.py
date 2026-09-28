"""Expand each documented scenario into a chronological event sequence."""

from datetime import datetime, timedelta
from typing import Any

from simulator.generators.device_generator import known_device_id, new_device_id
from simulator.generators.event_generator import make_event
from simulator.generators.user_generator import generate_identity
from simulator.scenarios.legitimate import scenario_definitions as legitimate_scenarios
from simulator.scenarios.suspicious import scenario_definitions as suspicious_scenarios


SCENARIOS = legitimate_scenarios() + suspicious_scenarios()
SCENARIO_TYPES = tuple(definition[0] for definition in SCENARIOS)
TRANSACTION_SCENARIOS = {"LEGITIMATE_SIM_REPLACEMENT", "SUSPICIOUS_SIM_REPLACEMENT", "SUSPICIOUS_ESIM_CHANGE"}


def _append(events: list[dict[str, Any]], scenario_index: int, identity: dict[str, str],
            scenario_type: str | None, label: int | None, category: str, event_type: str,
            minutes: int, device_id: str, timestamp: datetime | None = None, **kwargs: Any) -> None:
    events.append(make_event(
        event_id=f"EVT-{scenario_index:06d}-{len(events) + 1:02d}",
        user_id=identity["user_id"], account_id=identity["account_id"],
        event_category=category, event_type=event_type,
        timestamp=timestamp or datetime(2026, 1, 1) + timedelta(days=scenario_index, minutes=minutes),
        device_id=device_id, scenario_type=scenario_type, ato_label=label, **kwargs,
    ))


def run_scenario(scenario_index: int, scenario_type: str,
                 transaction: dict[str, str] | None = None,
                 historical_transactions: list[dict[str, str]] | None = None,
                 history_length: int = 5, history_window_days: int = 30) -> list[dict[str, Any]]:
    """Return one scenario's events, with scenario label defined by its template."""
    definitions = {name: (label, category, telecom_type) for name, label, category, telecom_type in SCENARIOS}
    if scenario_type not in definitions:
        raise ValueError(f"Unsupported scenario_type: {scenario_type}")
    if historical_transactions is None or len(historical_transactions) != history_length:
        raise ValueError("one historical PaySim transaction is required per history_length")
    label, _, telecom_type = definitions[scenario_type]
    identity = generate_identity(scenario_index)
    known = known_device_id(scenario_index)
    changed = new_device_id(scenario_index)
    is_legitimate = label == 0
    events: list[dict[str, Any]] = []
    trigger_time = datetime(2026, 1, 1) + timedelta(days=scenario_index)

    # History is identical in structure for every scenario and carries no future ground-truth label.
    history_event_count = history_length * 2
    for history_index in range(history_event_count):
        fraction = history_index / history_event_count
        history_time = trigger_time - timedelta(days=history_window_days * (1 - fraction))
        is_login = history_index % 2 == 0
        _append(
            events, scenario_index, identity, None, None,
            "AUTHENTICATION" if is_login else "TRANSACTION",
            "LOGIN" if is_login else "TRANSACTION", 0, known,
            timestamp=history_time,
            transaction=None if is_login else historical_transactions[history_index // 2],
        )

    _append(events, scenario_index, identity, scenario_type, label, "TELECOM", telecom_type, 0, known)

    if is_legitimate:
        _append(events, scenario_index, identity, scenario_type, label, "AUTHENTICATION", "LOGIN", 10, known)
        if scenario_type == "LEGITIMATE_SIM_REPLACEMENT":
            _append(events, scenario_index, identity, scenario_type, label, "TRANSACTION", "TRANSACTION", 30, known, transaction=transaction)
        return events

    _append(events, scenario_index, identity, scenario_type, label, "DEVICE", "NEW_DEVICE", 5, changed, status="UNRECOGNIZED")
    if scenario_type == "SUSPICIOUS_SIM_REPLACEMENT":
        _append(events, scenario_index, identity, scenario_type, label, "AUTHENTICATION", "FAILED_LOGIN", 10, changed, status="FAILURE")
        _append(events, scenario_index, identity, scenario_type, label, "RECOVERY", "PASSWORD_RESET", 15, changed)
        _append(events, scenario_index, identity, scenario_type, label, "ACCOUNT", "NEW_BENEFICIARY", 20, changed,
                beneficiary_id=f"BEN-{scenario_index:06d}-01")
        _append(events, scenario_index, identity, scenario_type, label, "TRANSACTION", "TRANSACTION", 30, changed, transaction=transaction)
    elif scenario_type == "SUSPICIOUS_ESIM_CHANGE":
        _append(events, scenario_index, identity, scenario_type, label, "DEVICE", "DEVICE_CHANGE", 8, changed, status="DEVIATION")
        _append(events, scenario_index, identity, scenario_type, label, "AUTHENTICATION", "FAILED_LOGIN", 10, changed, status="FAILURE")
        _append(events, scenario_index, identity, scenario_type, label, "BEHAVIOUR", "DEVICE_DEVIATION", 15, changed, status="DEVIATION")
        _append(events, scenario_index, identity, scenario_type, label, "TRANSACTION", "TRANSACTION", 30, changed, transaction=transaction)
    else:
        _append(events, scenario_index, identity, scenario_type, label, "AUTHENTICATION", "FAILED_LOGIN", 10, changed, status="FAILURE")
        _append(events, scenario_index, identity, scenario_type, label, "AUTHENTICATION", "FAILED_LOGIN", 12, changed, status="FAILURE")
        _append(events, scenario_index, identity, scenario_type, label, "BEHAVIOUR", "DEVICE_DEVIATION", 15, changed, status="DEVIATION")
    return events
