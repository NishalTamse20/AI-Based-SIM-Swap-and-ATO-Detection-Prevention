"""Build common unified event records."""

from datetime import datetime
from typing import Any


EVENT_FIELDS = (
    "event_id", "user_id", "account_id", "event_category", "event_type", "timestamp",
    "device_id", "ip_address", "location", "channel", "status", "amount",
    "beneficiary_id", "source", "scenario_type", "ato_label", "transaction_type",
    "source_balance_before", "source_balance_after", "destination_balance_before",
    "destination_balance_after", "paysim_step", "paysim_nameOrig", "paysim_nameDest",
    "paysim_isFraud", "paysim_isFlaggedFraud",
)


def make_event(
    *, event_id: str, user_id: str, account_id: str, event_category: str, event_type: str,
    timestamp: datetime, device_id: str, scenario_type: str, ato_label: int,
    status: str = "SUCCESS", channel: str | None = None, location: str | None = None,
    amount: str | None = None, beneficiary_id: str | None = None,
    transaction: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a row with the common event schema and optional PaySim source data."""
    record: dict[str, Any] = {
        "event_id": event_id,
        "user_id": user_id,
        "account_id": account_id,
        "event_category": event_category,
        "event_type": event_type,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "device_id": device_id,
        "ip_address": None,
        "location": location,
        "channel": channel or event_category.lower(),
        "status": status,
        "amount": amount,
        "beneficiary_id": beneficiary_id,
        "source": "synthetic",
        "scenario_type": scenario_type,
        "ato_label": ato_label,
        "transaction_type": None,
        "source_balance_before": None,
        "source_balance_after": None,
        "destination_balance_before": None,
        "destination_balance_after": None,
        "paysim_step": None,
        "paysim_nameOrig": None,
        "paysim_nameDest": None,
        "paysim_isFraud": None,
        "paysim_isFlaggedFraud": None,
    }
    if transaction is not None:
        record.update({
            "source": "paysim",
            "amount": transaction["amount"],
            "transaction_type": transaction["type"],
            "source_balance_before": transaction["oldbalanceOrg"],
            "source_balance_after": transaction["newbalanceOrig"],
            "destination_balance_before": transaction["oldbalanceDest"],
            "destination_balance_after": transaction["newbalanceDest"],
            "paysim_step": transaction["step"],
            "paysim_nameOrig": transaction["nameOrig"],
            "paysim_nameDest": transaction["nameDest"],
            "paysim_isFraud": transaction["isFraud"],
            "paysim_isFlaggedFraud": transaction["isFlaggedFraud"],
        })
    return record
