"""Simple, explainable Phase 7 baselines over Phase 6 features."""

from collections.abc import Iterable, Mapping
from typing import Any


NO_INDICATOR = "NO_ATO_RISK_INDICATOR"
ATO_RISK_INDICATOR = "ATO_RISK_INDICATOR"
AMBIGUOUS = "AMBIGUOUS_INSUFFICIENT_EVIDENCE"

TELECOM_INPUTS = (
    "telecom_event_present",
    "sim_change",
    "esim_change",
)
MULTI_SIGNAL_INPUTS = (
    "new_device",
    "authentication_anomaly",
    "password_reset",
    "failed_login_count",
    "device_change",
    "device_deviation",
)
IDENTIFIER_FIELDS = ("user_id", "account_id", "event_id")


def _integer(row: Mapping[str, Any], feature: str) -> int:
    """Read a required integer feature without consulting any other columns."""
    value = row[feature]
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature {feature!r} must contain an integer, got {value!r}") from exc
    if str(value).strip() not in {str(normalized), f"{normalized}.0"}:
        raise ValueError(f"feature {feature!r} must contain an integer, got {value!r}")
    return normalized


def _result(
    row: Mapping[str, Any],
    prediction: str,
    matched_rule: str,
    feature_evidence: dict[str, int],
    explanation: str,
) -> dict[str, Any]:
    return {
        **{field: row[field] for field in IDENTIFIER_FIELDS if field in row},
        "prediction": prediction,
        "matched_rules": [matched_rule],
        "feature_evidence": feature_evidence,
        "explanation": explanation,
    }


def telecom_context_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return T0/T1 context-only reference output for one feature row."""
    evidence = {feature: _integer(row, feature) for feature in TELECOM_INPUTS}
    trigger_observed = (
        evidence["telecom_event_present"] == 1
        and (evidence["sim_change"] == 1 or evidence["esim_change"] == 1)
    )
    if trigger_observed:
        return _result(
            row,
            NO_INDICATOR,
            "T0_TRIGGER_CONTEXT_ONLY",
            evidence,
            "A SIM/eSIM event is present as context; the trigger alone does not indicate ATO.",
        )
    return _result(
        row,
        NO_INDICATOR,
        "T1_NO_TRIGGER_CONTEXT",
        evidence,
        "No SIM/eSIM trigger is represented by these features, so this context-only baseline raises no indicator.",
    )


def multi_signal_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return M1/M2/M3 multi-signal output for one feature row."""
    evidence = {feature: _integer(row, feature) for feature in MULTI_SIGNAL_INPUTS}
    new_device = evidence["new_device"] == 1
    auth_anomaly = evidence["authentication_anomaly"] == 1
    password_reset = evidence["password_reset"] == 1

    # M1 precedes M2 so overlapping recovery evidence remains explicitly ambiguous.
    if new_device and auth_anomaly and password_reset:
        return _result(
            row,
            AMBIGUOUS,
            "M1_AMBIGUOUS_DEVICE_AUTH_RECOVERY",
            evidence,
            "A new device, authentication anomaly, and password reset co-occur. This combination can also arise during authorized recovery and is insufficient to resolve ATO risk.",
        )

    if (
        new_device
        and auth_anomaly
        and not password_reset
        and (evidence["device_change"] == 1 or evidence["device_deviation"] == 1)
    ):
        return _result(
            row,
            ATO_RISK_INDICATOR,
            "M2_DEVICE_AUTH_CORROBORATION",
            evidence,
            "A new device and authentication anomaly co-occur with a device change or deviation; this is an ATO-risk indicator, not confirmed fraud.",
        )

    return _result(
        row,
        NO_INDICATOR,
        "M3_NO_CONFIGURED_MULTI_SIGNAL_MATCH",
        evidence,
        "Neither the ambiguous recovery combination nor the configured device/authentication corroboration matched; this does not establish legitimate activity.",
    )


def generate_rule_baselines(
    feature_rows: Iterable[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Generate both independent baseline outputs without reading target fields."""
    telecom_results: list[dict[str, Any]] = []
    multi_signal_results: list[dict[str, Any]] = []
    for row in feature_rows:
        telecom_results.append(telecom_context_prediction(row))
        multi_signal_results.append(multi_signal_prediction(row))
    return {
        "telecom_context": telecom_results,
        "multi_signal": multi_signal_results,
    }
