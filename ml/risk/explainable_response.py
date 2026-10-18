"""Explain supplied Phase 7–9 evidence and return conservative prototype guidance."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ml.risk.evidence_assembler import assemble_evidence
from ml.rules.rule_baseline import AMBIGUOUS, ATO_RISK_INDICATOR, NO_INDICATOR


def build_explainable_response(evidence_record: Mapping[str, Any]) -> dict[str, Any]:
    """Return reason codes and a simulated response without scoring or serving.

    Only a rule/model agreement or a model-only positive indicator receives a
    response recommendation. No-indicator, ambiguous, incomplete, and
    conflicting evidence do not imply legitimate activity or safety.
    """
    if not isinstance(evidence_record, Mapping):
        raise ValueError("evidence_record must be a mapping returned by assemble_evidence")

    try:
        validated = assemble_evidence(
            evidence_record["assessment_id"],
            evidence_record["identifiers"],
            rule_output=evidence_record["rule_evidence"],
            model_output=evidence_record["model_evidence"],
            temporal_evidence=evidence_record["temporal_evidence"],
            provenance=evidence_record["provenance"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("evidence_record is missing a required assembler field") from exc

    if evidence_record.get("evidence_status") != validated["evidence_status"]:
        raise ValueError("evidence_record.evidence_status does not match its component evidence")

    rule = validated["rule_evidence"]
    model = validated["model_evidence"]
    temporal = validated["temporal_evidence"]
    reason_codes: list[str] = []

    if rule is not None:
        reason_codes.append(f"RULE_OUTCOME_{rule['prediction']}")
    if model is not None:
        reason_codes.append("MODEL_PREDICTION_POSITIVE" if model["prediction"] == 1
                            else "MODEL_PREDICTION_NEGATIVE")
    if temporal is None:
        reason_codes.append("TEMPORAL_COMPONENT_MISSING")
    else:
        reason_codes.extend(
            f"TEMPORAL_VALUE_MISSING_{name.upper()}"
            for name, value in temporal.items()
            if value is None
        )

    outcome: str
    recommended_action: str | None
    if validated["evidence_status"] == "AMBIGUOUS":
        outcome = "AMBIGUOUS"
        recommended_action = None
        reason_codes.append("RULE_EVIDENCE_AMBIGUOUS")
        explanation = (
            "The rule evidence is ambiguous. This prototype does not resolve it "
            "as legitimate or suspicious and makes no response recommendation."
        )
    elif validated["evidence_status"] == "CONFLICTING":
        outcome = "CONFLICTING_EVIDENCE"
        recommended_action = None
        reason_codes.append("RULE_MODEL_CONFLICT")
        explanation = (
            "The rule risk indicator conflicts with the negative model prediction. "
            "Both are retained and no response recommendation is made."
        )
    elif validated["evidence_status"] == "INCOMPLETE":
        outcome = "INSUFFICIENT_EVIDENCE"
        recommended_action = None
        reason_codes.append("REQUIRED_COMPONENT_MISSING")
        explanation = (
            "Required component evidence is missing. Missing evidence is not "
            "treated as evidence of safety; no response recommendation is made."
        )
    elif rule["prediction"] == ATO_RISK_INDICATOR and model["prediction"] == 1:
        outcome = "CORROBORATED_RISK_INDICATOR"
        recommended_action = "Protect/Security Review"
        reason_codes.append("RULE_AND_MODEL_RISK_INDICATORS_AGREE")
        explanation = (
            "The Phase 7 rule and supplied model prediction both indicate risk. "
            "This is an experimental indicator, not confirmed ATO; the suggested "
            "security review is simulated only."
        )
    elif rule["prediction"] == NO_INDICATOR and model["prediction"] == 1:
        outcome = "MODEL_ONLY_RISK_INDICATOR"
        recommended_action = "Step-Up Authentication"
        reason_codes.append("MODEL_ONLY_RISK_INDICATOR")
        explanation = (
            "The model indicates risk while no configured Phase 7 risk rule "
            "matched. The step-up recommendation is simulated; the rule outcome "
            "does not establish legitimate activity."
        )
    elif rule["prediction"] == NO_INDICATOR and model["prediction"] == 0:
        outcome = NO_INDICATOR
        recommended_action = None
        reason_codes.append("NO_CONFIGURED_RISK_INDICATOR")
        explanation = (
            "No configured rule matched and the model prediction is negative. "
            "This does not establish legitimate activity, so no normal-access "
            "recommendation is inferred."
        )
    else:  # Defensive: validated assembler output should be covered above.
        raise ValueError("unsupported component combination in evidence_record")

    return {
        "assessment_id": validated["assessment_id"],
        "identifiers": validated["identifiers"],
        "outcome": outcome,
        "evidence_status": validated["evidence_status"],
        "reason_codes": reason_codes,
        "explanation": explanation,
        "risk_score_0_100": None,
        "risk_level": "UNDETERMINED",
        "recommended_action": recommended_action,
        "component_evidence": {
            "rule_evidence": validated["rule_evidence"],
            "model_evidence": validated["model_evidence"],
            "temporal_evidence": validated["temporal_evidence"],
        },
        "provenance": validated["provenance"],
    }
