"""Assemble supplied Phase 7–9 evidence without fusing it into a score."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from typing import Any

from ml.rules.rule_baseline import AMBIGUOUS, ATO_RISK_INDICATOR, NO_INDICATOR

_RULE_OUTCOMES = {AMBIGUOUS, ATO_RISK_INDICATOR, NO_INDICATOR}


def _mapping(value: Any, name: str, *, optional: bool = False) -> dict[str, Any] | None:
    if value is None and optional:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping" + (" or None" if optional else ""))
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"{name} keys must be non-empty strings")
    return deepcopy(dict(value))


def _validate_rule(value: Any) -> dict[str, Any] | None:
    rule = _mapping(value, "rule_output", optional=True)
    if rule is None:
        return None
    prediction = rule.get("prediction")
    if not isinstance(prediction, str) or prediction not in _RULE_OUTCOMES:
        raise ValueError("rule_output.prediction is not a supported Phase 7 outcome")
    rules = rule.get("matched_rules")
    if not isinstance(rules, list) or not rules or any(not isinstance(item, str) or not item for item in rules):
        raise ValueError("rule_output.matched_rules must be a non-empty list of non-empty strings")
    feature_evidence = rule.get("feature_evidence")
    if not isinstance(feature_evidence, Mapping):
        raise ValueError("rule_output.feature_evidence must be a mapping")
    if any(not isinstance(key, str) or not key for key in feature_evidence):
        raise ValueError("rule_output.feature_evidence keys must be non-empty strings")
    if not isinstance(rule.get("explanation"), str) or not rule["explanation"].strip():
        raise ValueError("rule_output.explanation must be a non-empty string")
    return rule


def _validate_model(value: Any) -> dict[str, Any] | None:
    model = _mapping(value, "model_output", optional=True)
    if model is None:
        return None
    prediction = model.get("prediction")
    if type(prediction) is not int or prediction not in (0, 1):
        raise ValueError("model_output.prediction must be 0 or 1")
    probability = model.get("probability_ato")
    if probability is not None:
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("model_output.probability_ato must be numeric or None")
        if not isfinite(float(probability)) or not 0 <= float(probability) <= 1:
            raise ValueError("model_output.probability_ato must be finite and between 0 and 1")
    return model


def _validate_temporal(value: Any) -> dict[str, Any] | None:
    temporal = _mapping(value, "temporal_evidence", optional=True)
    if temporal is None:
        return None
    for name, item in temporal.items():
        if item is None:
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"temporal_evidence.{name} must be numeric or None")
        if not isfinite(float(item)):
            raise ValueError(f"temporal_evidence.{name} must be finite or None")
    return temporal


def assemble_evidence(
    assessment_id: str,
    identifiers: Mapping[str, Any],
    *,
    rule_output: Mapping[str, Any] | None = None,
    model_output: Mapping[str, Any] | None = None,
    temporal_evidence: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic evidence record; no score or decision is inferred.

    ``model_output`` accepts explicitly supplied ``prediction`` and optional
    ``probability_ato`` values. A probability is retained as supplied and is
    not represented as calibrated. Missing components and temporal values
    remain missing.
    """
    if not isinstance(assessment_id, str) or not assessment_id.strip():
        raise ValueError("assessment_id must be a non-empty string")
    ids = _mapping(identifiers, "identifiers")
    if not ids:
        raise ValueError("identifiers must contain at least one traceability identifier")
    if any(not isinstance(value, str) or not value.strip() for value in ids.values()):
        raise ValueError("identifier values must be non-empty strings")
    rule = _validate_rule(rule_output)
    model = _validate_model(model_output)
    temporal = _validate_temporal(temporal_evidence)
    source_provenance = _mapping(provenance, "provenance", optional=True) or {}

    if rule is not None and rule["prediction"] == AMBIGUOUS:
        status = "AMBIGUOUS"
    elif rule is not None and model is not None:
        # Phase 7 NO_ATO_RISK_INDICATOR means no configured rule matched;
        # it is not a legitimate/negative classification.
        if rule["prediction"] == ATO_RISK_INDICATOR and model["prediction"] == 0:
            status = "CONFLICTING"
        elif temporal is None:
            status = "INCOMPLETE"
        else:
            status = "ASSEMBLED"
    elif rule is None or model is None or temporal is None:
        status = "INCOMPLETE"
    else:
        status = "ASSEMBLED"

    return {
        "assessment_id": assessment_id,
        "identifiers": ids,
        "evidence_status": status,
        "rule_evidence": rule,
        "model_evidence": model,
        "temporal_evidence": temporal,
        "provenance": source_provenance,
    }
