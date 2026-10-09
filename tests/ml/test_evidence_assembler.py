from copy import deepcopy

import pytest

from ml.risk.evidence_assembler import assemble_evidence
from ml.rules.rule_baseline import AMBIGUOUS, ATO_RISK_INDICATOR, NO_INDICATOR


def rule(prediction=NO_INDICATOR):
    return {
        "user_id": "U1",
        "prediction": prediction,
        "matched_rules": ["M3_NO_CONFIGURED_MULTI_SIGNAL_MATCH"],
        "feature_evidence": {"new_device": 0},
        "explanation": "Example evidence.",
    }


def model(prediction=0):
    return {"prediction": prediction, "probability_ato": 0.2, "model_name": "phase8_rf"}


def test_assembles_component_outputs_without_fusion():
    result = assemble_evidence(
        "ASSESS-1", {"user_id": "U1", "account_id": "A1", "event_id": "E1"},
        rule_output=rule(), model_output=model(),
        temporal_evidence={"time_to_new_device_seconds": None},
    )
    assert result["evidence_status"] == "ASSEMBLED"
    assert result["rule_evidence"]["prediction"] == NO_INDICATOR
    assert result["model_evidence"]["probability_ato"] == 0.2
    assert result["temporal_evidence"]["time_to_new_device_seconds"] is None
    assert "score" not in result and "risk_level" not in result


def test_ambiguous_rule_outcome_is_preserved():
    evidence = rule(AMBIGUOUS)
    result = assemble_evidence("A1", {"user_id": "U1"}, rule_output=evidence, model_output=model(1))
    assert result["evidence_status"] == "AMBIGUOUS"
    assert result["rule_evidence"]["prediction"] == AMBIGUOUS


def test_missing_temporal_component_is_not_converted_to_negative_evidence():
    result = assemble_evidence("A1", {"user_id": "U1"}, rule_output=rule(), model_output=model())
    assert result["temporal_evidence"] is None
    assert result["evidence_status"] == "INCOMPLETE"


@pytest.mark.parametrize("missing", ["rule", "model", "temporal"])
def test_each_missing_component_is_explicitly_incomplete(missing):
    components = {
        "rule": rule(), "model": model(), "temporal": {"time_to_recovery_seconds": None},
    }
    components[missing] = None
    result = assemble_evidence(
        "A1", {"user_id": "U1"}, rule_output=components["rule"],
        model_output=components["model"], temporal_evidence=components["temporal"],
    )
    assert result[f"{missing}_evidence"] is None
    assert result["evidence_status"] == "INCOMPLETE"


def test_conflicting_rule_and_model_are_retained():
    result = assemble_evidence(
        "A1", {"user_id": "U1"}, rule_output=rule(ATO_RISK_INDICATOR),
        model_output=model(0), temporal_evidence={},
    )
    assert result["evidence_status"] == "CONFLICTING"
    assert result["rule_evidence"]["prediction"] == ATO_RISK_INDICATOR
    assert result["model_evidence"]["prediction"] == 0


def test_no_rule_indicator_is_not_treated_as_a_negative_prediction():
    result = assemble_evidence(
        "A1", {"user_id": "U1"}, rule_output=rule(NO_INDICATOR),
        model_output=model(1), temporal_evidence={},
    )
    assert result["evidence_status"] == "ASSEMBLED"


def test_ato_rule_indicator_against_model_positive_is_not_conflicting():
    result = assemble_evidence(
        "A1", {"user_id": "U1"}, rule_output=rule(ATO_RISK_INDICATOR),
        model_output=model(1), temporal_evidence={},
    )
    assert result["evidence_status"] == "ASSEMBLED"


@pytest.mark.parametrize("kwargs", [
    {"assessment_id": "", "identifiers": {"user_id": "U1"}},
    {"assessment_id": "A1", "identifiers": {}},
    {"assessment_id": "A1", "identifiers": {"user_id": "U1"}, "model_output": {"prediction": 2}},
    {"assessment_id": "A1", "identifiers": {"user_id": "U1"}, "rule_output": {"prediction": "LEGITIMATE"}},
    {"assessment_id": "A1", "identifiers": {"user_id": "U1"}, "temporal_evidence": {"x": float("nan")}},
    {"assessment_id": "A1", "identifiers": {"user_id": "U1"}, "temporal_evidence": {"x": "missing"}},
    {"assessment_id": "A1", "identifiers": {"user_id": "U1"}, "temporal_evidence": {"x": True}},
    {"assessment_id": "A1", "identifiers": {"user_id": "U1"}, "temporal_evidence": {"x": [1]}},
    {"assessment_id": "A1", "identifiers": {"user_id": "U1"}, "temporal_evidence": {"x": float("inf")}},
    {"assessment_id": "A1", "identifiers": {"user_id": "U1"}, "model_output": {"prediction": 1.0}},
    {"assessment_id": "A1", "identifiers": {"user_id": ""}},
])
def test_malformed_inputs_raise_instead_of_becoming_negative_evidence(kwargs):
    with pytest.raises(ValueError):
        assemble_evidence(**kwargs)


def test_repeated_calls_are_deterministic_and_do_not_mutate_inputs():
    supplied_rule, supplied_model = rule(), model()
    supplied_ids = {"user_id": "U1"}
    supplied_temporal = {"time_to_recovery_seconds": None}
    supplied_provenance = {"component_versions": {"rule": "phase7"}}
    original_inputs = deepcopy((supplied_rule, supplied_model, supplied_ids, supplied_temporal, supplied_provenance))
    args = dict(assessment_id="A1", identifiers=supplied_ids, rule_output=supplied_rule,
                model_output=supplied_model, temporal_evidence=supplied_temporal,
                provenance=supplied_provenance)
    assert assemble_evidence(**args) == assemble_evidence(**args)
    assert (supplied_rule, supplied_model, supplied_ids, supplied_temporal, supplied_provenance) == original_inputs


def test_component_provenance_is_preserved():
    result = assemble_evidence(
        "A1", {"user_id": "U1"}, rule_output=rule(), model_output=model(),
        temporal_evidence={},
        provenance={"rule_version": "phase7", "model_version": "phase8", "temporal_version": "phase9"},
    )
    assert result["provenance"] == {
        "rule_version": "phase7", "model_version": "phase8", "temporal_version": "phase9"
    }
