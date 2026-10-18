from copy import deepcopy

import pytest

from ml.risk.evidence_assembler import assemble_evidence
from ml.risk.explainable_response import build_explainable_response
from ml.rules.rule_baseline import AMBIGUOUS, ATO_RISK_INDICATOR, NO_INDICATOR


def _rule(prediction):
    return {
        "prediction": prediction,
        "matched_rules": ["M3_NO_CONFIGURED_MULTI_SIGNAL_MATCH"],
        "feature_evidence": {"new_device": 0, "authentication_anomaly": 0},
        "explanation": "Existing Phase 7 rule explanation.",
    }


def _assessment(rule_prediction=NO_INDICATOR, model_prediction=0, temporal=None):
    return assemble_evidence(
        "ASSESS-1",
        {"user_id": "U1", "account_id": "A1", "event_id": "E1"},
        rule_output=_rule(rule_prediction),
        model_output={"prediction": model_prediction, "model_name": "phase9_model_b"},
        temporal_evidence={"time_to_new_device_seconds": None} if temporal is None else temporal,
        provenance={"rule_version": "phase7", "model_version": "phase9_model_b"},
    )


def test_no_indicator_does_not_become_legitimate_or_normal_access():
    result = build_explainable_response(_assessment(NO_INDICATOR, 0))

    assert result["outcome"] == NO_INDICATOR
    assert result["recommended_action"] is None
    assert "NO_CONFIGURED_RISK_INDICATOR" in result["reason_codes"]
    assert "does not establish legitimate activity" in result["explanation"]


def test_corroborated_suspicious_evidence_recommends_simulated_review():
    result = build_explainable_response(_assessment(ATO_RISK_INDICATOR, 1))

    assert result["outcome"] == "CORROBORATED_RISK_INDICATOR"
    assert result["recommended_action"] == "Protect/Security Review"
    assert "RULE_AND_MODEL_RISK_INDICATORS_AGREE" in result["reason_codes"]
    assert "not confirmed ATO" in result["explanation"]


def test_model_only_indicator_recommends_simulated_step_up():
    result = build_explainable_response(_assessment(NO_INDICATOR, 1))

    assert result["outcome"] == "MODEL_ONLY_RISK_INDICATOR"
    assert result["recommended_action"] == "Step-Up Authentication"
    assert result["component_evidence"]["rule_evidence"]["prediction"] == NO_INDICATOR
    assert result["component_evidence"]["model_evidence"]["prediction"] == 1


def test_ambiguous_rule_stays_ambiguous_without_response_recommendation():
    result = build_explainable_response(_assessment(AMBIGUOUS, 1))

    assert result["outcome"] == "AMBIGUOUS"
    assert result["recommended_action"] is None
    assert "RULE_EVIDENCE_AMBIGUOUS" in result["reason_codes"]


def test_missing_component_and_missing_temporal_values_are_preserved():
    assessment = assemble_evidence(
        "ASSESS-1", {"user_id": "U1"}, rule_output=_rule(ATO_RISK_INDICATOR),
        model_output={"prediction": 1}, temporal_evidence=None,
    )
    result = build_explainable_response(assessment)

    assert result["outcome"] == "INSUFFICIENT_EVIDENCE"
    assert result["recommended_action"] is None
    assert result["component_evidence"]["temporal_evidence"] is None
    assert "TEMPORAL_COMPONENT_MISSING" in result["reason_codes"]

    partial = build_explainable_response(_assessment(
        ATO_RISK_INDICATOR, 1, {"time_to_recovery_seconds": None}
    ))
    assert partial["component_evidence"]["temporal_evidence"]["time_to_recovery_seconds"] is None
    assert "TEMPORAL_VALUE_MISSING_TIME_TO_RECOVERY_SECONDS" in partial["reason_codes"]


def test_conflicting_rule_and_model_evidence_gets_no_action():
    result = build_explainable_response(_assessment(ATO_RISK_INDICATOR, 0))

    assert result["outcome"] == "CONFLICTING_EVIDENCE"
    assert result["recommended_action"] is None
    assert result["component_evidence"]["rule_evidence"]["prediction"] == ATO_RISK_INDICATOR
    assert result["component_evidence"]["model_evidence"]["prediction"] == 0
    assert "RULE_MODEL_CONFLICT" in result["reason_codes"]


def test_response_is_deterministic_and_does_not_mutate_evidence():
    assessment = _assessment(ATO_RISK_INDICATOR, 1)
    before = deepcopy(assessment)

    assert build_explainable_response(assessment) == build_explainable_response(assessment)
    assert assessment == before


def test_unvalidated_score_and_risk_level_are_never_fabricated():
    result = build_explainable_response(_assessment(ATO_RISK_INDICATOR, 1))

    assert result["risk_score_0_100"] is None
    assert result["risk_level"] == "UNDETERMINED"


def test_malformed_or_inconsistent_assembled_record_is_rejected():
    with pytest.raises(ValueError, match="missing a required assembler field"):
        build_explainable_response({"assessment_id": "A1"})

    assessment = _assessment()
    assessment["evidence_status"] = "CONFLICTING"
    with pytest.raises(ValueError, match="does not match"):
        build_explainable_response(assessment)
