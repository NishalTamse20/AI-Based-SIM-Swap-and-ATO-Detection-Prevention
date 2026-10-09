import json

import numpy as np
import pandas as pd
import pytest

from ml.models.hybrid_policy_evaluation import (
    EXPECTED_REPLACEMENT_ROWS,
    _hard_metrics,
    _policy_details,
    _require_matching_keys,
    _validate_fold_records,
    _validate_label_fold_alignment,
    apply_policy_1,
    apply_policy_2,
    run_hybrid_policy_evaluation,
)
from ml.rules.rule_baseline import AMBIGUOUS, ATO_RISK_INDICATOR, NO_INDICATOR
from ml.models.hybrid_policy_evaluation import DEFAULT_TEMPORAL_PATH


@pytest.mark.parametrize("rule,model,expected", [
    (ATO_RISK_INDICATOR, 1, "ATO_RISK_INDICATOR"),
    (ATO_RISK_INDICATOR, 0, "CONFLICTING_EVIDENCE"),
    (ATO_RISK_INDICATOR, None, "INSUFFICIENT_EVIDENCE"),
    (NO_INDICATOR, 1, "MODEL_ONLY_RISK_INDICATOR"),
    (NO_INDICATOR, 0, "NO_ATO_RISK_INDICATOR"),
    (NO_INDICATOR, None, "INSUFFICIENT_EVIDENCE"),
    (AMBIGUOUS, 1, "AMBIGUOUS_INSUFFICIENT_EVIDENCE"),
    (AMBIGUOUS, 0, "AMBIGUOUS_INSUFFICIENT_EVIDENCE"),
    (AMBIGUOUS, None, "AMBIGUOUS_INSUFFICIENT_EVIDENCE"),
    (None, 1, "INSUFFICIENT_EVIDENCE"),
    (None, 0, "INSUFFICIENT_EVIDENCE"),
    (None, None, "INSUFFICIENT_EVIDENCE"),
])
def test_policy_1_complete_decision_table(rule, model, expected):
    assert apply_policy_1(rule, model) == expected


@pytest.mark.parametrize("rule,model,expected", [
    (ATO_RISK_INDICATOR, 1, "ATO_RISK_INDICATOR"),
    (ATO_RISK_INDICATOR, 0, "CONFLICTING_EVIDENCE"),
    (ATO_RISK_INDICATOR, None, "INSUFFICIENT_EVIDENCE"),
    (NO_INDICATOR, 1, "AMBIGUOUS_INSUFFICIENT_EVIDENCE"),
    (NO_INDICATOR, 0, "NO_ATO_RISK_INDICATOR"),
    (NO_INDICATOR, None, "INSUFFICIENT_EVIDENCE"),
    (AMBIGUOUS, 1, "AMBIGUOUS_INSUFFICIENT_EVIDENCE"),
    (AMBIGUOUS, 0, "AMBIGUOUS_INSUFFICIENT_EVIDENCE"),
    (AMBIGUOUS, None, "AMBIGUOUS_INSUFFICIENT_EVIDENCE"),
    (None, 1, "INSUFFICIENT_EVIDENCE"),
    (None, 0, "INSUFFICIENT_EVIDENCE"),
    (None, None, "INSUFFICIENT_EVIDENCE"),
])
def test_policy_2_complete_decision_table(rule, model, expected):
    assert apply_policy_2(rule, model) == expected


def test_no_rule_indicator_is_not_legitimate_and_policy_coverage_is_explicit():
    assert apply_policy_1(NO_INDICATOR, 0) == "NO_ATO_RISK_INDICATOR"
    rows = pd.DataFrame({
        "actual_label_evaluation_only": [0, 1, 1, 0],
        "policy": ["NO_ATO_RISK_INDICATOR", "AMBIGUOUS_INSUFFICIENT_EVIDENCE",
                   "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE"],
    })
    report = _policy_details(rows, "policy")
    assert report["determinate_count"] == 1
    assert report["coverage"] == 0.25
    assert report["uncertain_counts"]["AMBIGUOUS_INSUFFICIENT_EVIDENCE"]["actual_label_counts"]["1"] == 1


def test_exact_key_join_rejects_duplicate_and_mismatched_rows():
    good = pd.DataFrame({"user_id": ["u1"], "account_id": ["a1"], "event_id": ["e1"]})
    duplicate = pd.concat([good, good], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        _require_matching_keys({"good": good, "duplicate": duplicate})
    other = pd.DataFrame({"user_id": ["u2"], "account_id": ["a2"], "event_id": ["e2"]})
    with pytest.raises(ValueError, match="key-set mismatch"):
        _require_matching_keys({"good": good, "other": other})


def test_metric_helper_uses_fixed_binary_confusion_matrix_and_null_auc():
    report = _hard_metrics(np.array([0, 1]), np.array([0, 1]))
    assert report["confusion_matrix"] == [[1, 0], [0, 1]]
    assert report["roc_auc"] is None and report["average_precision"] is None


def test_label_and_held_out_fold_alignment_are_checked_after_prediction():
    evaluation = pd.DataFrame({
        "user_id": ["u1"], "account_id": ["a1"], "event_id": ["e1"],
        "ato_label": [1], "ato_label_evaluation_only": [1], "actual_label_evaluation_only": [1],
        "scenario_type_evaluation_only": ["SUSPICIOUS_X"],
        "held_out_scenario_type": ["SUSPICIOUS_X"],
    })
    reference = pd.DataFrame({
        "user_id": ["u1"], "account_id": ["a1"], "event_id": ["e1"],
        "actual_label": [1], "scenario_type": ["SUSPICIOUS_X"],
        "held_out_scenario_type": ["SUSPICIOUS_X"],
    })
    _validate_label_fold_alignment(evaluation, {"reference": reference})
    wrong_label = reference.assign(actual_label=0)
    with pytest.raises(ValueError, match="evaluation label"):
        _validate_label_fold_alignment(evaluation, {"reference": wrong_label})
    wrong_fold = reference.assign(held_out_scenario_type="LEGITIMATE_X")
    with pytest.raises(ValueError, match="row-to-fold"):
        _validate_label_fold_alignment(evaluation, {"reference": wrong_fold})


def test_fold_records_require_each_scenario_to_be_held_out_once():
    scenarios = [f"SCENARIO_{index}" for index in range(8)]
    evaluation = pd.DataFrame({
        "scenario_type_evaluation_only": scenarios,
        "held_out_scenario_type": scenarios,
    })
    folds = [{
        "held_out_scenario_type": held,
        "test_rows": 1,
        "train_scenario_types": sorted(set(scenarios) - {held}),
        "test_scenario_types": [held],
    } for held in scenarios]
    _validate_fold_records(evaluation, folds)
    folds[0]["test_scenario_types"] = [scenarios[1]]
    with pytest.raises(ValueError, match="test scenario assignment"):
        _validate_fold_records(evaluation, folds)


def test_saved_fold_evaluation_exports_all_rows_deterministically(tmp_path):
    first = run_hybrid_policy_evaluation(output_dir=tmp_path / "one")
    second = run_hybrid_policy_evaluation(output_dir=tmp_path / "two")
    assert len(first["predictions"]) == 1000
    assert len(first["replacement_cases"]) == EXPECTED_REPLACEMENT_ROWS
    assert (first["replacement_cases"]["phase9_prediction"].astype(int) == 0).all()
    for filename in ("metrics.json", "predictions.csv", "scenario_error_summary.csv",
                     "suspicious_sim_replacement_cases.csv", "run_metadata.json"):
        assert (tmp_path / "one" / filename).read_bytes() == (tmp_path / "two" / filename).read_bytes()
    assert len(pd.read_csv(tmp_path / "one" / "suspicious_sim_replacement_cases.csv")) == 125
    source_temporal = pd.read_csv(DEFAULT_TEMPORAL_PATH)
    exported = pd.read_csv(tmp_path / "one" / "predictions.csv")
    null_fields = ["time_to_new_device_seconds", "time_to_failed_login_seconds",
                   "time_to_password_reset_seconds", "time_to_recovery_seconds",
                   "time_to_auth_anomaly_seconds"]
    assert exported[null_fields].isna().sum().to_dict() == source_temporal[null_fields].isna().sum().to_dict()
    metrics = json.loads((tmp_path / "one" / "metrics.json").read_text())
    for name in ("POLICY_1_RF_LED", "POLICY_2_RULE_CORROBORATION"):
        policy = metrics["policies"][name]
        assert policy["determinate_count"] + sum(
            record["count"] for record in policy["uncertain_counts"].values()
        ) == 1000
        assert policy["reference_model_metrics_on_same_determinate_subset"]["model_a_phase8"]["n"] == policy["determinate_count"]
        assert policy["reference_model_metrics_on_same_determinate_subset"]["model_b_phase9"]["n"] == policy["determinate_count"]
