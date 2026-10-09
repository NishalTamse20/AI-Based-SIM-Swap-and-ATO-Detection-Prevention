import csv

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from ml.features.temporal_features import FEATURE_COLUMNS as ALL_TEMPORAL_FEATURES
from ml.models.random_forest_experiment import (
    DEFAULT_FEATURES_PATH,
    DEFAULT_OUTPUT_DIR as PHASE8_OUTPUT_DIR,
    FEATURE_COLUMNS as PHASE8_FEATURES,
    IDENTIFIER_COLUMNS,
    MODEL_PARAMETERS,
    TARGET_COLUMN,
    make_classifier,
)
from ml.models.unseen_scenario_experiment import (
    DEFAULT_OUTPUT_DIR as UNSEEN_OUTPUT_DIR,
    DEFAULT_TEMPORAL_PATH,
)
from ml.models.temporal_ablation_experiment import (
    COUNT_FEATURES,
    DEFAULT_OUTPUT_DIR,
    EXPERIMENT_FEATURES,
    EXPERIMENT_ORDER,
    SEQUENCE_FEATURES,
    TIMING_FEATURES,
    _ensure_output_path_safe,
    run_temporal_ablation,
)
from ml.models.unseen_scenario_experiment import prepare_model_inputs


def _frames():
    scenario_names = ["LEGIT_A", "LEGIT_B", "SUSP_A", "SUSP_B"]
    base_rows = []
    temporal_rows = []
    scenario_by_user = {}
    for index in range(32):
        user_id, account_id, event_id = f"U{index:03d}", f"A{index:03d}", f"E{index:03d}"
        scenario = scenario_names[index // 8]
        label = int(scenario.startswith("SUSP"))
        base = {
            "user_id": user_id, "account_id": account_id, "event_id": event_id,
            TARGET_COLUMN: label, "scenario_type": scenario,
            "paysim_isFraud": label, "paysim_isFlaggedFraud": label,
            "after_t15_sentinel": 999999,
        }
        for feature_index, feature in enumerate(PHASE8_FEATURES):
            base[feature] = 1 if feature == "telecom_event_present" else (index + feature_index) % 2
        base_rows.append(base)
        temporal = {"user_id": user_id, "account_id": account_id, "event_id": event_id,
                    "after_t15_temporal_sentinel": 999999}
        for feature_index, feature in enumerate(ALL_TEMPORAL_FEATURES):
            temporal[feature] = np.nan if feature.startswith("time_to_") and index % 3 == 0 else index + feature_index
        temporal_rows.append(temporal)
        scenario_by_user[user_id] = scenario
    return pd.DataFrame(base_rows), pd.DataFrame(temporal_rows), scenario_by_user


def test_groups_are_exact_partition_of_defined_temporal_features():
    assert set(TIMING_FEATURES) == {
        "time_to_new_device_seconds", "time_to_failed_login_seconds",
        "time_to_password_reset_seconds", "time_to_recovery_seconds",
        "time_to_auth_anomaly_seconds",
    }
    assert set(COUNT_FEATURES) == {
        "post_trigger_failed_login_count", "post_trigger_device_event_count",
        "post_trigger_recovery_event_count",
    }
    assert set(SEQUENCE_FEATURES) == {
        "telecom_to_device_sequence", "telecom_to_auth_sequence",
        "telecom_to_recovery_sequence", "device_before_auth_sequence",
        "auth_before_recovery_sequence",
    }
    grouped = (*TIMING_FEATURES, *COUNT_FEATURES, *SEQUENCE_FEATURES)
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(ALL_TEMPORAL_FEATURES)
    assert "temporal_sequence_score" not in grouped
    assert EXPERIMENT_ORDER == ("BASELINE", "TIMING", "COUNTS", "SEQUENCES", "ALL_TEMPORAL")
    assert EXPERIMENT_FEATURES["ALL_TEMPORAL"] == ALL_TEMPORAL_FEATURES


def test_prepared_experiment_inputs_exclude_scenario_label_ids_fraud_and_future():
    base, temporal, scenarios = _frames()
    X8, X9, y, identifiers, grouping = prepare_model_inputs(base, temporal, scenarios)
    assert tuple(X8.columns) == PHASE8_FEATURES
    assert tuple(X9.columns) == (*PHASE8_FEATURES, *ALL_TEMPORAL_FEATURES)
    assert y.name == TARGET_COLUMN
    assert identifiers.columns.tolist() == list(IDENTIFIER_COLUMNS)
    assert grouping.name == "scenario_type"
    excluded = {
        TARGET_COLUMN, "scenario_type", *IDENTIFIER_COLUMNS,
        "paysim_isFraud", "paysim_isFlaggedFraud",
        "after_t15_sentinel", "after_t15_temporal_sentinel",
    }
    assert not (excluded & (set(X8.columns) | set(X9.columns)))


def test_output_path_guard_protects_phase8_and_phase9_artifacts(tmp_path):
    with pytest.raises(ValueError, match="protected"):
        _ensure_output_path_safe(PHASE8_OUTPUT_DIR)
    with pytest.raises(ValueError, match="protected"):
        _ensure_output_path_safe(UNSEEN_OUTPUT_DIR)
    with pytest.raises(ValueError, match="protected"):
        _ensure_output_path_safe(DEFAULT_FEATURES_PATH)
    with pytest.raises(ValueError, match="protected"):
        _ensure_output_path_safe(DEFAULT_TEMPORAL_PATH)
    assert _ensure_output_path_safe(tmp_path / "temporal_ablation") == (tmp_path / "temporal_ablation").resolve()
    assert DEFAULT_OUTPUT_DIR.name == "temporal_ablation"


def test_five_experiments_share_folds_and_predict_every_row_once(tmp_path, monkeypatch):
    import ml.models.temporal_ablation_experiment as experiment

    # The production run uses the imported Phase 8 forest unchanged. This test
    # exercises fold reuse and artifact behavior with a cheap deterministic estimator.
    monkeypatch.setattr(experiment, "make_classifier", lambda: DummyClassifier(strategy="prior"))
    base, temporal, scenarios = _frames()
    features_path, temporal_path, events_path = (
        tmp_path / "features.csv", tmp_path / "temporal.csv", tmp_path / "events.csv"
    )
    output_dir = tmp_path / "temporal_ablation"
    base.to_csv(features_path, index=False)
    temporal.to_csv(temporal_path, index=False)
    with events_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("user_id", "event_category", "scenario_type", "extra_after_window"))
        writer.writeheader()
        for user_id, scenario_type in scenarios.items():
            writer.writerow({"user_id": user_id, "event_category": "TELECOM",
                             "scenario_type": scenario_type, "extra_after_window": 123})
            writer.writerow({"user_id": user_id, "event_category": "DEVICE",
                             "scenario_type": "", "extra_after_window": 999999})

    result = run_temporal_ablation(features_path, temporal_path, events_path, output_dir)
    predictions = pd.read_csv(output_dir / "predictions.csv")
    assert set(predictions["experiment_name"]) == set(EXPERIMENT_ORDER)
    assert len(predictions) == len(base) * len(EXPERIMENT_ORDER)
    assert predictions.groupby("experiment_name")["user_id"].nunique().to_dict() == {
        name: len(base) for name in EXPERIMENT_ORDER
    }
    assert predictions.groupby(["experiment_name", "user_id"]).size().eq(1).all()
    assert (predictions["scenario_type"] == predictions["held_out_scenario_type"]).all()
    assert set(predictions.columns) >= {
        "user_id", "scenario_type", "actual_label", "experiment_name",
        "predicted_label", "prediction_probability",
    }
    common_fold_signature = [
        (fold["held_out_scenario_type"], tuple(fold["train_scenario_types"]), tuple(fold["test_scenario_types"]))
        for fold in result["metadata"]["folds"]
    ]
    for name in EXPERIMENT_ORDER:
        model_fold_signature = [
            (fold["held_out_scenario_type"], tuple(fold["train_scenario_types"]), tuple(fold["test_scenario_types"]))
            for fold in result["metrics"]["experiments"][name]["folds"]
        ]
        assert model_fold_signature == common_fold_signature
    assert result["metadata"]["same_folds_for_all_experiments"] is True
    assert set(path.name for path in output_dir.iterdir()) == {
        "metrics.json", "predictions.csv", "scenario_error_summary.csv", "run_metadata.json"
    }


def test_all_group_models_use_only_group_features_plus_phase8_base():
    for name, temporal_columns in EXPERIMENT_FEATURES.items():
        expected = (*PHASE8_FEATURES, *temporal_columns)
        assert len(expected) == len(set(expected))
        if name == "BASELINE":
            assert expected == PHASE8_FEATURES
        else:
            assert set(expected) - set(PHASE8_FEATURES) == set(temporal_columns)


def test_estimator_factory_keeps_phase8_configuration():
    estimator = make_classifier()
    for name, value in MODEL_PARAMETERS.items():
        assert estimator.get_params()[name] == value
