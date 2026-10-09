import csv

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ml.features.temporal_features import FEATURE_COLUMNS as TEMPORAL_FEATURE_COLUMNS
from ml.models.random_forest_experiment import (
    FEATURE_COLUMNS as PHASE8_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
)
from ml.models.unseen_scenario_experiment import (
    build_scenario_folds,
    prepare_model_inputs,
    run_unseen_scenario_experiment,
)


def _frames():
    scenarios = ["LEGIT_A", "LEGIT_B", "SUSP_A", "SUSP_B"]
    feature_rows = []
    temporal_rows = []
    scenario_by_user = {}
    for index in range(32):
        user_id = f"U{index:03d}"
        account_id = f"A{index:03d}"
        event_id = f"E{index:03d}"
        scenario = scenarios[index // 8]
        label = int(scenario.startswith("SUSP"))
        base = {"user_id": user_id, "account_id": account_id, "event_id": event_id, TARGET_COLUMN: label}
        for feature_index, feature in enumerate(PHASE8_FEATURE_COLUMNS):
            base[feature] = 1 if feature == "telecom_event_present" else (index + feature_index) % 2
        base.update({
            "scenario_type": scenario,
            "paysim_isFraud": 1 - label,
            "paysim_isFlaggedFraud": 1 - label,
            "future_event_value": 999999,
        })
        feature_rows.append(base)
        temporal = {"user_id": user_id, "account_id": account_id, "event_id": event_id}
        for feature_index, feature in enumerate(TEMPORAL_FEATURE_COLUMNS):
            temporal[feature] = np.nan if feature.startswith("time_to_") and index % 3 == 0 else index + feature_index
        temporal["future_window_value"] = 999999
        temporal_rows.append(temporal)
        scenario_by_user[user_id] = scenario
    return pd.DataFrame(feature_rows), pd.DataFrame(temporal_rows), scenario_by_user


def test_each_scenario_type_is_held_out_once_and_absent_from_training():
    scenario_types = ["A", "A", "B", "C", "C"]
    folds = build_scenario_folds(scenario_types)
    assert [fold["held_out_scenario_type"] for fold in folds] == ["A", "B", "C"]
    test_indices = []
    for fold in folds:
        assert fold["test_scenario_types"] == [fold["held_out_scenario_type"]]
        assert fold["held_out_scenario_type"] not in fold["train_scenario_types"]
        assert not set(fold["train_indices"]) & set(fold["test_indices"])
        test_indices.extend(fold["test_indices"].tolist())
    assert sorted(test_indices) == list(range(len(scenario_types)))


def test_model_inputs_exclude_group_target_ids_fraud_flags_and_future_fields():
    base, temporal, scenarios = _frames()
    X8, X9, y, identifiers, grouping = prepare_model_inputs(base, temporal, scenarios)
    assert tuple(X8.columns) == PHASE8_FEATURE_COLUMNS
    assert tuple(X9.columns) == (*PHASE8_FEATURE_COLUMNS, *TEMPORAL_FEATURE_COLUMNS)
    for excluded in (TARGET_COLUMN, "scenario_type", "paysim_isFraud", "paysim_isFlaggedFraud", "future_event_value"):
        assert excluded not in X8.columns and excluded not in X9.columns
    assert not set(IDENTIFIER_COLUMNS) & (set(X8.columns) | set(X9.columns))
    assert y.name == TARGET_COLUMN
    assert identifiers.columns.tolist() == list(IDENTIFIER_COLUMNS)
    assert grouping.name == "scenario_type"
    assert "future_window_value" not in X9


def test_changing_labels_or_group_labels_does_not_change_feature_matrices():
    base, temporal, scenarios = _frames()
    X8, X9, _, _, _ = prepare_model_inputs(base, temporal, scenarios)
    changed = base.copy()
    changed[TARGET_COLUMN] = 1 - changed[TARGET_COLUMN]
    changed["scenario_type"] = "CHANGED"
    changed["paysim_isFraud"] = 999
    changed["paysim_isFlaggedFraud"] = 999
    X8_changed, X9_changed, _, _, _ = prepare_model_inputs(changed, temporal, scenarios)
    pd.testing.assert_frame_equal(X8, X8_changed)
    pd.testing.assert_frame_equal(X9, X9_changed)


def test_experiment_predicts_every_row_once_with_identical_model_folds(tmp_path, monkeypatch):
    import ml.models.unseen_scenario_experiment as experiment

    # Keep this structural test fast; the real experiment uses the unchanged
    # Phase 8 forest factory and is run separately on the project artifacts.
    monkeypatch.setattr(experiment, "make_classifier", lambda: DummyClassifier(strategy="prior"))
    base, temporal, scenario_by_user = _frames()
    feature_path = tmp_path / "features.csv"
    temporal_path = tmp_path / "temporal.csv"
    events_path = tmp_path / "events.csv"
    output_dir = tmp_path / "unseen"
    base.to_csv(feature_path, index=False)
    temporal.to_csv(temporal_path, index=False)
    with events_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("user_id", "event_category", "scenario_type", "future_event_value"))
        writer.writeheader()
        for user_id, scenario_type in scenario_by_user.items():
            writer.writerow({"user_id": user_id, "event_category": "TELECOM", "scenario_type": scenario_type,
                             "future_event_value": "post-window sentinel"})
            writer.writerow({"user_id": user_id, "event_category": "AUTHENTICATION", "scenario_type": "",
                             "future_event_value": "post-window sentinel"})

    result = run_unseen_scenario_experiment(feature_path, temporal_path, events_path, output_dir)
    predictions = pd.read_csv(output_dir / "predictions.csv")
    folds = result["metadata"]["folds"]
    assert len(predictions) == len(base)
    assert predictions["user_id"].nunique() == len(base)
    assert (predictions["scenario_type_evaluation_only"] == predictions["held_out_scenario_type"]).all()
    assert len(folds) == len(set(scenario_by_user.values()))
    assert all(fold["test_scenario_types"] == [fold["held_out_scenario_type"]] for fold in folds)
    assert all(fold["held_out_scenario_type"] not in fold["train_scenario_types"] for fold in folds)
    assert result["metadata"]["model_a_X_columns"] == list(PHASE8_FEATURE_COLUMNS)
    assert result["metadata"]["model_b_X_columns"] == [*PHASE8_FEATURE_COLUMNS, *TEMPORAL_FEATURE_COLUMNS]
    assert set(path.name for path in output_dir.iterdir()) == {
        "metrics.json", "predictions.csv", "scenario_error_summary.csv", "feature_audit.csv", "run_metadata.json"
    }


def test_audit_contains_scenario_level_temporal_distributions():
    base, temporal, scenario_by_user = _frames()
    _, _, _, _, scenarios = prepare_model_inputs(base, temporal, scenario_by_user)
    from ml.models.unseen_scenario_experiment import _feature_audit

    audit = _feature_audit(temporal, scenarios)
    assert len(audit) == len(set(scenario_by_user.values())) * len(TEMPORAL_FEATURE_COLUMNS)
    assert {row["scenario_type_evaluation_only"] for row in audit} == set(scenario_by_user.values())


def test_one_class_fold_metrics_keep_undefined_auc_values_missing():
    base, temporal, scenario_by_user = _frames()
    _, _, y, _, scenarios = prepare_model_inputs(base, temporal, scenario_by_user)
    folds = build_scenario_folds(scenarios)
    legit_fold = next(fold for fold in folds if fold["held_out_scenario_type"] == "LEGIT_A")
    assert set(y.iloc[legit_fold["test_indices"]]) == {0}
    # The fold metric path uses the Phase 8 metric helper, which marks ROC-AUC
    # and average precision unavailable when there are no positive examples.
    from ml.models.random_forest_experiment import binary_metrics

    metrics = binary_metrics(
        y.iloc[legit_fold["test_indices"]],
        np.zeros(len(legit_fold["test_indices"]), dtype=int),
        np.zeros(len(legit_fold["test_indices"]), dtype=float),
    )
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc_average_precision"] is None
