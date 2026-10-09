import csv

import numpy as np
import pandas as pd

from ml.models.random_forest_experiment import (
    FEATURE_COLUMNS,
    MODEL_PARAMETERS,
    TARGET_COLUMN,
    binary_metrics,
    extract_xy,
    make_classifier,
    run_experiment,
    stratified_split,
)


def _feature_frame(count=40):
    rows = []
    for index in range(count):
        target = index % 2
        row = {
            "user_id": f"USR-{index:04d}",
            "account_id": f"ACC-{index:04d}",
            "event_id": f"EVT-{index:04d}",
            TARGET_COLUMN: target,
            "scenario_type": f"SCENARIO-{target}",
            "paysim_isFraud": index % 2,
            "paysim_isFlaggedFraud": (index + 1) % 2,
            "transaction_amount": np.nan,
            "transaction_count": 0,
            "transaction_type": np.nan,
            "transaction_deviation": np.nan,
        }
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            if feature == "baseline_event_frequency":
                value = 0.2 + (index % 4) / 100
            elif feature.startswith("baseline_transaction_amount_"):
                value = (feature_index + 1) * 100.0 + index
            elif feature == "failed_login_count":
                value = target
            elif feature == "telecom_event_present":
                value = 1
            else:
                value = (index + feature_index) % 2
            row[feature] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_extract_xy_selects_exact_21_features_and_target_only():
    frame = _feature_frame()
    X, y = extract_xy(frame)
    assert tuple(X.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 21
    assert y.name == "ato_label"
    assert "ato_label" not in X
    assert "scenario_type" not in X
    assert "paysim_isFraud" not in X
    assert "paysim_isFlaggedFraud" not in X
    assert not ({"transaction_amount", "transaction_count", "transaction_type", "transaction_deviation"} & set(X.columns))


def test_excluded_fields_do_not_change_model_inputs():
    original = _feature_frame()
    changed = original.copy()
    changed["scenario_type"] = "CHANGED"
    changed["paysim_isFraud"] = 999
    changed["paysim_isFlaggedFraud"] = 999
    changed["transaction_amount"] = 999
    changed["transaction_count"] = 999
    changed["transaction_type"] = "TRANSFER"
    changed["transaction_deviation"] = 999
    original_X, _ = extract_xy(original)
    changed_X, _ = extract_xy(changed)
    pd.testing.assert_frame_equal(original_X, changed_X)


def test_extract_xy_rejects_repeated_account_to_prevent_cross_split_overlap():
    frame = _feature_frame()
    frame.loc[1, "account_id"] = frame.loc[0, "account_id"]
    try:
        extract_xy(frame)
    except ValueError as error:
        assert "one row per account_id" in str(error)
    else:
        raise AssertionError("duplicate accounts must be rejected")


def test_split_is_reproducible_stratified_and_80_20():
    labels = pd.Series([0] * 625 + [1] * 375, name=TARGET_COLUMN)
    train_1, test_1 = stratified_split(labels)
    train_2, test_2 = stratified_split(labels)
    assert np.array_equal(train_1, train_2)
    assert np.array_equal(test_1, test_2)
    assert len(train_1) == 800
    assert len(test_1) == 200
    assert labels.iloc[train_1].value_counts().to_dict() == {0: 500, 1: 300}
    assert labels.iloc[test_1].value_counts().to_dict() == {0: 125, 1: 75}


def test_classifier_uses_approved_fixed_configuration():
    model = make_classifier()
    for parameter, value in MODEL_PARAMETERS.items():
        assert model.get_params()[parameter] == value
    assert model.class_weight == "balanced"


def test_binary_metrics_use_fixed_class_order_and_expected_formulas():
    metrics = binary_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 1, 0, 1]),
        np.asarray([0.1, 0.8, 0.2, 0.9]),
    )
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["tp"] == metrics["fp"] == metrics["tn"] == metrics["fn"] == 1
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["fpr"] == metrics["fnr"] == 0.5
    assert metrics["roc_auc"] == 0.75


def test_experiment_writes_reproducible_artifacts_and_scenario_analysis(tmp_path):
    feature_path = tmp_path / "features.csv"
    event_path = tmp_path / "events.csv"
    output_dir = tmp_path / "experiment"
    frame = _feature_frame()
    frame.to_csv(feature_path, index=False)
    with event_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=("user_id", "event_category", "scenario_type"))
        writer.writeheader()
        for index, user_id in enumerate(frame["user_id"]):
            writer.writerow({
                "user_id": user_id,
                "event_category": "TELECOM",
                "scenario_type": "LEGITIMATE_SIM_REPLACEMENT" if index % 2 == 0 else "SUSPICIOUS_SIM_REPLACEMENT",
            })

    first = run_experiment(feature_path, event_path, output_dir)
    first_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    second = run_experiment(feature_path, event_path, output_dir)
    second_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    assert first_bytes == second_bytes
    assert set(first_bytes) == {
        "metrics.json", "run_metadata.json", "predictions.csv",
        "feature_importance.csv", "scenario_error_summary.csv",
    }
    assert len(first["predictions"]) == 8
    assert len(first["feature_importance"]) == 21
    assert len(first["scenario_error_summary"]) == 2
    assert first["metadata"]["split"]["train_test_counts"]["total"] == {"train": 32, "test": 8}
    assert first["metadata"]["feature_columns"] == list(FEATURE_COLUMNS)
    assert first["metadata"]["ablation"]["same_split_and_model_parameters"] is True
    assert first["metrics"]["phase7_comparison_same_test_rows"]["multi_signal_ambiguous_count"] >= 0
