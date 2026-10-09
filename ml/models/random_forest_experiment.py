"""Reproducible Phase 8 Random Forest baseline experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from ml.rules.rule_baseline import (
    AMBIGUOUS,
    ATO_RISK_INDICATOR,
    NO_INDICATOR,
    generate_rule_baselines,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "ato_synthetic_events.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "random_forest"

TARGET_COLUMN = "ato_label"
IDENTIFIER_COLUMNS = ("user_id", "account_id", "event_id")
FEATURE_COLUMNS = (
    "sim_change",
    "esim_change",
    "telecom_event_present",
    "new_device",
    "device_change",
    "device_deviation",
    "known_device",
    "failed_login_count",
    "authentication_anomaly",
    "password_reset",
    "account_recovery",
    "recovery_event_present",
    "new_beneficiary",
    "account_change",
    "baseline_event_count",
    "baseline_event_frequency",
    "baseline_transaction_count",
    "baseline_transaction_amount_mean",
    "baseline_transaction_amount_min",
    "baseline_transaction_amount_max",
    "baseline_successful_login_count",
)
POST_TRIGGER_TRANSACTION_COLUMNS = (
    "transaction_amount",
    "transaction_count",
    "transaction_type",
    "transaction_deviation",
)
RANDOM_STATE = 42
TEST_SIZE = 0.20
PERMUTATION_REPEATS = 10

MODEL_PARAMETERS: dict[str, Any] = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def extract_xy(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select exactly the approved model inputs and target, validating values."""
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN} | set(IDENTIFIER_COLUMNS)
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"features.csv is missing required columns: {', '.join(missing)}")
    if features["user_id"].duplicated().any():
        raise ValueError("features.csv must contain one row per user_id")
    if features["account_id"].duplicated().any():
        raise ValueError("features.csv must contain one row per account_id")

    X = features.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("approved model input features must be numeric, finite, and non-missing")
    y_numeric = pd.to_numeric(features[TARGET_COLUMN], errors="raise")
    if y_numeric.isna().any() or not set(y_numeric.unique()).issubset({0, 1}):
        raise ValueError("ato_label must contain only non-missing binary values 0 and 1")
    y = y_numeric.astype(int).rename(TARGET_COLUMN)
    return X, y


def make_classifier() -> RandomForestClassifier:
    """Construct the fixed, un-tuned Phase 8 Random Forest configuration."""
    return RandomForestClassifier(**MODEL_PARAMETERS)


def stratified_split(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible stratified row indices for the fixed 80/20 split."""
    indices = np.arange(len(y))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y.to_numpy(),
    )
    return train_indices, test_indices


def _class_counts(values: pd.Series | np.ndarray) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(label): counts.get(label, 0) for label in (0, 1)}


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def binary_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    positive_probability: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Calculate class-1 metrics and a [0, 1]-ordered confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    truth = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(positive_probability, dtype=float)
    has_both_classes = len(np.unique(truth)) == 2
    roc_auc = float(roc_auc_score(truth, probabilities)) if has_both_classes else None
    pr_auc = float(average_precision_score(truth, probabilities)) if np.any(truth == 1) else None
    return {
        "n": int(len(truth)),
        "class_counts": _class_counts(truth),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _safe_ratio(fp, fp + tn),
        "fnr": _safe_ratio(fn, fn + tp),
        "roc_auc": roc_auc,
        "pr_auc_average_precision": pr_auc,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [0, 1],
    }


def _positive_probabilities(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, class_index]


def _metrics_for_model(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    indices: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    subset = X.iloc[indices]
    truth = y.iloc[indices].to_numpy()
    predictions = model.predict(subset).astype(int)
    probabilities = _positive_probabilities(model, subset)
    return binary_metrics(truth, predictions, probabilities), predictions, probabilities


def _rule_binary(outcome: str) -> int:
    if outcome == ATO_RISK_INDICATOR:
        return 1
    if outcome == NO_INDICATOR:
        return 0
    raise ValueError(f"ambiguous Phase 7 outcome cannot be converted to binary: {outcome}")


def _scenario_types_by_user(events_path: Path) -> dict[str, str]:
    """Read scenario labels for post-prediction analysis only."""
    scenario_by_user: dict[str, str] = {}
    with events_path.open("r", newline="", encoding="utf-8") as source:
        for event in csv.DictReader(source):
            if event.get("event_category") != "TELECOM":
                continue
            user_id = event["user_id"]
            if user_id in scenario_by_user:
                raise ValueError(f"multiple telecom trigger rows for user {user_id}")
            scenario_by_user[user_id] = event["scenario_type"]
    return scenario_by_user


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_label(path: Path) -> str:
    """Use a stable project-relative path, or a basename for external test inputs."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return resolved.name


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def run_experiment(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Fit/evaluate the primary and ablated forests and save deterministic outputs."""
    features_path = Path(features_path)
    events_path = Path(events_path)
    output_dir = Path(output_dir)
    feature_rows = pd.read_csv(features_path)
    X, y = extract_xy(feature_rows)
    train_indices, test_indices = stratified_split(y)

    # Fit and predict before joining scenario_type or other evaluation metadata.
    model = make_classifier()
    model.fit(X.iloc[train_indices], y.iloc[train_indices])
    primary_metrics, predictions, probabilities = _metrics_for_model(model, X, y, test_indices)

    importance = permutation_importance(
        model,
        X.iloc[test_indices],
        y.iloc[test_indices],
        scoring="average_precision",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance_rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(FEATURE_COLUMNS, importance.importances_mean, importance.importances_std)
    ]
    importance_rows.sort(key=lambda row: (-row["importance_mean"], row["feature"]))

    ablated_features = tuple(feature for feature in FEATURE_COLUMNS if feature != "device_deviation")
    ablated_model = make_classifier()
    ablated_X = X.loc[:, list(ablated_features)]
    ablated_model.fit(ablated_X.iloc[train_indices], y.iloc[train_indices])
    ablation_metrics, _, _ = _metrics_for_model(
        ablated_model,
        ablated_X,
        y,
        test_indices,
    )
    metric_names = (
        "precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc_average_precision",
    )
    ablation_delta = {
        name: (
            ablation_metrics[name] - primary_metrics[name]
            if ablation_metrics[name] is not None and primary_metrics[name] is not None
            else None
        )
        for name in metric_names
    }

    test_frame = feature_rows.iloc[test_indices].copy()
    rule_inputs = test_frame.loc[:, [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS]].to_dict(orient="records")
    rule_outputs = generate_rule_baselines(rule_inputs)
    telecom_outcomes = [result["prediction"] for result in rule_outputs["telecom_context"]]
    multi_outcomes = [result["prediction"] for result in rule_outputs["multi_signal"]]
    truth_test = y.iloc[test_indices].to_numpy()

    telecom_binary = np.asarray([_rule_binary(value) for value in telecom_outcomes], dtype=int)
    telecom_metrics = binary_metrics(
        truth_test,
        telecom_binary,
        telecom_binary.astype(float),
    )
    determinate_positions = np.asarray(
        [position for position, outcome in enumerate(multi_outcomes) if outcome != AMBIGUOUS],
        dtype=int,
    )
    multi_binary = np.asarray(
        [_rule_binary(multi_outcomes[position]) for position in determinate_positions],
        dtype=int,
    )
    phase7_determinate_metrics = binary_metrics(
        truth_test[determinate_positions],
        multi_binary,
        multi_binary.astype(float),
    )
    rf_determinate_metrics = binary_metrics(
        truth_test[determinate_positions],
        predictions[determinate_positions],
        probabilities[determinate_positions],
    )
    multi_outcome_counts = Counter(multi_outcomes)
    ambiguous_count = multi_outcome_counts.get(AMBIGUOUS, 0)

    # Scenario types are loaded only after both model and rule predictions exist.
    scenario_by_user = _scenario_types_by_user(events_path)
    if set(test_frame["user_id"]) - set(scenario_by_user):
        raise ValueError("scenario metadata is missing one or more test users")

    prediction_rows: list[dict[str, Any]] = []
    scenario_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for position, (_, row) in enumerate(test_frame.iterrows()):
        user_id = row["user_id"]
        scenario_type = scenario_by_user[user_id]
        actual = int(truth_test[position])
        predicted = int(predictions[position])
        error_type = "FP" if predicted == 1 and actual == 0 else "FN" if predicted == 0 and actual == 1 else "CORRECT"
        output = {
            "user_id": user_id,
            "account_id": row["account_id"],
            "event_id": row["event_id"],
            "scenario_type_evaluation_only": scenario_type,
            "y_true_evaluation_only": actual,
            "rf_prediction": predicted,
            "rf_probability_ato": float(probabilities[position]),
            "rf_error_type": error_type,
            "phase7_telecom_prediction": telecom_outcomes[position],
            "phase7_multi_signal_prediction": multi_outcomes[position],
        }
        prediction_rows.append(output)
        scenario_rows[scenario_type].append(output)

    scenario_error_rows: list[dict[str, Any]] = []
    for scenario_type in sorted(scenario_rows):
        rows = scenario_rows[scenario_type]
        actuals = [int(row["y_true_evaluation_only"]) for row in rows]
        rf_predictions = [int(row["rf_prediction"]) for row in rows]
        determinate_rows = [row for row in rows if row["phase7_multi_signal_prediction"] != AMBIGUOUS]
        phase7_errors = sum(
            _rule_binary(row["phase7_multi_signal_prediction"]) != int(row["y_true_evaluation_only"])
            for row in determinate_rows
        )
        scenario_error_rows.append({
            "scenario_type_evaluation_only": scenario_type,
            "test_rows": len(rows),
            "actual_legitimate": sum(actual == 0 for actual in actuals),
            "actual_suspicious": sum(actual == 1 for actual in actuals),
            "rf_tp": sum(actual == 1 and pred == 1 for actual, pred in zip(actuals, rf_predictions)),
            "rf_fp": sum(actual == 0 and pred == 1 for actual, pred in zip(actuals, rf_predictions)),
            "rf_tn": sum(actual == 0 and pred == 0 for actual, pred in zip(actuals, rf_predictions)),
            "rf_fn": sum(actual == 1 and pred == 0 for actual, pred in zip(actuals, rf_predictions)),
            "rf_error_count": sum(row["rf_error_type"] != "CORRECT" for row in rows),
            "phase7_multi_signal_ambiguous": sum(row["phase7_multi_signal_prediction"] == AMBIGUOUS for row in rows),
            "phase7_multi_signal_determinate": len(determinate_rows),
            "phase7_multi_signal_errors_on_determinate": phase7_errors,
        })

    train_class_counts = _class_counts(y.iloc[train_indices])
    test_class_counts = _class_counts(y.iloc[test_indices])
    metrics = {
        "target_definition": {"column": TARGET_COLUMN, "positive_class": 1, "negative_class": 0},
        "primary_test": primary_metrics,
        "device_deviation_ablation": {
            "removed_feature": "device_deviation",
            "test_metrics": ablation_metrics,
            "delta_ablated_minus_full": ablation_delta,
        },
        "phase7_comparison_same_test_rows": {
            "telecom_context_all_test_rows": telecom_metrics,
            "multi_signal_outcome_counts": {key: multi_outcome_counts.get(key, 0) for key in (
                NO_INDICATOR, ATO_RISK_INDICATOR, AMBIGUOUS,
            )},
            "multi_signal_ambiguous_count": ambiguous_count,
            "multi_signal_ambiguity_rate": ambiguous_count / len(test_indices),
            "multi_signal_coverage": (len(test_indices) - ambiguous_count) / len(test_indices),
            "multi_signal_metrics_on_determinate_rows": phase7_determinate_metrics,
            "random_forest_metrics_on_same_determinate_rows": rf_determinate_metrics,
        },
    }
    train_test_counts = {
        "total": {"train": int(len(train_indices)), "test": int(len(test_indices))},
        "by_ato_label": {"train": train_class_counts, "test": test_class_counts},
    }
    metadata = {
        "experiment": "Phase 8 Random Forest baseline",
        "input_features_path": _path_label(features_path),
        "input_features_sha256": _sha256(features_path),
        "scenario_metadata_path_evaluation_only": _path_label(events_path),
        "scenario_metadata_sha256": _sha256(events_path),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "identifier_columns_not_in_X": list(IDENTIFIER_COLUMNS),
        "excluded_post_trigger_transaction_columns": list(POST_TRIGGER_TRANSACTION_COLUMNS),
        "split": {
            "method": "train_test_split with stratify=y",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_test_counts": train_test_counts,
            "train_user_ids": sorted(str(value) for value in feature_rows.iloc[train_indices]["user_id"]),
            "test_user_ids": sorted(str(value) for value in feature_rows.iloc[test_indices]["user_id"]),
        },
        "random_forest_parameters": MODEL_PARAMETERS,
        "decision_rule": "RandomForestClassifier.predict default; no custom threshold",
        "permutation_importance": {
            "scoring": "average_precision",
            "n_repeats": PERMUTATION_REPEATS,
            "random_state": RANDOM_STATE,
            "evaluation_set": "held-out test set",
        },
        "ablation": {
            "removed_feature": "device_deviation",
            "same_split_and_model_parameters": True,
        },
        "phase7_comparison": {
            "same_test_rows": True,
            "ambiguous_outcome": "excluded from binary metrics; separately reported, not mapped",
        },
        "software_versions": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        (
            "user_id", "account_id", "event_id", "scenario_type_evaluation_only",
            "y_true_evaluation_only", "rf_prediction", "rf_probability_ato", "rf_error_type",
            "phase7_telecom_prediction", "phase7_multi_signal_prediction",
        ),
    )
    _write_csv(
        output_dir / "feature_importance.csv",
        importance_rows,
        ("feature", "importance_mean", "importance_std"),
    )
    _write_csv(
        output_dir / "scenario_error_summary.csv",
        scenario_error_rows,
        (
            "scenario_type_evaluation_only", "test_rows", "actual_legitimate", "actual_suspicious",
            "rf_tp", "rf_fp", "rf_tn", "rf_fn", "rf_error_count",
            "phase7_multi_signal_ambiguous", "phase7_multi_signal_determinate",
            "phase7_multi_signal_errors_on_determinate",
        ),
    )
    return {
        "metrics": metrics,
        "metadata": metadata,
        "feature_importance": importance_rows,
        "predictions": prediction_rows,
        "scenario_error_summary": scenario_error_rows,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reproducible Phase 8 Random Forest experiment.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_experiment(args.features, args.events, args.output_dir)
    print(json.dumps({
        "output_dir": str(result["output_dir"]),
        "train_test_counts": result["metadata"]["split"]["train_test_counts"],
        "test_metrics": result["metrics"]["primary_test"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
