"""Phase 9B leave-one-scenario-type-out temporal feature ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from joblib import cpu_count
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.features.temporal_features import FEATURE_COLUMNS as ALL_TEMPORAL_FEATURES
from ml.models.random_forest_experiment import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_FEATURES_PATH,
    DEFAULT_OUTPUT_DIR as PHASE8_OUTPUT_DIR,
    FEATURE_COLUMNS as PHASE8_FEATURES,
    MODEL_PARAMETERS,
    TARGET_COLUMN,
    binary_metrics,
    make_classifier,
)
from ml.models.unseen_scenario_experiment import (
    DEFAULT_OUTPUT_DIR as UNSEEN_SCENARIO_OUTPUT_DIR,
    DEFAULT_TEMPORAL_PATH,
    _scenario_types_from_triggers,
    build_scenario_folds,
    prepare_model_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "temporal_ablation"

TIMING_FEATURES = (
    "time_to_new_device_seconds",
    "time_to_failed_login_seconds",
    "time_to_password_reset_seconds",
    "time_to_recovery_seconds",
    "time_to_auth_anomaly_seconds",
)
COUNT_FEATURES = (
    "post_trigger_failed_login_count",
    "post_trigger_device_event_count",
    "post_trigger_recovery_event_count",
)
SEQUENCE_FEATURES = (
    "telecom_to_device_sequence",
    "telecom_to_auth_sequence",
    "telecom_to_recovery_sequence",
    "device_before_auth_sequence",
    "auth_before_recovery_sequence",
)
FEATURE_GROUPS = {
    "TIMING": TIMING_FEATURES,
    "COUNTS": COUNT_FEATURES,
    "SEQUENCES": SEQUENCE_FEATURES,
}
EXPERIMENT_FEATURES = {
    "BASELINE": (),
    "TIMING": TIMING_FEATURES,
    "COUNTS": COUNT_FEATURES,
    "SEQUENCES": SEQUENCE_FEATURES,
    "ALL_TEMPORAL": ALL_TEMPORAL_FEATURES,
}
EXPERIMENT_ORDER = ("BASELINE", "TIMING", "COUNTS", "SEQUENCES", "ALL_TEMPORAL")


def _validate_feature_groups() -> None:
    listed = (*TIMING_FEATURES, *COUNT_FEATURES, *SEQUENCE_FEATURES)
    if len(listed) != len(set(listed)) or set(listed) != set(ALL_TEMPORAL_FEATURES):
        raise ValueError("Phase 9 group definitions must partition the existing temporal features exactly")
    if "temporal_sequence_score" in listed:
        raise ValueError("temporal_sequence_score is excluded from Phase 9")


def _ensure_output_path_safe(output_dir: str | Path) -> Path:
    """Reject output paths that could overwrite Phase 8 or Phase 9 artifacts."""
    target = Path(output_dir).resolve()
    protected_dirs = (PHASE8_OUTPUT_DIR.resolve(), UNSEEN_SCENARIO_OUTPUT_DIR.resolve())
    protected_files = (
        (PROJECT_ROOT / "data" / "processed" / "features.csv").resolve(),
        DEFAULT_TEMPORAL_PATH.resolve(),
    )
    for protected in protected_dirs:
        if target == protected or protected in target.parents or target in protected.parents:
            raise ValueError(f"output directory overlaps a protected Phase 8/9 output: {protected}")
    if target in protected_files:
        raise ValueError(f"output directory points to a protected Phase 8/9 artifact: {target}")
    return target


def _experiment_pipeline() -> Pipeline:
    """Build the Phase 9 preprocessing path with unchanged Phase 8 RF settings."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("classifier", make_classifier()),
    ])


def _positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
    class_index = list(classifier.classes_).index(1)
    return model.predict_proba(X)[:, class_index]


def _metrics(
    y_true: pd.Series | np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    return binary_metrics(np.asarray(y_true, dtype=int), predictions, probabilities)


def _hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_temporal_ablation(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    temporal_path: str | Path = DEFAULT_TEMPORAL_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the five fixed feature-set experiments on identical LOSO folds."""
    _validate_feature_groups()
    output_dir = _ensure_output_path_safe(output_dir)
    feature_rows = pd.read_csv(features_path)
    temporal_rows = pd.read_csv(temporal_path)
    users = set(feature_rows["user_id"].astype(str))
    scenario_by_user = _scenario_types_from_triggers(events_path, users)
    X_base, X_with_temporal, y, identifiers, scenarios = prepare_model_inputs(
        feature_rows, temporal_rows, scenario_by_user
    )
    folds = build_scenario_folds(scenarios)

    experiment_columns = {
        name: [*PHASE8_FEATURES, *EXPERIMENT_FEATURES[name]]
        for name in EXPERIMENT_ORDER
    }
    model_inputs: dict[str, pd.DataFrame] = {}
    for name, columns in experiment_columns.items():
        if EXPERIMENT_FEATURES[name]:
            model_inputs[name] = X_with_temporal.loc[:, columns].copy()
        else:
            model_inputs[name] = X_base.copy()
    n_rows = len(y)
    outputs: dict[str, dict[str, np.ndarray]] = {
        name: {
            "prediction": np.full(n_rows, -1, dtype=int),
            "probability": np.full(n_rows, np.nan, dtype=float),
        }
        for name in EXPERIMENT_ORDER
    }
    per_fold_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in EXPERIMENT_ORDER}

    for fold_number, fold in enumerate(folds):
        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]
        train_y = y.iloc[train_idx]
        if len(train_y.unique()) != 2:
            raise ValueError(f"training complement for {fold['held_out_scenario_type']} lacks both classes")
        for experiment_name in EXPERIMENT_ORDER:
            X = model_inputs[experiment_name]
            has_temporal = bool(EXPERIMENT_FEATURES[experiment_name])
            model = _experiment_pipeline() if has_temporal else make_classifier()
            model.fit(X.iloc[train_idx], train_y)
            prediction = model.predict(X.iloc[test_idx]).astype(int)
            probability = _positive_probabilities(model, X.iloc[test_idx])
            if np.any(outputs[experiment_name]["prediction"][test_idx] != -1):
                raise ValueError(f"{experiment_name}: a row was assigned out-of-fold predictions more than once")
            outputs[experiment_name]["prediction"][test_idx] = prediction
            outputs[experiment_name]["probability"][test_idx] = probability
            per_fold_metrics[experiment_name].append({
                "fold_number": fold_number,
                "held_out_scenario_type": fold["held_out_scenario_type"],
                "train_scenario_types": fold["train_scenario_types"],
                "test_scenario_types": fold["test_scenario_types"],
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_class_counts": {str(k): int(v) for k, v in train_y.value_counts().sort_index().items()},
                "test_class_counts": {
                    str(k): int(v) for k, v in y.iloc[test_idx].value_counts().sort_index().items()
                },
                "metrics": _metrics(y.iloc[test_idx], prediction, probability),
            })

    for name, result in outputs.items():
        if np.any(result["prediction"] == -1) or not np.isfinite(result["probability"]).all():
            raise ValueError(f"{name}: every row must have exactly one finite out-of-fold prediction")

    actual = y.to_numpy(dtype=int)
    scenario_values = scenarios.astype(str).to_numpy()
    metrics_by_experiment: dict[str, Any] = {}
    error_rows: list[dict[str, Any]] = []
    for name in EXPERIMENT_ORDER:
        predictions = outputs[name]["prediction"]
        probabilities = outputs[name]["probability"]
        per_scenario: dict[str, Any] = {}
        for scenario_type in sorted(set(scenario_values)):
            indices = np.flatnonzero(scenario_values == scenario_type)
            labels = actual[indices]
            scenario_predictions = predictions[indices]
            per_scenario[scenario_type] = _metrics(
                labels, scenario_predictions, probabilities[indices]
            )
            false_positives = int(np.sum((labels == 0) & (scenario_predictions == 1)))
            false_negatives = int(np.sum((labels == 1) & (scenario_predictions == 0)))
            correct = int(np.sum(labels == scenario_predictions))
            error_rows.append({
                "experiment_name": name,
                "scenario_type": scenario_type,
                "rows": int(len(indices)),
                "legitimate_count": int(np.sum(labels == 0)),
                "suspicious_count": int(np.sum(labels == 1)),
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "correct_predictions": correct,
                "errors": false_positives + false_negatives,
                "precision": per_scenario[scenario_type]["precision"],
                "recall": per_scenario[scenario_type]["recall"],
                "f1": per_scenario[scenario_type]["f1"],
                "fpr": per_scenario[scenario_type]["fpr"],
                "fnr": per_scenario[scenario_type]["fnr"],
                "roc_auc": per_scenario[scenario_type]["roc_auc"],
                "pr_auc_average_precision": per_scenario[scenario_type]["pr_auc_average_precision"],
                "confusion_matrix": json.dumps(per_scenario[scenario_type]["confusion_matrix"]),
            })
        metrics_by_experiment[name] = {
            "feature_columns": experiment_columns[name],
            "pooled_out_of_fold": _metrics(actual, predictions, probabilities),
            "per_scenario_type": per_scenario,
            "folds": per_fold_metrics[name],
        }

    prediction_rows: list[dict[str, Any]] = []
    for experiment_name in EXPERIMENT_ORDER:
        for index in range(n_rows):
            prediction_rows.append({
                "user_id": str(identifiers.iloc[index]["user_id"]),
                "account_id": str(identifiers.iloc[index]["account_id"]),
                "event_id": str(identifiers.iloc[index]["event_id"]),
                "scenario_type": str(scenarios.iloc[index]),
                "actual_label": int(actual[index]),
                "held_out_scenario_type": str(scenarios.iloc[index]),
                "experiment_name": experiment_name,
                "predicted_label": int(outputs[experiment_name]["prediction"][index]),
                "prediction_probability": float(outputs[experiment_name]["probability"][index]),
            })

    scenario_class_counts = {
        scenario: {str(k): int(v) for k, v in y[scenarios == scenario].value_counts().sort_index().items()}
        for scenario in sorted(set(scenario_values))
    }
    metadata = {
        "experiment": "Phase 9B temporal feature group ablation",
        "evaluation": "leave-one-scenario-type-out; the same eight folds are used by every experiment",
        "row_count": int(n_rows),
        "scenario_type_count": int(len(folds)),
        "scenario_class_counts": scenario_class_counts,
        "experiment_order": list(EXPERIMENT_ORDER),
        "feature_groups": {
            "PHASE8_BASELINE": list(PHASE8_FEATURES),
            **{name: list(columns) for name, columns in FEATURE_GROUPS.items()},
            "ALL_TEMPORAL": list(ALL_TEMPORAL_FEATURES),
        },
        "experiment_features": experiment_columns,
        "target_column": TARGET_COLUMN,
        "scenario_type_role": "grouping and evaluation metadata only; never an ML input",
        "excluded_inputs": [
            TARGET_COLUMN, "scenario_type", "user_id", "account_id", "event_id",
            "paysim_isFraud", "paysim_isFlaggedFraud", "Phase 7 predictions",
            "temporal_sequence_score",
        ],
        "temporal_snapshot": "existing T through T+15 inclusive feature artifact; no data is generated or extended",
        "random_forest_parameters": MODEL_PARAMETERS,
        "phase9_missing_value_handling": (
            "for experiments with temporal inputs, median imputation and missingness indicators fit on each "
            "training fold only; BASELINE uses the Phase 8 classifier directly"
        ),
        "threshold": "RandomForestClassifier default predict behavior; no custom threshold",
        "same_folds_for_all_experiments": True,
        "same_folds_for_all_experiments": True,
        "folds": [{
            "fold_number": index,
            "held_out_scenario_type": fold["held_out_scenario_type"],
            "train_scenario_types": fold["train_scenario_types"],
            "test_scenario_types": fold["test_scenario_types"],
            "train_rows": int(len(fold["train_indices"])),
            "test_rows": int(len(fold["test_indices"])),
        } for index, fold in enumerate(folds)],
        "effective_joblib_cpu_count": cpu_count(),
        "input_sha256": {
            "phase8_features": _hash(features_path),
            "phase9_temporal_features": _hash(temporal_path),
            "scenario_group_metadata_source": _hash(events_path),
        },
        "software_versions": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "metric_definitions": "class 1 is suspicious; confusion matrix labels are [0, 1]",
        "experiments": metrics_by_experiment,
    }
    _write_json(output_dir / "metrics.json", metrics_payload)
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        (
            "user_id", "account_id", "event_id", "scenario_type", "actual_label",
            "held_out_scenario_type", "experiment_name", "predicted_label", "prediction_probability",
        ),
    )
    _write_csv(
        output_dir / "scenario_error_summary.csv",
        error_rows,
        (
            "experiment_name", "scenario_type", "rows", "legitimate_count", "suspicious_count",
            "false_positives", "false_negatives", "correct_predictions", "errors",
            "precision", "recall", "f1", "fpr", "fnr", "roc_auc",
            "pr_auc_average_precision", "confusion_matrix",
        ),
    )
    return {
        "metrics": metrics_payload,
        "predictions": prediction_rows,
        "scenario_error_summary": error_rows,
        "metadata": metadata,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 9B temporal feature group ablations.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--temporal-features", type=Path, default=DEFAULT_TEMPORAL_PATH)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_temporal_ablation(args.features, args.temporal_features, args.events, args.output_dir)
    print(json.dumps({
        name: result["metrics"]["experiments"][name]["pooled_out_of_fold"]
        for name in EXPERIMENT_ORDER
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
