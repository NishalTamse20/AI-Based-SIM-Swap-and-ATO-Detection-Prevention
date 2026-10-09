"""Leave-one-scenario-type-out comparison for Phase 8 and Phase 9 inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
from joblib import cpu_count
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.features.temporal_features import FEATURE_COLUMNS as TEMPORAL_FEATURE_COLUMNS
from ml.models.random_forest_experiment import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_FEATURES_PATH,
    FEATURE_COLUMNS as PHASE8_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_PARAMETERS,
    TARGET_COLUMN,
    binary_metrics,
    make_classifier,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPORAL_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_features.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "unseen_scenario"
SCENARIO_COLUMN = "scenario_type"


def build_scenario_folds(scenario_types: Iterable[str]) -> list[dict[str, Any]]:
    """Construct deterministic leave-one-scenario-type-out row indices."""
    values = np.asarray(list(scenario_types), dtype=object)
    if values.ndim != 1 or len(values) == 0 or pd.isna(values).any():
        raise ValueError("scenario_type must contain non-missing values")
    unique_types = sorted(set(str(value) for value in values))
    folds: list[dict[str, Any]] = []
    for held_out in unique_types:
        test_indices = np.flatnonzero(values == held_out)
        train_indices = np.flatnonzero(values != held_out)
        if not len(train_indices) or not len(test_indices):
            raise ValueError(f"invalid fold for scenario type {held_out}")
        folds.append({
            "held_out_scenario_type": held_out,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_scenario_types": sorted(set(str(value) for value in values[train_indices])),
            "test_scenario_types": sorted(set(str(value) for value in values[test_indices])),
        })
    return folds


def _scenario_types_from_triggers(events_path: str | Path, users: set[str]) -> dict[str, str]:
    """Read only the trigger's scenario_type as grouping/evaluation metadata."""
    result: dict[str, str] = {}
    with Path(events_path).open("r", newline="", encoding="utf-8") as source:
        for event in csv.DictReader(source):
            if event.get("event_category") != "TELECOM" or event.get("user_id") not in users:
                continue
            user_id = str(event["user_id"])
            if user_id in result:
                raise ValueError(f"multiple telecom trigger rows for user {user_id}")
            scenario_type = event.get(SCENARIO_COLUMN)
            if not scenario_type:
                raise ValueError(f"telecom trigger for user {user_id} has no scenario_type")
            result[user_id] = str(scenario_type)
    if set(result) != users:
        missing = sorted(users - set(result))
        raise ValueError(f"scenario metadata missing for users: {missing[:5]}")
    return result


def prepare_model_inputs(
    feature_rows: pd.DataFrame,
    temporal_rows: pd.DataFrame,
    scenario_by_user: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Align artifacts and expose exact X matrices separately from metadata."""
    required_base = set(IDENTIFIER_COLUMNS) | set(PHASE8_FEATURE_COLUMNS) | {TARGET_COLUMN}
    required_temporal = set(IDENTIFIER_COLUMNS) | set(TEMPORAL_FEATURE_COLUMNS)
    missing_base = sorted(required_base - set(feature_rows.columns))
    missing_temporal = sorted(required_temporal - set(temporal_rows.columns))
    if missing_base:
        raise ValueError(f"Phase 8 feature artifact missing columns: {missing_base}")
    if missing_temporal:
        raise ValueError(f"temporal feature artifact missing columns: {missing_temporal}")
    for frame, name in ((feature_rows, "Phase 8"), (temporal_rows, "temporal")):
        if frame["user_id"].duplicated().any() or frame["account_id"].duplicated().any():
            raise ValueError(f"{name} artifact must contain unique users/accounts")

    identifiers = feature_rows.loc[:, list(IDENTIFIER_COLUMNS)].copy()
    base = feature_rows.loc[:, [*IDENTIFIER_COLUMNS, TARGET_COLUMN, *PHASE8_FEATURE_COLUMNS]]
    temporal = temporal_rows.loc[:, [*IDENTIFIER_COLUMNS, *TEMPORAL_FEATURE_COLUMNS]]
    aligned = base.merge(
        temporal,
        on=list(IDENTIFIER_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if len(aligned) != len(feature_rows) or not (aligned["_merge"] == "both").all():
        raise ValueError("Phase 8 and temporal artifacts must match exactly by identifiers")
    aligned = aligned.drop(columns="_merge")
    users = set(identifiers["user_id"].astype(str))
    if users != set(scenario_by_user):
        raise ValueError("scenario grouping metadata must match feature users exactly")
    scenarios = identifiers["user_id"].astype(str).map(scenario_by_user).rename(SCENARIO_COLUMN)

    y_numeric = pd.to_numeric(aligned[TARGET_COLUMN], errors="raise")
    if y_numeric.isna().any() or not set(y_numeric.unique()).issubset({0, 1}):
        raise ValueError("ato_label must contain only non-missing binary values")
    y = y_numeric.astype(int).rename(TARGET_COLUMN)
    X_phase8 = aligned.loc[:, list(PHASE8_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise")
    X_phase9 = aligned.loc[:, [*PHASE8_FEATURE_COLUMNS, *TEMPORAL_FEATURE_COLUMNS]].apply(
        pd.to_numeric, errors="raise"
    )
    if X_phase8.isna().any().any() or not np.isfinite(X_phase8.to_numpy(dtype=float)).all():
        raise ValueError("Phase 8 model inputs must be finite and non-missing")
    phase9_values = X_phase9.to_numpy(dtype=float)
    observed_values = phase9_values[~X_phase9.isna().to_numpy()]
    if not np.isfinite(observed_values).all():
        raise ValueError("observed Phase 9 model inputs must be finite")
    return X_phase8, X_phase9, y, identifiers, scenarios


def _phase9_pipeline() -> Pipeline:
    """Use train-fold-only median imputation and missingness indicators."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("classifier", make_classifier()),
    ])


def _positive_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    class_index = list(model.classes_).index(1)
    return model.predict_proba(X)[:, class_index]


def _fold_metrics(
    y_true: pd.Series,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    # binary_metrics reports ROC-AUC only when both classes are present and AP
    # only when positive examples exist; other zero-denominator rates are null.
    return binary_metrics(y_true.to_numpy(), prediction, probability)


def _feature_audit(temporal_rows: pd.DataFrame, scenarios: pd.Series) -> list[dict[str, Any]]:
    """Summarize temporal inputs by scenario only after OOF predictions exist."""
    audit: list[dict[str, Any]] = []
    for scenario_type in sorted(scenarios.unique()):
        positions = np.flatnonzero(scenarios.to_numpy() == scenario_type)
        for feature in TEMPORAL_FEATURE_COLUMNS:
            values = pd.to_numeric(temporal_rows.iloc[positions][feature], errors="raise")
            observed = values.dropna().astype(float)
            audit.append({
                "scenario_type_evaluation_only": scenario_type,
                "feature": feature,
                "row_count": int(len(values)),
                "non_missing_count": int(observed.size),
                "missing_count": int(values.isna().sum()),
                "zero_count": int((observed == 0).sum()),
                "one_count": int((observed == 1).sum()),
                "nonzero_count": int((observed != 0).sum()),
                "nonzero_rate_among_non_missing": float((observed != 0).mean()) if observed.size else None,
                "minimum": float(observed.min()) if observed.size else None,
                "mean": float(observed.mean()) if observed.size else None,
                "median": float(observed.median()) if observed.size else None,
                "maximum": float(observed.max()) if observed.size else None,
            })
    return audit


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def run_unseen_scenario_experiment(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    temporal_path: str | Path = DEFAULT_TEMPORAL_PATH,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run both RF variants on identical leave-one-scenario-type-out folds."""
    features_path, temporal_path, events_path, output_dir = map(
        Path, (features_path, temporal_path, events_path, output_dir)
    )
    feature_rows = pd.read_csv(features_path)
    temporal_rows = pd.read_csv(temporal_path)
    users = set(feature_rows["user_id"].astype(str))
    scenario_by_user = _scenario_types_from_triggers(events_path, users)
    X_phase8, X_phase9, y, identifiers, scenarios = prepare_model_inputs(
        feature_rows, temporal_rows, scenario_by_user
    )
    folds = build_scenario_folds(scenarios)

    n_rows = len(y)
    model_predictions: dict[str, dict[str, np.ndarray]] = {
        "model_a_phase8": {
            "prediction": np.full(n_rows, -1, dtype=int),
            "probability": np.full(n_rows, np.nan, dtype=float),
        },
        "model_b_phase9": {
            "prediction": np.full(n_rows, -1, dtype=int),
            "probability": np.full(n_rows, np.nan, dtype=float),
        },
    }
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]
        train_y = y.iloc[train_idx]
        if len(train_y.unique()) != 2:
            raise ValueError(f"training complement for {fold['held_out_scenario_type']} lacks both classes")

        model_a = make_classifier()
        model_a.fit(X_phase8.iloc[train_idx], train_y)
        pred_a = model_a.predict(X_phase8.iloc[test_idx]).astype(int)
        prob_a = _positive_probabilities(model_a, X_phase8.iloc[test_idx])

        model_b = _phase9_pipeline()
        model_b.fit(X_phase9.iloc[train_idx], train_y)
        pred_b = model_b.predict(X_phase9.iloc[test_idx]).astype(int)
        prob_b = _positive_probabilities(model_b, X_phase9.iloc[test_idx])

        for model_key, predictions, probabilities in (
            ("model_a_phase8", pred_a, prob_a),
            ("model_b_phase9", pred_b, prob_b),
        ):
            if np.any(model_predictions[model_key]["prediction"][test_idx] != -1):
                raise ValueError("a row was assigned predictions more than once")
            model_predictions[model_key]["prediction"][test_idx] = predictions
            model_predictions[model_key]["probability"][test_idx] = probabilities

        fold_records.append({
            "held_out_scenario_type": fold["held_out_scenario_type"],
            "train_scenario_types": fold["train_scenario_types"],
            "test_scenario_types": fold["test_scenario_types"],
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_class_counts": {str(k): int(v) for k, v in train_y.value_counts().sort_index().items()},
            "test_class_counts": {str(k): int(v) for k, v in y.iloc[test_idx].value_counts().sort_index().items()},
            "model_a_phase8": _fold_metrics(y.iloc[test_idx], pred_a, prob_a),
            "model_b_phase9": _fold_metrics(y.iloc[test_idx], pred_b, prob_b),
        })

    for model_key, outputs in model_predictions.items():
        if np.any(outputs["prediction"] == -1) or not np.isfinite(outputs["probability"]).all():
            raise ValueError(f"not every row received exactly one {model_key} prediction")

    truth = y.to_numpy(dtype=int)
    aggregated = {
        key: _fold_metrics(y, outputs["prediction"], outputs["probability"])
        for key, outputs in model_predictions.items()
    }
    prediction_rows: list[dict[str, Any]] = []
    scenario_errors: list[dict[str, Any]] = []
    for i in range(n_rows):
        scenario_type = str(scenarios.iloc[i])
        actual = int(truth[i])
        row: dict[str, Any] = {
            **{column: str(identifiers.iloc[i][column]) for column in IDENTIFIER_COLUMNS},
            "scenario_type_evaluation_only": scenario_type,
            "ato_label_evaluation_only": actual,
            "held_out_scenario_type": scenario_type,
        }
        for key, prefix in (("model_a_phase8", "model_a"), ("model_b_phase9", "model_b")):
            prediction = int(model_predictions[key]["prediction"][i])
            error_type = "FP" if prediction == 1 and actual == 0 else "FN" if prediction == 0 and actual == 1 else "CORRECT"
            row[f"{prefix}_prediction"] = prediction
            row[f"{prefix}_probability_ato"] = float(model_predictions[key]["probability"][i])
            row[f"{prefix}_error_type"] = error_type
        prediction_rows.append(row)

    for scenario_type in sorted(scenarios.unique()):
        indices = np.flatnonzero(scenarios.to_numpy() == scenario_type)
        labels = truth[indices]
        item: dict[str, Any] = {
            "scenario_type_evaluation_only": str(scenario_type),
            "test_rows": int(len(indices)),
            "legitimate_count": int(np.sum(labels == 0)),
            "suspicious_count": int(np.sum(labels == 1)),
        }
        for key, prefix in (("model_a_phase8", "model_a"), ("model_b_phase9", "model_b")):
            preds = model_predictions[key]["prediction"][indices]
            fp = int(np.sum((labels == 0) & (preds == 1)))
            fn = int(np.sum((labels == 1) & (preds == 0)))
            correct = int(np.sum(labels == preds))
            item.update({
                f"{prefix}_false_positives": fp,
                f"{prefix}_false_negatives": fn,
                f"{prefix}_correct_predictions": correct,
                f"{prefix}_errors": fp + fn,
            })
        scenario_errors.append(item)

    # Audit is computed after all held-out predictions, grouped for descriptive
    # template association only; it does not influence fitting or evaluation.
    audit_temporal_rows = X_phase9.loc[:, list(TEMPORAL_FEATURE_COLUMNS)]
    audit_rows = _feature_audit(audit_temporal_rows, scenarios)
    feature_delta = {
        metric: aggregated["model_b_phase9"][metric] - aggregated["model_a_phase8"][metric]
        if aggregated["model_b_phase9"][metric] is not None and aggregated["model_a_phase8"][metric] is not None
        else None
        for metric in ("precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc_average_precision")
    }
    metrics = {
        "evaluation": "leave-one-scenario-type-out, aggregated out-of-fold",
        "positive_class": 1,
        "negative_class": 0,
        "aggregated_out_of_fold": aggregated,
        "phase9_minus_phase8": feature_delta,
        "per_held_out_scenario_type": fold_records,
    }
    scenario_class_counts = {
        str(scenario): {str(k): int(v) for k, v in y[scenarios == scenario].value_counts().sort_index().items()}
        for scenario in sorted(scenarios.unique())
    }
    metadata = {
        "experiment": "unseen-scenario generalization: leave one scenario_type out",
        "row_count": n_rows,
        "scenario_type_count": len(folds),
        "scenario_class_counts": scenario_class_counts,
        "scenario_type_role": "grouping and post-prediction evaluation metadata only; excluded from X",
        "target_column": TARGET_COLUMN,
        "identifier_columns_not_in_X": list(IDENTIFIER_COLUMNS),
        "phase8_feature_columns": list(PHASE8_FEATURE_COLUMNS),
        "temporal_feature_columns": list(TEMPORAL_FEATURE_COLUMNS),
        "model_a_X_columns": list(PHASE8_FEATURE_COLUMNS),
        "model_b_X_columns": [*PHASE8_FEATURE_COLUMNS, *TEMPORAL_FEATURE_COLUMNS],
        "excluded_fields": [TARGET_COLUMN, SCENARIO_COLUMN, *IDENTIFIER_COLUMNS,
                            "paysim_isFraud", "paysim_isFlaggedFraud", "Phase 7 predictions"],
        "folds": [{
            "held_out_scenario_type": fold["held_out_scenario_type"],
            "train_scenario_types": fold["train_scenario_types"],
            "test_scenario_types": fold["test_scenario_types"],
            "train_rows": int(len(fold["train_indices"])),
            "test_rows": int(len(fold["test_indices"])),
        } for fold in folds],
        "random_forest_parameters": MODEL_PARAMETERS,
        "effective_joblib_cpu_count": cpu_count(),
        "phase9_preprocessing": "median imputer with missingness indicators fit separately on each training fold; raw temporal CSV missingness retained",
        "temporal_snapshot": "existing Phase 9 T through T+15 inclusive artifact; this experiment reads no events to create inputs",
        "scenario_metadata_source": "only scenario_type from each TELECOM trigger row in the existing event dataset",
        "input_sha256": {
            "phase8_features": _sha256(features_path),
            "phase9_temporal_features": _sha256(temporal_path),
            "scenario_group_metadata_source": _sha256(events_path),
        },
        "software_versions": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "Each current scenario_type contains only one ato_label class, so many per-fold class metrics are undefined.",
            "The eight scenario types are synthetic templates and do not represent independent real-world populations.",
            "Feature audit distributions describe association with templates and are not causal evidence.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_csv(
        output_dir / "predictions.csv", prediction_rows,
        (*IDENTIFIER_COLUMNS, "scenario_type_evaluation_only", "ato_label_evaluation_only",
         "held_out_scenario_type", "model_a_prediction", "model_a_probability_ato", "model_a_error_type",
         "model_b_prediction", "model_b_probability_ato", "model_b_error_type"),
    )
    _write_csv(
        output_dir / "scenario_error_summary.csv", scenario_errors,
        ("scenario_type_evaluation_only", "test_rows", "legitimate_count", "suspicious_count",
         "model_a_false_positives", "model_a_false_negatives", "model_a_correct_predictions", "model_a_errors",
         "model_b_false_positives", "model_b_false_negatives", "model_b_correct_predictions", "model_b_errors"),
    )
    audit_columns = (
        "scenario_type_evaluation_only", "feature", "row_count", "non_missing_count", "missing_count",
        "zero_count", "one_count", "nonzero_count", "nonzero_rate_among_non_missing",
        "minimum", "mean", "median", "maximum",
    )
    _write_csv(output_dir / "feature_audit.csv", audit_rows, audit_columns)
    return {
        "metrics": metrics,
        "predictions": prediction_rows,
        "scenario_error_summary": scenario_errors,
        "feature_audit": audit_rows,
        "metadata": metadata,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unseen-scenario Phase 8/9 generalization evaluation.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--temporal-features", type=Path, default=DEFAULT_TEMPORAL_PATH)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_unseen_scenario_experiment(args.features, args.temporal_features, args.events, args.output_dir)
    print(json.dumps({
        "output_dir": str(result["output_dir"]),
        "aggregated_out_of_fold": result["metrics"]["aggregated_out_of_fold"],
        "scenario_type_count": result["metadata"]["scenario_type_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
