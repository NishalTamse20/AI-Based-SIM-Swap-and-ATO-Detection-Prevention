"""Evaluate two fixed Phase 10.3 policies on saved Phase 8/9 OOF predictions.

This module never fits a model. Phase 7 rules are regenerated deterministically
from frozen Phase 6 features; saved leave-one-scenario-type-out predictions
provide the model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ml.models.random_forest_experiment import (
    DEFAULT_FEATURES_PATH,
    IDENTIFIER_COLUMNS,
    binary_metrics,
)
from ml.models.unseen_scenario_experiment import DEFAULT_OUTPUT_DIR as UNSEEN_DIR
from ml.models.temporal_ablation_experiment import DEFAULT_OUTPUT_DIR as ABLATION_DIR
from ml.features.temporal_features import FEATURE_COLUMNS as TEMPORAL_COLUMNS
from ml.rules.rule_baseline import (
    AMBIGUOUS,
    ATO_RISK_INDICATOR,
    NO_INDICATOR,
    generate_rule_baselines,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPORAL_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_features.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "hybrid_policy_evaluation"
KEYS = tuple(IDENTIFIER_COLUMNS)
RULE_FEATURES = (
    "telecom_event_present", "sim_change", "esim_change",
    "new_device", "authentication_anomaly", "password_reset", "failed_login_count",
    "device_change", "device_deviation",
)
REPLACEMENT_SCENARIO = "SUSPICIOUS_SIM_REPLACEMENT"
EXPECTED_REPLACEMENT_ROWS = 125

POLICY_DEFINITIONS = {
    "POLICY_1_RF_LED": {
        "decision_table": {
            ATO_RISK_INDICATOR: {"1": "ATO_RISK_INDICATOR", "0": "CONFLICTING_EVIDENCE", "MISSING": "INSUFFICIENT_EVIDENCE"},
            NO_INDICATOR: {"1": "MODEL_ONLY_RISK_INDICATOR", "0": NO_INDICATOR, "MISSING": "INSUFFICIENT_EVIDENCE"},
            AMBIGUOUS: {"1": AMBIGUOUS, "0": AMBIGUOUS, "MISSING": AMBIGUOUS},
            "RULE_MISSING": {"1": "INSUFFICIENT_EVIDENCE", "0": "INSUFFICIENT_EVIDENCE", "MISSING": "INSUFFICIENT_EVIDENCE"},
        },
    },
    "POLICY_2_RULE_CORROBORATION": {
        "decision_table": {
            ATO_RISK_INDICATOR: {"1": "ATO_RISK_INDICATOR", "0": "CONFLICTING_EVIDENCE", "MISSING": "INSUFFICIENT_EVIDENCE"},
            NO_INDICATOR: {"1": AMBIGUOUS, "0": NO_INDICATOR, "MISSING": "INSUFFICIENT_EVIDENCE"},
            AMBIGUOUS: {"1": AMBIGUOUS, "0": AMBIGUOUS, "MISSING": AMBIGUOUS},
            "RULE_MISSING": {"1": "INSUFFICIENT_EVIDENCE", "0": "INSUFFICIENT_EVIDENCE", "MISSING": "INSUFFICIENT_EVIDENCE"},
        },
    },
}

_DETERMINATE_POSITIVE = {"ATO_RISK_INDICATOR", "MODEL_ONLY_RISK_INDICATOR"}
_DETERMINATE = _DETERMINATE_POSITIVE | {"NO_ATO_RISK_INDICATOR"}
_UNCERTAIN = {
    "AMBIGUOUS_INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE", "INSUFFICIENT_EVIDENCE",
}


def apply_policy_1(rule_outcome: str | None, model_prediction: int | None) -> str:
    """RF-led policy; keep rule ambiguity/conflict and model-only positives explicit."""
    if rule_outcome == AMBIGUOUS:
        return "AMBIGUOUS_INSUFFICIENT_EVIDENCE"
    if rule_outcome not in (ATO_RISK_INDICATOR, NO_INDICATOR, None):
        raise ValueError(f"unsupported Phase 7 outcome: {rule_outcome!r}")
    if model_prediction not in (0, 1, None) or isinstance(model_prediction, bool):
        raise ValueError(f"model prediction must be 0, 1, or None; got {model_prediction!r}")
    if rule_outcome is None or model_prediction is None:
        return "INSUFFICIENT_EVIDENCE"
    if rule_outcome == ATO_RISK_INDICATOR:
        return "ATO_RISK_INDICATOR" if model_prediction == 1 else "CONFLICTING_EVIDENCE"
    return "MODEL_ONLY_RISK_INDICATOR" if model_prediction == 1 else "NO_ATO_RISK_INDICATOR"


def apply_policy_2(rule_outcome: str | None, model_prediction: int | None) -> str:
    """Require rule corroboration for a model positive; retain unresolved cases."""
    if rule_outcome == AMBIGUOUS:
        return "AMBIGUOUS_INSUFFICIENT_EVIDENCE"
    if rule_outcome not in (ATO_RISK_INDICATOR, NO_INDICATOR, None):
        raise ValueError(f"unsupported Phase 7 outcome: {rule_outcome!r}")
    if model_prediction not in (0, 1, None) or isinstance(model_prediction, bool):
        raise ValueError(f"model prediction must be 0, 1, or None; got {model_prediction!r}")
    if rule_outcome is None or model_prediction is None:
        return "INSUFFICIENT_EVIDENCE"
    if rule_outcome == ATO_RISK_INDICATOR:
        return "ATO_RISK_INDICATOR" if model_prediction == 1 else "CONFLICTING_EVIDENCE"
    return "AMBIGUOUS_INSUFFICIENT_EVIDENCE" if model_prediction == 1 else "NO_ATO_RISK_INDICATOR"


def _validate_key_frame(frame: pd.DataFrame, name: str) -> set[tuple[str, ...]]:
    missing = sorted(set(KEYS) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing identifier columns: {missing}")
    if frame.loc[:, list(KEYS)].isna().any().any():
        raise ValueError(f"{name} has missing identifier values")
    keys_frame = frame.loc[:, list(KEYS)].astype(str)
    if keys_frame.duplicated().any():
        duplicate = keys_frame.loc[keys_frame.duplicated(keep=False)].iloc[0].to_dict()
        raise ValueError(f"{name} has duplicate (user_id, account_id, event_id) key: {duplicate}")
    return set(map(tuple, keys_frame.to_numpy().tolist()))


def _require_matching_keys(frames: dict[str, pd.DataFrame]) -> set[tuple[str, ...]]:
    key_sets = {name: _validate_key_frame(frame, name) for name, frame in frames.items()}
    reference_name = next(iter(frames))
    reference = key_sets[reference_name]
    for name, keys in key_sets.items():
        if keys != reference:
            missing = sorted(reference - keys)[:5]
            extra = sorted(keys - reference)[:5]
            raise ValueError(
                f"{name} key-set mismatch against {reference_name}: "
                f"row_count={len(frames[name])} expected={len(reference)}, "
                f"missing_keys={missing}, extra_keys={extra}"
            )
    return reference


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _same_values(left: pd.Series, right: pd.Series, name: str) -> None:
    a = pd.to_numeric(left, errors="raise").to_numpy(dtype=float)
    b = pd.to_numeric(right, errors="raise").to_numpy(dtype=float)
    if not np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError(f"Phase 9B ALL_TEMPORAL {name} does not match saved unseen-scenario Model B")


def _validate_label_fold_alignment(evaluation: pd.DataFrame,
                                   comparisons: dict[str, pd.DataFrame]) -> None:
    """Check evaluation metadata only after policy outcomes have been produced."""
    target = pd.to_numeric(evaluation["ato_label"], errors="raise")
    saved_target = pd.to_numeric(evaluation["ato_label_evaluation_only"], errors="raise")
    if target.isna().any() or not target.isin([0, 1]).all() or not target.astype(int).equals(saved_target.astype(int)):
        raise ValueError("ato_label alignment mismatch after policy prediction; no rows were dropped")
    if not (evaluation["scenario_type_evaluation_only"] == evaluation["held_out_scenario_type"]).all():
        raise ValueError("held-out fold is misaligned with scenario_type evaluation metadata")
    for comparison_name, comparison in comparisons.items():
        compare = evaluation.loc[:, [*KEYS, "actual_label_evaluation_only", "scenario_type_evaluation_only",
                                     "held_out_scenario_type"]].merge(
            comparison.loc[:, [*KEYS, "actual_label", "scenario_type", "held_out_scenario_type"]],
            on=list(KEYS), how="left", validate="one_to_one", sort=False, suffixes=("_eval", "_saved"),
        )
        _same_values(compare["actual_label_evaluation_only"], compare["actual_label"],
                     f"{comparison_name} evaluation label")
        if not (compare["scenario_type_evaluation_only"] == compare["scenario_type"]).all():
            raise ValueError(f"{comparison_name} scenario metadata is misaligned")
        if not (compare["held_out_scenario_type_eval"] == compare["held_out_scenario_type_saved"]).all():
            raise ValueError(f"{comparison_name} row-to-fold assignment is misaligned")


def _validate_fold_records(evaluation: pd.DataFrame, folds: list[dict[str, Any]]) -> None:
    scenarios = set(evaluation["scenario_type_evaluation_only"].astype(str))
    if len(folds) != 8 or {fold["held_out_scenario_type"] for fold in folds} != scenarios:
        raise ValueError("fold records do not cover the eight observed scenario types exactly")
    for fold in folds:
        held_out = fold["held_out_scenario_type"]
        rows = evaluation[evaluation["held_out_scenario_type"] == held_out]
        expected_train = sorted(scenarios - {held_out})
        if len(rows) != int(fold["test_rows"]):
            raise ValueError(f"held-out row count mismatch for fold {held_out}")
        if sorted(set(rows["scenario_type_evaluation_only"].astype(str))) != [held_out]:
            raise ValueError(f"held-out scenario mismatch for fold {held_out}")
        if sorted(fold["train_scenario_types"]) != expected_train:
            raise ValueError(f"training scenario assignment mismatch for fold {held_out}")
        if sorted(fold["test_scenario_types"]) != [held_out]:
            raise ValueError(f"test scenario assignment mismatch for fold {held_out}")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def _hard_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "n": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "precision": None, "recall": None, "f1": None, "fpr": None, "fnr": None,
            "confusion_matrix": [[0, 0], [0, 0]], "confusion_matrix_labels": [0, 1],
            "roc_auc": None, "average_precision": None,
            "auc_note": "No determinate rows are available for this subset.",
        }
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "n": int(len(y_true)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "fpr": _ratio(fp, fp + tn),
        "fnr": _ratio(fn, fn + tp),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": [0, 1],
        "roc_auc": None,
        "average_precision": None,
        "auc_note": "Policy emits categorical outcomes, not a continuous score.",
    }


def _reference_metrics(rows: pd.DataFrame, prediction_column: str, probability_column: str) -> dict[str, Any]:
    return binary_metrics(
        rows["actual_label_evaluation_only"].to_numpy(dtype=int),
        rows[prediction_column].to_numpy(dtype=int),
        rows[probability_column].to_numpy(dtype=float),
    )


def _policy_details(rows: pd.DataFrame, outcome_column: str) -> dict[str, Any]:
    outcomes = rows[outcome_column].astype(str)
    determinate = outcomes.isin(_DETERMINATE).to_numpy()
    truth = rows["actual_label_evaluation_only"].to_numpy(dtype=int)
    binary_prediction = np.asarray([int(value in _DETERMINATE_POSITIVE) for value in outcomes], dtype=int)
    determinate_metrics = _hard_metrics(truth[determinate], binary_prediction[determinate])
    outcome_counts: dict[str, Any] = {}
    for outcome in sorted(set(outcomes)):
        mask = (outcomes == outcome).to_numpy()
        outcome_counts[outcome] = {
            "count": int(mask.sum()),
            "actual_label_counts": {"0": int(np.sum(truth[mask] == 0)), "1": int(np.sum(truth[mask] == 1))},
        }
    return {
        "row_count": int(len(rows)),
        "determinate_count": int(determinate.sum()),
        "coverage": _ratio(int(determinate.sum()), len(rows)),
        "coverage_definition": "determinate policy outcomes divided by all expected joined rows",
        "determinate_metrics": determinate_metrics,
        "outcome_counts_and_label_composition": outcome_counts,
        "uncertain_counts": {key: outcome_counts.get(key, {"count": 0, "actual_label_counts": {"0": 0, "1": 0}})
                              for key in sorted(_UNCERTAIN)},
        "metric_mapping": (
            "For determinate-case evaluation only, risk-indicator outcomes map to 1 and "
            "NO_ATO_RISK_INDICATOR maps to 0. This is not a legitimate-account classification."
        ),
        "determinate_mask": determinate,
    }


def _scenario_summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scenario in sorted(rows["scenario_type_evaluation_only"].unique()):
        subset = rows[rows["scenario_type_evaluation_only"] == scenario]
        truth = subset["actual_label_evaluation_only"].to_numpy(dtype=int)
        item: dict[str, Any] = {
            "scenario_type_evaluation_only": scenario,
            "rows": int(len(subset)),
            "legitimate_count": int(np.sum(truth == 0)),
            "suspicious_count": int(np.sum(truth == 1)),
            "phase8_false_positives": int(((subset["phase8_error"] == "FP")).sum()),
            "phase8_false_negatives": int(((subset["phase8_error"] == "FN")).sum()),
            "phase9_false_positives": int(((subset["phase9_error"] == "FP")).sum()),
            "phase9_false_negatives": int(((subset["phase9_error"] == "FN")).sum()),
        }
        for name, column in (("policy_1", "policy_1_outcome"), ("policy_2", "policy_2_outcome")):
            outcomes = subset[column].astype(str)
            mask = outcomes.isin(_DETERMINATE).to_numpy()
            pred = np.asarray([int(value in _DETERMINATE_POSITIVE) for value in outcomes], dtype=int)
            item[f"{name}_coverage"] = _ratio(int(mask.sum()), len(subset))
            item[f"{name}_determinate_count"] = int(mask.sum())
            item[f"{name}_false_positives"] = int(np.sum(mask & (pred == 1) & (truth == 0)))
            item[f"{name}_false_negatives"] = int(np.sum(mask & (pred == 0) & (truth == 1)))
            item[f"{name}_errors"] = item[f"{name}_false_positives"] + item[f"{name}_false_negatives"]
            item[f"{name}_ambiguous"] = int((outcomes == "AMBIGUOUS_INSUFFICIENT_EVIDENCE").sum())
            item[f"{name}_conflicting"] = int((outcomes == "CONFLICTING_EVIDENCE").sum())
            item[f"{name}_insufficient"] = int((outcomes == "INSUFFICIENT_EVIDENCE").sum())
        result.append(item)
    return result


def _scenario_metrics(rows: pd.DataFrame, outcome_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scenario in sorted(rows["scenario_type_evaluation_only"].unique()):
        subset = rows[rows["scenario_type_evaluation_only"] == scenario]
        actual = subset["actual_label_evaluation_only"].to_numpy(dtype=int)
        outcomes = subset[outcome_column].astype(str)
        mask = outcomes.isin(_DETERMINATE).to_numpy()
        pred = np.asarray([int(value in _DETERMINATE_POSITIVE) for value in outcomes], dtype=int)
        result[scenario] = {
            "rows": int(len(subset)),
            "determinate_rows": int(mask.sum()),
            "coverage": _ratio(int(mask.sum()), len(subset)),
            "determinate_metrics": _hard_metrics(actual[mask], pred[mask]),
            "uncertain_outcome_counts": {
                key: int((outcomes == key).sum()) for key in sorted(_UNCERTAIN)
            },
        }
    return result


def _write_outputs(output_dir: Path, metrics: dict[str, Any], predictions: pd.DataFrame,
                   scenario_summary: list[dict[str, Any]], special: pd.DataFrame,
                   metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False, lineterminator="\n")
    pd.DataFrame(scenario_summary).to_csv(output_dir / "scenario_error_summary.csv", index=False, lineterminator="\n")
    special.to_csv(output_dir / "suspicious_sim_replacement_cases.csv", index=False, lineterminator="\n")
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def run_hybrid_policy_evaluation(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    temporal_path: str | Path = DEFAULT_TEMPORAL_PATH,
    unseen_dir: str | Path = UNSEEN_DIR,
    ablation_dir: str | Path = ABLATION_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Evaluate fixed policies using saved OOF predictions; never fit a model."""
    features_path, temporal_path = Path(features_path), Path(temporal_path)
    unseen_dir, ablation_dir, output_dir = Path(unseen_dir), Path(ablation_dir), Path(output_dir)
    protected = [PROJECT_ROOT / "data" / "processed" / name for name in
                 ("features.csv", "temporal_features.csv", "random_forest", "unseen_scenario", "temporal_ablation")]
    resolved_output = output_dir.resolve()
    for path in protected:
        p = path.resolve()
        if resolved_output == p or p in resolved_output.parents or resolved_output in p.parents:
            raise ValueError(f"output directory overlaps frozen Phase 5–9 artifact: {p}")

    features = pd.read_csv(features_path)
    temporal = pd.read_csv(temporal_path)
    unseen = pd.read_csv(unseen_dir / "predictions.csv")
    ablation = pd.read_csv(ablation_dir / "predictions.csv")
    unseen_metadata = json.loads((unseen_dir / "run_metadata.json").read_text(encoding="utf-8"))
    ablation_metadata = json.loads((ablation_dir / "run_metadata.json").read_text(encoding="utf-8"))
    all_temporal = ablation[ablation["experiment_name"] == "ALL_TEMPORAL"].copy()
    baseline_ablation = ablation[ablation["experiment_name"] == "BASELINE"].copy()

    _require_columns(features, set(KEYS) | set(RULE_FEATURES) | {"ato_label"}, "Phase 6 features")
    _require_columns(temporal, set(KEYS) | set(TEMPORAL_COLUMNS), "Phase 9 temporal features")
    _require_columns(unseen, set(KEYS) | {
        "scenario_type_evaluation_only", "ato_label_evaluation_only", "held_out_scenario_type",
        "model_a_prediction", "model_a_probability_ato", "model_b_prediction", "model_b_probability_ato",
    }, "unseen-scenario predictions")
    _require_columns(all_temporal, set(KEYS) | {
        "scenario_type", "actual_label", "held_out_scenario_type", "predicted_label", "prediction_probability",
    }, "Phase 9B ALL_TEMPORAL predictions")
    _require_columns(baseline_ablation, set(KEYS) | {"predicted_label", "prediction_probability"},
                     "Phase 9B BASELINE predictions")
    if len(unseen_metadata.get("folds", [])) != 8 or len(ablation_metadata.get("folds", [])) != 8:
        raise ValueError("expected the existing eight leave-one-scenario-type-out folds")
    fold_fields = ("held_out_scenario_type", "train_scenario_types", "test_scenario_types", "train_rows", "test_rows")
    unseen_folds = [{key: fold[key] for key in fold_fields} for fold in unseen_metadata["folds"]]
    ablation_folds = [{key: fold[key] for key in fold_fields} for fold in ablation_metadata["folds"]]
    if unseen_folds != ablation_folds:
        raise ValueError("Phase 8/9 and Phase 9B saved fold assignments differ")

    # Select one row per source before join; the complete key is always used.
    sources = {
        "Phase 6 features": features,
        "Phase 9 temporal features": temporal,
        "unseen-scenario predictions": unseen,
        "Phase 9B ALL_TEMPORAL predictions": all_temporal,
        "Phase 9B BASELINE predictions": baseline_ablation,
    }
    key_set = _require_matching_keys(sources)
    expected_rows = len(key_set)
    if expected_rows != len(features) or expected_rows != len(unseen):
        raise ValueError("input row counts differ after exact-key validation")

    # Phase 7 and policy outputs are constructed without target/scenario columns.
    rule_input = features.loc[:, [*KEYS, *RULE_FEATURES]].to_dict(orient="records")
    rules = generate_rule_baselines(rule_input)["multi_signal"]
    rule_frame = pd.DataFrame([{
        **{key: row[key] for key in KEYS},
        "phase7_prediction": row["prediction"],
        "phase7_matched_rules": json.dumps(row["matched_rules"], separators=(",", ":")),
        "phase7_feature_evidence": json.dumps(row["feature_evidence"], sort_keys=True, separators=(",", ":")),
        "phase7_explanation": row["explanation"],
    } for row in rules])
    model_inputs = unseen.loc[:, [*KEYS, "model_a_prediction", "model_a_probability_ato",
                                  "model_b_prediction", "model_b_probability_ato"]]
    for column in ("model_a_prediction", "model_b_prediction"):
        values = pd.to_numeric(model_inputs[column], errors="raise")
        if values.isna().any() or not values.isin([0, 1]).all():
            raise ValueError(f"saved {column} must contain only non-missing binary predictions")
        model_inputs[column] = values.astype(int)
    for column in ("model_a_probability_ato", "model_b_probability_ato"):
        values = pd.to_numeric(model_inputs[column], errors="raise")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all() or not values.between(0, 1).all():
            raise ValueError(f"saved {column} must be finite and between 0 and 1")
        model_inputs[column] = values.astype(float)
    outcomes = rule_frame.merge(model_inputs, on=list(KEYS), how="left", validate="one_to_one", sort=False)
    outcomes["policy_1_outcome"] = [
        apply_policy_1(rule, int(model)) for rule, model in
        zip(outcomes["phase7_prediction"], outcomes["model_b_prediction"])
    ]
    outcomes["policy_2_outcome"] = [
        apply_policy_2(rule, int(model)) for rule, model in
        zip(outcomes["phase7_prediction"], outcomes["model_b_prediction"])
    ]

    # Validate that all saved Phase 9B predictions reproduce the corresponding
    # saved unseen-scenario predictions before attaching evaluation labels.
    unseen_ordered = unseen.set_index(list(KEYS)).sort_index()
    temporal_ordered = all_temporal.set_index(list(KEYS)).reindex(unseen_ordered.index)
    baseline_ordered = baseline_ablation.set_index(list(KEYS)).reindex(unseen_ordered.index)
    _same_values(unseen_ordered["model_b_prediction"], temporal_ordered["predicted_label"], "Model B prediction")
    _same_values(unseen_ordered["model_b_probability_ato"], temporal_ordered["prediction_probability"], "Model B probability")
    _same_values(unseen_ordered["model_a_prediction"], baseline_ordered["predicted_label"], "Model A prediction")
    _same_values(unseen_ordered["model_a_probability_ato"], baseline_ordered["prediction_probability"], "Model A probability")

    # Only now attach labels and scenario/fold metadata for evaluation.
    evaluation = outcomes.merge(
        features.loc[:, [*KEYS, "ato_label"]], on=list(KEYS), how="left", validate="one_to_one", sort=False,
    ).merge(
        unseen.loc[:, [*KEYS, "scenario_type_evaluation_only", "ato_label_evaluation_only", "held_out_scenario_type",
                       "model_a_error_type", "model_b_error_type"]],
        on=list(KEYS), how="left", validate="one_to_one", sort=False,
    )
    evaluation["actual_label_evaluation_only"] = pd.to_numeric(evaluation["ato_label"], errors="raise").astype(int)
    _validate_label_fold_alignment(evaluation, {"ALL_TEMPORAL": all_temporal, "BASELINE": baseline_ablation})
    for comparison_name, comparison in (("ALL_TEMPORAL", all_temporal), ("BASELINE", baseline_ablation)):
        compare = evaluation.loc[:, [*KEYS, "actual_label_evaluation_only", "scenario_type_evaluation_only",
                                     "held_out_scenario_type"]].merge(
            comparison.loc[:, [*KEYS, "actual_label", "scenario_type", "held_out_scenario_type"]],
            on=list(KEYS), how="left", validate="one_to_one", sort=False, suffixes=("_eval", "_saved"),
        )
        _same_values(compare["actual_label_evaluation_only"], compare["actual_label"],
                     f"{comparison_name} evaluation label")
        if not (compare["scenario_type_evaluation_only"] == compare["scenario_type"]).all():
            raise ValueError(f"{comparison_name} scenario metadata is misaligned")
        if not (compare["held_out_scenario_type_eval"] == compare["held_out_scenario_type_saved"]).all():
            raise ValueError(f"{comparison_name} row-to-fold assignment is misaligned")
    scenario_types = sorted(evaluation["scenario_type_evaluation_only"].astype(str).unique())
    if len(scenario_types) != 8 or scenario_types != sorted(
        fold["held_out_scenario_type"] for fold in unseen_metadata["folds"]
    ):
        raise ValueError("scenario metadata does not match the eight saved held-out folds")
    _validate_fold_records(evaluation, unseen_metadata["folds"])

    # Attach temporal values for traceability only; individual nulls remain null.
    evaluation = evaluation.merge(temporal.loc[:, [*KEYS, *TEMPORAL_COLUMNS]],
                                  on=list(KEYS), how="left", validate="one_to_one", sort=False)
    if evaluation[list(TEMPORAL_COLUMNS)].isna().to_numpy().sum() != temporal[list(TEMPORAL_COLUMNS)].isna().to_numpy().sum():
        raise ValueError("temporal null values changed during key-based output join")
    evaluation = evaluation.rename(columns={
        "model_a_prediction": "phase8_prediction",
        "model_a_probability_ato": "phase8_probability_ato",
        "model_b_prediction": "phase9_prediction",
        "model_b_probability_ato": "phase9_probability_ato",
    })
    evaluation["phase8_error"] = np.where(
        evaluation["phase8_prediction"].astype(int) == evaluation["actual_label_evaluation_only"], "CORRECT",
        np.where(evaluation["phase8_prediction"].astype(int) == 1, "FP", "FN"),
    )
    evaluation["phase9_error"] = np.where(
        evaluation["phase9_prediction"].astype(int) == evaluation["actual_label_evaluation_only"], "CORRECT",
        np.where(evaluation["phase9_prediction"].astype(int) == 1, "FP", "FN"),
    )
    for policy, column in (("policy_1", "policy_1_outcome"), ("policy_2", "policy_2_outcome")):
        is_det = evaluation[column].isin(_DETERMINATE)
        pred = evaluation[column].isin(_DETERMINATE_POSITIVE).astype(int)
        evaluation[f"{policy}_error"] = np.where(
            ~is_det, "NOT_DETERMINATE",
            np.where(pred == evaluation["actual_label_evaluation_only"], "CORRECT",
                     np.where(pred == 1, "FP", "FN")),
        )

    full_refs = {
        "model_a_phase8": _reference_metrics(evaluation, "phase8_prediction", "phase8_probability_ato"),
        "model_b_phase9": _reference_metrics(evaluation, "phase9_prediction", "phase9_probability_ato"),
    }
    policy_outputs: dict[str, Any] = {}
    for policy, outcome_column in (("POLICY_1_RF_LED", "policy_1_outcome"),
                                   ("POLICY_2_RULE_CORROBORATION", "policy_2_outcome")):
        detail = _policy_details(evaluation, outcome_column)
        mask = detail.pop("determinate_mask")
        same_subset = evaluation.loc[mask]
        detail["reference_model_metrics_on_same_determinate_subset"] = {
            "model_a_phase8": _reference_metrics(same_subset, "phase8_prediction", "phase8_probability_ato"),
            "model_b_phase9": _reference_metrics(same_subset, "phase9_prediction", "phase9_probability_ato"),
        }
        detail["per_scenario_type"] = _scenario_metrics(evaluation, outcome_column)
        policy_outputs[policy] = detail

    special = evaluation[evaluation["scenario_type_evaluation_only"] == REPLACEMENT_SCENARIO].copy()
    if len(special) != EXPECTED_REPLACEMENT_ROWS:
        raise ValueError(f"expected {EXPECTED_REPLACEMENT_ROWS} {REPLACEMENT_SCENARIO} rows; found {len(special)}")
    if not (special["actual_label_evaluation_only"] == 1).all():
        raise ValueError(f"{REPLACEMENT_SCENARIO} label composition differs from frozen evaluation")
    if not (special["phase9_prediction"].astype(int) == 0).all():
        raise ValueError("frozen Phase 9 all-125-missed finding no longer matches the saved predictions")

    metrics = {
        "evaluation": "fixed policies over existing eight-fold leave-one-scenario-type-out predictions",
        "row_count": int(len(evaluation)),
        "full_set_reference_metrics": full_refs,
        "policies": policy_outputs,
        "fold_metric_limitation": (
            "Each held-out scenario fold contains one class; per-fold ROC-AUC is undefined. "
            "Pooled OOF probabilities come from separately fit fold models, are not calibrated, "
            "and may not share comparable scales. Pooled AUC/AP are descriptive only."
        ),
        "phase9_note": "Model B already includes Phase 8 inputs plus the 13 Phase 9 temporal features; it is the sole learned policy signal.",
    }
    scenario_summary = _scenario_summary(evaluation)
    special_columns = [*KEYS, "actual_label_evaluation_only", "scenario_type_evaluation_only",
                       "held_out_scenario_type", "phase7_prediction", "phase7_matched_rules",
                       "phase8_prediction", "phase8_probability_ato", "phase8_error",
                       "phase9_prediction", "phase9_probability_ato", "phase9_error",
                       "policy_1_outcome", "policy_1_error", "policy_2_outcome", "policy_2_error"]
    metadata = {
        "experiment": "Phase 10.3 hybrid policy evaluation",
        "model_fitting": False,
        "row_count": int(len(evaluation)),
        "expected_replacement_rows": EXPECTED_REPLACEMENT_ROWS,
        "replacement_rows_exported": int(len(special)),
        "unique_key": list(KEYS),
        "fold_count": 8,
        "fold_assignments": unseen_metadata["folds"],
        "policy_definitions": POLICY_DEFINITIONS,
        "phase9b_all_temporal_matches_phase9_model_b": True,
        "random_forest_parameters": unseen_metadata.get("random_forest_parameters"),
        "phase9_preprocessing": unseen_metadata.get("phase9_preprocessing"),
        "prediction_threshold": "saved RandomForestClassifier predict output; no custom threshold",
        "target_and_scenario_usage": "ato_label and scenario metadata are attached only after policy outcomes are constructed; evaluation metadata only",
        "metric_convention": metrics["policies"]["POLICY_1_RF_LED"]["metric_mapping"],
        "input_sha256": {
            "phase6_features": _hash(features_path),
            "phase9_temporal_features": _hash(temporal_path),
            "phase8_phase9_unseen_predictions": _hash(unseen_dir / "predictions.csv"),
            "phase8_phase9_unseen_metadata": _hash(unseen_dir / "run_metadata.json"),
            "phase9b_ablation_predictions": _hash(ablation_dir / "predictions.csv"),
            "phase9b_ablation_metadata": _hash(ablation_dir / "run_metadata.json"),
        },
        "input_row_counts": {name: int(len(frame)) for name, frame in sources.items()},
        "output_schemas": {
            "predictions.csv": [*KEYS, "phase7_prediction", "phase7_matched_rules", "phase7_feature_evidence",
                                "phase7_explanation",
                                "phase8_prediction", "phase8_probability_ato", "phase9_prediction",
                                "phase9_probability_ato", "policy_1_outcome", "policy_1_error",
                                "policy_2_outcome", "policy_2_error", "actual_label_evaluation_only",
                                "scenario_type_evaluation_only", "held_out_scenario_type", "phase8_error",
                                "phase9_error", *TEMPORAL_COLUMNS],
            "suspicious_sim_replacement_cases.csv": special_columns,
        },
        "phase9_replacement_failure": {
            "scenario_type": REPLACEMENT_SCENARIO,
            "rows": int(len(special)),
            "phase9_false_negatives": int(((special["phase9_prediction"].astype(int) == 0) &
                                            (special["actual_label_evaluation_only"] == 1)).sum()),
        },
    }
    ordered_columns = [*KEYS, "phase7_prediction", "phase7_matched_rules", "phase7_feature_evidence",
                       "phase7_explanation", "phase8_prediction", "phase8_probability_ato", "phase9_prediction",
                       "phase9_probability_ato", "policy_1_outcome", "policy_1_error", "policy_2_outcome",
                       "policy_2_error", "actual_label_evaluation_only", "scenario_type_evaluation_only",
                       "held_out_scenario_type", "phase8_error", "phase9_error", *TEMPORAL_COLUMNS]
    _write_outputs(output_dir, metrics, evaluation.loc[:, ordered_columns], scenario_summary,
                   special.loc[:, special_columns], metadata)
    return {"metrics": metrics, "predictions": evaluation, "scenario_error_summary": scenario_summary,
            "replacement_cases": special, "metadata": metadata, "output_dir": output_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed Phase 10.3 policies on saved OOF predictions.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--temporal-features", type=Path, default=DEFAULT_TEMPORAL_PATH)
    parser.add_argument("--unseen-dir", type=Path, default=UNSEEN_DIR)
    parser.add_argument("--ablation-dir", type=Path, default=ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_hybrid_policy_evaluation(args.features, args.temporal_features, args.unseen_dir,
                                          args.ablation_dir, args.output_dir)
    print(json.dumps({"output_dir": str(result["output_dir"]), "full_set_reference_metrics": result["metrics"]["full_set_reference_metrics"],
                      "policy_summary": {name: {"coverage": value["coverage"], "metrics": value["determinate_metrics"]}
                                        for name, value in result["metrics"]["policies"].items()},
                      "replacement_rows": len(result["replacement_cases"])}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
