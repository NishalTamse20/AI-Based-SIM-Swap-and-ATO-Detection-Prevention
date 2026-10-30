"""Held-out permutation attribution for the v3 leave-one-family-out models.

The fold models are not persisted, so each is refit with the identical Phase 8
configuration on the identical folds, and the refit predictions must match the
saved family_holdout_v3 predictions exactly before any attribution runs.
Attribution is measured by permuting a feature (or a feature group, jointly)
within each held-out family and recording the drop in accuracy and average
precision. A device-group drop-and-refit ablation is a diagnostic, not tuning.
Results describe these models on synthetic controlled cohorts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, average_precision_score

from ml.models.family_holdout_experiment import (
    COHORTS,
    DEFAULT_COHORT_ROOT,
    DEFAULT_OUTPUT_ROOT as HOLDOUT_OUTPUT_ROOT,
    DISCLOSURES,
    MODEL_B_FEATURE_COLUMNS,
    TRANSACTION_NUMERIC_COLUMNS,
    TRANSACTION_TYPE_COLUMNS,
    load_cohort,
    prepare_model_inputs,
)
from ml.models.random_forest_experiment import (
    FEATURE_COLUMNS as PHASE8_FEATURE_COLUMNS,
    MODEL_PARAMETERS,
    PERMUTATION_REPEATS,
    RANDOM_STATE,
    binary_metrics,
    make_classifier,
)
from ml.models.unseen_scenario_experiment import (
    _phase9_pipeline,
    _sha256,
    _write_csv,
    _write_json,
    build_scenario_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "feature_attribution_v3"
DEVICE_GROUP = ("new_device", "device_change", "known_device")
TRANSACTION_GROUP = (*TRANSACTION_NUMERIC_COLUMNS, *TRANSACTION_TYPE_COLUMNS)
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "model_a_phase8": {"prefix": "model_a", "columns": PHASE8_FEATURE_COLUMNS, "factory": "classifier",
                       "groups": {"device_group": DEVICE_GROUP}},
    "model_b_transactions": {"prefix": "model_b", "columns": MODEL_B_FEATURE_COLUMNS, "factory": "pipeline",
                             "groups": {"device_group": DEVICE_GROUP, "transaction_group": TRANSACTION_GROUP}},
}
ATTRIBUTION_FIELDS = (
    "model", "unit", "unit_type", "columns", "pooled_accuracy_drop_mean", "pooled_accuracy_drop_std",
    "pooled_ap_drop_mean", "pooled_ap_drop_std", "per_family_accuracy_drop_median",
    "per_family_accuracy_drop_min", "per_family_accuracy_drop_max", "per_family_ap_drop_median",
    "per_family_ap_drop_min", "per_family_ap_drop_max", "rank_by_pooled_ap_drop",
)
PER_FAMILY_FIELDS = ("model", "unit", "template_family_id_evaluation_only",
                     "accuracy_drop_mean", "ap_drop_mean")
LIMITATIONS = (
    *DISCLOSURES,
    "Permutation importance measures how these fitted models depend on a feature on this synthetic "
    "data; it does not show that the feature causes account takeover.",
    "Correlated features share credit when permuted one at a time; grouped permutation is reported "
    "for the device and transaction groups for that reason.",
    "Per-family drops are computed on 20 held-out cases each and are noisy.",
)


def _factory(kind: str) -> Callable[[], Any]:
    # Resolved at call time so tests can substitute the same factories.
    return make_classifier if kind == "classifier" else _phase9_pipeline


def _predict(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    probability = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    return classes[np.argmax(probability, axis=1)].astype(int), probability[:, list(classes).index(1)]


def fit_fold_models(X: pd.DataFrame, y: pd.Series, families: pd.Series,
                    factory: Callable[[], Any]) -> list[dict[str, Any]]:
    """Refit one model per held-out family using the identical LOFO folds."""
    fitted = []
    for fold in build_scenario_folds(families):
        train_idx, test_idx = fold["train_indices"], fold["test_indices"]
        if fold["held_out_scenario_type"] in set(families.iloc[train_idx]):
            raise ValueError("held-out family appears in training")
        model = factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fitted.append({"family": fold["held_out_scenario_type"], "train_idx": train_idx,
                       "test_idx": test_idx, "model": model})
    return fitted


def verify_against_saved(fitted: list[dict[str, Any]], X: pd.DataFrame, features: pd.DataFrame,
                         saved: pd.DataFrame, prefix: str) -> dict[str, Any]:
    """Require refit predictions and probabilities to equal the saved LOFO outputs exactly."""
    saved = saved.set_index("user_id")
    mismatched_predictions = mismatched_probabilities = 0
    for fold in fitted:
        X_test = X.iloc[fold["test_idx"]]
        prediction = fold["model"].predict(X_test).astype(int)
        argmax_prediction, probability = _predict(fold["model"], X_test)
        if not np.array_equal(prediction, argmax_prediction):
            raise ValueError("predict and argmax(predict_proba) disagree; attribution would be invalid")
        users = features.iloc[fold["test_idx"]]["user_id"].astype(str)
        expected = saved.loc[users]
        mismatched_predictions += int((expected[f"{prefix}_prediction"].to_numpy() != prediction).sum())
        mismatched_probabilities += int((expected[f"{prefix}_probability_ato"].to_numpy() != probability).sum())
    result = {"cases": int(sum(len(f["test_idx"]) for f in fitted)),
              "prediction_mismatches": mismatched_predictions,
              "probability_mismatches": mismatched_probabilities}
    if mismatched_predictions or mismatched_probabilities:
        raise ValueError(f"refit {prefix} does not reproduce saved LOFO outputs: {result}")
    return {**result, "matches_saved_outputs": True}


def permutation_attribution(fitted: list[dict[str, Any]], X: pd.DataFrame, y: pd.Series,
                            units: dict[str, tuple[str, ...]], repeats: int = PERMUTATION_REPEATS,
                            seed: int = RANDOM_STATE) -> dict[str, Any]:
    """Permute each unit's columns jointly within every held-out family; pool and per-fold drops."""
    y_all = y.to_numpy()
    base_pred = np.empty(len(y), dtype=int)
    base_prob = np.empty(len(y), dtype=float)
    for fold in fitted:
        base_pred[fold["test_idx"]], base_prob[fold["test_idx"]] = _predict(fold["model"], X.iloc[fold["test_idx"]])
    base_acc, base_ap = accuracy_score(y_all, base_pred), average_precision_score(y_all, base_prob)
    fold_base = {f["family"]: (accuracy_score(y_all[f["test_idx"]], base_pred[f["test_idx"]]),
                               average_precision_score(y_all[f["test_idx"]], base_prob[f["test_idx"]]))
                 for f in fitted}
    results: dict[str, Any] = {}
    for unit_index, (unit, columns) in enumerate(units.items()):
        rng = np.random.RandomState(seed + unit_index)
        pooled_pred = np.empty((repeats, len(y)), dtype=int)
        pooled_prob = np.empty((repeats, len(y)), dtype=float)
        per_family: dict[str, list[tuple[float, float]]] = {}
        positions = [X.columns.get_loc(c) for c in columns]
        for fold in fitted:
            test_idx = fold["test_idx"]
            X_test = X.iloc[test_idx]
            # All repeats are stacked into one matrix so each fold model predicts once per unit.
            stacked = pd.concat([X_test] * repeats, ignore_index=True)
            for r in range(repeats):
                order = rng.permutation(len(test_idx))
                block = slice(r * len(test_idx), (r + 1) * len(test_idx))
                stacked.iloc[block, positions] = X_test.iloc[order, positions].to_numpy()
            prediction, probability = _predict(fold["model"], stacked)
            prediction = prediction.reshape(repeats, len(test_idx))
            probability = probability.reshape(repeats, len(test_idx))
            pooled_pred[:, test_idx], pooled_prob[:, test_idx] = prediction, probability
            acc0, ap0 = fold_base[fold["family"]]
            truth = y_all[test_idx]
            per_family[fold["family"]] = [
                (acc0 - accuracy_score(truth, prediction[r]), ap0 - average_precision_score(truth, probability[r]))
                for r in range(repeats)]
        acc_drops = [base_acc - accuracy_score(y_all, pooled_pred[r]) for r in range(repeats)]
        ap_drops = [base_ap - average_precision_score(y_all, pooled_prob[r]) for r in range(repeats)]
        family_means = {family: (float(np.mean([d[0] for d in drops])), float(np.mean([d[1] for d in drops])))
                        for family, drops in per_family.items()}
        results[unit] = {"columns": list(columns), "pooled_accuracy_drops": acc_drops, "pooled_ap_drops": ap_drops,
                         "per_family_mean_drops": family_means}
    return {"baseline": {"accuracy": base_acc, "average_precision": base_ap}, "units": results}


def _attribution_rows(model_name: str, attribution: dict[str, Any], groups: set[str]) -> list[dict[str, Any]]:
    rows = []
    for unit, entry in attribution["units"].items():
        acc_family = [v[0] for v in entry["per_family_mean_drops"].values()]
        ap_family = [v[1] for v in entry["per_family_mean_drops"].values()]
        rows.append({
            "model": model_name, "unit": unit, "unit_type": "group" if unit in groups else "feature",
            "columns": "|".join(entry["columns"]),
            "pooled_accuracy_drop_mean": float(np.mean(entry["pooled_accuracy_drops"])),
            "pooled_accuracy_drop_std": float(np.std(entry["pooled_accuracy_drops"])),
            "pooled_ap_drop_mean": float(np.mean(entry["pooled_ap_drops"])),
            "pooled_ap_drop_std": float(np.std(entry["pooled_ap_drops"])),
            "per_family_accuracy_drop_median": median(acc_family),
            "per_family_accuracy_drop_min": min(acc_family), "per_family_accuracy_drop_max": max(acc_family),
            "per_family_ap_drop_median": median(ap_family),
            "per_family_ap_drop_min": min(ap_family), "per_family_ap_drop_max": max(ap_family),
        })
    rows.sort(key=lambda r: (-r["pooled_ap_drop_mean"], r["unit"]))
    for rank, row in enumerate(rows, start=1):
        row["rank_by_pooled_ap_drop"] = rank
    return rows


def _pooled_metrics(fitted: list[dict[str, Any]], X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, Any], np.ndarray]:
    prediction = np.empty(len(y), dtype=int)
    probability = np.empty(len(y), dtype=float)
    for fold in fitted:
        prediction[fold["test_idx"]], probability[fold["test_idx"]] = _predict(fold["model"], X.iloc[fold["test_idx"]])
    return binary_metrics(y.to_numpy(), prediction, probability), prediction


def _correlation_diagnostics(prediction: np.ndarray, features: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Descriptive associations only; not attribution and not causal."""
    new_device = features["new_device"].astype(int).to_numpy()
    truth = y.to_numpy()
    by_value = {}
    for value in (0, 1):
        mask = new_device == value
        by_value[f"new_device_{value}"] = {
            "cases": int(mask.sum()), "label_1_rate": float(truth[mask].mean()) if mask.any() else None,
            "predicted_1_rate": float(prediction[mask].mean()) if mask.any() else None,
            "model_accuracy": float((prediction[mask] == truth[mask]).mean()) if mask.any() else None}
    return {"kind": "correlation (descriptive, not attribution)",
            "prediction_equals_new_device_rate": float((prediction == new_device).mean()),
            "by_new_device_value": by_value}


def run_feature_attribution(cohort_dir: str | Path, holdout_dir: str | Path,
                            output_dir: str | Path) -> dict[str, Any]:
    cohort_dir, holdout_dir, output_dir = Path(cohort_dir).resolve(), Path(holdout_dir), Path(output_dir)
    features, families = load_cohort(cohort_dir)
    X_a, X_b, y = prepare_model_inputs(features)
    matrices = {"model_a_phase8": X_a, "model_b_transactions": X_b}
    saved = pd.read_csv(holdout_dir / "predictions.csv", float_precision="round_trip", dtype={"user_id": str})
    audit = json.loads((cohort_dir / "design_audit.json").read_text(encoding="utf-8"))

    attribution_rows: list[dict[str, Any]] = []
    per_family_rows: list[dict[str, Any]] = []
    models_payload: dict[str, Any] = {}
    for name, spec in MODEL_SPECS.items():
        X = matrices[name]
        factory = _factory(spec["factory"])
        fitted = fit_fold_models(X, y, families, factory)
        verification = verify_against_saved(fitted, X, features, saved, spec["prefix"])
        units = {column: (column,) for column in spec["columns"]} | dict(spec["groups"])
        attribution = permutation_attribution(fitted, X, y, units)
        rows = _attribution_rows(name, attribution, set(spec["groups"]))
        attribution_rows += rows
        for unit, entry in attribution["units"].items():
            for family, (acc, ap) in sorted(entry["per_family_mean_drops"].items()):
                per_family_rows.append({"model": name, "unit": unit, "template_family_id_evaluation_only": family,
                                        "accuracy_drop_mean": acc, "ap_drop_mean": ap})
        full_metrics, prediction = _pooled_metrics(fitted, X, y)
        ablated_columns = [c for c in X.columns if c not in DEVICE_GROUP]
        ablated_fitted = fit_fold_models(X.loc[:, ablated_columns], y, families, factory)
        ablated_metrics, _ = _pooled_metrics(ablated_fitted, X.loc[:, ablated_columns], y)
        per_family_ablation = {}
        for full, ablated in zip(fitted, ablated_fitted):
            idx = full["test_idx"]
            full_m = binary_metrics(y.iloc[idx].to_numpy(), *_predict(full["model"], X.iloc[idx]))
            abl_m = binary_metrics(y.iloc[idx].to_numpy(), *_predict(ablated["model"], X.loc[:, ablated_columns].iloc[idx]))
            per_family_ablation[full["family"]] = {
                "full_accuracy": (full_m["tp"] + full_m["tn"]) / full_m["n"],
                "ablated_accuracy": (abl_m["tp"] + abl_m["tn"]) / abl_m["n"]}
        models_payload[name] = {
            "verification": verification,
            "baseline_out_of_fold": attribution["baseline"],
            "top_units_by_pooled_ap_drop": [{k: r[k] for k in ("unit", "unit_type", "pooled_ap_drop_mean",
                                                               "pooled_accuracy_drop_mean")} for r in rows[:5]],
            "device_group_ablation": {
                "kind": "measured diagnostic refit (same settings and folds, device group removed); not tuning",
                "removed_columns": list(DEVICE_GROUP),
                "full": full_metrics, "ablated": ablated_metrics,
                "accuracy_full": (full_metrics["tp"] + full_metrics["tn"]) / full_metrics["n"],
                "accuracy_ablated": (ablated_metrics["tp"] + ablated_metrics["tn"]) / ablated_metrics["n"],
                "ablated_minus_full": {k: (ablated_metrics[k] - full_metrics[k])
                                       if ablated_metrics[k] is not None and full_metrics[k] is not None else None
                                       for k in ("precision", "recall", "f1", "fpr", "fnr", "roc_auc",
                                                 "pr_auc_average_precision")},
                "per_family": per_family_ablation},
            "correlation_diagnostics": _correlation_diagnostics(prediction, features, y),
        }

    lookup = audit["component_diagnostics"]
    payload = {
        "cohort": cohort_dir.name,
        "method": ("held-out permutation importance on leave-one-family-out folds; scoring accuracy and "
                   f"average precision; {PERMUTATION_REPEATS} repeats; seed {RANDOM_STATE}; grouped units permute "
                   "their columns jointly; pooled drops use all 400 out-of-fold predictions"),
        "models": models_payload,
        "single_feature_lookup_baselines": {
            "kind": "design-audit lookup baselines (abstentions counted as 0.5)",
            "new_device": lookup["new_device"]["single_feature_lofo_lookup"],
            "full_vector": audit["lofo_lookup_baseline"]["pooled"]},
        "limitations": list(LIMITATIONS),
    }
    metadata = {
        "experiment": "feature_attribution_v3", "cohort": cohort_dir.name,
        "model_parameters": dict(MODEL_PARAMETERS), "permutation_repeats": PERMUTATION_REPEATS,
        "random_state": RANDOM_STATE, "groups": {"device_group": list(DEVICE_GROUP),
                                                 "transaction_group": list(TRANSACTION_GROUP)},
        "decision_rule": "argmax of predict_proba, verified equal to RandomForestClassifier.predict; no threshold",
        "input_sha256": {"features.csv": _sha256(cohort_dir / "features.csv"),
                         "case_metadata.csv": _sha256(cohort_dir / "case_metadata.csv"),
                         "design_audit.json": _sha256(cohort_dir / "design_audit.json"),
                         "family_holdout_predictions.csv": _sha256(holdout_dir / "predictions.csv")},
        "software_versions": {"python": ".".join(map(str, sys.version_info[:3])),
                              "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "feature_attribution.csv", attribution_rows, ATTRIBUTION_FIELDS)
    _write_csv(output_dir / "feature_attribution_per_family.csv", per_family_rows, PER_FAMILY_FIELDS)
    _write_json(output_dir / "metrics.json", payload)
    _write_json(output_dir / "run_metadata.json", metadata)
    return {"output_dir": output_dir, "metrics": payload, "rows": attribution_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out permutation attribution for v3 LOFO models.")
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--holdout-root", type=Path, default=HOLDOUT_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    for cohort in COHORTS:
        result = run_feature_attribution(args.cohort_root / cohort, args.holdout_root / cohort,
                                         args.output_root / cohort)
        print(json.dumps({"cohort": cohort, "output_dir": str(result["output_dir"]),
                          "verification": {m: v["verification"] for m, v in result["metrics"]["models"].items()}},
                         indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
