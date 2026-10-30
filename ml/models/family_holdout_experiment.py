"""Leave-one-family-out evaluation of the fixed Phase 8 Random Forest on v3 cohorts.

Each fold trains on 19 template families and predicts the held-out family.
template_family_id is only the grouping key and evaluation metadata. Model A
uses the exact Phase 8 feature list; Model B adds the post-trigger transaction
columns with train-fold-only median imputation and missingness indicators.
Results describe synthetic controlled cohorts only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from ml.models.random_forest_experiment import (
    FEATURE_COLUMNS as PHASE8_FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_PARAMETERS,
    TARGET_COLUMN,
    binary_metrics,
    make_classifier,
)
from ml.models.unseen_scenario_experiment import (
    _phase9_pipeline,
    _positive_probabilities,
    _sha256,
    _write_csv,
    _write_json,
    build_scenario_folds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COHORT_ROOT = PROJECT_ROOT / "data" / "processed" / "controlled_sim_replacement_v3"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "family_holdout_v3"
COHORTS = ("predictive", "negative_control")
FAMILY_COLUMN = "template_family_id"
TRANSACTION_NUMERIC_COLUMNS = ("transaction_count", "transaction_amount", "transaction_deviation")
TRANSACTION_TYPES = ("PAYMENT", "TRANSFER", "CASH_OUT")
TRANSACTION_TYPE_COLUMNS = tuple(f"transaction_type_{t}" for t in TRANSACTION_TYPES)
MODEL_B_FEATURE_COLUMNS = (*PHASE8_FEATURE_COLUMNS, *TRANSACTION_NUMERIC_COLUMNS, *TRANSACTION_TYPE_COLUMNS)
FORBIDDEN_INPUTS = {TARGET_COLUMN, FAMILY_COLUMN, "scenario_type", "declared_scenario_intent",
                    "recipe_source", "is_anchor", "components_json", "case_id", "timestamp",
                    "paysim_isFraud", "paysim_isFlaggedFraud", *IDENTIFIER_COLUMNS}
METRIC_KEYS = ("precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc_average_precision")
DISCLOSURES = (
    "Synthetic controlled cohorts (v3, seed 42) only; not real-world detection performance.",
    "p_new_device = 0.5 (legitimate) vs 0.9 (suspicious) is a provisional synthetic design "
    "assumption, not an empirical rate; other version probabilities and slot quotas are also provisional.",
    "Families are combinations of shared building blocks, so a held-out family is not truly novel.",
    "Each held-out family has 20 test cases; per-family metrics are noisy (one case moves recall by 10 points).",
    "predict_proba outputs are uncalibrated and descriptive only; decisions use RandomForestClassifier.predict.",
)


def load_cohort(cohort_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature rows and the family grouping read from evaluation metadata."""
    cohort_dir = Path(cohort_dir)
    features = pd.read_csv(cohort_dir / "features.csv", keep_default_na=True)
    cases = pd.read_csv(cohort_dir / "case_metadata.csv", usecols=["user_id", FAMILY_COLUMN])
    if features["user_id"].duplicated().any() or cases["user_id"].duplicated().any():
        raise ValueError("features and case metadata must have one row per user")
    family_by_user = dict(zip(cases["user_id"].astype(str), cases[FAMILY_COLUMN].astype(str)))
    if set(features["user_id"].astype(str)) != set(family_by_user):
        raise ValueError("family metadata must match feature users exactly")
    families = features["user_id"].astype(str).map(family_by_user).rename(FAMILY_COLUMN)
    return features, families


def prepare_model_inputs(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build Model A/B matrices; missing transaction values stay missing (NaN)."""
    required = {*IDENTIFIER_COLUMNS, TARGET_COLUMN, *PHASE8_FEATURE_COLUMNS,
                *TRANSACTION_NUMERIC_COLUMNS, "transaction_type"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"features.csv is missing required columns: {missing}")
    X_a = features.loc[:, list(PHASE8_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if X_a.isna().any().any() or not np.isfinite(X_a.to_numpy(dtype=float)).all():
        raise ValueError("Model A inputs must be numeric, finite and non-missing")
    numeric = features.loc[:, list(TRANSACTION_NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="raise")
    types = features["transaction_type"].fillna("").astype(str)
    unknown = set(types) - {"", *TRANSACTION_TYPES}
    if unknown:
        raise ValueError(f"unexpected transaction_type values: {sorted(unknown)}")
    observed = features["transaction_count"] > 0
    if numeric.loc[~observed, ["transaction_amount", "transaction_deviation"]].notna().any().any():
        raise ValueError("transaction amount/deviation must be missing when no transaction is observed")
    # One-hot type columns are an encoding of the observed category, not an imputed value:
    # all three are 0 when no transaction exists, which the amount missingness indicator flags.
    one_hot = pd.DataFrame({f"transaction_type_{t}": (types == t).astype(int) for t in TRANSACTION_TYPES})
    X_b = pd.concat([X_a, numeric, one_hot], axis=1).loc[:, list(MODEL_B_FEATURE_COLUMNS)]
    if FORBIDDEN_INPUTS.intersection(X_a.columns) or FORBIDDEN_INPUTS.intersection(X_b.columns):
        raise ValueError("forbidden metadata entered the model inputs")
    y = pd.to_numeric(features[TARGET_COLUMN], errors="raise")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("ato_label must contain only binary values")
    return X_a, X_b, y.astype(int).rename(TARGET_COLUMN)


def _undefined(metrics: dict[str, Any]) -> list[str]:
    return [key for key in METRIC_KEYS if metrics[key] is None]


def _spread(per_family: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [m[key] for m in per_family if m[key] is not None]
    return {"defined_families": len(values), "undefined_families": len(per_family) - len(values),
            "median": median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def run_family_holdout(cohort_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run Model A and Model B on identical leave-one-family-out folds for one cohort."""
    cohort_dir, output_dir = Path(cohort_dir).resolve(), Path(output_dir)
    features, families = load_cohort(cohort_dir)
    X_a, X_b, y = prepare_model_inputs(features)
    folds = build_scenario_folds(families)
    models = {"model_a_phase8": (X_a, make_classifier), "model_b_transactions": (X_b, _phase9_pipeline)}
    outputs = {name: {"prediction": np.full(len(y), -1), "probability": np.full(len(y), np.nan)}
               for name in models}
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        train_idx, test_idx = fold["train_indices"], fold["test_indices"]
        held_out = fold["held_out_scenario_type"]
        if held_out in set(families.iloc[train_idx]):
            raise ValueError(f"held-out family {held_out} appears in training")
        if y.iloc[train_idx].nunique() != 2:
            raise ValueError(f"training complement for {held_out} lacks both classes")
        record: dict[str, Any] = {"held_out_family": held_out, "train_rows": int(len(train_idx)),
                                  "test_rows": int(len(test_idx))}
        for name, (X, factory) in models.items():
            model = factory()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            prediction = model.predict(X.iloc[test_idx]).astype(int)
            probability = _positive_probabilities(model, X.iloc[test_idx])
            outputs[name]["prediction"][test_idx] = prediction
            outputs[name]["probability"][test_idx] = probability
            metrics = binary_metrics(y.iloc[test_idx].to_numpy(), prediction, probability)
            record[name] = {**metrics, "undefined_metrics": _undefined(metrics)}
        fold_records.append(record)
    for name, values in outputs.items():
        if (values["prediction"] < 0).any() or np.isnan(values["probability"]).any():
            raise ValueError(f"{name} did not predict every case exactly once")

    pooled = {name: binary_metrics(y.to_numpy(), v["prediction"], v["probability"]) for name, v in outputs.items()}
    summary = {name: {key: _spread([r[name] for r in fold_records], key)
                      for key in ("precision", "recall", "f1", "fpr", "fnr")} for name in models}
    audit = json.loads((cohort_dir / "design_audit.json").read_text(encoding="utf-8"))
    lookup = audit["lofo_lookup_baseline"]["pooled"]
    comparison = {
        "full_vector_lookup_baseline": lookup,
        "single_feature_lookup_baselines": {
            name: diag["single_feature_lofo_lookup"]
            for name, diag in audit["component_diagnostics"].items()},
        "model_accuracy": {name: (m["tp"] + m["tn"]) / m["n"] for name, m in pooled.items()},
        "note": "Lookup baselines abstain on unseen or tied keys (abstentions counted as 0.5); "
                "the Random Forest decides every case.",
    }

    prediction_rows: list[dict[str, Any]] = []
    for i in range(len(y)):
        row = {**{col: features.iloc[i][col] for col in IDENTIFIER_COLUMNS},
               "template_family_id_evaluation_only": families.iloc[i],
               "ato_label_evaluation_only": int(y.iloc[i]), "held_out_family": families.iloc[i]}
        for name, values in outputs.items():
            prefix = name.split("_")[0] + "_" + name.split("_")[1]
            prediction, actual = int(values["prediction"][i]), int(y.iloc[i])
            row[f"{prefix}_prediction"] = prediction
            row[f"{prefix}_probability_ato"] = float(values["probability"][i])
            row[f"{prefix}_error_type"] = ("FP" if prediction == 1 and actual == 0
                                           else "FN" if prediction == 0 and actual == 1 else "CORRECT")
        prediction_rows.append(row)
    error_rows = []
    for record in fold_records:
        row = {"template_family_id_evaluation_only": record["held_out_family"], "test_rows": record["test_rows"]}
        for name in models:
            m = record[name]
            prefix = name.split("_")[0] + "_" + name.split("_")[1]
            row.update({f"{prefix}_true_positives": m["tp"], f"{prefix}_false_positives": m["fp"],
                        f"{prefix}_true_negatives": m["tn"], f"{prefix}_false_negatives": m["fn"],
                        f"{prefix}_undefined_metrics": "|".join(m["undefined_metrics"])})
        error_rows.append(row)

    metrics_payload = {
        "cohort": cohort_dir.name,
        "evaluation": "leave-one-family-out, aggregated out-of-fold",
        "coverage": {"cases": int(len(y)), "predicted": int(len(y)), "abstained": 0},
        "aggregated_out_of_fold": pooled,
        "per_family_spread": summary,
        "per_held_out_family": fold_records,
        "design_audit_comparison": comparison,
        "disclosures": list(DISCLOSURES),
    }
    metadata = {
        "experiment": "family_holdout_v3", "cohort_dir": str(cohort_dir.relative_to(PROJECT_ROOT))
        if cohort_dir.is_relative_to(PROJECT_ROOT) else str(cohort_dir),
        "families": len(folds), "folds": [{k: r[k] for k in ("held_out_family", "train_rows", "test_rows")}
                                          for r in fold_records],
        "model_a_feature_columns": list(PHASE8_FEATURE_COLUMNS),
        "model_b_feature_columns": list(MODEL_B_FEATURE_COLUMNS),
        "model_parameters": {k: v for k, v in MODEL_PARAMETERS.items()},
        "decision_rule": "RandomForestClassifier.predict default; no custom threshold or tuning",
        "model_b_preprocessing": "median imputer with missingness indicators fit on each training fold only; "
                                 "transaction_type one-hot encoded (all zero when no transaction)",
        "grouping_metadata": "template_family_id from case_metadata.csv; grouping and evaluation only, excluded from X",
        "input_sha256": {name: _sha256(cohort_dir / name)
                         for name in ("features.csv", "case_metadata.csv", "design_audit.json")},
        "software_versions": {"python": ".".join(map(str, sys.version_info[:3])),
                              "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "disclosures": list(DISCLOSURES),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "metrics.json", metrics_payload)
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_csv(output_dir / "predictions.csv", prediction_rows, tuple(prediction_rows[0]))
    _write_csv(output_dir / "family_error_summary.csv", error_rows, tuple(error_rows[0]))
    return {"output_dir": output_dir, "metrics": metrics_payload, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leave-one-family-out evaluation on v3 cohorts.")
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    for cohort in COHORTS:
        result = run_family_holdout(args.cohort_root / cohort, args.output_root / cohort)
        print(json.dumps({"cohort": cohort, "output_dir": str(result["output_dir"]),
                          "aggregated_out_of_fold": result["metrics"]["aggregated_out_of_fold"]},
                         indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
