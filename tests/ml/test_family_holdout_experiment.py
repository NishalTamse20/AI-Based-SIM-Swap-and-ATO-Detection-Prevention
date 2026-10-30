import csv
import json

import numpy as np
import pandas as pd
import pytest

import ml.models.family_holdout_experiment as experiment
from ml.models.family_holdout_experiment import (
    FORBIDDEN_INPUTS,
    MODEL_B_FEATURE_COLUMNS,
    TRANSACTION_TYPE_COLUMNS,
    _undefined,
    load_cohort,
    prepare_model_inputs,
    run_family_holdout,
)
from ml.models.random_forest_experiment import FEATURE_COLUMNS as PHASE8_FEATURE_COLUMNS, binary_metrics
from simulator.generators.paysim_loader import PAYSIM_FIELDS
from simulator.generators.sim_replacement_v3 import generate_experiment


@pytest.fixture(scope="module")
def cohort_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("family_holdout")
    paysim = root / "paysim.csv"
    with paysim.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=PAYSIM_FIELDS)
        writer.writeheader()
        for i in range(5000):
            writer.writerow({
                "step": str(i), "type": ("CASH_OUT", "TRANSFER", "PAYMENT")[i % 3],
                "amount": str(10 + i), "nameOrig": f"C{i}", "oldbalanceOrg": str(10000 + i),
                "newbalanceOrig": str(9000 + i), "nameDest": f"D{i}", "oldbalanceDest": str(2000 + i),
                "newbalanceDest": str(2010 + i), "isFraud": str(i % 2), "isFlaggedFraud": "0",
            })
    generate_experiment("predictive", output_dir=root / "predictive", paysim_path=paysim, seed=42)
    return root / "predictive"


@pytest.fixture(scope="module")
def result(cohort_dir, tmp_path_factory):
    return run_family_holdout(cohort_dir, tmp_path_factory.mktemp("out") / "predictive")


def test_model_inputs_exclude_metadata_and_keep_missing_transactions_missing(cohort_dir):
    features, _ = load_cohort(cohort_dir)
    features["template_family_id"] = "F01"
    X_a, X_b, y = prepare_model_inputs(features)
    assert list(X_a.columns) == list(PHASE8_FEATURE_COLUMNS)
    assert list(X_b.columns) == list(MODEL_B_FEATURE_COLUMNS)
    assert not FORBIDDEN_INPUTS.intersection(X_b.columns)
    no_transaction = features["transaction_count"] == 0
    assert no_transaction.any() and (~no_transaction).any()
    assert X_b.loc[no_transaction, "transaction_amount"].isna().all()
    assert X_b.loc[no_transaction, "transaction_deviation"].isna().all()
    assert (X_b.loc[no_transaction, list(TRANSACTION_TYPE_COLUMNS)] == 0).all().all()
    assert X_b.loc[~no_transaction, "transaction_amount"].notna().all()
    assert set(y.unique()) == {0, 1}


def test_held_out_family_is_never_in_training_and_every_case_is_predicted_once(result, cohort_dir):
    folds = result["metadata"]["folds"]
    assert len(folds) == 20 and all(f["train_rows"] == 380 and f["test_rows"] == 20 for f in folds)
    predictions = pd.read_csv(result["output_dir"] / "predictions.csv")
    assert len(predictions) == 400 and predictions["user_id"].is_unique
    assert (predictions["held_out_family"] == predictions["template_family_id_evaluation_only"]).all()
    assert result["metrics"]["coverage"] == {"cases": 400, "predicted": 400, "abstained": 0}
    for model in ("model_a_phase8", "model_b_transactions"):
        assert result["metrics"]["aggregated_out_of_fold"][model]["n"] == 400


def test_model_b_imputer_is_fit_on_training_rows_only(cohort_dir, tmp_path, monkeypatch):
    fitted_sizes = []
    original = experiment._phase9_pipeline

    def spy():
        pipeline = original()
        fit = pipeline.fit

        def recording_fit(X, y):
            fitted_sizes.append((len(X), set(X.index)))
            return fit(X, y)

        pipeline.fit = recording_fit
        return pipeline

    monkeypatch.setattr(experiment, "_phase9_pipeline", spy)
    run = run_family_holdout(cohort_dir, tmp_path / "spy")
    _, families = load_cohort(cohort_dir)
    assert len(fitted_sizes) == 20 and all(size == 380 for size, _ in fitted_sizes)
    for record, (_, indices) in zip(run["metrics"]["per_held_out_family"], fitted_sizes):
        assert record["held_out_family"] not in set(families.iloc[sorted(indices)])


def test_undefined_metrics_are_reported_as_null_not_zero():
    metrics = binary_metrics(np.array([0, 1]), np.array([0, 0]), np.array([0.1, 0.2]))
    assert metrics["precision"] is None and "precision" in _undefined(metrics)
    assert metrics["recall"] == 0.0


def test_outputs_carry_disclosures_and_design_audit_comparison(result):
    metrics = json.loads((result["output_dir"] / "metrics.json").read_text())
    assert any("p_new_device" in d and "provisional" in d for d in metrics["disclosures"])
    assert any("not real-world" in d for d in metrics["disclosures"])
    comparison = metrics["design_audit_comparison"]
    assert "full_vector_lookup_baseline" in comparison and "new_device" in comparison["single_feature_lookup_baselines"]
    metadata = json.loads((result["output_dir"] / "run_metadata.json").read_text())
    assert "no custom threshold" in metadata["decision_rule"]
    assert {p.name for p in result["output_dir"].iterdir()} == {
        "metrics.json", "run_metadata.json", "predictions.csv", "family_error_summary.csv"}


def test_rerun_is_deterministic(result, cohort_dir, tmp_path):
    rerun = run_family_holdout(cohort_dir, tmp_path / "rerun")
    for name in ("metrics.json", "predictions.csv", "family_error_summary.csv"):
        assert (rerun["output_dir"] / name).read_bytes() == (result["output_dir"] / name).read_bytes()
