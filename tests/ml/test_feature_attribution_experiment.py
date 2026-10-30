import csv
import json

import numpy as np
import pandas as pd
import pytest

from ml.models.family_holdout_experiment import load_cohort, prepare_model_inputs, run_family_holdout
from ml.models.feature_attribution_experiment import (
    DEVICE_GROUP,
    MODEL_SPECS,
    fit_fold_models,
    permutation_attribution,
    run_feature_attribution,
    verify_against_saved,
)
from ml.models.random_forest_experiment import make_classifier
from simulator.generators.paysim_loader import PAYSIM_FIELDS
from simulator.generators.sim_replacement_v3 import generate_experiment


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    root = tmp_path_factory.mktemp("attribution")
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
    cohort_dir = root / "predictive"
    generate_experiment("predictive", output_dir=cohort_dir, paysim_path=paysim, seed=42)
    holdout_dir = root / "holdout"
    run_family_holdout(cohort_dir, holdout_dir)
    return {"cohort_dir": cohort_dir, "holdout_dir": holdout_dir, "root": root}


@pytest.fixture(scope="module")
def result(cohort):
    return run_feature_attribution(cohort["cohort_dir"], cohort["holdout_dir"], cohort["root"] / "attribution")


class RecordingModel:
    """Deterministic stand-in that records every matrix it is asked to score."""

    classes_ = np.array([0, 1])

    def __init__(self):
        self.seen = []

    def predict_proba(self, X):
        self.seen.append(X.copy())
        score = X["new_device"].to_numpy(dtype=float) * 0.8 + 0.1
        return np.column_stack([1 - score, score])


def _stub_folds(X, families):
    folds = []
    for family in sorted(set(families)):
        test_idx = np.flatnonzero(families.to_numpy() == family)
        folds.append({"family": family, "train_idx": np.flatnonzero(families.to_numpy() != family),
                      "test_idx": test_idx, "model": RecordingModel()})
    return folds


def test_refit_reproduces_saved_predictions_and_mismatch_stops(cohort, tmp_path):
    features, families = load_cohort(cohort["cohort_dir"])
    X_a, _, y = prepare_model_inputs(features)
    fitted = fit_fold_models(X_a, y, families, make_classifier)
    saved = pd.read_csv(cohort["holdout_dir"] / "predictions.csv", float_precision="round_trip",
                        dtype={"user_id": str})
    assert verify_against_saved(fitted, X_a, features, saved, "model_a")["matches_saved_outputs"]
    tampered = saved.copy()
    tampered.loc[0, "model_a_prediction"] = 1 - tampered.loc[0, "model_a_prediction"]
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_against_saved(fitted, X_a, features, tampered, "model_a")
    for fold in fitted:
        assert fold["family"] not in set(families.iloc[fold["train_idx"]])


def test_group_units_permute_columns_jointly_and_rows_stay_within_family(cohort):
    features, families = load_cohort(cohort["cohort_dir"])
    X_a, _, y = prepare_model_inputs(features)
    folds = _stub_folds(X_a, families)
    permutation_attribution(folds, X_a, y, {"device_group": DEVICE_GROUP}, repeats=3)
    for fold in folds:
        original = X_a.iloc[fold["test_idx"]]
        pairs = sorted(map(tuple, original.loc[:, list(DEVICE_GROUP)].to_numpy()))
        stacked = fold["model"].seen[-1]
        for r in range(3):
            block = stacked.iloc[r * len(original):(r + 1) * len(original)]
            assert sorted(map(tuple, block.loc[:, list(DEVICE_GROUP)].to_numpy())) == pairs
            others = [c for c in X_a.columns if c not in DEVICE_GROUP]
            assert (block.loc[:, others].to_numpy() == original.loc[:, others].to_numpy()).all()


def test_constant_feature_has_zero_drop_and_attribution_is_deterministic(cohort):
    features, families = load_cohort(cohort["cohort_dir"])
    X_a, _, y = prepare_model_inputs(features)
    first = permutation_attribution(_stub_folds(X_a, families), X_a, y,
                                    {"sim_change": ("sim_change",), "new_device": ("new_device",)}, repeats=4)
    second = permutation_attribution(_stub_folds(X_a, families), X_a, y,
                                     {"sim_change": ("sim_change",), "new_device": ("new_device",)}, repeats=4)
    assert first == second
    assert all(d == 0 for d in first["units"]["sim_change"]["pooled_accuracy_drops"])
    assert np.mean(first["units"]["new_device"]["pooled_ap_drops"]) > 0


def test_full_run_writes_labelled_outputs_for_both_models(result):
    output_dir = result["output_dir"]
    assert {p.name for p in output_dir.iterdir()} == {
        "feature_attribution.csv", "feature_attribution_per_family.csv", "metrics.json", "run_metadata.json"}
    metrics = json.loads((output_dir / "metrics.json").read_text())
    for name, spec in MODEL_SPECS.items():
        model = metrics["models"][name]
        assert model["verification"]["matches_saved_outputs"] is True
        assert model["correlation_diagnostics"]["kind"].startswith("correlation")
        assert "not tuning" in model["device_group_ablation"]["kind"]
        units = set(spec["columns"]) | set(spec["groups"])
        rows = [r for r in result["rows"] if r["model"] == name]
        assert {r["unit"] for r in rows} == units
    per_family = pd.read_csv(output_dir / "feature_attribution_per_family.csv")
    expected = sum(len(s["columns"]) + len(s["groups"]) for s in MODEL_SPECS.values()) * 20
    assert len(per_family) == expected
    assert any("not show that the feature causes" in item for item in metrics["limitations"])
