import json

import numpy as np
import pandas as pd
import pytest

import ml.models.reference_model as reference
from ml.models.random_forest_experiment import (
    FEATURE_COLUMNS,
    MODEL_PARAMETERS,
    extract_xy,
    stratified_split,
)
from ml.models.reference_model import (
    ReproducibilityError,
    check_training_inputs,
    fit_reference_model,
    load_reference_model,
    predict_case,
    save_reference_model,
    verify_phase8_parity,
)
from ml.risk.evidence_assembler import assemble_evidence


def _fixture_features(path, n=60):
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        label = i % 2
        row = {"user_id": f"USR-{i:03d}", "account_id": f"ACC-{i:03d}", "event_id": f"EVT-{i:03d}"}
        for j, column in enumerate(FEATURE_COLUMNS):
            row[column] = int(rng.integers(0, 3)) + (label if j % 3 == 0 else 0)
        row["ato_label"] = label
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def phase8_fixture(tmp_path):
    features_path = tmp_path / "features.csv"
    _fixture_features(features_path)
    features = pd.read_csv(features_path)
    _, y = extract_xy(features)
    train_idx, test_idx = stratified_split(y)
    users = features["user_id"].astype(str)
    metadata = {
        "input_features_sha256": reference._sha256(features_path),
        "feature_columns": list(FEATURE_COLUMNS),
        "random_forest_parameters": dict(MODEL_PARAMETERS),
        "software_versions": reference.current_versions(),
        "split": {"test_user_ids": sorted(users.iloc[test_idx]), "train_user_ids": sorted(users.iloc[train_idx])},
    }
    metadata_path = tmp_path / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    model, _ = fit_reference_model(features_path, metadata_path)
    X, _ = extract_xy(features)
    predictions = pd.DataFrame({
        "user_id": users.iloc[test_idx].tolist(),
        "rf_prediction": model.predict(X.iloc[test_idx]).astype(int),
        "rf_probability_ato": model.predict_proba(X.iloc[test_idx])[:, list(model.classes_).index(1)],
    })
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    return {"features": features_path, "metadata": metadata_path, "predictions": predictions_path,
            "model": model, "tmp": tmp_path}


def _rewrite_metadata(path, **changes):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        metadata[key] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")


def test_real_phase8_inputs_match_recorded_hash_versions_and_split():
    metadata = check_training_inputs()
    features = pd.read_csv(reference.DEFAULT_FEATURES_PATH)
    _, y = extract_xy(features)
    train_idx, test_idx = stratified_split(y)
    users = features["user_id"].astype(str)
    assert set(users.iloc[test_idx]) == set(metadata["split"]["test_user_ids"])
    assert set(users.iloc[train_idx]) == set(metadata["split"]["train_user_ids"])
    saved = pd.read_csv(reference.DEFAULT_PREDICTIONS_PATH, dtype={"user_id": str})
    assert saved["user_id"].tolist() == users.iloc[test_idx].tolist()


@pytest.mark.parametrize("key, value, message", [
    ("input_features_sha256", "0" * 64, "SHA-256"),
    ("random_forest_parameters", {**MODEL_PARAMETERS, "n_estimators": 99}, "parameters"),
    ("software_versions", {"numpy": "0", "pandas": "0", "python": "0", "scikit_learn": "0"}, "library versions"),
])
def test_input_check_rejects_any_difference_from_phase8(phase8_fixture, key, value, message):
    _rewrite_metadata(phase8_fixture["metadata"], **{key: value})
    with pytest.raises(ReproducibilityError, match=message):
        check_training_inputs(phase8_fixture["features"], phase8_fixture["metadata"])


def test_fit_rejects_a_split_that_differs_from_phase8(phase8_fixture):
    metadata = json.loads(phase8_fixture["metadata"].read_text(encoding="utf-8"))
    split = metadata["split"]
    split["test_user_ids"], split["train_user_ids"] = split["test_user_ids"][1:], [*split["train_user_ids"], split["test_user_ids"][0]]
    _rewrite_metadata(phase8_fixture["metadata"], split=split)
    with pytest.raises(ReproducibilityError, match="split"):
        fit_reference_model(phase8_fixture["features"], phase8_fixture["metadata"])


def test_parity_is_exact_and_any_probability_difference_fails(phase8_fixture):
    report = verify_phase8_parity(phase8_fixture["model"], phase8_fixture["features"], phase8_fixture["predictions"])
    assert report["prediction_mismatches"] == 0 and report["probability_mismatches"] == 0
    saved = pd.read_csv(phase8_fixture["predictions"], float_precision="round_trip")
    saved.loc[0, "rf_probability_ato"] = saved.loc[0, "rf_probability_ato"] + 1e-12
    saved.to_csv(phase8_fixture["predictions"], index=False)
    with pytest.raises(ReproducibilityError, match="does not reproduce"):
        verify_phase8_parity(phase8_fixture["model"], phase8_fixture["features"], phase8_fixture["predictions"])


def test_save_load_round_trip_and_safety_checks(phase8_fixture):
    model, metadata = fit_reference_model(phase8_fixture["features"], phase8_fixture["metadata"])
    parity = verify_phase8_parity(model, phase8_fixture["features"], phase8_fixture["predictions"])
    artifact_dir = phase8_fixture["tmp"] / "artifacts"
    with pytest.raises(ReproducibilityError, match="without exact Phase 8 parity"):
        save_reference_model(model, metadata, {**parity, "probability_mismatches": 1}, artifact_dir)
    save_reference_model(model, metadata, parity, artifact_dir)
    loaded, record = load_reference_model(artifact_dir)
    X, _ = extract_xy(pd.read_csv(phase8_fixture["features"]))
    assert (loaded.predict_proba(X) == model.predict_proba(X)).all()
    assert record["phase8_parity"] == parity and "uncalibrated" in record["probability_note"]
    with pytest.raises(FileExistsError):
        save_reference_model(model, metadata, parity, artifact_dir)
    model_path = artifact_dir / reference.MODEL_FILENAME
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ReproducibilityError, match="hash"):
        load_reference_model(artifact_dir)


def test_predict_case_matches_evidence_contract_and_ignores_labels(phase8_fixture):
    row = pd.read_csv(phase8_fixture["features"]).iloc[0].to_dict()
    output = predict_case(phase8_fixture["model"], row)
    assert type(output["prediction"]) is int and 0 <= output["probability_ato"] <= 1
    relabelled = {**row, "ato_label": 1 - row["ato_label"], "scenario_type": "CHANGED"}
    assert predict_case(phase8_fixture["model"], relabelled) == output
    unlabelled = {k: v for k, v in row.items() if k != "ato_label"}
    assert predict_case(phase8_fixture["model"], unlabelled) == output
    record = assemble_evidence("A-1", {"user_id": row["user_id"]}, model_output=output)
    assert record["model_evidence"] == output


def test_saved_reference_model_reproduces_phase8_exactly():
    if not (reference.DEFAULT_ARTIFACT_DIR / reference.METADATA_FILENAME).exists():
        pytest.skip("reference model not yet trained/saved (awaiting approval)")
    model, record = load_reference_model()
    report = verify_phase8_parity(model)
    assert report == {**report, "rows_compared": 200, "prediction_mismatches": 0, "probability_mismatches": 0}
    assert record["phase8_parity"]["probability_mismatches"] == 0
