"""Fixed Phase 8 reference model for single-case assessment.

The model is the Phase 8 Random Forest reproduced exactly: same features.csv
(verified by SHA-256), row order, stratified 80/20 split (seed 42), estimator
parameters and library versions recorded in the Phase 8 run metadata. Before it
is saved, its test-split predictions and probabilities must equal the saved
Phase 8 predictions exactly. Probabilities are uncalibrated and are passed on
only as descriptive model evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from ml.models.random_forest_experiment import (
    DEFAULT_FEATURES_PATH,
    FEATURE_COLUMNS,
    MODEL_PARAMETERS,
    extract_xy,
    make_classifier,
    stratified_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE8_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "random_forest"
DEFAULT_RUN_METADATA_PATH = PHASE8_OUTPUT_DIR / "run_metadata.json"
DEFAULT_PREDICTIONS_PATH = PHASE8_OUTPUT_DIR / "predictions.csv"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"
MODEL_FILENAME = "phase8_reference_model.joblib"
METADATA_FILENAME = "phase8_reference_model.json"
MODEL_ID = "phase8-random-forest-reference"


class ReproducibilityError(ValueError):
    """Raised when the reference model cannot be shown to equal Phase 8."""


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_versions() -> dict[str, str]:
    return {"numpy": np.__version__, "pandas": pd.__version__,
            "python": ".".join(map(str, sys.version_info[:3])), "scikit_learn": sklearn.__version__}


def check_training_inputs(features_path: str | Path = DEFAULT_FEATURES_PATH,
                          run_metadata_path: str | Path = DEFAULT_RUN_METADATA_PATH) -> dict[str, Any]:
    """Require the exact Phase 8 data, features, parameters and library versions."""
    metadata = json.loads(Path(run_metadata_path).read_text(encoding="utf-8"))
    problems = []
    if _sha256(features_path) != metadata["input_features_sha256"]:
        problems.append("features.csv SHA-256 differs from the Phase 8 run")
    if list(metadata["feature_columns"]) != list(FEATURE_COLUMNS):
        problems.append("feature columns differ from the Phase 8 run")
    if dict(metadata["random_forest_parameters"]) != dict(MODEL_PARAMETERS):
        problems.append("Random Forest parameters differ from the Phase 8 run")
    if metadata["software_versions"] != current_versions():
        problems.append(f"library versions {current_versions()} differ from Phase 8 {metadata['software_versions']}")
    if problems:
        raise ReproducibilityError("; ".join(problems))
    return metadata


def fit_reference_model(features_path: str | Path = DEFAULT_FEATURES_PATH,
                        run_metadata_path: str | Path = DEFAULT_RUN_METADATA_PATH) -> tuple[Any, dict[str, Any]]:
    """Refit the Phase 8 forest on the identical training split (no tuning)."""
    run_metadata = check_training_inputs(features_path, run_metadata_path)
    features = pd.read_csv(features_path)
    X, y = extract_xy(features)
    train_indices, test_indices = stratified_split(y)
    users = features["user_id"].astype(str)
    if set(users.iloc[test_indices]) != set(run_metadata["split"]["test_user_ids"]) or \
            set(users.iloc[train_indices]) != set(run_metadata["split"]["train_user_ids"]):
        raise ReproducibilityError("train/test split differs from the Phase 8 run")
    model = make_classifier()
    model.fit(X.iloc[train_indices], y.iloc[train_indices])
    metadata = {
        "model_id": MODEL_ID,
        "feature_columns": list(FEATURE_COLUMNS),
        "model_parameters": dict(MODEL_PARAMETERS),
        "training_features_sha256": run_metadata["input_features_sha256"],
        "train_user_ids": users.iloc[train_indices].tolist(),
        "test_user_ids": users.iloc[test_indices].tolist(),
        "software_versions": current_versions(),
        "decision_rule": "RandomForestClassifier.predict default; no custom threshold",
        "probability_note": "predict_proba output is uncalibrated and is not a risk probability",
    }
    return model, metadata


def verify_phase8_parity(model: Any, features_path: str | Path = DEFAULT_FEATURES_PATH,
                         predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH) -> dict[str, Any]:
    """Require exact equality with the saved Phase 8 test predictions and probabilities."""
    saved = pd.read_csv(predictions_path, float_precision="round_trip", dtype={"user_id": str})
    features = pd.read_csv(features_path)
    X, _ = extract_xy(features)
    position = {user: i for i, user in enumerate(features["user_id"].astype(str))}
    rows = [position[user] for user in saved["user_id"]]
    prediction = model.predict(X.iloc[rows]).astype(int)
    probability = model.predict_proba(X.iloc[rows])[:, list(model.classes_).index(1)]
    prediction_mismatches = int((prediction != saved["rf_prediction"].to_numpy()).sum())
    probability_mismatches = int((probability != saved["rf_probability_ato"].to_numpy()).sum())
    report = {"rows_compared": len(rows), "prediction_mismatches": prediction_mismatches,
              "probability_mismatches": probability_mismatches,
              "max_abs_probability_difference": float(np.max(np.abs(probability - saved["rf_probability_ato"].to_numpy())))}
    if prediction_mismatches or probability_mismatches:
        raise ReproducibilityError(f"reference model does not reproduce Phase 8: {report}")
    return report


def save_reference_model(model: Any, metadata: Mapping[str, Any], parity: Mapping[str, Any],
                         artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    """Write the model and its metadata; refuses to overwrite an existing artifact."""
    if parity.get("prediction_mismatches") != 0 or parity.get("probability_mismatches") != 0:
        raise ReproducibilityError("refusing to save a model without exact Phase 8 parity")
    artifact_dir = Path(artifact_dir)
    model_path, metadata_path = artifact_dir / MODEL_FILENAME, artifact_dir / METADATA_FILENAME
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite existing reference model in {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    record = {**metadata, "phase8_parity": dict(parity), "model_sha256": _sha256(model_path)}
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def load_reference_model(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> tuple[Any, dict[str, Any]]:
    """Load only our own artifact: hash, versions and feature list must match before unpickling."""
    artifact_dir = Path(artifact_dir)
    model_path, metadata_path = artifact_dir / MODEL_FILENAME, artifact_dir / METADATA_FILENAME
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if _sha256(model_path) != metadata["model_sha256"]:
        raise ReproducibilityError("reference model file hash does not match its metadata")
    if metadata["software_versions"] != current_versions():
        raise ReproducibilityError("current library versions differ from those the model was saved with")
    if metadata["feature_columns"] != list(FEATURE_COLUMNS):
        raise ReproducibilityError("reference model feature columns differ from Phase 8 FEATURE_COLUMNS")
    return joblib.load(model_path), metadata


def predict_case(model: Any, feature_row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the model evidence contract expected by assemble_evidence for one feature row.

    Only FEATURE_COLUMNS are read; labels, scenario_type and identifiers are ignored.
    """
    frame = pd.DataFrame([{column: feature_row[column] for column in FEATURE_COLUMNS}])
    X = frame.apply(pd.to_numeric, errors="raise")
    if X.isna().any().any() or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("reference model inputs must be numeric, finite and non-missing")
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, list(model.classes_).index(1)])
    return {"prediction": prediction, "probability_ato": probability}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit, verify and save the fixed Phase 8 reference model.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    model, metadata = fit_reference_model()
    parity = verify_phase8_parity(model)
    record = save_reference_model(model, metadata, parity, args.artifact_dir)
    print(json.dumps({"artifact_dir": str(args.artifact_dir), "phase8_parity": record["phase8_parity"],
                      "model_sha256": record["model_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
