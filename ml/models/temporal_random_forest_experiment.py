"""Phase 9 comparison using the frozen Phase 8 split and forest settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ml.features.temporal_features import FEATURE_COLUMNS as TEMPORAL_FEATURE_COLUMNS
from ml.models.random_forest_experiment import (
    DEFAULT_FEATURES_PATH,
    FEATURE_COLUMNS as PHASE8_FEATURE_COLUMNS,
    TARGET_COLUMN,
    binary_metrics,
    make_classifier,
    stratified_split,
)

DEFAULT_TEMPORAL_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "temporal_features.csv"


def run_temporal_comparison(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    temporal_path: str | Path = DEFAULT_TEMPORAL_PATH,
) -> dict[str, Any]:
    """Evaluate Phase 8 inputs and Phase 8+temporal inputs on identical rows.

    This function is intentionally read-only: it returns results and does not
    write or update Phase 8 artifacts or any other output file.
    """
    base = pd.read_csv(features_path)
    temporal = pd.read_csv(temporal_path)
    required_ids = ["user_id", "account_id", "event_id"]
    if base["user_id"].duplicated().any() or temporal["user_id"].duplicated().any():
        raise ValueError("both feature artifacts must contain one row per user")
    merged = base[required_ids + [TARGET_COLUMN, *PHASE8_FEATURE_COLUMNS]].merge(
        temporal[required_ids + list(TEMPORAL_FEATURE_COLUMNS)],
        on=required_ids,
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if len(merged) != len(base) or not (merged["_merge"] == "both").all():
        raise ValueError("temporal and Phase 8 feature rows must match exactly by identifiers")
    merged = merged.drop(columns="_merge")
    if merged[TARGET_COLUMN].isna().any() or not set(merged[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("ato_label must contain only non-missing binary values")
    y = merged[TARGET_COLUMN].astype(int)
    X_base = merged.loc[:, list(PHASE8_FEATURE_COLUMNS)].apply(pd.to_numeric, errors="raise")
    X_temporal = merged.loc[:, [*PHASE8_FEATURE_COLUMNS, *TEMPORAL_FEATURE_COLUMNS]].apply(
        pd.to_numeric, errors="raise"
    )
    if X_base.isna().any().any() or not np.isfinite(X_base.to_numpy(dtype=float)).all():
        raise ValueError("Phase 8 inputs must be finite and non-missing")
    temporal_values = X_temporal.to_numpy(dtype=float)
    observed_values = temporal_values[~X_temporal.isna().to_numpy()]
    if not np.isfinite(observed_values).all():
        raise ValueError("observed Phase 9 input values must be finite")

    train_indices, test_indices = stratified_split(y)
    phase8_model = make_classifier()
    phase8_model.fit(X_base.iloc[train_indices], y.iloc[train_indices])
    phase8_pred = phase8_model.predict(X_base.iloc[test_indices]).astype(int)
    phase8_prob = phase8_model.predict_proba(X_base.iloc[test_indices])[:, list(phase8_model.classes_).index(1)]
    phase8_metrics = binary_metrics(y.iloc[test_indices], phase8_pred, phase8_prob)

    phase9_model = Pipeline([
        # Keep feature CSV missingness intact; train-only median imputation plus
        # indicators is confined to the model pipeline.
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("classifier", make_classifier()),
    ])
    phase9_model.fit(X_temporal.iloc[train_indices], y.iloc[train_indices])
    phase9_pred = phase9_model.predict(X_temporal.iloc[test_indices]).astype(int)
    phase9_prob = phase9_model.predict_proba(X_temporal.iloc[test_indices])[:, list(phase9_model.classes_).index(1)]
    phase9_metrics = binary_metrics(y.iloc[test_indices], phase9_pred, phase9_prob)

    return {
        "split": {
            "random_state": 42,
            "method": "same stratified 80/20 split as Phase 8",
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_class_counts": y.iloc[train_indices].value_counts().sort_index().to_dict(),
            "test_class_counts": y.iloc[test_indices].value_counts().sort_index().to_dict(),
        },
        "phase8_metrics": phase8_metrics,
        "phase9_metrics": phase9_metrics,
        "delta_phase9_minus_phase8": {
            key: phase9_metrics[key] - phase8_metrics[key]
            if phase9_metrics[key] is not None and phase8_metrics[key] is not None else None
            for key in ("precision", "recall", "f1", "fpr", "fnr", "roc_auc", "pr_auc_average_precision")
        },
        "phase8_feature_columns": list(PHASE8_FEATURE_COLUMNS),
        "temporal_feature_columns": list(TEMPORAL_FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "phase9_model_pipeline": "train-only median imputer with missingness indicators, then Phase 8 RandomForestClassifier",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_temporal_comparison(), indent=2, sort_keys=True))
