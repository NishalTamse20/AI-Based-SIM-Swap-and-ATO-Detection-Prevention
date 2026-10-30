"""Assess one SIM/eSIM-triggered case by chaining the existing Phase 5–11 components.

events → pre-trigger baseline (Phase 5) → T..T+15 features (Phase 6) and temporal
evidence (Phase 9) → multi-signal rule (Phase 7) → fixed reference model (Phase 8)
→ evidence record (Phase 10.2) → explainable response (Phase 11.1).

No decision rule is added here: the outcome comes from build_explainable_response.
Labels and scenario_type are removed before any component runs, risk_score_0_100
stays null and risk_level stays UNDETERMINED (Phase 11.2). Model probabilities are
uncalibrated. Outputs are prototype indicators on synthetic data, not fraud verdicts.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.features.feature_generator import FEATURE_COLUMNS, IDENTIFIER_COLUMNS, PREDICTION_WINDOW_MINUTES, build_feature_rows
from ml.features.temporal_features import FEATURE_COLUMNS as TEMPORAL_FEATURE_COLUMNS, build_temporal_feature_rows
from ml.models.reference_model import predict_case
from ml.risk.evidence_assembler import assemble_evidence
from ml.risk.explainable_response import build_explainable_response
from ml.rules.rule_baseline import multi_signal_prediction
from simulator.baseline.behavioural_baseline import build_baseline_profiles, validate_baseline_cutoffs
from simulator.validation.dataset_validator import CATEGORY_EVENT_TYPES


PIPELINE_VERSION = "single-case-assessment-v1"
EVALUATION_ONLY_FIELDS = ("ato_label", "scenario_type")


def load_events_by_user(events_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Read an event CSV once and group its rows by user_id."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(events_path).open("r", newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            grouped[row["user_id"]].append(row)
    return dict(grouped)


def load_case_events(events_path: str | Path, user_id: str) -> list[dict[str, Any]]:
    events = load_events_by_user(events_path).get(user_id)
    if not events:
        raise ValueError(f"no events for user {user_id!r}")
    return events


def _unlabelled(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy events with evaluation-only fields blanked so they cannot influence any component."""
    return [{**dict(event), **{field: None for field in EVALUATION_ONLY_FIELDS}} for event in events]


def _validate_case(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        raise ValueError("a case needs at least its telecom trigger and pre-trigger history")
    if len({event.get("user_id") for event in events}) != 1 or len({event.get("account_id") for event in events}) != 1:
        raise ValueError("a case must contain exactly one user_id and one account_id")
    # event_id is the schema's unique event identifier; within one case a repeated id is a
    # duplicate record and would be double-counted by the baseline, so it is rejected.
    seen_event_ids: set[str] = set()
    for event in events:
        for field in ("event_id", "timestamp"):
            if not str(event.get(field) or "").strip():
                raise ValueError(f"event {event.get('event_id')!r} is missing required field {field!r}")
        event_id = str(event["event_id"])
        if event_id in seen_event_ids:
            raise ValueError(f"duplicate event_id {event_id!r}: each event must appear once in a case")
        seen_event_ids.add(event_id)
        category, event_type = event.get("event_category"), event.get("event_type")
        if event_type not in CATEGORY_EVENT_TYPES.get(category, ()):
            raise ValueError(f"invalid event_type {category}/{event_type} in {event.get('event_id')}")
        datetime.fromisoformat(str(event["timestamp"]))
    triggers = [event for event in events if event["event_category"] == "TELECOM"]
    if len(triggers) != 1:
        raise ValueError(f"a case must contain exactly one SIM/eSIM trigger, found {len(triggers)}")
    return triggers[0]


def assess_case(events: Iterable[Mapping[str, Any]], *, model: Any | None = None,
                model_metadata: Mapping[str, Any] | None = None,
                assessment_id: str | None = None) -> dict[str, Any]:
    """Return the explainable response plus the evidence snapshots it was built from."""
    case = _unlabelled(events)
    trigger = _validate_case(case)
    trigger_time = datetime.fromisoformat(str(trigger["timestamp"]))
    history = [event for event in case if datetime.fromisoformat(str(event["timestamp"])) < trigger_time]

    # Phase 5 sees only pre-trigger history plus the trigger itself.
    baseline_input = [*history, trigger]
    profiles = build_baseline_profiles(baseline_input)
    cutoff_errors = validate_baseline_cutoffs(baseline_input, profiles)
    if cutoff_errors:
        raise ValueError("baseline cutoff validation failed: " + "; ".join(cutoff_errors))

    feature_row = build_feature_rows(case, profiles)[0]
    temporal_row = build_temporal_feature_rows(case)[0]
    temporal_evidence = {column: temporal_row[column] for column in TEMPORAL_FEATURE_COLUMNS}
    rule_output = multi_signal_prediction(feature_row)
    model_output = predict_case(model, feature_row) if model is not None else None

    identifiers = {column: str(feature_row[column]) for column in IDENTIFIER_COLUMNS}
    provenance = {
        "pipeline_version": PIPELINE_VERSION,
        "prediction_window_minutes": PREDICTION_WINDOW_MINUTES,
        "rule_component": "Phase 7 multi_signal_prediction",
        "model_id": (model_metadata or {}).get("model_id") if model is not None else None,
        "model_sha256": (model_metadata or {}).get("model_sha256") if model is not None else None,
        "model_probability_calibrated": False,
        "evaluation_only_fields_removed": list(EVALUATION_ONLY_FIELDS),
    }
    evidence = assemble_evidence(assessment_id or f"ASSESS-{identifiers['event_id']}", identifiers,
                                 rule_output=rule_output, model_output=model_output,
                                 temporal_evidence=temporal_evidence, provenance=provenance)
    return {
        "response": build_explainable_response(evidence),
        "feature_snapshot": {column: feature_row[column] for column in (*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS)},
        "temporal_evidence": temporal_evidence,
        "baseline_profile": profiles[0],
    }
