"""FastAPI demo service: list synthetic cases, run assess_case on one, store and return the result.

All assessment logic is the existing ml.pipeline.assess_case with the fixed Phase 8 reference
model. This module adds no decision rule, score or level, and never acts on a real account.
Run from the repository root: .venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --port 8010
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from sqlalchemy import func, select

from backend.app.db import Assessment, DemoCase, create_db_engine, init_db
from ml.features.feature_generator import PREDICTION_WINDOW_MINUTES
from ml.models.reference_model import DEFAULT_ARTIFACT_DIR, load_reference_model
from ml.pipeline import assess_case, load_events_by_user

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "synthetic" / "ato_synthetic_events.csv"

# Shown to the dashboard. Evaluation-only fields (labels, scenario_type, PaySim fraud flags)
# are never returned by the API and are blanked by assess_case before any component runs.
PUBLIC_EVENT_FIELDS = ("event_id", "event_category", "event_type", "timestamp", "device_id", "channel",
                       "status", "amount", "beneficiary_id", "transaction_type", "source")

DISCLAIMER = ("Prototype on synthetic data. Outcomes are research indicators, not fraud verdicts; "
              "recommended actions are simulated. No validated risk score or level exists, and the "
              "model probability is uncalibrated.")


def _json_safe(value: Any) -> Any:
    """Make pipeline snapshots storable as JSON (Decimal baseline amounts become floats)."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    return value


def _trigger(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(event for event in events if event["event_category"] == "TELECOM")


def _event_phase(timestamp: datetime, trigger_time: datetime) -> str:
    if timestamp < trigger_time:
        return "PRE_TRIGGER"
    if timestamp == trigger_time:
        return "TRIGGER"
    if timestamp <= trigger_time + timedelta(minutes=PREDICTION_WINDOW_MINUTES):
        return "FEATURE_WINDOW"
    return "AFTER_WINDOW"


def _assessment_summary(row: Assessment) -> dict[str, Any]:
    return {
        "id": row.id, "user_id": row.user_id, "created_at": row.created_at.isoformat(),
        "outcome": row.outcome, "evidence_status": row.evidence_status,
        "recommended_action": row.recommended_action, "risk_level": row.risk_level,
        "risk_score_0_100": row.risk_score_0_100,
    }


def _assessment_detail(row: Assessment) -> dict[str, Any]:
    return {
        **_assessment_summary(row),
        "pipeline_assessment_id": row.pipeline_assessment_id, "pipeline_version": row.pipeline_version,
        "model_id": row.model_id, "model_sha256": row.model_sha256, "response": row.response,
        "feature_snapshot": row.feature_snapshot, "temporal_evidence": row.temporal_evidence,
        "baseline_profile": row.baseline_profile, "disclaimer": DISCLAIMER,
    }


def create_app(database_url: str | None = None, events_path: str | Path = DEFAULT_EVENTS_PATH,
               artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> FastAPI:
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        engine, backend = create_db_engine(database_url)
        state["sessions"], state["database_backend"] = init_db(engine), backend
        state["events"] = load_events_by_user(events_path)
        try:
            state["model"], state["model_metadata"] = load_reference_model(artifact_dir)
            state["model_error"] = None
        except Exception as exc:  # noqa: BLE001 - reported by /api/health; assessments then refuse
            state["model"], state["model_metadata"], state["model_error"] = None, None, str(exc)
        _seed_demo_cases(state)
        yield
        engine.dispose()

    app = FastAPI(title="SIM-Swap ATO Assessment Demo API", lifespan=lifespan)

    def case_events(user_id: str) -> list[dict[str, Any]]:
        with state["sessions"]() as session:
            if session.get(DemoCase, user_id) is None:
                raise HTTPException(404, f"{user_id} is not a demo case (Phase 8 test-split users only)")
        return state["events"][user_id]

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        metadata = state["model_metadata"] or {}
        return {
            "status": "ok" if state["model"] is not None else "degraded",
            "database_backend": state["database_backend"],
            "model_loaded": state["model"] is not None, "model_error": state["model_error"],
            "model_id": metadata.get("model_id"), "model_sha256": metadata.get("model_sha256"),
            "prediction_window_minutes": PREDICTION_WINDOW_MINUTES, "disclaimer": DISCLAIMER,
        }

    @app.get("/api/cases")
    def list_cases() -> list[dict[str, Any]]:
        with state["sessions"]() as session:
            cases = session.scalars(select(DemoCase).order_by(DemoCase.user_id)).all()
            latest_ids = select(func.max(Assessment.id)).group_by(Assessment.user_id)
            latest = {row.user_id: row for row in
                      session.scalars(select(Assessment).where(Assessment.id.in_(latest_ids)))}
        return [{
            "user_id": case.user_id, "account_id": case.account_id, "trigger_type": case.trigger_type,
            "trigger_time": case.trigger_time, "event_count": case.event_count,
            "latest_outcome": latest[case.user_id].outcome if case.user_id in latest else None,
        } for case in cases]

    @app.get("/api/cases/{user_id}/events")
    def list_case_events(user_id: str) -> dict[str, Any]:
        events = case_events(user_id)
        trigger_time = datetime.fromisoformat(_trigger(events)["timestamp"])
        rows = sorted(events, key=lambda event: event["timestamp"])
        return {
            "user_id": user_id, "trigger_time": trigger_time.isoformat(),
            "prediction_window_minutes": PREDICTION_WINDOW_MINUTES,
            "events": [{**{field: event.get(field) or None for field in PUBLIC_EVENT_FIELDS},
                        "phase": _event_phase(datetime.fromisoformat(event["timestamp"]), trigger_time)}
                       for event in rows],
        }

    @app.post("/api/cases/{user_id}/assess")
    def assess(user_id: str) -> dict[str, Any]:
        events = case_events(user_id)
        if state["model"] is None:
            raise HTTPException(503, f"reference model not loaded: {state['model_error']}")
        result = _json_safe(assess_case(events, model=state["model"], model_metadata=state["model_metadata"]))
        response = result["response"]
        row = Assessment(
            user_id=user_id, pipeline_assessment_id=response["assessment_id"], outcome=response["outcome"],
            evidence_status=response["evidence_status"], recommended_action=response["recommended_action"],
            risk_level=response["risk_level"], risk_score_0_100=response["risk_score_0_100"],
            pipeline_version=response["provenance"]["pipeline_version"],
            model_id=response["provenance"]["model_id"], model_sha256=response["provenance"]["model_sha256"],
            response=response, feature_snapshot=result["feature_snapshot"],
            temporal_evidence=result["temporal_evidence"], baseline_profile=result["baseline_profile"],
        )
        with state["sessions"]() as session:
            session.add(row)
            session.commit()
        return _assessment_detail(row)

    @app.get("/api/assessments")
    def list_assessments(limit: int = 50) -> list[dict[str, Any]]:
        with state["sessions"]() as session:
            rows = session.scalars(select(Assessment).order_by(Assessment.id.desc()).limit(limit)).all()
        return [_assessment_summary(row) for row in rows]

    @app.get("/api/assessments/{assessment_id}")
    def get_assessment(assessment_id: int) -> dict[str, Any]:
        with state["sessions"]() as session:
            row = session.get(Assessment, assessment_id)
        if row is None:
            raise HTTPException(404, f"assessment {assessment_id} not found")
        return _assessment_detail(row)

    return app


def _seed_demo_cases(state: dict[str, Any]) -> None:
    """Insert the Phase 8 test-split users once; the training split stays out (in-sample model)."""
    metadata = state["model_metadata"]
    if metadata is None:
        return
    with state["sessions"]() as session:
        existing = set(session.scalars(select(DemoCase.user_id)))
        for user_id in metadata["test_user_ids"]:
            events = state["events"].get(user_id)
            if user_id in existing or not events:
                continue
            trigger = _trigger(events)
            session.add(DemoCase(user_id=user_id, account_id=trigger["account_id"],
                                 trigger_event_id=trigger["event_id"], trigger_type=trigger["event_type"],
                                 trigger_time=trigger["timestamp"], event_count=len(events)))
        session.commit()


app = create_app()
