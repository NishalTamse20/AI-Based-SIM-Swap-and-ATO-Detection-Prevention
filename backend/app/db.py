"""Database tables and engine selection: PostgreSQL via DATABASE_URL, SQLite fallback for the demo.

Only demo case metadata and stored assessment results live here. Events stay in the research
CSV, so no research dataset is copied or modified.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

log = logging.getLogger(__name__)

SQLITE_FALLBACK_PATH = Path(__file__).resolve().parents[1] / "demo_fallback.db"
SQLITE_FALLBACK_URL = f"sqlite:///{SQLITE_FALLBACK_PATH.as_posix()}"


class Base(DeclarativeBase):
    pass


class DemoCase(Base):
    """One synthetic case the dashboard may assess (Phase 8 test-split users only)."""

    __tablename__ = "demo_cases"

    user_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(32))
    trigger_event_id: Mapped[str] = mapped_column(String(48))
    trigger_type: Mapped[str] = mapped_column(String(32))
    trigger_time: Mapped[str] = mapped_column(String(32))
    event_count: Mapped[int] = mapped_column(Integer)


class Assessment(Base):
    """One stored run of assess_case. risk_score_0_100 stays null by design (Phase 11.2)."""

    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 default=lambda: datetime.now(timezone.utc))
    pipeline_assessment_id: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(48))
    evidence_status: Mapped[str] = mapped_column(String(32))
    recommended_action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    risk_level: Mapped[str] = mapped_column(String(32))
    risk_score_0_100: Mapped[float | None] = mapped_column(Float, nullable=True)
    pipeline_version: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response: Mapped[dict] = mapped_column(JSON)
    feature_snapshot: Mapped[dict] = mapped_column(JSON)
    temporal_evidence: Mapped[dict] = mapped_column(JSON)
    baseline_profile: Mapped[dict] = mapped_column(JSON)


def _connect(url: str) -> Engine:
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return engine


def create_db_engine(url: str | None = None) -> tuple[Engine, str]:
    """Return (engine, backend name). An unreachable DATABASE_URL falls back to SQLite, loudly."""
    url = url or os.environ.get("DATABASE_URL")
    if url:
        try:
            engine = _connect(url)
            return engine, engine.dialect.name
        except Exception as exc:  # noqa: BLE001 - any connection failure triggers the fallback
            log.warning("DATABASE_URL unreachable (%s); falling back to SQLite at %s",
                        exc.__class__.__name__, SQLITE_FALLBACK_PATH)
    engine = _connect(SQLITE_FALLBACK_URL)
    return engine, "sqlite-fallback"


def init_db(engine: Engine) -> sessionmaker:
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
