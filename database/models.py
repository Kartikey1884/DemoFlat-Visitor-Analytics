from __future__ import annotations

import json
from datetime import datetime, date
from typing import Any, List, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from utils.timeutils import utcnow


class Base(DeclarativeBase):
    pass


class JSONEncoded(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return json.loads(value)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


# =========================================================================
# REAL ESTATE FLAT VISITS & SALES PERSON ANALYTICS
# =========================================================================

class FlatVisitSessionModel(Base, TimestampMixin):
    """Stores full session records for flat visit tours led by sales agents or unassigned groups."""
    __tablename__ = "flat_visit_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    sales_person_id: Mapped[str] = mapped_column(String(32), index=True)
    sales_person_name: Mapped[str] = mapped_column(String(64))
    accompanying_visitor_ids: Mapped[Optional[list]] = mapped_column(JSONEncoded, nullable=True)
    visitor_count: Mapped[int] = mapped_column(Integer, default=0)
    start_time: Mapped[datetime] = mapped_column(DateTime, index=True, default=utcnow)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    peak_group_size: Mapped[int] = mapped_column(Integer, default=1)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_session_active_sales", "sales_person_id", "is_active"),
        Index("idx_session_start_end", "start_time", "end_time"),
    )


class GlobalPersonModel(Base, TimestampMixin):
    """Stores persistent identity, AI clothing signatures, and visit history for each unique person."""
    __tablename__ = "global_persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    person_number: Mapped[int] = mapped_column(Integer, index=True)
    display_name: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(24), default="visitor", index=True)  # "visitor" or "sales_person"
    visit_count: Mapped[int] = mapped_column(Integer, default=1)
    total_dwell_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    semantic_profile: Mapped[Optional[dict]] = mapped_column(JSONEncoded, nullable=True)
    llm_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_person_role_active", "role", "is_active"),
        Index("idx_person_seen_duration", "first_seen", "total_dwell_seconds"),
    )


class LLMDecisionLogModel(Base, TimestampMixin):
    """Audit log of real-time AI/LLM biometric Re-ID decisions and visual reasoning."""
    __tablename__ = "llm_decision_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    decision: Mapped[str] = mapped_column(String(24), index=True)  # "MATCH" or "NEW_PERSON"
    matched_global_id: Mapped[Optional[str]] = mapped_column(String(32), index=True, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    reasoning: Mapped[str] = mapped_column(Text)
    persona_summary: Mapped[str] = mapped_column(Text)
    engine: Mapped[str] = mapped_column(String(64))

    __table_args__ = (
        Index("idx_decision_track_time", "track_id", "timestamp"),
    )


# =========================================================================
# CORE TRACKING & ANALYTICS MODELS
# =========================================================================

class Visit(Base, TimestampMixin):
    __tablename__ = "visits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    role: Mapped[str] = mapped_column(String(16), default="customer")
    entry_time: Mapped[datetime] = mapped_column(DateTime, index=True)
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_occupancy_seen: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    trajectory: Mapped[Optional[list]] = mapped_column(JSONEncoded, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class TrackEvent(Base, TimestampMixin):
    __tablename__ = "track_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    role: Mapped[str] = mapped_column(String(16), default="customer")
    zone_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, default=utcnow)
    details: Mapped[Optional[dict]] = mapped_column(JSONEncoded, nullable=True)


class AnalyticsSnapshot(Base, TimestampMixin):
    __tablename__ = "analytics_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, default=utcnow)
    current_customers: Mapped[int] = mapped_column(Integer, default=0)
    current_staff: Mapped[int] = mapped_column(Integer, default=0)
    occupied_tables: Mapped[int] = mapped_column(Integer, default=0)
    empty_tables: Mapped[int] = mapped_column(Integer, default=0)
    queue_length: Mapped[int] = mapped_column(Integer, default=0)
    avg_wait_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    extra: Mapped[Optional[dict]] = mapped_column(JSONEncoded, nullable=True)


class DailySummary(Base, TimestampMixin):
    __tablename__ = "daily_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    summary_date: Mapped[date] = mapped_column(Date, unique=True, index=True)
    total_visitors: Mapped[int] = mapped_column(Integer, default=0)
    peak_occupancy: Mapped[int] = mapped_column(Integer, default=0)
    peak_hour: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_stay_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    max_queue_length: Mapped[int] = mapped_column(Integer, default=0)
    avg_wait_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    total_alerts: Mapped[int] = mapped_column(Integer, default=0)
    hourly_visitors: Mapped[Optional[dict]] = mapped_column(JSONEncoded, nullable=True)


class Alert(Base, TimestampMixin):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    alert_type: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="warning")
    message: Mapped[str] = mapped_column(String(255))
    timestamp: Mapped[datetime] = mapped_column(DateTime, index=True, default=utcnow)
    snapshot_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class Zone(Base, TimestampMixin):
    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    zone_type: Mapped[str] = mapped_column(String(24), index=True)
    points: Mapped[list] = mapped_column(JSONEncoded)
    reserved: Mapped[bool] = mapped_column(Boolean, default=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSONEncoded, nullable=True)


class Report(Base, TimestampMixin):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(128))
    report_type: Mapped[str] = mapped_column(String(16))
    file_path: Mapped[str] = mapped_column(String(255))
    period_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    period_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSONEncoded)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow
    )


__all__ = [
    "Base",
    "JSONEncoded",
    "FlatVisitSessionModel",
    "GlobalPersonModel",
    "LLMDecisionLogModel",
    "Visit",
    "TrackEvent",
    "AnalyticsSnapshot",
    "DailySummary",
    "Alert",
    "Zone",
    "Report",
    "Setting",
]
