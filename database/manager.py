from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime

from utils.timeutils import utcnow
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from config import Config, get_config
from database.models import (
    Alert,
    AnalyticsSnapshot,
    Base,
    DailySummary,
    FlatVisitSessionModel,
    GlobalPersonModel,
    LLMDecisionLogModel,
    Report,
    Setting,
    TrackEvent,
    Visit,
    Zone,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        url = self.config.database_url

        connect_args: Dict[str, Any] = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False

        self.engine = create_engine(
            url,
            echo=self.config.database.echo,
            pool_pre_ping=self.config.database.pool_pre_ping,
            connect_args=connect_args,
            future=True,
        )
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        logger.info("Database engine created (%s)", url)


    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        logger.info("Database schema ensured (%d tables).", len(Base.metadata.tables))

    def drop_all(self) -> None:
        Base.metadata.drop_all(self.engine)
        logger.warning("All database tables dropped.")

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Database transaction rolled back.")
            raise
        finally:
            session.close()


    def open_visit(
        self,
        session_id: str,
        track_id: int,
        entry_time: Optional[datetime] = None,
        role: str = "customer",
    ) -> int:
        with self.session() as s:
            visit = Visit(
                session_id=session_id,
                track_id=track_id,
                role=role,
                entry_time=entry_time or utcnow(),
                is_active=True,
            )
            s.add(visit)
            s.flush()
            return visit.id

    def close_visit(
        self,
        visit_id: int,
        exit_time: Optional[datetime] = None,
        trajectory: Optional[List] = None,
        max_occupancy_seen: Optional[int] = None,
    ) -> None:
        with self.session() as s:
            visit = s.get(Visit, visit_id)
            if visit is None:
                logger.warning("close_visit: no visit with id=%s", visit_id)
                return
            visit.exit_time = exit_time or utcnow()
            visit.duration_seconds = (visit.exit_time - visit.entry_time).total_seconds()
            visit.is_active = False
            if trajectory is not None:
                visit.trajectory = trajectory
            if max_occupancy_seen is not None:
                visit.max_occupancy_seen = max_occupancy_seen

    def close_stale_visits(self, session_id: str) -> int:
        with self.session() as s:
            stmt = select(Visit).where(Visit.session_id == session_id, Visit.is_active.is_(True))
            stale = s.scalars(stmt).all()
            now = utcnow()
            for visit in stale:
                visit.exit_time = now
                visit.duration_seconds = (now - visit.entry_time).total_seconds()
                visit.is_active = False
            return len(stale)

    def get_visits(
        self, session_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Visit]:
        with self.session() as s:
            stmt = select(Visit).order_by(Visit.entry_time.desc())
            if session_id:
                stmt = stmt.where(Visit.session_id == session_id)
            if limit:
                stmt = stmt.limit(limit)
            return list(s.scalars(stmt).all())


    def log_event(
        self,
        session_id: str,
        track_id: int,
        event_type: str,
        role: str = "customer",
        zone_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        with self.session() as s:
            s.add(
                TrackEvent(
                    session_id=session_id,
                    track_id=track_id,
                    event_type=event_type,
                    role=role,
                    zone_name=zone_name,
                    details=details,
                    timestamp=timestamp or utcnow(),
                )
            )


    def add_snapshot(self, session_id: str, **kwargs: Any) -> None:
        known = {c.name for c in AnalyticsSnapshot.__table__.columns}
        fields = {k: v for k, v in kwargs.items() if k in known}
        extra = {k: v for k, v in kwargs.items() if k not in known}
        if extra:
            fields["extra"] = {**(fields.get("extra") or {}), **extra}
        with self.session() as s:
            s.add(AnalyticsSnapshot(session_id=session_id, **fields))

    def get_snapshots(
        self,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[AnalyticsSnapshot]:
        with self.session() as s:
            stmt = select(AnalyticsSnapshot).order_by(AnalyticsSnapshot.timestamp.asc())
            if session_id:
                stmt = stmt.where(AnalyticsSnapshot.session_id == session_id)
            if since:
                stmt = stmt.where(AnalyticsSnapshot.timestamp >= since)
            if limit:
                stmt = stmt.limit(limit)
            return list(s.scalars(stmt).all())


    def add_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        session_id: Optional[str] = None,
        snapshot_path: Optional[str] = None,
    ) -> int:
        with self.session() as s:
            alert = Alert(
                alert_type=alert_type,
                message=message,
                severity=severity,
                session_id=session_id,
                snapshot_path=snapshot_path,
            )
            s.add(alert)
            s.flush()
            logger.info("Alert [%s/%s]: %s", severity, alert_type, message)
            return alert.id

    def get_alerts(
        self, unacknowledged_only: bool = False, limit: Optional[int] = 100
    ) -> List[Alert]:
        with self.session() as s:
            stmt = select(Alert).order_by(Alert.timestamp.desc())
            if unacknowledged_only:
                stmt = stmt.where(Alert.acknowledged.is_(False))
            if limit:
                stmt = stmt.limit(limit)
            return list(s.scalars(stmt).all())

    def acknowledge_alert(self, alert_id: int) -> None:
        with self.session() as s:
            alert = s.get(Alert, alert_id)
            if alert:
                alert.acknowledged = True


    def upsert_zone(
        self,
        name: str,
        zone_type: str,
        points: List,
        reserved: bool = False,
        enabled: bool = True,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self.session() as s:
            zone = s.scalars(select(Zone).where(Zone.name == name)).first()
            if zone is None:
                zone = Zone(name=name, zone_type=zone_type, points=points)
                s.add(zone)
            zone.zone_type = zone_type
            zone.points = points
            zone.reserved = reserved
            zone.enabled = enabled
            zone.meta = meta
            s.flush()
            return zone.id

    def get_zones(self, zone_type: Optional[str] = None, enabled_only: bool = True) -> List[Zone]:
        with self.session() as s:
            stmt = select(Zone).order_by(Zone.name.asc())
            if zone_type:
                stmt = stmt.where(Zone.zone_type == zone_type)
            if enabled_only:
                stmt = stmt.where(Zone.enabled.is_(True))
            return list(s.scalars(stmt).all())

    def delete_zone(self, zone_id: int) -> None:
        with self.session() as s:
            zone = s.get(Zone, zone_id)
            if zone:
                s.delete(zone)


    def upsert_daily_summary(self, summary_date: date, **kwargs: Any) -> None:
        with self.session() as s:
            summary = s.scalars(
                select(DailySummary).where(DailySummary.summary_date == summary_date)
            ).first()
            if summary is None:
                summary = DailySummary(summary_date=summary_date)
                s.add(summary)
            for key, value in kwargs.items():
                if hasattr(summary, key):
                    setattr(summary, key, value)

    def get_daily_summaries(self, limit: Optional[int] = 90) -> List[DailySummary]:
        with self.session() as s:
            stmt = select(DailySummary).order_by(DailySummary.summary_date.desc())
            if limit:
                stmt = stmt.limit(limit)
            return list(s.scalars(stmt).all())


    def add_report(
        self,
        title: str,
        report_type: str,
        file_path: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> int:
        with self.session() as s:
            report = Report(
                title=title,
                report_type=report_type,
                file_path=file_path,
                period_start=period_start,
                period_end=period_end,
            )
            s.add(report)
            s.flush()
            return report.id

    def get_reports(self, limit: Optional[int] = 100) -> List[Report]:
        with self.session() as s:
            stmt = select(Report).order_by(Report.generated_at.desc())
            if limit:
                stmt = stmt.limit(limit)
            return list(s.scalars(stmt).all())


    def set_setting(self, key: str, value: Any) -> None:
        with self.session() as s:
            setting = s.get(Setting, key)
            if setting is None:
                s.add(Setting(key=key, value=value))
            else:
                setting.value = value

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.session() as s:
            setting = s.get(Setting, key)
            return setting.value if setting is not None else default

    def get_all_settings(self) -> Dict[str, Any]:
        with self.session() as s:
            return {row.key: row.value for row in s.scalars(select(Setting)).all()}


    def clear_analytics(self) -> Dict[str, int]:
        from sqlalchemy import delete

        removed: Dict[str, int] = {}
        with self.session() as s:
            for model in (Visit, TrackEvent, AnalyticsSnapshot, Alert, DailySummary):
                result = s.execute(delete(model))
                removed[model.__tablename__] = result.rowcount or 0
        logger.warning("Cleared analytics data: %s", removed)
        return removed

    def table_counts(self) -> Dict[str, int]:
        with self.session() as s:
            return {
                "visits": int(s.scalar(select(func.count(Visit.id))) or 0),
                "snapshots": int(s.scalar(select(func.count(AnalyticsSnapshot.id))) or 0),
                "alerts": int(s.scalar(select(func.count(Alert.id))) or 0),
                "zones": int(s.scalar(select(func.count(Zone.id))) or 0),
                "reports": int(s.scalar(select(func.count(Report.id))) or 0),
            }

    def count_visits(self, session_id: Optional[str] = None, role: Optional[str] = None) -> int:
        with self.session() as s:
            stmt = select(func.count(Visit.id))
            if session_id:
                stmt = stmt.where(Visit.session_id == session_id)
            if role:
                stmt = stmt.where(Visit.role == role)
            return int(s.scalar(stmt) or 0)


_DB: Optional[DatabaseManager] = None


def get_db() -> DatabaseManager:
    global _DB
    if _DB is None:
        _DB = DatabaseManager()
        _DB.create_all()
    return _DB


def reset_db() -> None:
    global _DB
    if _DB is not None:
        _DB.engine.dispose()
    _DB = None
