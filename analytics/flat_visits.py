from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config import Config, get_config
from tracking.reid import GlobalPerson, PersonGallery
from utils.logger import get_logger
from utils.timeutils import utcnow

logger = get_logger(__name__)


@dataclass
class FlatVisitSession:
    session_id: str
    sales_person_id: str
    sales_person_name: str
    accompanying_visitor_ids: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=utcnow)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    is_active: bool = True
    peak_group_size: int = 1
    last_activity_time: datetime = field(default_factory=utcnow)
    notes: str = ""

    @property
    def visitor_count(self) -> int:
        return len(self.accompanying_visitor_ids)

    @property
    def total_group_size(self) -> int:
        return self.visitor_count + (1 if self.sales_person_id else 0)

    def update_activity(self, timestamp: datetime, active_visitor_ids: List[str]) -> None:
        self.last_activity_time = timestamp
        self.duration_seconds = max(0.0, (timestamp - self.start_time).total_seconds())

        for vid in active_visitor_ids:
            if vid not in self.accompanying_visitor_ids:
                self.accompanying_visitor_ids.append(vid)

        current_size = len(active_visitor_ids) + 1
        if current_size > self.peak_group_size:
            self.peak_group_size = current_size

    def close(self, timestamp: datetime) -> None:
        self.is_active = False
        self.end_time = timestamp
        self.duration_seconds = max(0.0, (timestamp - self.start_time).total_seconds())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "sales_person_id": self.sales_person_id,
            "sales_person_name": self.sales_person_name,
            "accompanying_visitor_ids": list(self.accompanying_visitor_ids),
            "visitor_count": self.visitor_count,
            "total_group_size": self.total_group_size,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S") if self.end_time else "In Progress",
            "duration_seconds": round(self.duration_seconds, 1),
            "duration_formatted": self.format_duration(),
            "is_active": self.is_active,
            "peak_group_size": self.peak_group_size,
            "notes": self.notes,
        }

    def format_duration(self) -> str:
        s = int(self.duration_seconds)
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {sec}s"
        return f"{m}m {sec}s"


@dataclass
class FlatVisitEvent:
    timestamp: datetime
    event_type: str  # "session_start", "visitor_joined", "visitor_left", "session_end", "re_entry"
    message: str
    session_id: Optional[str] = None
    person_id: Optional[str] = None
    role: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.strftime("%H:%M:%S"),
            "event_type": self.event_type,
            "message": self.message,
            "session_id": self.session_id,
            "person_id": self.person_id,
            "role": self.role,
        }


class FlatVisitTracker:
    """Tracks Flat Visit Sessions, Sales Person interactions, accompanying visitors, and dwell time."""

    def __init__(self, config: Optional[Config] = None, db: Optional[Any] = None) -> None:
        self.config = config or get_config()
        self.flat_cfg = self.config.flat_visit
        self.db = db
        self.inactivity_timeout = self.flat_cfg.session_inactivity_timeout_seconds
        self.min_visit_seconds = self.flat_cfg.min_visit_seconds

        self._active_sessions: Dict[str, FlatVisitSession] = {}  # sales_person_id -> session
        self._completed_sessions: List[FlatVisitSession] = []
        self._session_counter: int = 1
        self._events: List[FlatVisitEvent] = []
        self._max_events: int = 200

    def log_event(
        self,
        event_type: str,
        message: str,
        timestamp: Optional[datetime] = None,
        session_id: Optional[str] = None,
        person_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> None:
        ts = timestamp or utcnow()
        evt = FlatVisitEvent(
            timestamp=ts,
            event_type=event_type,
            message=message,
            session_id=session_id,
            person_id=person_id,
            role=role,
        )
        self._events.insert(0, evt)
        if len(self._events) > self._max_events:
            self._events.pop()
        logger.info("[EVENT] %s - %s", event_type.upper(), message)

    def update(
        self,
        gallery: PersonGallery,
        timestamp: Optional[datetime] = None,
    ) -> None:
        ts = timestamp or utcnow()

        all_persons = gallery.get_all_persons()
        active_sales = [p for p in all_persons if p.is_active and p.role == "sales_person"]
        active_visitors = [p for p in all_persons if p.is_active and p.role == "visitor"]
        active_visitor_ids = [p.global_id for p in active_visitors]

        # Case A: If there are designated sales persons present
        if active_sales:
            for sp in active_sales:
                session = self._active_sessions.get(sp.global_id)
                if session is None:
                    # Start new Flat Visit Session
                    sid = f"VISIT-{ts.strftime('%Y%m%d')}-{self._session_counter:03d}"
                    self._session_counter += 1
                    session = FlatVisitSession(
                        session_id=sid,
                        sales_person_id=sp.global_id,
                        sales_person_name=sp.display_name,
                        accompanying_visitor_ids=list(active_visitor_ids),
                        start_time=ts,
                        last_activity_time=ts,
                        is_active=True,
                    )
                    self._active_sessions[sp.global_id] = session
                    self.log_event(
                        "session_start",
                        f"Sales Person {sp.display_name} started flat visit tour ({sid}) with {len(active_visitor_ids)} visitor(s).",
                        timestamp=ts,
                        session_id=sid,
                        person_id=sp.global_id,
                        role="sales_person",
                    )
                else:
                    # Update active session
                    prev_visitors = set(session.accompanying_visitor_ids)
                    session.update_activity(ts, active_visitor_ids)
                    new_joined = set(session.accompanying_visitor_ids) - prev_visitors
                    for n_vid in new_joined:
                        v_person = gallery.get_person(n_vid)
                        v_name = v_person.display_name if v_person else n_vid
                        self.log_event(
                            "visitor_joined",
                            f"{v_name} ({n_vid}) joined flat visit session {session.session_id}.",
                            timestamp=ts,
                            session_id=session.session_id,
                            person_id=n_vid,
                            role="visitor",
                        )

        # Case B: If no sales person is designated yet, but visitors are present in the flat
        elif active_visitors and self.flat_cfg.auto_detect_sessions:
            # Check for general visitor session
            general_key = "__general_visit__"
            session = self._active_sessions.get(general_key)
            if session is None:
                sid = f"VISIT-{ts.strftime('%Y%m%d')}-{self._session_counter:03d}"
                self._session_counter += 1
                session = FlatVisitSession(
                    session_id=sid,
                    sales_person_id="UNASSIGNED",
                    sales_person_name="Self / Unassigned Agent",
                    accompanying_visitor_ids=list(active_visitor_ids),
                    start_time=ts,
                    last_activity_time=ts,
                    is_active=True,
                )
                self._active_sessions[general_key] = session
                self.log_event(
                    "session_start",
                    f"New Flat Visit Session started ({sid}) with {len(active_visitor_ids)} visitor(s).",
                    timestamp=ts,
                    session_id=sid,
                    role="visitor",
                )
            else:
                session.update_activity(ts, active_visitor_ids)

        # Check for sessions that should be closed (inactivity or absence)
        sessions_to_close = []
        for key, session in list(self._active_sessions.items()):
            # If sales person and all visitors left, or timeout passed
            is_sales_active = gallery.get_person(session.sales_person_id)
            sales_present = is_sales_active.is_active if is_sales_active else False
            visitors_present = any(
                gallery.get_person(vid).is_active
                for vid in session.accompanying_visitor_ids
                if gallery.get_person(vid)
            )

            inactive_duration = (ts - session.last_activity_time).total_seconds()

            if (not sales_present and not visitors_present) or inactive_duration > self.inactivity_timeout:
                sessions_to_close.append(key)

        for key in sessions_to_close:
            session = self._active_sessions.pop(key)
            session.close(ts)
            if session.duration_seconds >= self.min_visit_seconds or session.visitor_count > 0:
                self._completed_sessions.append(session)
                self.log_event(
                    "session_end",
                    f"Flat Visit Session {session.session_id} completed. Duration: {session.format_duration()} · Total Visitors: {session.visitor_count}.",
                    timestamp=ts,
                    session_id=session.session_id,
                )

    def get_active_sessions(self) -> List[FlatVisitSession]:
        return list(self._active_sessions.values())

    def get_completed_sessions(self) -> List[FlatVisitSession]:
        return list(self._completed_sessions)

    def get_all_sessions(self) -> List[FlatVisitSession]:
        # Active sessions first, then completed sessions (most recent first)
        active = list(self._active_sessions.values())
        completed = list(reversed(self._completed_sessions))
        return active + completed

    def get_events(self) -> List[dict]:
        return [e.to_dict() for e in self._events]

    def compute_kpis(self, gallery: PersonGallery) -> Dict[str, Any]:
        all_sessions = self.get_all_sessions()
        completed = self._completed_sessions
        durations = [s.duration_seconds for s in completed if s.duration_seconds > 0]
        avg_duration = (sum(durations) / len(durations)) if durations else 0.0

        visitor_counts = [s.visitor_count for s in all_sessions if s.visitor_count > 0]
        avg_visitors_per_tour = (sum(visitor_counts) / len(visitor_counts)) if visitor_counts else 0.0

        total_unique_visitors = gallery.total_unique_visitors_count()
        active_visitors = gallery.active_visitors_count()
        active_sales = gallery.active_sales_count()

        returning_visitors = sum(1 for p in gallery.get_all_persons() if p.role == "visitor" and p.visit_count > 1)
        return_rate = (returning_visitors / total_unique_visitors * 100.0) if total_unique_visitors > 0 else 0.0

        return {
            "total_visits": len(all_sessions),
            "active_visits": len(self._active_sessions),
            "total_unique_visitors": total_unique_visitors,
            "active_visitors": active_visitors,
            "active_sales_agents": active_sales,
            "avg_visit_duration_seconds": avg_duration,
            "avg_visitors_per_tour": round(avg_visitors_per_tour, 1),
            "returning_visitors_count": returning_visitors,
            "returning_visitor_rate": round(return_rate, 1),
        }

    def get_sales_person_analytics(self, gallery: PersonGallery) -> List[Dict[str, Any]]:
        """Computes comprehensive metrics and tour logs for each Sales Person."""
        all_sessions = self.get_all_sessions()
        all_persons = gallery.get_all_persons()
        sales_persons = [p for p in all_persons if p.role == "sales_person"]

        records = []
        for p in sales_persons:
            p_sessions = [s for s in all_sessions if s.sales_person_id == p.global_id]
            total_visits_conducted = len(p_sessions)

            visitors_set = set()
            for s in p_sessions:
                visitors_set.update(s.accompanying_visitor_ids)
            total_visitors_shown = len(visitors_set)

            durations = [s.duration_seconds for s in p_sessions if s.duration_seconds > 0]
            total_duty_sec = max(p.total_dwell_seconds, sum(durations))
            avg_tour_sec = (sum(durations) / len(durations)) if durations else 0.0

            active_s = self._active_sessions.get(p.global_id)

            records.append({
                "sales_person_id": p.global_id,
                "name": p.display_name,
                "role": "Sales Agent",
                "is_active": p.is_active,
                "current_status": f"Leading Tour ({active_s.session_id})" if active_s else ("In Flat (Standby)" if p.is_active else "Away / Off-duty"),
                "total_visits_conducted": total_visits_conducted,
                "total_visitors_shown": total_visitors_shown,
                "total_duty_seconds": total_duty_sec,
                "total_duty_formatted": self._format_sec(total_duty_sec),
                "avg_tour_seconds": avg_tour_sec,
                "avg_tour_formatted": self._format_sec(avg_tour_sec),
                "thumbnail_path": p.thumbnail_path,
                "ai_persona": p.semantic_profile.persona_summary if p.semantic_profile else "Sales Agent",
                "sessions_list": p_sessions,
            })
        return records

    def get_visitor_logs(self, gallery: PersonGallery) -> List[Dict[str, Any]]:
        """Computes detailed visitor logs with assigned sales agent, dwell time, and AI persona."""
        all_sessions = self.get_all_sessions()
        all_persons = gallery.get_all_persons()
        visitors = [p for p in all_persons if p.role == "visitor"]

        visitor_agent_map: Dict[str, List[str]] = {}
        for s in all_sessions:
            for vid in s.accompanying_visitor_ids:
                if vid not in visitor_agent_map:
                    visitor_agent_map[vid] = []
                if s.sales_person_name not in visitor_agent_map[vid]:
                    visitor_agent_map[vid].append(s.sales_person_name)

        records = []
        for p in visitors:
            agents = visitor_agent_map.get(p.global_id, ["Self / Unassigned"])
            prof = p.semantic_profile
            records.append({
                "visitor_id": p.global_id,
                "name": p.display_name,
                "role": "Visitor",
                "is_active": p.is_active,
                "status": "🟢 In Flat" if p.is_active else "⚪ Left Flat",
                "assigned_agents": ", ".join(agents),
                "visits_count": p.visit_count,
                "is_returning": p.visit_count > 1,
                "first_seen": p.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
                "last_seen": p.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
                "total_dwell_seconds": p.total_dwell_seconds,
                "dwell_formatted": self._format_sec(p.total_dwell_seconds),
                "ai_persona": prof.persona_summary if prof else "Visitor in frame",
                "top_clothing": prof.upper_clothing if prof else "—",
                "bottom_clothing": prof.lower_clothing if prof else "—",
                "accessories": ", ".join(prof.accessories) if (prof and prof.accessories) else "None",
                "ai_reasoning": p.llm_reasoning or "Standard entry",
                "thumbnail_path": p.thumbnail_path,
            })
        return records

    @staticmethod
    def _format_sec(seconds: float) -> str:
        s = int(seconds or 0)
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {sec}s"
        if m > 0:
            return f"{m}m {sec}s"
        return f"{sec}s"

    def reset(self) -> None:
        self._active_sessions.clear()
        self._completed_sessions.clear()
        self._session_counter = 1
        self._events.clear()
        logger.info("FlatVisitTracker reset.")
