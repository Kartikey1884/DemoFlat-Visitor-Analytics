from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from analytics.base import BaseAnalyzer, RunningStat
from config import Config, get_config
from tracking.track_state import TrackFrameResult, TrackState
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CustomerMetrics:
    current_customers: int = 0
    total_visitors: int = 0
    entries: int = 0
    exits: int = 0
    avg_stay_seconds: float = 0.0
    min_stay_seconds: float = 0.0
    max_stay_seconds: float = 0.0
    completed_visits: int = 0
    current_occupancy: int = 0
    max_occupancy: int = 0
    min_occupancy: int = 0
    avg_occupancy: float = 0.0
    hourly_visitors: Dict[int, int] = field(default_factory=dict)
    peak_hour: Optional[int] = None
    repeat_visits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hourly_visitors"] = {str(k): v for k, v in self.hourly_visitors.items()}
        return data


class CustomerAnalytics(BaseAnalyzer):
    def __init__(
        self,
        config: Optional[Config] = None,
        db: Optional[Any] = None,
        session_id: str = "session",
        role: str = "customer",
    ) -> None:
        self.config = config or get_config()
        self.db = db
        self.session_id = session_id
        self.role = role
        self.min_stay_seconds = self.config.analytics.min_stay_seconds
        self._reset_state()

    def _reset_state(self) -> None:
        self._entries = 0
        self._exits = 0
        self._current = 0
        self._unique_ids: set[int] = set()
        self._durations: List[float] = []
        self._hourly: Counter[int] = Counter()
        self._occupancy = RunningStat()
        self._open_visits: Dict[int, int] = {}


    def update(self, track_result: TrackFrameResult) -> None:
        active = [t for t in track_result.active_tracks if t.role == self.role]
        self._current = len(active)
        self._occupancy.add(self._current)

        for track in track_result.entered_tracks:
            if track.role != self.role:
                continue
            self._on_entry(track)

        for track in track_result.exited_tracks:
            if track.role != self.role:
                continue
            self._on_exit(track)

    def _on_entry(self, track: TrackState) -> None:
        self._entries += 1
        self._unique_ids.add(track.track_id)
        self._hourly[track.first_seen.hour] += 1
        if self.db is not None and track.track_id not in self._open_visits:
            visit_id = self.db.open_visit(
                self.session_id,
                track_id=track.track_id,
                entry_time=track.first_seen,
                role=self.role,
            )
            self._open_visits[track.track_id] = visit_id
            self.db.log_event(
                self.session_id, track.track_id, "enter", role=self.role,
                timestamp=track.first_seen,
            )

    def _on_exit(self, track: TrackState) -> None:
        self._exits += 1
        if track.duration_seconds >= self.min_stay_seconds:
            self._durations.append(track.duration_seconds)
        if self.db is not None:
            visit_id = self._open_visits.pop(track.track_id, None)
            if visit_id is not None:
                self.db.close_visit(
                    visit_id,
                    exit_time=track.last_seen,
                    trajectory=[[round(x, 1), round(y, 1)] for x, y in track.trajectory],
                    max_occupancy_seen=self._occupancy.maximum,
                )
            self.db.log_event(
                self.session_id, track.track_id, "exit", role=self.role,
                timestamp=track.last_seen,
            )


    def compute(self) -> CustomerMetrics:
        durations = self._durations
        peak_hour = max(self._hourly, key=self._hourly.get) if self._hourly else None
        return CustomerMetrics(
            current_customers=self._current,
            total_visitors=len(self._unique_ids),
            entries=self._entries,
            exits=self._exits,
            avg_stay_seconds=(sum(durations) / len(durations)) if durations else 0.0,
            min_stay_seconds=min(durations) if durations else 0.0,
            max_stay_seconds=max(durations) if durations else 0.0,
            completed_visits=len(durations),
            current_occupancy=self._current,
            max_occupancy=int(self._occupancy.maximum),
            min_occupancy=int(self._occupancy.minimum),
            avg_occupancy=round(self._occupancy.mean, 2),
            hourly_visitors=dict(sorted(self._hourly.items())),
            peak_hour=peak_hour,
            repeat_visits=0,
        )

    def metrics(self) -> Dict[str, Any]:
        return self.compute().to_dict()


    def persist_snapshot(self, extra: Optional[Dict[str, Any]] = None, fps: float = 0.0) -> None:
        if self.db is None:
            return
        m = self.compute()
        self.db.add_snapshot(
            self.session_id,
            current_customers=m.current_customers,
            avg_wait_seconds=0.0,
            fps=fps,
            **(extra or {}),
        )

    def finalize(self) -> None:
        if self.db is not None:
            closed = self.db.close_stale_visits(self.session_id)
            if closed:
                logger.info("Finalised %d open customer visit(s).", closed)
        self._open_visits.clear()

    def reset(self) -> None:
        self._reset_state()
        logger.debug("CustomerAnalytics reset (session=%s).", self.session_id)
