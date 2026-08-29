from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from analytics.base import BaseAnalyzer
from config import Config, get_config
from tracking.track_state import TrackFrameResult
from utils.geometry import Polygon, point_in_polygon
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AlertEvent:
    alert_type: str
    severity: str
    message: str
    zone_name: str


@dataclass
class QueueZoneState:
    name: str
    polygon: Polygon
    kind: str = "counter"
    zone_id: Optional[int] = None
    current_ids: set = field(default_factory=set)
    member_since: Dict[int, datetime] = field(default_factory=dict)
    max_length: int = 0

    @property
    def length(self) -> int:
        return len(self.current_ids)


@dataclass
class QueueMetrics:
    total_queue_length: int = 0
    max_queue_length: int = 0
    avg_wait_seconds: float = 0.0
    max_wait_seconds: float = 0.0
    longest_waiting_seconds: float = 0.0
    longest_waiting_id: Optional[int] = None
    active_alerts: int = 0
    per_queue: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueueAnalytics(BaseAnalyzer):
    def __init__(
        self,
        config: Optional[Config] = None,
        db: Optional[Any] = None,
        queues: Optional[List[QueueZoneState]] = None,
        session_id: str = "session",
        role: str = "customer",
    ) -> None:
        self.config = config or get_config()
        self.acfg = self.config.analytics
        self.db = db
        self.session_id = session_id
        self.role = role
        self._queues: List[QueueZoneState] = queues or []
        if not self._queues and db is not None:
            self.reload_queues()
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self._completed_waits: List[float] = []
        self._max_queue_length = 0
        self._trend: List[tuple] = []
        self._longest_now: float = 0.0
        self._longest_id: Optional[int] = None
        self._active_alerts = 0
        self._last_alert: Dict[tuple, datetime] = {}


    def reload_queues(self) -> None:
        if self.db is None:
            return
        zones = self.db.get_zones(zone_type="queue")
        self._queues = [
            QueueZoneState(
                name=z.name,
                polygon=[(float(x), float(y)) for x, y in z.points],
                kind=(z.meta or {}).get("kind", "counter") if z.meta else "counter",
                zone_id=z.id,
            )
            for z in zones
        ]
        logger.info("Loaded %d queue zone(s).", len(self._queues))

    @property
    def queues(self) -> List[QueueZoneState]:
        return self._queues


    def update(self, track_result: TrackFrameResult) -> List[AlertEvent]:
        ts = track_result.timestamp
        customers = [
            t for t in track_result.active_tracks
            if t.role == self.role and t.current_anchor is not None
        ]

        self._longest_now = 0.0
        self._longest_id = None
        alerts: List[AlertEvent] = []

        for queue in self._queues:
            present = {
                t.track_id for t in customers
                if point_in_polygon(t.current_anchor, queue.polygon)
            }
            for tid in present - queue.current_ids:
                queue.member_since[tid] = ts
            for tid in queue.current_ids - present:
                since = queue.member_since.pop(tid, None)
                if since is not None:
                    self._completed_waits.append((ts - since).total_seconds())

            queue.current_ids = present
            queue.max_length = max(queue.max_length, queue.length)
            self._max_queue_length = max(self._max_queue_length, queue.length)

            for tid, since in queue.member_since.items():
                waited = (ts - since).total_seconds()
                if waited > self._longest_now:
                    self._longest_now = waited
                    self._longest_id = tid

            alerts.extend(self._evaluate_alerts(queue, ts))

        self._trend.append((ts, sum(q.length for q in self._queues)))
        self._active_alerts = len(alerts)
        return alerts

    def _evaluate_alerts(self, queue: QueueZoneState, ts: datetime) -> List[AlertEvent]:
        raised: List[AlertEvent] = []

        if queue.length > self.acfg.queue_length_alert:
            raised.extend(
                self._maybe_alert(
                    queue, ts, "long_queue", "warning",
                    f"Queue '{queue.name}' has {queue.length} people "
                    f"(threshold {self.acfg.queue_length_alert}).",
                )
            )

        longest = max(
            ((ts - s).total_seconds() for s in queue.member_since.values()), default=0.0
        )
        if longest > self.acfg.queue_wait_alert_seconds:
            raised.extend(
                self._maybe_alert(
                    queue, ts, "long_wait", "critical",
                    f"A customer has waited {longest:.0f}s in '{queue.name}' "
                    f"(threshold {self.acfg.queue_wait_alert_seconds:.0f}s).",
                )
            )
        return raised

    def _maybe_alert(
        self, queue: QueueZoneState, ts: datetime, atype: str, severity: str, message: str
    ) -> List[AlertEvent]:
        key = (queue.name, atype)
        last = self._last_alert.get(key)
        if last is not None and (ts - last).total_seconds() < self.acfg.alert_cooldown_seconds:
            return []
        self._last_alert[key] = ts
        event = AlertEvent(alert_type=atype, severity=severity, message=message, zone_name=queue.name)
        if self.db is not None:
            self.db.add_alert(atype, message, severity=severity, session_id=self.session_id)
        logger.info("Queue alert [%s] %s", severity, message)
        return [event]


    def compute(self) -> QueueMetrics:
        waits = self._completed_waits
        per_queue = [
            {
                "name": q.name,
                "kind": q.kind,
                "length": q.length,
                "max_length": q.max_length,
            }
            for q in self._queues
        ]
        return QueueMetrics(
            total_queue_length=sum(q.length for q in self._queues),
            max_queue_length=self._max_queue_length,
            avg_wait_seconds=round(sum(waits) / len(waits), 1) if waits else 0.0,
            max_wait_seconds=round(max(waits), 1) if waits else 0.0,
            longest_waiting_seconds=round(self._longest_now, 1),
            longest_waiting_id=self._longest_id,
            active_alerts=self._active_alerts,
            per_queue=per_queue,
        )

    def metrics(self) -> Dict[str, Any]:
        return self.compute().to_dict()

    def queue_trend(self) -> List[tuple]:
        return list(self._trend)

    def reset(self) -> None:
        for queue in self._queues:
            queue.current_ids = set()
            queue.member_since = {}
            queue.max_length = 0
        self._reset_metrics()
        logger.debug("QueueAnalytics reset.")
