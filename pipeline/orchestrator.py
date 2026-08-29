from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from utils.timeutils import utcnow
from typing import Any, Dict, List, Optional

import numpy as np

from analytics.customer import CustomerAnalytics
from analytics.flat_visits import FlatVisitTracker
from analytics.heatmap import HeatmapGenerator, TrajectoryRenderer
from analytics.queue import AlertEvent, QueueAnalytics
from analytics.staff import StaffAnalytics, StaffClassifier
from analytics.table import TableOccupancy
from config import Config, get_config
from engine.engine import DetectionEngine, FrameResult
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OrchestratorMetrics:
    customer: Dict[str, Any] = field(default_factory=dict)
    staff: Dict[str, Any] = field(default_factory=dict)
    table: Dict[str, Any] = field(default_factory=dict)
    queue: Dict[str, Any] = field(default_factory=dict)
    flat_visits: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    fps: float = 0.0

    def kpis(self) -> Dict[str, Any]:
        c, s, t, q, fv = self.customer, self.staff, self.table, self.queue, self.flat_visits
        return {
            "current_customers": fv.get("active_visitors", c.get("current_customers", 0)),
            "total_visitors": fv.get("total_unique_visitors", c.get("total_visitors", 0)),
            "total_flat_visits": fv.get("total_visits", 0),
            "active_flat_visits": fv.get("active_visits", 0),
            "active_sales_agents": fv.get("active_sales_agents", s.get("current_staff", 0)),
            "avg_visitors_per_tour": fv.get("avg_visitors_per_tour", 0.0),
            "returning_visitors": fv.get("returning_visitors_count", 0),
            "returning_visitor_rate": fv.get("returning_visitor_rate", 0.0),
            "current_occupancy": c.get("current_occupancy", 0),
            "avg_stay_seconds": fv.get("avg_visit_duration_seconds", c.get("avg_stay_seconds", 0.0)),
            "max_occupancy": c.get("max_occupancy", 0),
            "peak_hour": c.get("peak_hour"),
            "occupied_tables": t.get("occupied_tables", 0),
            "empty_tables": t.get("empty_tables", 0),
            "available_tables": t.get("available_tables", 0),
            "occupancy_percentage": t.get("occupancy_percentage", 0.0),
            "queue_length": q.get("total_queue_length", 0),
            "avg_wait_seconds": q.get("avg_wait_seconds", 0.0),
            "longest_wait_seconds": q.get("longest_waiting_seconds", 0.0),
            "current_staff": s.get("current_staff", 0),
            "staff_ratio": s.get("staff_to_customer_ratio", 0.0),
            "fps": round(self.fps, 1),
        }


class AnalyticsOrchestrator:
    def __init__(
        self,
        config: Optional[Config] = None,
        db: Optional[Any] = None,
        session_id: Optional[str] = None,
        staff_mode: Optional[str] = None,
        snapshot_interval_seconds: float = 3.0,
        engine: Optional[DetectionEngine] = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db
        self.session_id = session_id or datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.snapshot_interval = snapshot_interval_seconds

        self.engine = engine or DetectionEngine(self.config)
        self.flat_visits = FlatVisitTracker(self.config, db=db)
        self.classifier = StaffClassifier(self.config, db=db, mode=staff_mode)
        self.customer = CustomerAnalytics(self.config, db=db, session_id=self.session_id)
        self.staff = StaffAnalytics(self.config, session_id=self.session_id)
        self.tables = TableOccupancy(self.config, db=db)
        self.queues = QueueAnalytics(self.config, db=db, session_id=self.session_id)
        self.trajectory = TrajectoryRenderer(self.config)

        self.heatmap_customer: Optional[HeatmapGenerator] = None
        self.heatmap_staff: Optional[HeatmapGenerator] = None

        self._last_frame: Optional[np.ndarray] = None
        self._last_alerts: List[AlertEvent] = []
        self._last_fps: float = 0.0
        self._last_snapshot_ts: Optional[datetime] = None
        self._global_alert_last: Dict[str, datetime] = {}
        self._no_staff_since: Optional[datetime] = None
        self._inference_frames = 0
        self._prune_every = 600


    def process(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: Optional[datetime] = None,
        run_inference: bool = True,
    ) -> FrameResult:
        result = self.engine.process(frame, frame_index, timestamp, run_inference)
        self._last_frame = frame
        self._last_fps = result.fps
        self._ensure_heatmaps(frame)

        tr = result.track_result
        if tr is not None:
            gallery = getattr(self.engine.manager, "gallery", None)
            if gallery is not None:
                self.flat_visits.update(gallery, result.timestamp)

            self.classifier.classify(tr, frame)
            customer_count = sum(1 for t in tr.active_tracks if t.role == "visitor" or t.role == "customer")
            self.customer.update(tr)
            self.staff.update(tr, customer_count=customer_count)
            self.tables.update(tr)
            alerts = list(self.queues.update(tr))
            self.heatmap_customer.update(tr, role="visitor")
            self.heatmap_staff.update(tr, role="sales_person")
            alerts.extend(self._evaluate_global_alerts(tr, result.timestamp))
            self._last_alerts = alerts
            self._maybe_snapshot(result.timestamp)
            self._inference_frames += 1
            if self._inference_frames % self._prune_every == 0:
                self.engine.manager.prune_finished(keep_last=50)

        return result

    def reset(self) -> None:
        self.engine.reset()
        self.flat_visits.reset()
        self.classifier.reset()
        self.customer.reset()
        self.staff.reset()
        self.tables.reset()
        self.queues.reset()
        if self.heatmap_customer is not None:
            self.heatmap_customer.reset()
        if self.heatmap_staff is not None:
            self.heatmap_staff.reset()
        self._last_alerts = []
        self._last_snapshot_ts = None
        self._global_alert_last = {}
        self._no_staff_since = None
        self._inference_frames = 0
        logger.info("Orchestrator reset (session=%s).", self.session_id)


    def _ensure_heatmaps(self, frame: np.ndarray) -> None:
        if self.heatmap_customer is None:
            h, w = frame.shape[:2]
            self.heatmap_customer = HeatmapGenerator(w, h, self.config)
            self.heatmap_staff = HeatmapGenerator(w, h, self.config)

    def _evaluate_global_alerts(self, tr, ts: datetime) -> List[AlertEvent]:
        raised: List[AlertEvent] = []
        acfg = self.config.analytics
        customers = sum(1 for t in tr.active_tracks if t.role == "customer")
        staff = sum(1 for t in tr.active_tracks if t.role == "staff")

        crowd_threshold = acfg.max_capacity * acfg.crowd_alert_ratio
        if customers >= crowd_threshold:
            raised.extend(
                self._maybe_global_alert(
                    ts, "overcrowd", "critical",
                    f"Cafe overcrowded: {customers} customers "
                    f"(capacity {acfg.max_capacity}).",
                )
            )

        if staff == 0 and customers > 0:
            if self._no_staff_since is None:
                self._no_staff_since = ts
            elif (ts - self._no_staff_since).total_seconds() >= 10:
                raised.extend(
                    self._maybe_global_alert(
                        ts, "no_staff", "warning",
                        "No staff detected on the floor while customers are present.",
                    )
                )
        else:
            self._no_staff_since = None

        return raised

    def _maybe_global_alert(
        self, ts: datetime, atype: str, severity: str, message: str
    ) -> List[AlertEvent]:
        last = self._global_alert_last.get(atype)
        cooldown = self.config.analytics.alert_cooldown_seconds
        if last is not None and (ts - last).total_seconds() < cooldown:
            return []
        self._global_alert_last[atype] = ts
        if self.db is not None:
            self.db.add_alert(atype, message, severity=severity, session_id=self.session_id)
        logger.info("Global alert [%s] %s", severity, message)
        return [AlertEvent(atype, severity, message, zone_name="global")]

    def _maybe_snapshot(self, ts: datetime) -> None:
        if self.db is None:
            return
        if (
            self._last_snapshot_ts is not None
            and (ts - self._last_snapshot_ts).total_seconds() < self.snapshot_interval
        ):
            return
        self._last_snapshot_ts = ts
        cm = self.customer.compute()
        sm = self.staff.compute()
        tm = self.tables.compute()
        qm = self.queues.compute()
        self.db.add_snapshot(
            self.session_id,
            current_customers=cm.current_customers,
            current_staff=sm.current_staff,
            occupied_tables=tm.occupied_tables,
            empty_tables=tm.empty_tables,
            queue_length=qm.total_queue_length,
            avg_wait_seconds=qm.avg_wait_seconds,
            fps=self._last_fps,
        )


    def metrics(self) -> OrchestratorMetrics:
        gallery = getattr(self.engine.manager, "gallery", None)
        fv_kpis = self.flat_visits.compute_kpis(gallery) if gallery else {}
        return OrchestratorMetrics(
            customer=self.customer.compute().to_dict(),
            staff=self.staff.compute().to_dict(),
            table=self.tables.compute().to_dict(),
            queue=self.queues.compute().to_dict(),
            flat_visits=fv_kpis,
            alerts=[
                {
                    "alert_type": a.alert_type,
                    "severity": a.severity,
                    "message": a.message,
                    "zone_name": a.zone_name,
                }
                for a in self._last_alerts
            ],
            fps=self._last_fps,
        )

    def render_heatmap(self, role: str = "customer", overlay: bool = True) -> Optional[np.ndarray]:
        hm = self.heatmap_customer if role == "customer" else self.heatmap_staff
        if hm is None:
            return None
        base = self._last_frame if overlay else None
        return hm.render(base_frame=base)

    def reload_zones(self) -> None:
        self.tables.reload_tables()
        self.queues.reload_queues()
        self.classifier.reload_zones()

    def finalize(self) -> None:
        self.customer.finalize()
        if self.db is not None and self._last_frame is not None:
            self._last_snapshot_ts = None
            self._maybe_snapshot(utcnow())
        logger.info("Orchestrator finalised (session=%s).", self.session_id)
