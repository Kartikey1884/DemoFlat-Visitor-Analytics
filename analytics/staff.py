from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from analytics.base import BaseAnalyzer, RunningStat
from config import Config, get_config
from tracking.track_state import TrackFrameResult, TrackState
from utils.geometry import Point, Polygon, euclidean, point_in_polygon
from utils.logger import get_logger

logger = get_logger(__name__)


class StaffClassifier:
    def __init__(
        self,
        config: Optional[Config] = None,
        db: Optional[Any] = None,
        mode: Optional[str] = None,
        staff_zones: Optional[Sequence[Polygon]] = None,
        manual_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self.config = config or get_config()
        self.acfg = self.config.analytics
        self.db = db
        self.mode = (mode or self.acfg.staff_classification).lower()
        self._staff_zones: List[Polygon] = [list(z) for z in (staff_zones or [])]
        if db is not None and not self._staff_zones:
            self.reload_zones()
        self.manual_ids: set[int] = set(manual_ids or [])
        self._evidence: Counter[int] = Counter()

    def reload_zones(self) -> None:
        if self.db is None:
            return
        zones = self.db.get_zones(zone_type="staff")
        self._staff_zones = [[(float(x), float(y)) for x, y in z.points] for z in zones]
        logger.info("Loaded %d staff zone(s).", len(self._staff_zones))

    def mark_manual(self, track_id: int, is_staff: bool = True) -> None:
        if is_staff:
            self.manual_ids.add(track_id)
        else:
            self.manual_ids.discard(track_id)


    def classify(
        self, track_result: TrackFrameResult, frame: Optional[np.ndarray] = None
    ) -> List[int]:
        for track in track_result.active_tracks:
            if track.role == "staff":
                continue
            if self._is_staff(track, frame):
                track.role = "staff"
                logger.debug("Track %d classified as staff.", track.track_id)

        return [t.track_id for t in track_result.active_tracks if t.role == "staff"]

    def _is_staff(self, track: TrackState, frame: Optional[np.ndarray]) -> bool:
        if self.mode == "manual":
            return track.track_id in self.manual_ids
        if self.mode == "zone":
            return self._classify_zone(track)
        if self.mode == "color":
            return self._classify_color(track, frame)
        return False

    def _classify_zone(self, track: TrackState) -> bool:
        anchor = track.current_anchor
        if anchor is None or not self._staff_zones:
            return False
        inside = any(point_in_polygon(anchor, zone) for zone in self._staff_zones)
        if inside:
            self._evidence[track.track_id] += 1
        else:
            self._evidence[track.track_id] = max(0, self._evidence[track.track_id] - 1)
        return self._evidence[track.track_id] >= self.acfg.staff_zone_min_frames

    def _classify_color(self, track: TrackState, frame: Optional[np.ndarray]) -> bool:
        lower = self.acfg.staff_color_hsv_lower
        upper = self.acfg.staff_color_hsv_upper
        if frame is None or lower is None or upper is None:
            return False
        import cv2

        x1, y1, x2, y2 = (int(v) for v in track.last_box)
        h = y2 - y1
        ty1, ty2 = y1 + int(0.15 * h), y1 + int(0.55 * h)
        roi = frame[max(0, ty1):max(0, ty2), max(0, x1):max(0, x2)]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        fraction = float(np.count_nonzero(mask)) / mask.size
        if fraction >= self.acfg.staff_color_min_fraction:
            self._evidence[track.track_id] += 1
        else:
            self._evidence[track.track_id] = max(0, self._evidence[track.track_id] - 1)
        return self._evidence[track.track_id] >= self.acfg.staff_color_min_frames

    def reset(self) -> None:
        self._evidence.clear()


@dataclass
class StaffMetrics:
    current_staff: int = 0
    total_staff: int = 0
    staff_to_customer_ratio: float = 0.0
    total_attendance_seconds: float = 0.0
    avg_attendance_seconds: float = 0.0
    working_seconds: float = 0.0
    idle_seconds: float = 0.0
    working_ratio: float = 0.0
    max_staff: int = 0
    avg_staff: float = 0.0
    per_staff: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["per_staff"] = {str(k): v for k, v in self.per_staff.items()}
        return data


class StaffAnalytics(BaseAnalyzer):
    def __init__(self, config: Optional[Config] = None, session_id: str = "session") -> None:
        self.config = config or get_config()
        self.idle_speed = self.config.analytics.idle_speed_px
        self.session_id = session_id
        self._reset_state()

    def _reset_state(self) -> None:
        self._states: Dict[int, TrackState] = {}
        self._working: Dict[int, float] = defaultdict(float)
        self._idle: Dict[int, float] = defaultdict(float)
        self._occupancy = RunningStat()
        self._ratio = 0.0
        self._current_staff = 0
        self._presence: List[tuple] = []
        self._last_ts: Optional[datetime] = None
        self._prev_anchor: Dict[int, Point] = {}


    def update(self, track_result: TrackFrameResult, customer_count: Optional[int] = None) -> None:
        ts = track_result.timestamp
        dt = (ts - self._last_ts).total_seconds() if self._last_ts else 0.0

        staff = [t for t in track_result.active_tracks if t.role == "staff"]
        customers = [t for t in track_result.active_tracks if t.role == "customer"]
        current_staff = len(staff)
        cust = customer_count if customer_count is not None else len(customers)

        self._occupancy.add(current_staff)
        self._current_staff = current_staff
        self._ratio = (current_staff / cust) if cust > 0 else float(current_staff)

        for track in staff:
            self._states[track.track_id] = track
            self._accumulate_activity(track, dt)

        self._presence.append((ts, current_staff))
        self._last_ts = ts

    def _accumulate_activity(self, track: TrackState, dt: float) -> None:
        anchor = track.current_anchor
        if anchor is None:
            return
        prev = self._prev_anchor.get(track.track_id)
        self._prev_anchor[track.track_id] = anchor
        if prev is None or dt <= 0:
            return
        speed = euclidean(prev, anchor) / dt
        if speed < self.idle_speed:
            self._idle[track.track_id] += dt
        else:
            self._working[track.track_id] += dt


    def compute(self) -> StaffMetrics:
        attendances = [s.duration_seconds for s in self._states.values()]
        total_att = sum(attendances)
        working = sum(self._working.values())
        idle = sum(self._idle.values())
        per_staff = {
            tid: {
                "attendance_seconds": round(state.duration_seconds, 1),
                "working_seconds": round(self._working.get(tid, 0.0), 1),
                "idle_seconds": round(self._idle.get(tid, 0.0), 1),
            }
            for tid, state in self._states.items()
        }
        return StaffMetrics(
            current_staff=self._current_staff,
            total_staff=len(self._states),
            staff_to_customer_ratio=round(self._ratio, 2),
            total_attendance_seconds=round(total_att, 1),
            avg_attendance_seconds=round(total_att / len(attendances), 1) if attendances else 0.0,
            working_seconds=round(working, 1),
            idle_seconds=round(idle, 1),
            working_ratio=round(working / (working + idle), 2) if (working + idle) > 0 else 0.0,
            max_staff=int(self._occupancy.maximum),
            avg_staff=round(self._occupancy.mean, 2),
            per_staff=per_staff,
        )

    def metrics(self) -> Dict[str, Any]:
        return self.compute().to_dict()

    def presence_timeline(self) -> List[tuple]:
        return list(self._presence)

    def reset(self) -> None:
        self._reset_state()
        logger.debug("StaffAnalytics reset (session=%s).", self.session_id)
