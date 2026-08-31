from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config import Config, get_config
from detection.detections import Detections
from tracking.reid import GlobalPerson, PersonGallery
from utils.geometry import BBox, Point
from utils.logger import get_logger
from utils.timeutils import utcnow

logger = get_logger(__name__)


@dataclass
class TrackState:
    track_id: int
    class_id: int
    first_seen: datetime
    last_seen: datetime
    first_frame: int
    last_frame: int
    last_box: BBox
    trajectory: List[Point] = field(default_factory=list)
    hits: int = 1
    misses: int = 0
    active: bool = True
    role: str = "visitor"
    global_id: Optional[str] = None
    global_person: Optional[GlobalPerson] = None
    max_trajectory: int = 1024

    def update(self, box: BBox, anchor: Point, timestamp: datetime, frame_index: int) -> None:
        self.last_box = box
        self.last_seen = timestamp
        self.last_frame = frame_index
        self.hits += 1
        self.misses = 0
        self.active = True
        self.trajectory.append(anchor)
        if len(self.trajectory) > self.max_trajectory:
            self.trajectory = self.trajectory[-self.max_trajectory :]
        if self.global_person is not None:
            self.global_person.update_dwell(timestamp)
            self.role = self.global_person.role

    def mark_missed(self) -> None:
        self.misses += 1

    @property
    def is_confirmed(self) -> bool:
        """Returns True if the person has been identified and confirmed via face/ReID."""
        return self.global_person is not None and self.global_id is not None

    @property
    def duration_seconds(self) -> float:
        if self.global_person is not None and self.global_person.total_dwell_seconds > 0:
            return self.global_person.total_dwell_seconds
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def current_anchor(self) -> Optional[Point]:
        return self.trajectory[-1] if self.trajectory else None

    @property
    def display_label(self) -> str:
        if self.global_person is not None:
            role_tag = "Sales Agent" if self.role == "sales_person" else "Visitor"
            dwell_str = self._format_seconds(self.duration_seconds)
            return f"[{role_tag} {self.global_person.global_id} | {dwell_str}]"
        return f"[Tracking #{self.track_id} · Awaiting Face]"

    def _format_seconds(self, s: float) -> str:
        sec = int(s)
        m, rsec = divmod(sec, 60)
        return f"{m}m {rsec}s" if m > 0 else f"{rsec}s"

    def to_record(self) -> dict:
        return {
            "track_id": self.track_id,
            "global_id": self.global_id,
            "class_id": self.class_id,
            "role": self.role,
            "is_confirmed": self.is_confirmed,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_seconds": self.duration_seconds,
            "hits": self.hits,
            "trajectory": [[round(x, 1), round(y, 1)] for x, y in self.trajectory],
        }


@dataclass
class TrackFrameResult:
    timestamp: datetime
    frame_index: int
    active_tracks: List[TrackState] = field(default_factory=list)
    entered_tracks: List[TrackState] = field(default_factory=list)
    exited_tracks: List[TrackState] = field(default_factory=list)
    gallery: Optional[PersonGallery] = None

    @property
    def active_count(self) -> int:
        return len(self.active_tracks)

    @property
    def confirmed_count(self) -> int:
        return sum(1 for t in self.active_tracks if t.is_confirmed)


class TrackManager:
    def __init__(
        self,
        config: Optional[Config] = None,
        max_trajectory: int = 1024,
        gallery: Optional[PersonGallery] = None,
    ) -> None:
        self.config = config or get_config()
        self.lost_buffer = self.config.tracking.lost_track_buffer
        self.max_trajectory = max_trajectory
        self.gallery = gallery or PersonGallery(self.config)
        self._states: Dict[int, TrackState] = {}

    def update(
        self,
        detections: Detections,
        frame: Optional[Any] = None,
        timestamp: Optional[datetime] = None,
        frame_index: int = 0,
    ) -> TrackFrameResult:
        ts = timestamp or utcnow()
        entered: List[TrackState] = []
        current_ids: set[int] = set()

        for det in detections:
            if det.tracker_id is None:
                continue
            tid = int(det.tracker_id)
            current_ids.add(tid)

            state = self._states.get(tid)
            if state is None:
                # Brand new track entering frame
                if frame is not None and len(frame.shape) >= 2:
                    gp, is_new, sim = self.gallery.match_or_create(frame, det.box, tid, ts)
                else:
                    gp = self.gallery.get_by_track(tid)

                role = gp.role if gp else "visitor"
                gid = gp.global_id if gp else None
                state = TrackState(
                    track_id=tid,
                    class_id=det.class_id,
                    first_seen=ts,
                    last_seen=ts,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    last_box=det.box,
                    trajectory=[det.anchor],
                    max_trajectory=self.max_trajectory,
                    role=role,
                    global_id=gid,
                    global_person=gp,
                )
                self._states[tid] = state
                entered.append(state)
                logger.debug(
                    "Track %d (Global: %s, Confirmed: %s) entered at frame %d.",
                    tid,
                    gid,
                    state.is_confirmed,
                    frame_index,
                )
            else:
                # Existing track continuing in frame
                state.update(det.box, det.anchor, ts, frame_index)
                if frame is not None and len(frame.shape) >= 2:
                    gp = self.gallery.update_active_person_appearance(frame, det.box, tid, ts)
                    # If this track was provisional and has just been confirmed via face view:
                    if state.global_person is None and gp is not None:
                        state.global_person = gp
                        state.global_id = gp.global_id
                        state.role = gp.role
                        logger.info(
                            "✅ Track #%d successfully confirmed as %s (%s) upon frontal face detection.",
                            tid,
                            gp.global_id,
                            gp.display_name,
                        )

        exited: List[TrackState] = []
        for tid, state in self._states.items():
            if tid in current_ids or not state.active:
                continue
            state.mark_missed()
            if state.misses > self.lost_buffer:
                state.active = False
                exited.append(state)
                self.gallery.on_track_lost(tid)
                logger.debug("Track %d exited (lost > %d frames).", tid, self.lost_buffer)

        active = [s for s in self._states.values() if s.active]
        return TrackFrameResult(
            timestamp=ts,
            frame_index=frame_index,
            active_tracks=active,
            entered_tracks=entered,
            exited_tracks=exited,
            gallery=self.gallery,
        )

    def get_active(self) -> List[TrackState]:
        return [s for s in self._states.values() if s.active]

    def get_track(self, track_id: int) -> Optional[TrackState]:
        return self._states.get(track_id)

    def all_tracks(self) -> List[TrackState]:
        return list(self._states.values())

    def active_count(self, role: Optional[str] = None) -> int:
        return sum(
            1 for s in self._states.values() if s.active and (role is None or s.role == role)
        )

    def total_count(self, role: Optional[str] = None) -> int:
        return sum(1 for s in self._states.values() if role is None or s.role == role)

    def set_role(self, track_id: int, role: str) -> None:
        state = self._states.get(track_id)
        if state is not None:
            state.role = role

    def prune_finished(self, keep_last: int = 0) -> int:
        finished = sorted(
            (s for s in self._states.values() if not s.active),
            key=lambda s: s.last_frame,
            reverse=True,
        )
        to_remove = finished[keep_last:]
        for state in to_remove:
            self._states.pop(state.track_id, None)
        return len(to_remove)

    def reset(self) -> None:
        self._states.clear()
        logger.debug("TrackManager state reset.")
