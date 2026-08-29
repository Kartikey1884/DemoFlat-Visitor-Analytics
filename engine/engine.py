from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from utils.timeutils import utcnow
from typing import Optional

import numpy as np

from config import Config, get_config
from detection.annotator import DetectionAnnotator
from detection.detections import Detections
from detection.detector import YOLODetector
from engine.fps import FPSMeter
from tracking.track_state import TrackFrameResult, TrackManager
from tracking.tracker import MultiObjectTracker
from utils import image as img_utils
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameResult:
    frame_index: int
    timestamp: datetime
    raw_frame: np.ndarray
    annotated_frame: np.ndarray
    detections: Detections
    track_result: Optional[TrackFrameResult] = None
    fps: float = 0.0
    inference_ms: float = 0.0
    inferred: bool = True

    @property
    def person_count(self) -> int:
        return self.detections.count_class(0)


class DetectionEngine:
    def __init__(
        self,
        config: Optional[Config] = None,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[MultiObjectTracker] = None,
        manager: Optional[TrackManager] = None,
        annotator: Optional[DetectionAnnotator] = None,
        fps_window: int = 30,
    ) -> None:
        self.config = config or get_config()
        self.detector = detector or YOLODetector(self.config)
        self.tracker = tracker or MultiObjectTracker(self.config)
        self.manager = manager or TrackManager(self.config)
        self.annotator = annotator or DetectionAnnotator(self.config, color_by="track")
        self._fps = FPSMeter(window=fps_window)
        self._last_tracked: Detections = Detections.empty(self.detector.class_names)
        logger.info("DetectionEngine ready (device=%s).", self.detector.device)


    def _infer(self, frame: np.ndarray) -> Detections:
        resize_width = self.config.performance.resize_width
        if resize_width and resize_width < frame.shape[1]:
            small = img_utils.resize_keep_aspect(frame, width=resize_width)
            dets = self.detector.predict(small)
            scale_x = frame.shape[1] / small.shape[1]
            scale_y = frame.shape[0] / small.shape[0]
            dets = dets.scale(scale_x, scale_y)
        else:
            dets = self.detector.predict(frame)
        return self.tracker.update(dets)

    def process(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: Optional[datetime] = None,
        run_inference: bool = True,
    ) -> FrameResult:
        ts = timestamp or utcnow()
        inference_ms = 0.0
        track_result: Optional[TrackFrameResult] = None

        if run_inference:
            start = time.perf_counter()
            tracked = self._infer(frame)
            inference_ms = (time.perf_counter() - start) * 1000.0
            track_result = self.manager.update(
                tracked, frame=frame, timestamp=ts, frame_index=frame_index
            )
            self._last_tracked = tracked
        else:
            tracked = self._last_tracked

        annotated = self.annotator.annotate(
            frame, tracked, track_result=track_result, copy=True
        )
        fps = self._fps.tick()
        self._overlay_hud(annotated, fps, tracked, track_result)

        return FrameResult(
            frame_index=frame_index,
            timestamp=ts,
            raw_frame=frame,
            annotated_frame=annotated,
            detections=tracked,
            track_result=track_result,
            fps=fps,
            inference_ms=inference_ms,
            inferred=run_inference,
        )

    def _overlay_hud(
        self,
        frame: np.ndarray,
        fps: float,
        detections: Detections,
        track_result: Optional[TrackFrameResult] = None,
    ) -> None:
        primary = img_utils.hex_to_bgr(self.config.dashboard.primary_color)
        accent = img_utils.hex_to_bgr(self.config.dashboard.accent_color)
        success = img_utils.hex_to_bgr(self.config.dashboard.success_color)

        img_utils.draw_label(frame, f"FPS: {fps:4.1f}", (8, 22), primary)

        gallery = getattr(self.manager, "gallery", None)
        if gallery is not None:
            active_v = gallery.active_visitors_count()
            uniq_v = gallery.total_unique_visitors_count()
            active_s = gallery.active_sales_count()
            img_utils.draw_label(frame, f"Visitors: {active_v} (Unique: {uniq_v})", (8, 50), accent)
            if active_s > 0:
                img_utils.draw_label(frame, f"Sales Agents: {active_s}", (8, 78), success)
        else:
            img_utils.draw_label(frame, f"Persons: {detections.count_class(0)}", (8, 50), accent)


    def reset(self) -> None:
        self.tracker.reset()
        self.manager.reset()
        self._fps.reset()
        self._last_tracked = Detections.empty(self.detector.class_names)
        logger.debug("DetectionEngine reset.")

    def warmup(self) -> float:
        return self.detector.warmup()
