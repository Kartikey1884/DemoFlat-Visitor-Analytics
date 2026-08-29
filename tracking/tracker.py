from __future__ import annotations

import warnings
from typing import Any, Optional

from config import Config, get_config
from detection.detections import Detections
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiObjectTracker:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.track_cfg = self.config.tracking
        self._backend_name = self.track_cfg.tracker.lower()
        self._tracker: Any = self._build_backend()


    def _build_backend(self) -> Any:
        try:
            import supervision as sv
        except ImportError as exc:
            raise ImportError(
                "supervision is required for tracking. Install it with "
                "`pip install supervision`."
            ) from exc

        if self._backend_name == "botsort":
            logger.warning(
                "BoT-SORT is not provided by supervision; use "
                "YOLODetector.predict_tracked for BoT-SORT. Falling back to "
                "ByteTrack for the standalone tracker."
            )
            self._backend_name = "bytetrack"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            tracker = sv.ByteTrack(
                track_activation_threshold=self.track_cfg.track_activation_threshold,
                lost_track_buffer=self.track_cfg.lost_track_buffer,
                minimum_matching_threshold=self.track_cfg.minimum_matching_threshold,
                frame_rate=self.track_cfg.frame_rate,
                minimum_consecutive_frames=self.track_cfg.minimum_consecutive_frames,
            )
        logger.info(
            "Tracker initialised (backend=bytetrack, activation=%.2f, "
            "lost_buffer=%d, min_consecutive=%d).",
            self.track_cfg.track_activation_threshold,
            self.track_cfg.lost_track_buffer,
            self.track_cfg.minimum_consecutive_frames,
        )
        return tracker


    def update(self, detections: Detections) -> Detections:
        sv_in = detections.to_supervision()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            sv_out = self._tracker.update_with_detections(sv_in)
        return Detections.from_supervision(sv_out, class_names=detections.class_names)

    def reset(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self._tracker.reset()
        logger.debug("Tracker state reset.")

    @property
    def backend(self) -> str:
        return self._backend_name
