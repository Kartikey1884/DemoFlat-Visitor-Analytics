from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from config import Config, get_config
from tracking.track_state import TrackFrameResult, TrackState
from utils import image as img_utils
from utils.geometry import Point
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ZoneScore:
    row: int
    col: int
    score: float
    bbox: Tuple[int, int, int, int]

    def to_dict(self) -> dict:
        return {"row": self.row, "col": self.col, "score": round(self.score, 2), "bbox": self.bbox}


class HeatmapGenerator:
    def __init__(
        self,
        width: int,
        height: int,
        config: Optional[Config] = None,
        decay: Optional[float] = None,
    ) -> None:
        self.config = config or get_config()
        self.width = int(width)
        self.height = int(height)
        self.decay = self.config.analytics.heatmap_decay if decay is None else decay
        self._blur = self._odd(self.config.analytics.heatmap_blur_kernel)
        self._acc = np.zeros((self.height, self.width), dtype=np.float32)

    @staticmethod
    def _odd(k: int) -> int:
        k = max(1, int(k))
        return k if k % 2 == 1 else k + 1


    def add_points(self, points: Sequence[Point], weight: float = 1.0) -> None:
        if self.decay < 1.0:
            self._acc *= self.decay
        for x, y in points:
            xi, yi = int(x), int(y)
            if 0 <= xi < self.width and 0 <= yi < self.height:
                self._acc[yi, xi] += weight

    def update(
        self, track_result: TrackFrameResult, role: Optional[str] = None
    ) -> None:
        points = [
            t.current_anchor
            for t in track_result.active_tracks
            if t.current_anchor is not None and (role is None or t.role == role)
        ]
        self.add_points(points)


    def _normalized(self) -> np.ndarray:
        import cv2

        blurred = cv2.GaussianBlur(self._acc, (self._blur, self._blur), 0)
        peak = float(blurred.max())
        if peak <= 0:
            return np.zeros_like(blurred)
        return blurred / peak

    def render(
        self,
        base_frame: Optional[np.ndarray] = None,
        opacity: Optional[float] = None,
        threshold: float = 0.05,
        colormap: int = 2,
    ) -> np.ndarray:
        import cv2

        alpha = self.config.analytics.heatmap_opacity if opacity is None else opacity
        norm = self._normalized()
        heat_color = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)

        if base_frame is None:
            base = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            base = base_frame
            if base.shape[:2] != (self.height, self.width):
                base = cv2.resize(base, (self.width, self.height))

        blended = cv2.addWeighted(base, 1.0 - alpha, heat_color, alpha, 0)
        mask = norm > threshold
        out = base.copy()
        out[mask] = blended[mask]
        return out


    def zone_scores(self, rows: int = 3, cols: int = 3) -> List[ZoneScore]:
        cell_h = self.height // rows
        cell_w = self.width // cols
        scores: List[ZoneScore] = []
        for r in range(rows):
            for c in range(cols):
                y1, x1 = r * cell_h, c * cell_w
                y2 = self.height if r == rows - 1 else (r + 1) * cell_h
                x2 = self.width if c == cols - 1 else (c + 1) * cell_w
                score = float(self._acc[y1:y2, x1:x2].sum())
                scores.append(ZoneScore(row=r, col=c, score=score, bbox=(x1, y1, x2, y2)))
        return scores

    def most_visited(self, rows: int = 3, cols: int = 3, top: int = 3) -> List[ZoneScore]:
        return sorted(self.zone_scores(rows, cols), key=lambda z: z.score, reverse=True)[:top]

    def least_visited(self, rows: int = 3, cols: int = 3, bottom: int = 3) -> List[ZoneScore]:
        return sorted(self.zone_scores(rows, cols), key=lambda z: z.score)[:bottom]


    def save(self, path: Optional[Path] = None, base_frame: Optional[np.ndarray] = None) -> Path:
        import cv2

        if path is None:
            from datetime import datetime

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.config.paths.heatmaps / f"heatmap_{ts}.jpg"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), self.render(base_frame))
        logger.info("Saved heatmap to %s", path)
        return path

    @property
    def peak(self) -> float:
        return float(self._acc.max())

    def reset(self) -> None:
        self._acc[:] = 0.0
        logger.debug("HeatmapGenerator reset.")


class TrajectoryRenderer:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()

    def _color(self, role: str):
        hex_color = (
            self.config.dashboard.warning_color
            if role == "staff"
            else self.config.dashboard.accent_color
        )
        return img_utils.hex_to_bgr(hex_color)

    def render(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackState],
        max_points: int = 64,
        copy: bool = True,
        draw_ids: bool = True,
    ) -> np.ndarray:
        canvas = frame.copy() if copy else frame
        for track in tracks:
            if len(track.trajectory) < 2:
                continue
            color = self._color(track.role)
            pts = track.trajectory[-max_points:]
            img_utils.draw_trajectory(canvas, pts, color=color, thickness=2)
            head = pts[-1]
            img_utils.draw_point(canvas, head, color=color, radius=4)
            if draw_ids:
                img_utils.draw_label(canvas, f"#{track.track_id}", head, color)
        return canvas
