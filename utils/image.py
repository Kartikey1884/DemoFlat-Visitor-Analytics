from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from utils.geometry import Point, Polygon, Segment
from utils.logger import get_logger

logger = get_logger(__name__)

Color = Tuple[int, int, int]


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def hex_to_bgr(hex_color: str) -> Color:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def resize_keep_aspect(
    frame: np.ndarray, width: Optional[int] = None, height: Optional[int] = None
) -> np.ndarray:
    if width is None and height is None:
        return frame
    h, w = frame.shape[:2]
    if width is not None and height is None:
        scale = width / float(w)
        new_size = (width, max(1, int(round(h * scale))))
    elif height is not None and width is None:
        scale = height / float(h)
        new_size = (max(1, int(round(w * scale))), height)
    else:
        new_size = (int(width), int(height))
    interp = cv2.INTER_AREA if new_size[0] < w else cv2.INTER_LINEAR
    return cv2.resize(frame, new_size, interpolation=interp)


def draw_polygon(
    frame: np.ndarray,
    polygon: Polygon,
    color: Color = (108, 92, 231),
    thickness: int = 2,
    fill_alpha: float = 0.0,
    label: Optional[str] = None,
) -> np.ndarray:
    if len(polygon) < 2:
        return frame
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    if fill_alpha > 0.0:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)

    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

    if label:
        x, y = int(polygon[0][0]), int(polygon[0][1])
        draw_label(frame, label, (x, y), color)
    return frame


def draw_line(
    frame: np.ndarray,
    line: Segment,
    color: Color = (0, 206, 201),
    thickness: int = 2,
    label: Optional[str] = None,
) -> np.ndarray:
    p1 = (int(line[0][0]), int(line[0][1]))
    p2 = (int(line[1][0]), int(line[1][1]))
    cv2.line(frame, p1, p2, color, thickness)
    if label:
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        draw_label(frame, label, mid, color)
    return frame


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: Point,
    color: Color = (108, 92, 231),
    text_color: Color = (255, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(origin[0]), int(origin[1])
    pad = 4
    top_left = (x, max(0, y - th - baseline - pad))
    bottom_right = (x + tw + 2 * pad, y)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness=cv2.FILLED)
    cv2.putText(
        frame,
        text,
        (x + pad, y - baseline),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def draw_point(
    frame: np.ndarray, point: Point, color: Color = (0, 206, 201), radius: int = 4
) -> np.ndarray:
    cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, thickness=cv2.FILLED)
    return frame


def draw_trajectory(
    frame: np.ndarray,
    points: Sequence[Point],
    color: Color = (0, 206, 201),
    thickness: int = 2,
) -> np.ndarray:
    if len(points) < 2:
        return frame
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=thickness)
    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return buffer.tobytes()


def save_snapshot(
    frame: np.ndarray,
    directory: Union[str, Path],
    prefix: str = "snapshot",
    timestamp: Optional[datetime] = None,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ts = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"{prefix}_{ts}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write snapshot to {path}")
    logger.debug("Saved snapshot %s", path)
    return path
