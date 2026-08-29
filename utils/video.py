from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

SourceType = Union[int, str, Path]


@dataclass
class VideoProperties:
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def is_stream(self) -> bool:
        return self.frame_count <= 0


def _normalise_source(source: SourceType) -> Union[int, str]:
    if isinstance(source, int):
        return source
    text = str(source)
    if text.isdigit():
        return int(text)
    return text


class VideoSource:
    def __init__(self, source: SourceType, api_preference: Optional[int] = None) -> None:
        self.source = source
        self._api_preference = api_preference
        self._cap: Optional[cv2.VideoCapture] = None
        self._props: Optional[VideoProperties] = None


    def open(self) -> "VideoSource":
        normalised = _normalise_source(self.source)
        if self._api_preference is not None:
            cap = cv2.VideoCapture(normalised, self._api_preference)
        else:
            if isinstance(normalised, int):
                cap = cv2.VideoCapture(normalised, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(normalised)
            else:
                cap = cv2.VideoCapture(normalised)

        if not cap.isOpened():
            raise IOError(f"Unable to open video source: {self.source!r}")

        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self._props = VideoProperties(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(fps) if fps and fps > 0 else 30.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        logger.info(
            "Opened video source %r (%dx%d @ %.1f fps, frames=%d)",
            self.source,
            self._props.width,
            self._props.height,
            self._props.fps,
            self._props.frame_count,
        )
        return self

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Released video source %r", self.source)

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.release()


    @property
    def properties(self) -> VideoProperties:
        if self._props is None:
            raise RuntimeError("VideoSource not opened; call open() first.")
        return self._props

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()


    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("VideoSource not opened; call open() first.")
        ok, frame = self._cap.read()
        if not ok:
            return None
        return frame

    def frames(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            self.open()
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame


class VideoWriter:
    def __init__(
        self,
        path: Union[str, Path],
        fps: float,
        frame_size: Tuple[int, int],
        fourcc: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.frame_size = frame_size
        self._fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self) -> "VideoWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(self.path), self._fourcc, self.fps, self.frame_size)
        if not writer.isOpened():
            raise IOError(f"Unable to open video writer at {self.path}")
        self._writer = writer
        logger.info("Recording to %s (%s @ %.1f fps)", self.path, self.frame_size, self.fps)
        return self

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoWriter not opened; call open() first.")
        h, w = frame.shape[:2]
        if (w, h) != self.frame_size:
            frame = cv2.resize(frame, self.frame_size)
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.debug("Finalised recording %s", self.path)

    def __enter__(self) -> "VideoWriter":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.release()
