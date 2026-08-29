from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from utils.timeutils import utcnow
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol, Union

from config import Config, get_config
from engine.engine import DetectionEngine, FrameResult
from utils.image import save_snapshot
from utils.logger import get_logger
from utils.video import VideoProperties, VideoSource, VideoWriter

logger = get_logger(__name__)

SourceSpec = Union[int, str, Path]


class EngineLike(Protocol):
    def process(
        self, frame: Any, frame_index: int, timestamp: datetime, run_inference: bool
    ) -> FrameResult: ...

    def reset(self) -> None: ...


class RunnerState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class RunnerStats:
    frames_read: int = 0
    frames_inferred: int = 0
    started_at: Optional[datetime] = None
    last_error: Optional[str] = None


class LiveRunner:
    def __init__(
        self,
        engine: EngineLike,
        source: Union[SourceSpec, Any],
        config: Optional[Config] = None,
        realtime: Optional[bool] = None,
        record_path: Optional[Union[str, Path]] = None,
        loop: bool = False,
    ) -> None:
        self.engine = engine
        self.source = source
        self.config = config or get_config()
        self._realtime_override = realtime
        self.record_path = Path(record_path) if record_path else None
        self.loop = loop

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[FrameResult] = None
        self._state = RunnerState.IDLE
        self.stats = RunnerStats()
        self._properties: Optional[VideoProperties] = None


    def _make_source(self):
        if hasattr(self.source, "read") and hasattr(self.source, "properties"):
            return self.source
        return VideoSource(self.source)

    def _is_realtime(self, props: VideoProperties) -> bool:
        if self._realtime_override is not None:
            return self._realtime_override
        return props.is_stream


    def start(self) -> "LiveRunner":
        if self._thread is not None and self._thread.is_alive():
            logger.warning("LiveRunner already running.")
            return self
        self._stop_event.clear()
        self._pause_event.clear()
        self.engine.reset()
        self.stats = RunnerStats(started_at=utcnow())
        self._state = RunnerState.RUNNING
        self._thread = threading.Thread(target=self._run, name="LiveRunner", daemon=True)
        self._thread.start()
        logger.info("LiveRunner started (source=%r).", self.source)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None
        if self._state not in (RunnerState.FINISHED, RunnerState.ERROR):
            self._state = RunnerState.STOPPED
        logger.info("LiveRunner stopped (%d frames read).", self.stats.frames_read)

    def pause(self) -> None:
        self._pause_event.set()
        if self._state == RunnerState.RUNNING:
            self._state = RunnerState.PAUSED

    def resume(self) -> None:
        self._pause_event.clear()
        if self._state == RunnerState.PAUSED:
            self._state = RunnerState.RUNNING


    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def properties(self) -> Optional[VideoProperties]:
        return self._properties

    def get_latest(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest

    def snapshot(self, prefix: str = "snapshot") -> Optional[Path]:
        result = self.get_latest()
        if result is None:
            return None
        return save_snapshot(result.annotated_frame, self.config.paths.snapshots, prefix=prefix)


    def _run(self) -> None:
        writer: Optional[VideoWriter] = None
        try:
            source = self._make_source()
            with source as src:
                self._properties = src.properties
                realtime = self._is_realtime(src.properties)
                fps = src.properties.fps or 30.0
                skip = max(0, self.config.performance.frame_skip)
                base_time = utcnow()

                if self.record_path is not None:
                    writer = VideoWriter(
                        self.record_path, fps=fps, frame_size=src.properties.size
                    ).open()

                idx = 0
                while not self._stop_event.is_set():
                    if self._pause_event.is_set():
                        self._stop_event.wait(0.05)
                        continue

                    frame = src.read()
                    if frame is None:
                        logger.warning("Failed to read frame from source %r. Stream ended or source disconnected.", self.source)
                        if self.loop and not src.properties.is_stream:
                            src.release()
                            src.open()
                            idx = 0
                            continue
                        self._state = RunnerState.FINISHED
                        break

                    self.stats.frames_read += 1
                    run_inference = (idx % (skip + 1)) == 0
                    ts = (
                        utcnow()
                        if realtime
                        else base_time + timedelta(seconds=idx / fps)
                    )
                    result = self.engine.process(
                        frame, frame_index=idx, timestamp=ts, run_inference=run_inference
                    )
                    if run_inference:
                        self.stats.frames_inferred += 1
                    if writer is not None:
                        writer.write(result.annotated_frame)
                    with self._lock:
                        self._latest = result
                    idx += 1
        except Exception as exc:
            self._state = RunnerState.ERROR
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("LiveRunner worker crashed.")
        finally:
            if writer is not None:
                writer.release()
            logger.debug("LiveRunner worker exiting (state=%s).", self._state)
