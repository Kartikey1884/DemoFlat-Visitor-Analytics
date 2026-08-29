from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional


class FPSMeter:
    def __init__(self, window: int = 30) -> None:
        self._window = max(1, window)
        self._timestamps: Deque[float] = deque(maxlen=self._window)

    def tick(self, now: Optional[float] = None) -> float:
        t = now if now is not None else time.perf_counter()
        self._timestamps.append(t)
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()
