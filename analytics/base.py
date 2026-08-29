from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from tracking.track_state import TrackFrameResult


class RunningStat:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self._min = value if self._min is None else min(self._min, value)
        self._max = value if self._max is None else max(self._max, value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def minimum(self) -> float:
        return self._min if self._min is not None else 0.0

    @property
    def maximum(self) -> float:
        return self._max if self._max is not None else 0.0

    def reset(self) -> None:
        self.__init__()


class BaseAnalyzer(ABC):
    @abstractmethod
    def update(self, track_result: TrackFrameResult) -> None:
        pass

    @abstractmethod
    def metrics(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
