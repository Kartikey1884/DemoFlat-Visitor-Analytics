from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from utils.geometry import BBox, bbox_bottom_center, bbox_center


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    tracker_id: Optional[int] = None
    class_name: Optional[str] = None

    @property
    def box(self) -> BBox:
        return self.xyxy

    @property
    def center(self) -> Tuple[float, float]:
        return bbox_center(self.xyxy)

    @property
    def anchor(self) -> Tuple[float, float]:
        return bbox_bottom_center(self.xyxy)


@dataclass
class Detections:
    xyxy: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    tracker_id: Optional[np.ndarray] = None
    class_names: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyxy = np.asarray(self.xyxy, dtype=np.float32).reshape(-1, 4)
        self.confidence = np.asarray(self.confidence, dtype=np.float32).reshape(-1)
        self.class_id = np.asarray(self.class_id, dtype=np.int32).reshape(-1)
        if self.tracker_id is not None:
            self.tracker_id = np.asarray(self.tracker_id, dtype=np.int32).reshape(-1)
        n = len(self.xyxy)
        if not (len(self.confidence) == len(self.class_id) == n):
            raise ValueError("Detections arrays must share the same length.")


    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    def __bool__(self) -> bool:
        return len(self) > 0

    def __iter__(self) -> Iterator[Detection]:
        for i in range(len(self)):
            yield Detection(
                xyxy=tuple(float(v) for v in self.xyxy[i]),
                confidence=float(self.confidence[i]),
                class_id=int(self.class_id[i]),
                tracker_id=(int(self.tracker_id[i]) if self.tracker_id is not None else None),
                class_name=self.class_names.get(int(self.class_id[i])),
            )

    def __getitem__(self, index: Any) -> "Detections":
        idx = index
        if isinstance(index, np.ndarray) and index.dtype == bool:
            idx = index
        return Detections(
            xyxy=self.xyxy[idx],
            confidence=self.confidence[idx],
            class_id=self.class_id[idx],
            tracker_id=(self.tracker_id[idx] if self.tracker_id is not None else None),
            class_names=self.class_names,
        )


    @classmethod
    def empty(cls, class_names: Optional[Dict[int, str]] = None) -> "Detections":
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
            tracker_id=None,
            class_names=class_names or {},
        )

    @classmethod
    def from_ultralytics(
        cls, result: Any, class_names: Optional[Dict[int, str]] = None
    ) -> "Detections":
        boxes = getattr(result, "boxes", None)
        names = class_names or getattr(result, "names", {}) or {}
        if boxes is None or boxes.shape[0] == 0:
            return cls.empty(names)
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls_id = boxes.cls.cpu().numpy().astype(np.int32)
        tracker_id = None
        if getattr(boxes, "id", None) is not None:
            tracker_id = boxes.id.cpu().numpy().astype(np.int32)
        return cls(
            xyxy=xyxy,
            confidence=conf,
            class_id=cls_id,
            tracker_id=tracker_id,
            class_names=names,
        )


    def filter_by_class(self, class_ids: List[int]) -> "Detections":
        if len(self) == 0:
            return self
        mask = np.isin(self.class_id, np.asarray(class_ids, dtype=np.int32))
        return self[mask]

    def filter_by_confidence(self, threshold: float) -> "Detections":
        if len(self) == 0:
            return self
        return self[self.confidence >= threshold]

    def scale(self, scale_x: float, scale_y: float) -> "Detections":
        if len(self) == 0:
            return self
        scaled = self.xyxy.copy()
        scaled[:, [0, 2]] *= scale_x
        scaled[:, [1, 3]] *= scale_y
        return Detections(
            xyxy=scaled,
            confidence=self.confidence.copy(),
            class_id=self.class_id.copy(),
            tracker_id=(self.tracker_id.copy() if self.tracker_id is not None else None),
            class_names=self.class_names,
        )

    def with_tracker_ids(self, tracker_id: np.ndarray) -> "Detections":
        return Detections(
            xyxy=self.xyxy.copy(),
            confidence=self.confidence.copy(),
            class_id=self.class_id.copy(),
            tracker_id=np.asarray(tracker_id, dtype=np.int32),
            class_names=self.class_names,
        )


    @property
    def centers(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, 2), dtype=np.float32)
        cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        cy = (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2.0
        return np.stack([cx, cy], axis=1)

    @property
    def anchors(self) -> np.ndarray:
        if len(self) == 0:
            return np.empty((0, 2), dtype=np.float32)
        cx = (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2.0
        by = self.xyxy[:, 3]
        return np.stack([cx, by], axis=1)

    def count_class(self, class_id: int) -> int:
        return int(np.count_nonzero(self.class_id == class_id))


    def to_supervision(self) -> Any:
        import supervision as sv

        if len(self) == 0:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=self.xyxy.copy(),
            confidence=self.confidence.copy(),
            class_id=self.class_id.copy(),
            tracker_id=(self.tracker_id.copy() if self.tracker_id is not None else None),
        )

    @classmethod
    def from_supervision(
        cls, sv_detections: Any, class_names: Optional[Dict[int, str]] = None
    ) -> "Detections":
        if sv_detections.xyxy is None or len(sv_detections.xyxy) == 0:
            return cls.empty(class_names)
        conf = sv_detections.confidence
        if conf is None:
            conf = np.ones((len(sv_detections.xyxy),), dtype=np.float32)
        cls_id = sv_detections.class_id
        if cls_id is None:
            cls_id = np.zeros((len(sv_detections.xyxy),), dtype=np.int32)
        return cls(
            xyxy=sv_detections.xyxy,
            confidence=conf,
            class_id=cls_id,
            tracker_id=sv_detections.tracker_id,
            class_names=class_names or {},
        )
