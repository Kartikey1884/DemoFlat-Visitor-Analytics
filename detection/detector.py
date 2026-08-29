from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from config import Config, get_config
from detection.detections import Detections
from utils.logger import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        logger.debug("Torch unavailable during device resolution; using CPU.")
    return "cpu"


class YOLODetector:
    def __init__(self, config: Optional[Config] = None, autoload: bool = True) -> None:
        self.config = config or get_config()
        self.det_cfg = self.config.detection
        self.device: str = resolve_device(self.det_cfg.device)
        self._model = None
        self._class_names: dict[int, str] = dict(self.det_cfg.class_names)
        if autoload:
            self.load()


    def _resolve_weights(self) -> str:
        local = self.config.model_file
        if local.exists():
            logger.info("Using local weights: %s", local)
            return str(local)
        logger.info(
            "Local weights %s not found; ultralytics will resolve/download %r.",
            local,
            self.det_cfg.model_path,
        )
        return self.det_cfg.model_path

    def load(self) -> "YOLODetector":
        if self._model is not None:
            return self
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for detection. Install it with "
                "`pip install ultralytics`."
            ) from exc

        weights = self._resolve_weights()
        logger.info("Loading YOLO model %r on device=%s ...", weights, self.device)
        model = YOLO(weights)
        try:
            model.to(self.device)
        except Exception:
            logger.warning("Could not move model to %s; falling back to CPU.", self.device)
            self.device = "cpu"
            model.to("cpu")

        self._model = model
        model_names = getattr(model, "names", None)
        if isinstance(model_names, dict):
            merged = {int(k): str(v) for k, v in model_names.items()}
            merged.update(self._class_names)
            self._class_names = merged

        self._cache_downloaded_weights(weights)
        logger.info("Model loaded (%d classes).", len(self._class_names))
        return self

    def _cache_downloaded_weights(self, weights: str) -> None:
        target = self.config.model_file
        if target.exists():
            return
        candidate = Path(weights)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate.name
        if candidate.exists() and candidate.resolve() != target.resolve():
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(target)
                logger.info("Cached weights to %s", target)
            except OSError as exc:
                logger.debug("Could not cache weights to models/: %s", exc)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)


    def _run_inference(self, frame: np.ndarray, persist: bool, use_tracker: bool):
        assert self._model is not None
        classes: Optional[List[int]] = self.det_cfg.target_classes or None
        common = dict(
            conf=self.det_cfg.confidence,
            iou=self.det_cfg.iou,
            imgsz=self.det_cfg.image_size,
            max_det=self.det_cfg.max_detections,
            classes=classes,
            device=self.device,
            verbose=False,
        )
        if self.det_cfg.half_precision:
            common["half"] = True
        if use_tracker:
            tracker_cfg = f"{self.config.tracking.tracker}.yaml"
            return self._model.track(frame, persist=persist, tracker=tracker_cfg, **common)
        return self._model.predict(frame, **common)

    def predict(self, frame: np.ndarray) -> Detections:
        if self._model is None:
            self.load()
        results = self._run_inference(frame, persist=False, use_tracker=False)
        result = results[0]
        return Detections.from_ultralytics(result, class_names=self._class_names)

    def predict_tracked(self, frame: np.ndarray, persist: bool = True) -> Detections:
        if self._model is None:
            self.load()
        results = self._run_inference(frame, persist=persist, use_tracker=True)
        result = results[0]
        return Detections.from_ultralytics(result, class_names=self._class_names)

    def warmup(self, image_size: Optional[int] = None) -> float:
        size = image_size or self.det_cfg.image_size
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        start = time.perf_counter()
        self.predict(blank)
        elapsed = time.perf_counter() - start
        logger.info("Warmup complete in %.3fs (device=%s).", elapsed, self.device)
        return elapsed
