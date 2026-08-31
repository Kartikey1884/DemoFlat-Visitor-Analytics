from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


PROJECT_ROOT: Path = Path(__file__).resolve().parent


@dataclass
class Paths:
    root: Path = PROJECT_ROOT
    models: Path = PROJECT_ROOT / "models"
    outputs: Path = PROJECT_ROOT / "outputs"
    snapshots: Path = PROJECT_ROOT / "outputs" / "snapshots"
    recordings: Path = PROJECT_ROOT / "outputs" / "recordings"
    heatmaps: Path = PROJECT_ROOT / "outputs" / "heatmaps"
    logs: Path = PROJECT_ROOT / "logs"
    reports: Path = PROJECT_ROOT / "reports" / "generated"
    videos: Path = PROJECT_ROOT / "videos"
    sample_videos: Path = PROJECT_ROOT / "videos" / "sample"
    uploads: Path = PROJECT_ROOT / "videos" / "uploads"
    assets: Path = PROJECT_ROOT / "assets"
    thumbnails: Path = PROJECT_ROOT / "outputs" / "thumbnails"
    database_file: Path = PROJECT_ROOT / "database" / "cafe_analytics.db"

    def ensure(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "database_file":
                value.parent.mkdir(parents=True, exist_ok=True)
            elif isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectionConfig:
    model_path: str = "yolo11s.pt"
    device: str = "auto"
    confidence: float = 0.35
    iou: float = 0.5
    image_size: int = 640
    max_detections: int = 300
    half_precision: bool = False

    target_classes: List[int] = field(default_factory=lambda: [0])
    class_names: Dict[int, str] = field(
        default_factory=lambda: {0: "person"}
    )

@dataclass
class TrackingConfig:
    tracker: str = "bytetrack"
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    frame_rate: int = 30
    minimum_consecutive_frames: int = 3

@dataclass
class ReIDConfig:
    enabled: bool = True
    similarity_threshold: float = 0.68
    model_name: str = "mobilenet_v3_large"
    feature_dim: int = 960
    max_gallery_size: int = 2000
    crop_min_size: int = 32
    save_thumbnails: bool = True
    momentum: float = 0.85


@dataclass
class LLMConfig:
    enabled: bool = True
    provider: str = "groq"  # "groq", "gemini", "openai", "claude", "ollama", "local"
    model_name: str = "llama-3.3-70b-versatile"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.1
    max_tokens: int = 300
    timeout_seconds: float = 10.0


@dataclass
class FlatVisitConfig:
    sales_role_name: str = "sales_person"
    visitor_role_name: str = "visitor"
    session_inactivity_timeout_seconds: float = 45.0
    min_visit_seconds: float = 3.0
    auto_detect_sessions: bool = True


@dataclass
class AnalyticsConfig:
    staff_classification: str = "zone"

    staff_zone_min_frames: int = 15

    staff_color_hsv_lower: Optional[List[int]] = None
    staff_color_hsv_upper: Optional[List[int]] = None
    staff_color_min_fraction: float = 0.30
    staff_color_min_frames: int = 10

    idle_speed_px: float = 20.0


    pixels_per_meter: Optional[float] = None

    max_capacity: int = 40
    crowd_alert_ratio: float = 0.9

    table_proximity_px: int = 60
    table_occupied_seconds: float = 5.0

    queue_length_alert: int = 6
    queue_wait_alert_seconds: float = 300.0

    alert_cooldown_seconds: float = 30.0

    min_stay_seconds: float = 5.0

    heatmap_decay: float = 0.98
    heatmap_blur_kernel: int = 25
    heatmap_opacity: float = 0.5


@dataclass
class DashboardConfig:
    title: str = "Flat & Sales Visit Analytics"
    icon: str = "🏢"
    layout: str = "wide"
    theme: str = "dark"
    primary_color: str = "#6C5CE7"
    accent_color: str = "#00CEC9"
    success_color: str = "#00B894"
    warning_color: str = "#FDCB6E"
    danger_color: str = "#FF7675"
    refresh_interval_ms: int = 1000


@dataclass
class PerformanceConfig:
    frame_skip: int = 0
    resize_width: Optional[int] = None
    max_queue_size: int = 64
    use_threading: bool = True


@dataclass
class DatabaseConfig:
    url: Optional[str] = None
    echo: bool = False
    pool_pre_ping: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file_name: str = "cafe_analytics.log"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 5
    console: bool = True
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_fmt: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    flat_visit: FlatVisitConfig = field(default_factory=FlatVisitConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


    @property
    def model_file(self) -> Path:
        return self.paths.models / self.detection.model_path

    @property
    def database_url(self) -> str:
        if self.database.url:
            return self.database.url
        return f"sqlite:///{self.paths.database_file}"

    def to_dict(self) -> Dict[str, Any]:
        return _to_serialisable(asdict(self))


def _to_serialisable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serialisable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _coerce(current: Any, new: Any) -> Any:
    if current is None:
        return new
    if isinstance(current, Path):
        return Path(str(new))
    if isinstance(current, bool):
        if isinstance(new, str):
            return new.strip().lower() in {"1", "true", "yes", "on"}
        return bool(new)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(new)
    if isinstance(current, float):
        return float(new)
    return new


def _apply_overrides(obj: Any, overrides: Dict[str, Any]) -> None:
    if not is_dataclass(obj) or not isinstance(overrides, dict):
        return
    valid = {f.name for f in fields(obj)}
    for key, value in overrides.items():
        if key not in valid:
            continue
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overrides(current, value)
        else:
            setattr(obj, key, _coerce(current, value))


def _load_yaml_overrides(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        env_path = os.environ.get("CAFE_CONFIG")
        config_path = Path(env_path) if env_path else PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping.")
    return data


def _load_env_overrides(cfg: Config) -> None:
    for section_field in fields(cfg):
        section = getattr(cfg, section_field.name)
        if not is_dataclass(section):
            continue
        prefix = f"CAFE_{section_field.name.upper()}_"
        for f in fields(section):
            env_key = f"{prefix}{f.name.upper()}"
            if env_key in os.environ:
                current = getattr(section, f.name)
                setattr(section, f.name, _coerce(current, os.environ[env_key]))


def load_config(config_path: Optional[Path] = None) -> Config:
    cfg = Config()
    _apply_overrides(cfg, _load_yaml_overrides(config_path))
    _load_env_overrides(cfg)
    cfg.paths.ensure()
    return cfg


_CONFIG: Optional[Config] = None


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def reload_config(config_path: Optional[Path] = None) -> Config:
    global _CONFIG
    _CONFIG = load_config(config_path)
    return _CONFIG


if __name__ == "__main__":
    import json

    print(json.dumps(get_config().to_dict(), indent=2))
