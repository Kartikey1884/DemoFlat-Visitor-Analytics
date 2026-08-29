from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import Config, get_config

ROOT_LOGGER_NAME = "cafeanalytics"

_configured = False


def _configure_root(cfg: Config) -> logging.Logger:
    global _configured
    root = logging.getLogger(ROOT_LOGGER_NAME)

    if _configured:
        return root

    log_cfg = cfg.logging
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(fmt=log_cfg.fmt, datefmt=log_cfg.date_fmt)

    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    file_path = cfg.paths.logs / log_cfg.file_name

    file_handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=log_cfg.max_bytes,
        backupCount=log_cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    if log_cfg.console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    _configured = True
    root.debug("Logging initialised (level=%s, file=%s)", log_cfg.level, file_path)
    return root


def get_logger(name: Optional[str] = None) -> logging.Logger:
    root = _configure_root(get_config())
    if name is None or name == ROOT_LOGGER_NAME:
        return root
    child_name = name if name.startswith(f"{ROOT_LOGGER_NAME}.") else f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(child_name)


def set_level(level: str) -> None:
    root = _configure_root(get_config())
    numeric = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric)
    for handler in root.handlers:
        handler.setLevel(numeric)
    root.info("Log level changed to %s", level.upper())


def reset() -> None:
    global _configured
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    _configured = False
