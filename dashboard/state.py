from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import streamlit as st

from config import get_config
from database.manager import DatabaseManager, get_db
from engine.runner import LiveRunner, RunnerState
from pipeline.orchestrator import AnalyticsOrchestrator
from utils.logger import get_logger

logger = get_logger(__name__)


def get_config_cached():
    return get_config()


@st.cache_resource(show_spinner=False)
def get_database() -> DatabaseManager:
    return get_db()


def apply_saved_settings(db, cfg) -> None:
    for key, value in db.get_all_settings().items():
        parts = key.split(".")
        if len(parts) != 2:
            continue
        section = getattr(cfg, parts[0], None)
        if section is not None and hasattr(section, parts[1]):
            try:
                setattr(section, parts[1], value)
            except Exception:
                logger.debug("Skipped invalid setting %s=%r", key, value)


def ensure_state() -> None:
    ss = st.session_state
    ss.setdefault("orchestrator", None)
    ss.setdefault("runner", None)
    ss.setdefault("source_label", "")
    ss.setdefault("running", False)
    if not ss.get("_settings_loaded"):
        try:
            apply_saved_settings(get_database(), get_config())
        except Exception:
            logger.exception("Failed to apply saved settings.")
        ss["_settings_loaded"] = True


def get_orchestrator(create: bool = True) -> Optional[AnalyticsOrchestrator]:
    ss = st.session_state
    if ss.get("orchestrator") is None and create:
        ss["orchestrator"] = AnalyticsOrchestrator(
            config=get_config(),
            db=get_database(),
            session_id=datetime.now().strftime("session_%Y%m%d_%H%M%S"),
        )
    return ss.get("orchestrator")


def start_source(
    source: Union[int, str, Path],
    label: str,
    realtime: Optional[bool] = None,
    record_path: Optional[Union[str, Path]] = None,
    loop: bool = False,
) -> LiveRunner:
    stop_source()
    orch = AnalyticsOrchestrator(
        config=get_config(),
        db=get_database(),
        session_id=datetime.now().strftime("session_%Y%m%d_%H%M%S"),
    )
    runner = LiveRunner(orch, source=source, realtime=realtime, record_path=record_path, loop=loop)
    runner.start()
    ss = st.session_state
    ss["orchestrator"] = orch
    ss["runner"] = runner
    ss["source_label"] = label
    ss["running"] = True
    logger.info("Dashboard started source: %s", label)
    return runner


def stop_source() -> None:
    ss = st.session_state
    runner: Optional[LiveRunner] = ss.get("runner")
    if runner is not None:
        runner.stop()
    orch: Optional[AnalyticsOrchestrator] = ss.get("orchestrator")
    if orch is not None:
        try:
            orch.finalize()
        except Exception:
            logger.exception("Error finalising orchestrator.")
    ss["runner"] = None
    ss["running"] = False


def reset_all() -> Dict[str, int]:
    stop_source()
    removed: Dict[str, int] = {}
    try:
        removed = get_database().clear_analytics()
    except Exception:
        logger.exception("Failed to clear analytics data.")
    ss = st.session_state
    ss["orchestrator"] = None
    ss["source_label"] = ""
    logger.info("Owner reset: cleared %s", removed)
    return removed


def is_running() -> bool:
    runner: Optional[LiveRunner] = st.session_state.get("runner")
    return runner is not None and runner.is_running


def runner_state() -> Optional[RunnerState]:
    runner: Optional[LiveRunner] = st.session_state.get("runner")
    return runner.state if runner is not None else None


def get_latest_frame_result() -> Optional[Any]:
    runner: Optional[LiveRunner] = st.session_state.get("runner")
    return runner.get_latest() if runner is not None else None
