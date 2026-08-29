from __future__ import annotations

from datetime import datetime

import streamlit as st

from dashboard import state
from dashboard.components.cards import app_header, empty_state, kpi_row, section_title, status_pill
from dashboard.theme import PALETTE
from utils.image import bgr_to_rgb


def _save_upload(uploaded) -> str:
    uploads = state.get_config_cached().paths.uploads
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / f"{datetime.now():%Y%m%d_%H%M%S}_{uploaded.name}"
    dest.write_bytes(uploaded.getbuffer())
    return str(dest)


@st.fragment(run_every=1.0)
def _progress_view() -> None:
    result = state.get_latest_frame_result()
    runner = st.session_state.get("runner")
    orch = state.get_orchestrator(create=False)

    if runner is None:
        return

    props = runner.properties
    total = props.frame_count if props else 0
    read = runner.stats.frames_read
    if total and total > 0:
        st.progress(min(1.0, read / total), text=f"Processed {read} / {total} frames")
    else:
        st.caption(f"Processed {read} frames")

    if result is not None:
        left, right = st.columns([3, 2])
        with left:
            st.image(bgr_to_rgb(result.annotated_frame), use_container_width=True,
                     caption=f"Frame {result.frame_index}")
        with right:
            k = orch.metrics().kpis() if orch else {}
            kpi_row(
                [
                    {"label": "Customers", "value": k.get("current_customers", 0), "icon": "🧑‍🤝‍🧑", "accent": PALETTE["primary"]},
                    {"label": "Visitors", "value": k.get("total_visitors", 0), "icon": "🚪", "accent": PALETTE["accent"]},
                    {"label": "Occupied", "value": k.get("occupied_tables", 0), "icon": "🪑", "accent": PALETTE["success"]},
                    {"label": "Queue", "value": k.get("queue_length", 0), "icon": "🧾", "accent": PALETTE["warning"]},
                ],
                columns=2,
            )

    if not runner.is_running:
        st.success("✅ Processing complete.")


def render() -> None:
    app_header("Video Upload", "Analyse a recorded video with full analytics")

    running = state.is_running()
    st.markdown(status_pill("Processing", "warn") if running else status_pill("Idle", "idle"),
                unsafe_allow_html=True)

    section_title("Upload", "📁")
    uploaded = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"],
                                disabled=running)
    c1, c2 = st.columns(2)
    loop = c1.toggle("Loop", value=False, disabled=running)
    if c1.button("▶ Analyse", use_container_width=True, disabled=running or uploaded is None):
        path = _save_upload(uploaded)
        state.start_source(path, label=uploaded.name, realtime=False, loop=loop)
        st.rerun()
    if c2.button("⏹ Stop", use_container_width=True, disabled=not running):
        state.stop_source()
        st.rerun()

    st.divider()
    if running or state.get_latest_frame_result() is not None:
        _progress_view()
    else:
        empty_state("No video loaded", "🎬", "Upload a file and press Analyse.")
