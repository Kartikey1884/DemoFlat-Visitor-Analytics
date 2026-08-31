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
            gallery = getattr(orch.engine.manager, "gallery", None) if orch else None
            tracker = getattr(orch, "flat_visits", None) if orch else None

            active_v = gallery.active_visitors_count() if gallery else result.person_count
            total_v = gallery.total_unique_visitors_count() if gallery else 0

            # Sales Agent info
            all_sales = [p for p in gallery.get_all_persons() if p.role == "sales_person"] if gallery else []
            active_sales = [p for p in all_sales if p.is_active]
            num_sales = len(active_sales) if active_sales else (1 if all_sales else 0)

            if active_sales:
                sales_names = ", ".join(p.display_name for p in active_sales)
            elif all_sales:
                sales_names = f"{all_sales[0].display_name} (Tour)"
            else:
                sales_names = "None"

            kpi_row(
                [
                    {"label": "Visitors in Flat", "value": active_v, "icon": "🧑‍🤝‍🧑", "accent": PALETTE["primary"]},
                    {"label": "Total Unique Visitors", "value": total_v, "icon": "👤", "accent": PALETTE["accent"]},
                    {"label": "Sales Persons Present", "value": len(active_sales), "icon": "👔", "accent": PALETTE["success"]},
                    {"label": "Sales Agent Name", "value": sales_names, "icon": "🏷️", "accent": PALETTE["warning"]},
                ],
                columns=2,
            )

            # Tour session summary if active
            if tracker and tracker.get_active_sessions():
                active_s_list = tracker.get_active_sessions()
                curr = active_s_list[0]
                with st.container(border=True):
                    st.markdown(f"**🏢 Active Tour:** `{curr.session_id}`")
                    st.caption(f"Sales Agent: **{curr.sales_person_name}** · Duration: {curr.format_duration()}")
                    st.caption(f"Clients in Flat ({curr.visitor_count}): {', '.join(curr.accompanying_visitor_ids) if curr.accompanying_visitor_ids else 'None'}")

    if runner.stats.last_error:
        st.error(f"❌ Processing stopped due to error: {runner.stats.last_error}")
    elif not runner.is_running:
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
