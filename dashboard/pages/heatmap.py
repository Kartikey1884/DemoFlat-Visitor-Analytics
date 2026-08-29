from __future__ import annotations

import streamlit as st

from dashboard import state
from dashboard.components.cards import app_header, empty_state, section_title
from dashboard.theme import PALETTE
from utils.image import bgr_to_rgb


def render() -> None:
    app_header("Heatmap", "Foot-traffic intensity and movement hotspots")
    orch = state.get_orchestrator(create=False)

    if orch is None or orch.heatmap_customer is None:
        empty_state("No heatmap data yet", "🔥",
                    "Start a camera or upload a video; the heatmap builds as people move.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    role = c1.radio("Role", ["customer", "staff"], horizontal=True)
    overlay = c2.toggle("Overlay on frame", value=True)

    heat = orch.render_heatmap(role=role, overlay=overlay)
    if heat is None:
        empty_state("Heatmap not ready", "🔥")
        return

    section_title(f"{role.title()} Movement Heatmap", "🔥")
    st.image(bgr_to_rgb(heat), use_container_width=True)

    if c3.button("💾 Save heatmap"):
        hm = orch.heatmap_customer if role == "customer" else orch.heatmap_staff
        path = hm.save(base_frame=orch._last_frame)
        st.success(f"Saved: {path.name}")

    hm = orch.heatmap_customer if role == "customer" else orch.heatmap_staff
    col_a, col_b = st.columns(2)
    with col_a:
        section_title("Most Visited Zones", "📍")
        for z in hm.most_visited(rows=3, cols=3, top=3):
            st.markdown(
                f"<span style='color:{PALETTE['danger']}'>●</span> "
                f"Row {z.row}, Col {z.col} — score {z.score:.0f}",
                unsafe_allow_html=True,
            )
    with col_b:
        section_title("Least Visited Zones", "🧊")
        for z in hm.least_visited(rows=3, cols=3, bottom=3):
            st.markdown(
                f"<span style='color:{PALETTE['accent']}'>●</span> "
                f"Row {z.row}, Col {z.col} — score {z.score:.0f}",
                unsafe_allow_html=True,
            )
