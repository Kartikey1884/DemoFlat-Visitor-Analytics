from __future__ import annotations

from datetime import datetime, timedelta

from utils.timeutils import utcnow

import streamlit as st

from analytics import aggregates
from dashboard import state
from dashboard.components.cards import (
    app_header,
    empty_state,
    kpi_row,
    render_alerts,
    section_title,
    status_pill,
)
from dashboard.components.charts import bar_chart, line_chart
from dashboard.theme import PALETTE


def _fmt_seconds(seconds: float) -> str:
    seconds = int(seconds or 0)
    if seconds < 60:
        return f"{seconds}s"
    return f"{seconds // 60}m {seconds % 60}s"


def render() -> None:
    app_header("Flat & Sales Visit Intelligence", "Real-time AI tracking for flat tours, visitor counts, and stay durations")

    db = state.get_database()
    orch = state.get_orchestrator(create=False)
    running = state.is_running()

    pill = status_pill("● Live", "ok") if running else status_pill("Idle", "idle")
    src = st.session_state.get("source_label") or "—"
    st.markdown(
        f"{pill} &nbsp; <span style='color:{PALETTE['muted']}'>Source: {src}</span>",
        unsafe_allow_html=True,
    )

    metrics = orch.metrics() if orch is not None else None
    k = metrics.kpis() if metrics is not None else {}

    total_visitors = k.get("total_visitors", 0)
    total_flat_visits = k.get("total_flat_visits", 0)
    active_visitors = k.get("current_customers", 0)
    active_sales = k.get("active_sales_agents", 0)
    active_visits = k.get("active_flat_visits", 0)
    avg_stay = k.get("avg_stay_seconds", 0)
    avg_per_tour = k.get("avg_visitors_per_tour", 0.0)
    return_rate = k.get("returning_visitor_rate", 0.0)

    section_title("Flat Visit KPIs", "📊")
    cards = [
        {"label": "Active Visitors in Flat", "value": active_visitors, "icon": "🧑‍🤝‍🧑", "accent": PALETTE["primary"]},
        {"label": "Total Unique Visitors", "value": total_visitors, "icon": "👤", "accent": PALETTE["accent"]},
        {"label": "Active Sales Tours", "value": active_visits, "icon": "🏢", "accent": PALETTE["success"]},
        {"label": "Sales Agents Present", "value": active_sales, "icon": "👔", "accent": PALETTE["warning"]},
        {"label": "Avg Visitors / Tour", "value": f"{avg_per_tour:.1f}", "icon": "👥", "accent": PALETTE["accent"]},
        {"label": "Avg Visit Stay Time", "value": _fmt_seconds(avg_stay), "icon": "⏱️", "accent": PALETTE["warning"]},
        {"label": "Total Flat Visits", "value": total_flat_visits, "icon": "📜", "accent": PALETTE["primary"]},
        {"label": "Return Visitor Rate", "value": f"{return_rate:.1f}%", "icon": "🔄", "accent": PALETTE["success"]},
        {"label": "System FPS", "value": k.get("fps", 0), "icon": "🎥", "accent": PALETTE["muted"]},
    ]
    kpi_row(cards, columns=3)

    st.divider()

    # Active Flat Visits section if any
    if orch is not None and hasattr(orch, "flat_visits"):
        active_sessions = orch.flat_visits.get_active_sessions()
        if active_sessions:
            section_title("Active Flat Visit Tours", "🔴")
            for s in active_sessions:
                with st.container(border=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"**Tour Session:** `{s.session_id}`")
                        st.markdown(f"**Sales Agent:** `{s.sales_person_name}`")
                    with c2:
                        st.markdown(f"**Accompanying Visitors:** `{s.visitor_count}`")
                        st.caption(f"Visitor IDs: {', '.join(s.accompanying_visitor_ids) if s.accompanying_visitor_ids else 'None'}")
                    with c3:
                        st.markdown(f"**Elapsed Stay Time:** `{s.format_duration()}`")
                        st.markdown(status_pill("In Progress", "ok"), unsafe_allow_html=True)
            st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        section_title("Flat Occupancy Trend", "📉")
        snaps = db.get_snapshots(
            session_id=orch.session_id if orch else None,
            since=utcnow() - timedelta(hours=6),
        )
        if snaps:
            xs = [s.timestamp for s in snaps]
            ys = [s.current_customers for s in snaps]
            st.plotly_chart(line_chart(xs, ys, name="Visitors in Flat"), use_container_width=True)
        else:
            empty_state("No occupancy data yet", "📉",
                        "Start a camera or upload a flat tour video to see live trends.")
    with col2:
        section_title("Recent Activity Events", "🔔")
        if orch is not None and hasattr(orch, "flat_visits"):
            recent_events = orch.flat_visits.get_events()
            if recent_events:
                for evt in recent_events[:5]:
                    st.markdown(
                        f"<div style='padding: 6px 10px; margin-bottom: 4px; background: rgba(255,255,255,0.03); border-radius: 4px;'>"
                        f"<span style='color: {PALETTE['muted']}; font-size: 0.8em;'>[{evt['timestamp']}]</span> {evt['message']}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No recent events logged.")
        else:
            st.caption("No active tracker.")

    section_title("Visitors by Hour", "🕒")
    hourly = aggregates.hourly_visitors(db, role="customer")
    if any(hourly.values()):
        labels = [f"{h:02d}" for h in hourly.keys()]
        st.plotly_chart(
            bar_chart(labels, list(hourly.values()), name="Visitors"),
            use_container_width=True,
        )
    else:
        empty_state("No visitor history yet", "🕒")
