from __future__ import annotations

import streamlit as st

from dashboard import state
from dashboard.components.cards import (
    app_header,
    empty_state,
    kpi_row,
    section_title,
    status_pill,
)
from dashboard.components.charts import bar_chart, gauge, line_chart
from dashboard.theme import PALETTE

_STATUS_KIND = {"occupied": "err", "empty": "ok", "reserved": "warn"}


def _tables_tab(orch) -> None:
    if orch is None:
        empty_state("Start a source to see table occupancy", "🪑")
        return
    tm = orch.metrics().table
    if tm.get("total_tables", 0) == 0:
        empty_state("No tables defined", "🪑", "Add table zones from Settings › Zones.")
        return

    kpi_row(
        [
            {"label": "Occupied", "value": tm["occupied_tables"], "icon": "🔴", "accent": PALETTE["danger"]},
            {"label": "Available", "value": tm["available_tables"], "icon": "🟢", "accent": PALETTE["success"]},
            {"label": "Reserved", "value": tm["reserved_tables"], "icon": "🟡", "accent": PALETTE["warning"]},
            {"label": "Occupancy", "value": f"{tm['occupancy_percentage']:.0f}%", "icon": "📊", "accent": PALETTE["primary"]},
        ],
        columns=4,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        section_title("Occupancy", "📊")
        st.plotly_chart(
            gauge(tm["occupancy_percentage"], 100, title="% Occupied"),
            use_container_width=True,
        )
    with col2:
        section_title("Tables", "🪑")
        for t in tm["per_table"]:
            kind = _STATUS_KIND.get(t["status"], "idle")
            cols = st.columns([2, 1, 1, 1])
            cols[0].markdown(f"**{t['name']}**", unsafe_allow_html=True)
            cols[1].markdown(status_pill(t["status"].title(), kind), unsafe_allow_html=True)
            cols[2].caption(f"{t['occupants']} 👤")
            cols[3].caption(f"{t['occupied_seconds']:.0f}s")

    section_title("Utilisation Over Time", "📉")
    series = orch.tables.utilization_over_time()
    if series:
        xs = [ts for ts, _ in series]
        ys = [n for _, n in series]
        st.plotly_chart(line_chart(xs, ys, name="Occupied tables", color=PALETTE["success"]),
                        use_container_width=True)
    else:
        empty_state("No utilisation data yet", "📉")


def _queues_tab(orch) -> None:
    if orch is None:
        empty_state("Start a source to see queue analytics", "🧾")
        return
    qm = orch.metrics().queue
    kpi_row(
        [
            {"label": "In Queue", "value": qm["total_queue_length"], "icon": "🧾", "accent": PALETTE["primary"]},
            {"label": "Max Queue", "value": qm["max_queue_length"], "icon": "🔝", "accent": PALETTE["warning"]},
            {"label": "Avg Wait (s)", "value": f"{qm['avg_wait_seconds']:.0f}", "icon": "⌛", "accent": PALETTE["accent"]},
            {"label": "Longest Wait (s)", "value": f"{qm['longest_waiting_seconds']:.0f}", "icon": "🕰️", "accent": PALETTE["danger"]},
        ],
        columns=4,
    )
    if not qm["per_queue"]:
        empty_state("No queue zones defined", "🧾", "Add queue zones from Settings › Zones.")
        return

    section_title("Queues", "🧾")
    names = [q["name"] for q in qm["per_queue"]]
    lengths = [q["length"] for q in qm["per_queue"]]
    st.plotly_chart(bar_chart(names, lengths, name="Length", color=PALETTE["danger"]),
                    use_container_width=True)

    section_title("Queue Trend", "📉")
    trend = orch.queues.queue_trend()
    if trend:
        xs = [ts for ts, _ in trend]
        ys = [n for _, n in trend]
        st.plotly_chart(line_chart(xs, ys, name="Total in queue", color=PALETTE["warning"]),
                        use_container_width=True)
    else:
        empty_state("No queue trend yet", "📉")


def render() -> None:
    app_header("Occupancy", "Table status and queue monitoring")
    orch = state.get_orchestrator(create=False)
    tab_tables, tab_queues = st.tabs(["🪑 Tables", "🧾 Queues"])
    with tab_tables:
        _tables_tab(orch)
    with tab_queues:
        _queues_tab(orch)
