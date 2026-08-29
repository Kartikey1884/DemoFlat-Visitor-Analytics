from __future__ import annotations

import streamlit as st

from analytics import aggregates
from dashboard import state
from dashboard.components.cards import app_header, empty_state, kpi_row, section_title
from dashboard.components.charts import bar_chart, labelled_histogram, line_chart, pie_chart
from dashboard.theme import PALETTE


def _customer_tab(db, orch) -> None:
    k = orch.metrics().kpis() if orch else {}
    kpi_row(
        [
            {"label": "Current", "value": k.get("current_customers", 0), "icon": "🧑‍🤝‍🧑", "accent": PALETTE["primary"]},
            {"label": "Total Visitors", "value": k.get("total_visitors", 0) or db.count_visits(role="customer"), "icon": "🚪", "accent": PALETTE["accent"]},
            {"label": "Max Occupancy", "value": k.get("max_occupancy", 0), "icon": "🔝", "accent": PALETTE["warning"]},
            {"label": "Avg Stay (s)", "value": f"{k.get('avg_stay_seconds', 0):.0f}", "icon": "⏱️", "accent": PALETTE["success"]},
        ],
        columns=4,
    )

    period = st.selectbox("Time range", ["Hourly (today)", "Daily (30d)", "Weekly (12w)", "Monthly (12m)"])
    if period.startswith("Hourly"):
        data = aggregates.hourly_visitors(db, role="customer")
        labels, values = [f"{h:02d}" for h in data], list(data.values())
    elif period.startswith("Daily"):
        series = aggregates.daily_visitors(db, days=30, role="customer")
        labels, values = [d.strftime("%m-%d") for d, _ in series], [v for _, v in series]
    elif period.startswith("Weekly"):
        series = aggregates.weekly_visitors(db, weeks=12, role="customer")
        labels, values = [w for w, _ in series], [v for _, v in series]
    else:
        series = aggregates.monthly_visitors(db, months=12, role="customer")
        labels, values = [m for m, _ in series], [v for _, v in series]

    section_title(f"Visitors — {period}", "📈")
    if any(values):
        chart = bar_chart(labels, values, name="Visitors") if period.startswith("Hourly") \
            else line_chart(labels, values, name="Visitors")
        st.plotly_chart(chart, use_container_width=True)
    else:
        empty_state("No data for this range", "📈")

    section_title("Stay Duration Distribution", "⏳")
    stats = aggregates.stay_duration_stats(db, role="customer")
    if stats.count:
        st.plotly_chart(labelled_histogram(stats.histogram, color=PALETTE["primary"]),
                        use_container_width=True)
        st.caption(f"n={stats.count} · mean {stats.mean:.0f}s · median {stats.median:.0f}s · max {stats.maximum:.0f}s")
    else:
        empty_state("No completed visits yet", "⏳")


def _staff_tab(db, orch) -> None:
    if orch is None:
        empty_state("Start a source to see live staff analytics", "👔")
        sm = {}
    else:
        sm = orch.metrics().staff
    kpi_row(
        [
            {"label": "On Floor", "value": sm.get("current_staff", 0), "icon": "👔", "accent": PALETTE["primary"]},
            {"label": "Staff:Customer", "value": sm.get("staff_to_customer_ratio", 0), "icon": "⚖️", "accent": PALETTE["accent"]},
            {"label": "Working (s)", "value": f"{sm.get('working_seconds', 0):.0f}", "icon": "🏃", "accent": PALETTE["success"]},
            {"label": "Idle (s)", "value": f"{sm.get('idle_seconds', 0):.0f}", "icon": "🛑", "accent": PALETTE["warning"]},
        ],
        columns=4,
    )
    working = sm.get("working_seconds", 0)
    idle = sm.get("idle_seconds", 0)
    if working or idle:
        section_title("Working vs Idle", "🥧")
        st.plotly_chart(
            pie_chart(["Working", "Idle"], [working, idle],
                      colors=[PALETTE["success"], PALETTE["warning"]]),
            use_container_width=True,
        )
    per_staff = sm.get("per_staff", {})
    if per_staff:
        section_title("Per-Staff Breakdown", "📋")
        rows = [
            {"Staff ID": sid, "Attendance (s)": v["attendance_seconds"],
             "Working (s)": v["working_seconds"], "Idle (s)": v["idle_seconds"]}
            for sid, v in per_staff.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        empty_state("No staff classified yet", "👔",
                    "Configure staff zones or manual IDs in Settings.")


def render() -> None:
    app_header("Analytics", "Customer flow and staff activity insights")
    db = state.get_database()
    orch = state.get_orchestrator(create=False)

    tab_cust, tab_staff = st.tabs(["👥 Customers", "👔 Staff"])
    with tab_cust:
        _customer_tab(db, orch)
    with tab_staff:
        _staff_tab(db, orch)
