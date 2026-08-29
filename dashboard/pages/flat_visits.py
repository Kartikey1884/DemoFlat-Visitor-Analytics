from __future__ import annotations

import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

from dashboard import state
from dashboard.components.cards import app_header, empty_state, kpi_row, section_title, status_pill
from dashboard.theme import PALETTE


def render() -> None:
    app_header("Flat Visits & Sales Analytics", "Track sales agent performance, tour visit counts, accompanying visitors, and dwell time")

    orch = state.get_orchestrator(create=False)
    if orch is None or not hasattr(orch, "flat_visits"):
        empty_state(
            "No active flat visit tracker",
            "📋",
            "Start a live camera stream or upload a video from the Live Flat Monitor to start tracking visits.",
        )
        return

    tracker = orch.flat_visits
    gallery = getattr(orch.engine.manager, "gallery", None)
    kpis = tracker.compute_kpis(gallery) if gallery else {}

    # Top KPI Row
    kpi_row(
        [
            {"label": "Total Flat Visits", "value": kpis.get("total_visits", 0), "icon": "🏢", "accent": PALETTE["primary"]},
            {"label": "Active Tours Now", "value": kpis.get("active_visits", 0), "icon": "🔴", "accent": PALETTE["accent"]},
            {"label": "Unique Visitors", "value": kpis.get("total_unique_visitors", 0), "icon": "👥", "accent": PALETTE["success"]},
            {"label": "Sales Agents", "value": kpis.get("active_sales_agents", 0), "icon": "👔", "accent": PALETTE["warning"]},
        ],
        columns=4,
    )

    st.divider()

    # Active Visits Alert Card if in progress
    active_sessions = tracker.get_active_sessions()
    if active_sessions:
        section_title("🔴 In-Progress Flat Tours", "🏢")
        for s in active_sessions:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                with c1:
                    st.markdown(f"**Session ID:** `{s.session_id}`")
                    st.markdown(f"**Sales Agent:** 👔 `{s.sales_person_name}` ({s.sales_person_id})")
                with c2:
                    st.markdown(f"**Accompanying Visitors ({s.visitor_count}):**")
                    v_str = ", ".join(s.accompanying_visitor_ids) if s.accompanying_visitor_ids else "None yet"
                    st.caption(f"IDs: {v_str}")
                with c3:
                    st.markdown(f"**Started At:** {s.start_time.strftime('%H:%M:%S')}")
                    st.markdown(f"**Tour Elapsed:** `{s.format_duration()}`")
                with c4:
                    st.markdown(status_pill("In Progress", "ok"), unsafe_allow_html=True)
        st.divider()

    # Main Tabbed Content
    tab_sales, tab_visits, tab_visitors, tab_timeline = st.tabs([
        "👔 Sales Agent Performance",
        "📜 Flat Visit Tour Logs",
        "🧑 Visitor Logs & Profiles",
        "⚡ Real-Time Activity Log",
    ])

    # =========================================================================
    # TAB 1: SALES AGENT PERFORMANCE & VISIT COUNTS
    # =========================================================================
    with tab_sales:
        section_title("Sales Agent Leaderboard & Visit Counters", "👔")
        st.markdown("Track the exact number of flat tours each sales agent makes, total visitors handled, and working dwell time.")

        if gallery:
            sp_analytics = tracker.get_sales_person_analytics(gallery)
            if not sp_analytics:
                st.info("💡 No Sales Agents designated yet. You can promote any detected person to a Sales Agent below.")
            else:
                for sp in sp_analytics:
                    with st.container(border=True):
                        col_img, col_metrics, col_status = st.columns([1, 3, 1])
                        with col_img:
                            if sp.get("thumbnail_path") and Path(sp["thumbnail_path"]).exists():
                                st.image(sp["thumbnail_path"], use_container_width=True)
                            else:
                                st.markdown("<div style='height:100px; display:flex; align-items:center; justify-content:center; background:#1e222d; border-radius:6px; font-size:2.5em;'>👔</div>", unsafe_allow_html=True)
                        with col_metrics:
                            st.markdown(f"### {sp['name']} `({sp['sales_person_id']})`")
                            st.caption(f"🤖 AI Persona: {sp.get('ai_persona', 'Sales Agent')}")
                            
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Visits / Tours Led", sp["total_visits_conducted"], help="Total number of flat showing sessions conducted by this sales agent.")
                            m2.metric("Total Clients Shown", sp["total_visitors_shown"], help="Cumulative unique visitors escorted by this agent.")
                            m3.metric("Avg Tour Duration", sp["avg_tour_formatted"], help="Average time spent per tour.")
                            m4.metric("Total Duty Time", sp["total_duty_formatted"], help="Total working duration inside the flat.")
                        with col_status:
                            st.markdown(f"**Status:**")
                            st.markdown(f"`{sp['current_status']}`")

                # Sales Person Summary Table
                st.markdown("#### 📊 Sales Agents Summary Table")
                sp_table_rows = []
                for sp in sp_analytics:
                    sp_table_rows.append({
                        "Agent ID": sp["sales_person_id"],
                        "Agent Name": sp["name"],
                        "Visits / Tours Led": sp["total_visits_conducted"],
                        "Visitors Handled": sp["total_visitors_shown"],
                        "Avg Tour Length": sp["avg_tour_formatted"],
                        "Total Duty Time": sp["total_duty_formatted"],
                        "Current Status": sp["current_status"],
                    })
                df_sp = pd.DataFrame(sp_table_rows)
                st.dataframe(df_sp, use_container_width=True, hide_index=True)

            # Quick Role Management Card
            st.divider()
            with st.expander("⚙️ Designate / Change Sales Person Role"):
                all_p = gallery.get_all_persons()
                if all_p:
                    sel_id = st.selectbox(
                        "Select Person",
                        [p.global_id for p in all_p],
                        format_func=lambda x: f"{x} - {gallery.get_person(x).display_name} ({gallery.get_person(x).role})",
                        key="flat_sp_select",
                    )
                    curr_p = gallery.get_person(sel_id)
                    col_rn, col_chk, col_save = st.columns([2, 1, 1])
                    with col_rn:
                        new_agent_name = st.text_input("Name", value=curr_p.display_name if curr_p else "", key="flat_sp_name")
                    with col_chk:
                        is_sales_role = st.checkbox("Sales Agent Role", value=(curr_p.role == "sales_person") if curr_p else False, key="flat_sp_chk")
                    with col_save:
                        if st.button("Save Role", key="flat_sp_save", use_container_width=True):
                            gallery.designate_sales_person(sel_id, is_sales=is_sales_role, name=new_agent_name)
                            st.success(f"Updated {sel_id} successfully!")
                            st.rerun()

    # =========================================================================
    # TAB 2: FLAT VISIT TOUR LOGS
    # =========================================================================
    with tab_visits:
        section_title("Complete Flat Visit Sessions Log", "📜")
        sessions = tracker.get_all_sessions()

        if not sessions:
            empty_state("No visit sessions recorded yet", "📋", "Visits will appear here automatically when persons enter the flat.")
        else:
            filter_col1, filter_col2 = st.columns([2, 1])
            with filter_col1:
                search_agent = st.text_input("Filter by Sales Agent or Session ID", placeholder="e.g. Agent or VISIT-", key="search_sess")
            with filter_col2:
                status_filter = st.selectbox("Status Filter", ["All", "Active Only", "Completed Only"], key="status_sess")

            filtered = sessions
            if search_agent:
                filtered = [
                    s for s in filtered
                    if search_agent.lower() in s.sales_person_name.lower() or search_agent.lower() in s.session_id.lower()
                ]
            if status_filter == "Active Only":
                filtered = [s for s in filtered if s.is_active]
            elif status_filter == "Completed Only":
                filtered = [s for s in filtered if not s.is_active]

            records = []
            for s in filtered:
                records.append({
                    "Session ID": s.session_id,
                    "Sales Agent": s.sales_person_name,
                    "Visitors Count": s.visitor_count,
                    "Accompanying Visitor IDs": ", ".join(s.accompanying_visitor_ids) if s.accompanying_visitor_ids else "None",
                    "Start Time": s.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "End Time": s.end_time.strftime("%Y-%m-%d %H:%M:%S") if s.end_time else "In Progress",
                    "Tour Duration": s.format_duration(),
                    "Status": "🟢 Active" if s.is_active else "✅ Completed",
                })

            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Quick CSV Download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Flat Visit Sessions Log (CSV)",
                data=csv,
                file_name=f"flat_visits_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="dl_sessions_csv",
            )

    # =========================================================================
    # TAB 3: VISITOR DIRECTORY & AI PERSONA LOGS
    # =========================================================================
    with tab_visitors:
        section_title("Visitor Directory & Stay Records", "🧑")
        if gallery:
            v_logs = tracker.get_visitor_logs(gallery)
            if not v_logs:
                empty_state("No visitors recorded yet", "👤", "Visitors will appear here when detected.")
            else:
                df_v = pd.DataFrame(v_logs)
                # Drop raw thumbnail path for display table
                disp_df = df_v.drop(columns=["thumbnail_path", "role", "total_dwell_seconds"])
                st.dataframe(disp_df, use_container_width=True, hide_index=True)

                csv_v = df_v.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Visitor Log (CSV)",
                    data=csv_v,
                    file_name=f"flat_visitors_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="dl_visitors_csv",
                )

    # =========================================================================
    # TAB 4: REAL-TIME ACTIVITY TIMELINE
    # =========================================================================
    with tab_timeline:
        section_title("Real-Time Flat Activity Timeline", "⚡")
        events = tracker.get_events()
        if events:
            for evt in events[:35]:
                role_icon = "👔" if evt.get("role") == "sales_person" else "🧑"
                st.markdown(
                    f"<div style='padding: 8px 12px; margin-bottom: 6px; background-color: rgba(255,255,255,0.03); border-left: 3px solid {PALETTE['accent']}; border-radius: 4px;'>"
                    f"<span style='color: {PALETTE['muted']}; font-size: 0.85em;'>[{evt['timestamp']}]</span> "
                    f"{role_icon} <b>{evt['event_type'].replace('_', ' ').title()}</b>: {evt['message']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No timeline events logged yet.")


def _fmt_duration(seconds: float) -> str:
    s = int(seconds or 0)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"
