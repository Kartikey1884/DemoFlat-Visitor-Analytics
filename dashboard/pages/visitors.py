from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

from dashboard import state
from dashboard.components.cards import app_header, empty_state, kpi_row, section_title, status_pill
from dashboard.theme import PALETTE


def render() -> None:
    app_header(
        "Unique Visitors & Dwell Profiles",
        "Biometric visual Re-Identification gallery, anti-duplicate counting, and individual stay duration logs",
    )

    orch = state.get_orchestrator(create=False)
    gallery = getattr(orch.engine.manager, "gallery", None) if orch else None

    if gallery is None:
        empty_state(
            "No active person gallery",
            "👥",
            "Start a video stream or camera from the Live or Video Upload pages to register unique visitors.",
        )
        return

    all_persons = gallery.get_all_persons()
    total_uniq = gallery.total_unique_visitors_count()
    active_now = gallery.active_visitors_count()
    active_sales = gallery.active_sales_count()
    returning = sum(1 for p in all_persons if p.role == "visitor" and p.visit_count > 1)

    kpi_row(
        [
            {"label": "Total Unique Visitors", "value": total_uniq, "icon": "👤", "accent": PALETTE["primary"]},
            {"label": "Active in Flat Now", "value": active_now, "icon": "🟢", "accent": PALETTE["accent"]},
            {"label": "Returning Visitors", "value": returning, "icon": "🔄", "accent": PALETTE["success"]},
            {"label": "Active Sales Agents", "value": active_sales, "icon": "👔", "accent": PALETTE["warning"]},
        ],
        columns=4,
    )

    st.divider()

    # Search & Filter Controls
    c_search, c_role, c_status = st.columns([2, 1, 1])
    with c_search:
        search_query = st.text_input("Search by ID or Name", placeholder="e.g. P-001 or Visitor")
    with c_role:
        role_filter = st.selectbox("Role", ["All Roles", "Visitors Only", "Sales Agents Only"])
    with c_status:
        status_filter = st.selectbox("Current Presence", ["All", "Currently Active", "Left / Inactive"])

    filtered = all_persons
    if search_query:
        filtered = [
            p for p in filtered
            if search_query.lower() in p.global_id.lower() or search_query.lower() in p.display_name.lower()
        ]
    if role_filter == "Visitors Only":
        filtered = [p for p in filtered if p.role == "visitor"]
    elif role_filter == "Sales Agents Only":
        filtered = [p for p in filtered if p.role == "sales_person"]

    if status_filter == "Currently Active":
        filtered = [p for p in filtered if p.is_active]
    elif status_filter == "Left / Inactive":
        filtered = [p for p in filtered if not p.is_active]

    if not filtered:
        empty_state("No matching persons found", "🔍", "Adjust your filters or allow more visitors to enter.")
        return

    section_title(f"Registered Persons Directory ({len(filtered)})", "🗂️")

    # Display gallery cards in a 3-column layout
    cols = st.columns(3)
    for idx, p in enumerate(filtered):
        col = cols[idx % 3]
        with col:
            with st.container(border=True):
                # Header with ID & Presence pill
                h1, h2 = st.columns([2, 1])
                with h1:
                    role_badge = "👔 Sales Agent" if p.role == "sales_person" else "🧑 Visitor"
                    st.markdown(f"**{p.global_id}** · `{role_badge}`")
                with h2:
                    st.markdown(
                        status_pill("In Flat", "ok") if p.is_active else status_pill("Away", "idle"),
                        unsafe_allow_html=True,
                    )

                # Thumbnail image
                if p.thumbnail_path and Path(p.thumbnail_path).exists():
                    st.image(p.thumbnail_path, use_container_width=True)
                elif p.thumbnail_base64:
                    st.markdown(
                        f'<img src="data:image/jpeg;base64,{p.thumbnail_base64}" style="width:100%; border-radius:6px; max-height:160px; object-fit:cover;"/>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='height:120px; display:flex; align-items:center; justify-content:center; background:#222; border-radius:6px; font-size:2.5em;'>👤</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(f"**Name:** {p.display_name}")
                st.markdown(f"**Total Stay Duration:** `{_fmt_duration(p.total_dwell_seconds)}`")
                st.markdown(f"**Visits Count:** `{p.visit_count}` {'(Returning)' if p.visit_count > 1 else '(First-time)'}")
                st.caption(f"First seen: {p.first_seen.strftime('%H:%M:%S')} · Last: {p.last_seen.strftime('%H:%M:%S')}")

                # LLM Semantic Personality & Clothing Signature Card
                if getattr(p, "semantic_profile", None) is not None:
                    prof = p.semantic_profile
                    with st.container(border=True):
                        st.markdown(f"<small>🤖 <b>AI Visual Persona</b> ({prof.extracted_by})</small>", unsafe_allow_html=True)
                        st.markdown(f"<span style='font-size:0.88em;'><b>Top:</b> {prof.upper_clothing}</span>", unsafe_allow_html=True)
                        st.markdown(f"<span style='font-size:0.88em;'><b>Bottom:</b> {prof.lower_clothing}</span>", unsafe_allow_html=True)
                        if prof.accessories:
                            st.markdown(f"<span style='font-size:0.85em; color:{PALETTE['accent']};'><b>Acc:</b> {', '.join(prof.accessories)}</span>", unsafe_allow_html=True)
                        st.caption(f"Signature: \"{prof.persona_summary}\"")
                        if getattr(p, "llm_reasoning", ""):
                            st.caption(f"🧠 **AI Reasoning:** *{p.llm_reasoning}*")

                # Action to designate role / change name
                with st.expander("⚙️ Edit Profile"):
                    new_name = st.text_input(f"Name for {p.global_id}", value=p.display_name, key=f"name_{p.global_id}")
                    is_sales = st.checkbox("Mark as Sales Person", value=(p.role == "sales_person"), key=f"role_{p.global_id}")
                    if st.button("Save Changes", key=f"btn_{p.global_id}"):
                        gallery.designate_sales_person(p.global_id, is_sales=is_sales, name=new_name)
                        st.success("Updated!")
                        st.rerun()

    st.divider()

    # Table View & Export
    section_title("Dwell Time & AI Persona Table", "📊")
    table_rows = []
    for p in filtered:
        prof = getattr(p, "semantic_profile", None)
        table_rows.append({
            "Global ID": p.global_id,
            "Name": p.display_name,
            "Role": "Sales Person" if p.role == "sales_person" else "Visitor",
            "AI Persona Summary": prof.persona_summary if prof else "Processing...",
            "Upper Clothing": prof.upper_clothing if prof else "—",
            "Lower Clothing": prof.lower_clothing if prof else "—",
            "Accessories": ", ".join(prof.accessories) if prof and prof.accessories else "None",
            "Total Stay Time": _fmt_duration(p.total_dwell_seconds),
            "Stay Seconds": round(p.total_dwell_seconds, 1),
            "Visits Count": p.visit_count,
            "Status": "🟢 Inside Flat" if p.is_active else "⚪ Left",
            "First Seen": p.first_seen.strftime("%Y-%m-%d %H:%M:%S"),
            "Last Seen": p.last_seen.strftime("%Y-%m-%d %H:%M:%S"),
        })

    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Export Visitor Dwell Profiles (CSV)",
        data=csv_data,
        file_name=f"visitor_dwell_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def _fmt_duration(seconds: float) -> str:
    s = int(seconds or 0)
    m, sec = divmod(s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {sec}s"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"
