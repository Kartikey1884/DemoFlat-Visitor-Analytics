from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard import state
from dashboard.components.cards import app_header, empty_state, section_title
from reports.generator import ReportGenerator


def _visits_dataframe(db, session_id=None) -> pd.DataFrame:
    visits = db.get_visits(session_id=session_id)
    return pd.DataFrame(
        [
            {
                "track_id": v.track_id,
                "role": v.role,
                "entry_time": v.entry_time,
                "exit_time": v.exit_time,
                "duration_seconds": v.duration_seconds,
                "session_id": v.session_id,
            }
            for v in visits
        ]
    )


def _snapshots_dataframe(db, session_id=None) -> pd.DataFrame:
    snaps = db.get_snapshots(session_id=session_id)
    return pd.DataFrame(
        [
            {
                "timestamp": s.timestamp,
                "current_customers": s.current_customers,
                "current_staff": s.current_staff,
                "occupied_tables": s.occupied_tables,
                "queue_length": s.queue_length,
                "avg_wait_seconds": s.avg_wait_seconds,
                "fps": s.fps,
            }
            for s in snaps
        ]
    )


def _report_builder(db, sid) -> None:
    section_title("Generate Report", "🧾")
    st.caption("Styled Excel workbook or branded PDF with summary KPIs and charts.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        gen = ReportGenerator(db, session_id=sid)
    except Exception as exc:
        st.error(f"Could not gather report data: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 Build Excel", use_container_width=True):
            st.session_state["_report_xlsx"] = gen.build_excel()
        if st.session_state.get("_report_xlsx"):
            st.download_button(
                "⬇ Download .xlsx", st.session_state["_report_xlsx"],
                file_name=f"cafe_report_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    with c2:
        if st.button("📄 Build PDF", use_container_width=True):
            st.session_state["_report_pdf"] = gen.build_pdf()
        if st.session_state.get("_report_pdf"):
            st.download_button(
                "⬇ Download .pdf", st.session_state["_report_pdf"],
                file_name=f"cafe_report_{stamp}.pdf", mime="application/pdf",
                use_container_width=True,
            )
    with c3:
        if st.button("💾 Save to disk", use_container_width=True,
                     help="Write PDF + Excel to reports/generated and record them"):
            pdf_path = gen.generate("pdf")
            xlsx_path = gen.generate("excel")
            st.success(f"Saved {pdf_path.name} and {xlsx_path.name}")


def render() -> None:
    app_header("Reports", "Export analytics data and download reports")
    db = state.get_database()
    orch = state.get_orchestrator(create=False)
    session_id = orch.session_id if orch else None

    scope = st.radio("Scope", ["Current session", "All history"], horizontal=True)
    sid = session_id if scope == "Current session" else None

    _report_builder(db, sid)

    visits_df = _visits_dataframe(db, sid)
    snaps_df = _snapshots_dataframe(db, sid)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    section_title("Visit History", "🚪")
    if not visits_df.empty:
        st.dataframe(visits_df, use_container_width=True, hide_index=True, height=280)
        st.download_button(
            "⬇ Download visits CSV",
            visits_df.to_csv(index=False).encode("utf-8"),
            file_name=f"visits_{stamp}.csv",
            mime="text/csv",
        )
    else:
        empty_state("No visits recorded", "🚪")

    section_title("KPI Snapshots", "📈")
    if not snaps_df.empty:
        st.dataframe(snaps_df, use_container_width=True, hide_index=True, height=240)
        st.download_button(
            "⬇ Download snapshots CSV",
            snaps_df.to_csv(index=False).encode("utf-8"),
            file_name=f"snapshots_{stamp}.csv",
            mime="text/csv",
        )
    else:
        empty_state("No snapshots recorded", "📈")

    section_title("Generated Reports", "📄")
    st.caption("Reports saved to disk from the builder above are listed here.")
    reports = db.get_reports(limit=20)
    if reports:
        st.dataframe(
            [{"title": r.title, "type": r.report_type, "path": r.file_path,
              "generated": r.generated_at} for r in reports],
            use_container_width=True, hide_index=True,
        )
    else:
        empty_state("No reports generated yet", "📄")
