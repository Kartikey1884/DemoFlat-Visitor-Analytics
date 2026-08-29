from __future__ import annotations

import streamlit as st

from dashboard.components.cards import app_header, section_title


def render() -> None:
    app_header("About", "Smart Cafe Analytics platform")

    st.markdown(
        """
**CafeAnalytics** is a real-time computer-vision analytics platform that
turns a camera feed or recorded video into business intelligence for cafe
owners — customer flow, staff activity, table occupancy, queue monitoring
and foot-traffic heatmaps.
"""
    )

    section_title("Key Features", "✨")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
- 🎯 Real-time detection (YOLO26n)
- 🔗 Multi-object tracking (ByteTrack)
- 🧑‍🤝‍🧑 Customer flow & dwell analytics
- 👔 Staff detection & working/idle time
- 🪑 Table occupancy (occupied/empty/reserved)
"""
        )
    with c2:
        st.markdown(
            """
- 🧾 Queue length & wait-time monitoring
- 🔥 Movement heatmaps & trajectories
- 🚨 Intelligent alerts (overcrowd, long queue…)
- 📊 Interactive Plotly dashboards
- 🗄️ SQLite history & CSV export
"""
        )

    section_title("Technology", "🛠️")
    st.markdown(
        """
| Layer | Technology |
|------|------------|
| Detection | Ultralytics YOLO26n |
| Tracking | Supervision ByteTrack |
| Backend | Python, NumPy, OpenCV |
| Database | SQLAlchemy + SQLite |
| Dashboard | Streamlit + Plotly |
"""
    )

    section_title("Architecture", "🧱")
    st.markdown(
        """
```
Video → detection → tracking → analytics → database
                                   ↓            ↓
                               dashboard  ←  reports
```
The detector sits behind a stable interface, so YOLO26n can be swapped
for a newer model without touching downstream analytics.
"""
    )
    st.caption("Built in Python-only · Streamlit UI")
