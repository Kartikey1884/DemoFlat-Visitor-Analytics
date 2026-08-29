from __future__ import annotations

from typing import List, Optional, Sequence

import streamlit as st

from dashboard.theme import PALETTE


def section_title(text: str, icon: str = "") -> None:
    icon_html = f"<span>{icon}</span>" if icon else ""
    st.markdown(
        f'<div class="ca-section"><span class="bar"></span>{icon_html}<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: object,
    icon: str = "",
    delta: Optional[str] = None,
    delta_dir: str = "flat",
    accent: Optional[str] = None,
) -> str:
    accent = accent or PALETTE["primary"]
    delta_html = ""
    if delta is not None:
        arrow = {"up": "▲", "down": "▼", "flat": "•"}.get(delta_dir, "•")
        delta_html = f'<div class="ca-delta {delta_dir}">{arrow} {delta}</div>'
    return f"""
<div class="ca-card">
  <div class="ca-card-top">
    <div class="ca-label">{label}</div>
    <div class="ca-icon">{icon}</div>
  </div>
  <div class="ca-value" style="color:{accent}">{value}</div>
  {delta_html}
</div>
"""


def kpi_row(cards: Sequence[dict], columns: int = 4) -> None:
    for start in range(0, len(cards), columns):
        row = cards[start : start + columns]
        cols = st.columns(len(row))
        for col, card in zip(cols, row):
            with col:
                st.markdown(metric_card(**card), unsafe_allow_html=True)


def status_pill(label: str, kind: str = "idle") -> str:
    return f'<span class="ca-pill {kind}"><span class="dot"></span>{label}</span>'


def alert_banner(message: str, severity: str = "info", icon: Optional[str] = None) -> None:
    icons = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
    ic = icon or icons.get(severity, "ℹ️")
    st.markdown(
        f'<div class="ca-alert {severity}"><span class="ca-alert-icon">{ic}</span>'
        f'<span>{message}</span></div>',
        unsafe_allow_html=True,
    )


def empty_state(message: str, icon: str = "📭", hint: str = "") -> None:
    hint_html = f'<div style="font-size:.85rem;margin-top:.3rem">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="ca-empty"><span class="ca-empty-icon">{icon}</span>'
        f'<div>{message}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def app_header(title: str, subtitle: str, logo: str = "☕") -> None:
    st.markdown(
        f'<div class="ca-header"><span class="ca-logo">{logo}</span>'
        f'<h1 class="ca-title">{title}</h1></div>'
        f'<p class="ca-subtitle">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def render_alerts(alerts: List[dict], limit: int = 6, empty_msg: str = "No active alerts") -> None:
    if not alerts:
        alert_banner(empty_msg, severity="info", icon="✅")
        return
    for alert in alerts[:limit]:
        alert_banner(alert.get("message", ""), severity=alert.get("severity", "info"))
