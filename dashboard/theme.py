from __future__ import annotations

from typing import Any

from config import get_config

_cfg = get_config().dashboard

PALETTE = {
    "primary": _cfg.primary_color,
    "accent": _cfg.accent_color,
    "success": _cfg.success_color,
    "warning": _cfg.warning_color,
    "danger": _cfg.danger_color,
    "bg": "#0E1117",
    "surface": "#161A23",
    "surface_alt": "#1E2330",
    "border": "#2A3140",
    "text": "#E6E9EF",
    "muted": "#9AA4B2",
}

CHART_SEQUENCE = [
    PALETTE["primary"],
    PALETTE["accent"],
    PALETTE["warning"],
    PALETTE["success"],
    PALETTE["danger"],
    "#74B9FF",
    "#A29BFE",
    "#55EFC4",
]


def inject_css() -> str:
    p = PALETTE
    return f"""
<style>
:root {{
  --primary: {p['primary']};
  --accent: {p['accent']};
  --success: {p['success']};
  --warning: {p['warning']};
  --danger: {p['danger']};
  --surface: {p['surface']};
  --surface-alt: {p['surface_alt']};
  --border: {p['border']};
  --muted: {p['muted']};
}}

/* Layout polish */
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
#MainMenu, footer {{ visibility: hidden; }}
h1, h2, h3, h4 {{ letter-spacing: -0.02em; font-weight: 700; }}

/* App header band */
.ca-header {{
  display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.25rem;
}}
.ca-header .ca-logo {{ font-size: 1.9rem; }}
.ca-header .ca-title {{ font-size: 1.6rem; font-weight: 800; margin: 0; }}
.ca-subtitle {{ color: var(--muted); margin: 0 0 1.2rem 0; font-size: 0.95rem; }}

/* Metric cards */
.ca-card {{
  background: linear-gradient(160deg, var(--surface) 0%, var(--surface-alt) 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.05rem 1.2rem;
  box-shadow: 0 6px 20px rgba(0,0,0,0.25);
  transition: transform .15s ease, border-color .15s ease;
  height: 100%;
}}
.ca-card:hover {{ transform: translateY(-3px); border-color: var(--primary); }}
.ca-card .ca-card-top {{ display:flex; justify-content:space-between; align-items:center; }}
.ca-card .ca-label {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase;
  letter-spacing: .06em; font-weight: 600; }}
.ca-card .ca-icon {{ font-size: 1.25rem; opacity: .9; }}
.ca-card .ca-value {{ font-size: 2rem; font-weight: 800; margin: .35rem 0 .1rem 0; line-height: 1; }}
.ca-card .ca-delta {{ font-size: 0.8rem; font-weight: 600; }}
.ca-card .ca-delta.up {{ color: var(--success); }}
.ca-card .ca-delta.down {{ color: var(--danger); }}
.ca-card .ca-delta.flat {{ color: var(--muted); }}

/* Status pills */
.ca-pill {{ display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .7rem;
  border-radius: 999px; font-size:.8rem; font-weight:700; }}
.ca-pill .dot {{ width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }}
.ca-pill.ok {{ background: rgba(0,184,148,.14); color: var(--success); }}
.ca-pill.ok .dot {{ background: var(--success); box-shadow:0 0 8px var(--success); }}
.ca-pill.warn {{ background: rgba(253,203,110,.14); color: var(--warning); }}
.ca-pill.warn .dot {{ background: var(--warning); }}
.ca-pill.err {{ background: rgba(255,118,117,.14); color: var(--danger); }}
.ca-pill.err .dot {{ background: var(--danger); box-shadow:0 0 8px var(--danger); }}
.ca-pill.idle {{ background: rgba(154,164,178,.14); color: var(--muted); }}
.ca-pill.idle .dot {{ background: var(--muted); }}

/* Alert banners */
.ca-alert {{ border-radius: 12px; padding:.8rem 1rem; margin:.4rem 0; font-size:.9rem;
  border-left: 4px solid var(--muted); background: var(--surface); display:flex; gap:.6rem; align-items:center; }}
.ca-alert.critical {{ border-left-color: var(--danger); background: rgba(255,118,117,.08); }}
.ca-alert.warning {{ border-left-color: var(--warning); background: rgba(253,203,110,.08); }}
.ca-alert.info {{ border-left-color: var(--accent); background: rgba(0,206,201,.08); }}
.ca-alert .ca-alert-icon {{ font-size:1.1rem; }}

/* Section titles */
.ca-section {{ display:flex; align-items:center; gap:.5rem; margin: 1.4rem 0 .6rem 0;
  font-size:1.1rem; font-weight:700; }}
.ca-section .bar {{ width:.28rem; height:1.1rem; border-radius:4px; background: var(--primary); }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {p['bg']}; border-right:1px solid var(--border); }}
section[data-testid="stSidebar"] .ca-brand {{ font-size:1.25rem; font-weight:800; padding:.4rem 0 1rem 0; }}

/* Empty state */
.ca-empty {{ text-align:center; color:var(--muted); padding:2.5rem 1rem; border:1px dashed var(--border);
  border-radius:14px; background: var(--surface); }}
.ca-empty .ca-empty-icon {{ font-size:2.2rem; display:block; margin-bottom:.4rem; }}

/* Buttons */
.stButton>button {{ border-radius:10px; font-weight:600; border:1px solid var(--border); }}
.stButton>button:hover {{ border-color: var(--primary); color: var(--primary); }}
</style>
"""


def style_fig(fig: Any, height: int = 320, show_legend: bool = True) -> Any:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PALETTE["text"], family="sans-serif", size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        height=height,
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=CHART_SEQUENCE,
    )
    fig.update_xaxes(gridcolor=PALETTE["border"], zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE["border"], zeroline=False)
    return fig
