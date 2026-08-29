from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import plotly.graph_objects as go

from dashboard.theme import CHART_SEQUENCE, PALETTE, style_fig


def line_chart(
    x: Sequence, y: Sequence, name: str = "", color: Optional[str] = None, height: int = 300
) -> go.Figure:
    color = color or PALETTE["primary"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(x), y=list(y), mode="lines", name=name, line=dict(color=color, width=3),
            fill="tozeroy", fillcolor=_alpha(color, 0.12),
        )
    )
    return style_fig(fig, height=height, show_legend=bool(name))


def multi_line_chart(
    x: Sequence, series: Dict[str, Sequence], height: int = 300
) -> go.Figure:
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        color = CHART_SEQUENCE[i % len(CHART_SEQUENCE)]
        fig.add_trace(
            go.Scatter(x=list(x), y=list(y), mode="lines", name=name, line=dict(color=color, width=3))
        )
    return style_fig(fig, height=height)


def bar_chart(
    x: Sequence, y: Sequence, name: str = "", color: Optional[str] = None,
    height: int = 300, horizontal: bool = False,
) -> go.Figure:
    color = color or PALETTE["accent"]
    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(y=list(x), x=list(y), name=name, orientation="h", marker_color=color))
    else:
        fig.add_trace(go.Bar(x=list(x), y=list(y), name=name, marker_color=color))
    return style_fig(fig, height=height, show_legend=bool(name))


def area_chart(x: Sequence, y: Sequence, color: Optional[str] = None, height: int = 300) -> go.Figure:
    color = color or PALETTE["accent"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(x), y=list(y), mode="lines", line=dict(color=color, width=2),
            fill="tozeroy", fillcolor=_alpha(color, 0.25),
        )
    )
    return style_fig(fig, height=height, show_legend=False)


def pie_chart(
    labels: Sequence[str], values: Sequence[float], height: int = 300,
    colors: Optional[Sequence[str]] = None, hole: float = 0.55,
) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=list(labels), values=list(values), hole=hole,
            marker=dict(colors=list(colors) if colors else CHART_SEQUENCE),
            textinfo="label+percent",
        )
    )
    return style_fig(fig, height=height)


def histogram(
    values: Sequence[float], nbins: int = 20, color: Optional[str] = None, height: int = 300
) -> go.Figure:
    color = color or PALETTE["primary"]
    fig = go.Figure(go.Histogram(x=list(values), nbinsx=nbins, marker_color=color))
    return style_fig(fig, height=height, show_legend=False)


def labelled_histogram(
    buckets: Dict[str, int], color: Optional[str] = None, height: int = 300
) -> go.Figure:
    return bar_chart(list(buckets.keys()), list(buckets.values()), color=color, height=height)


def gauge(
    value: float, maximum: float, title: str = "", height: int = 260,
    color: Optional[str] = None,
) -> go.Figure:
    color = color or PALETTE["primary"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, maximum]},
                "bar": {"color": color},
                "bordercolor": PALETTE["border"],
                "bgcolor": PALETTE["surface"],
            },
        )
    )
    return style_fig(fig, height=height, show_legend=False)


def _alpha(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
