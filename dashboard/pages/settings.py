from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import state
from dashboard.components.cards import app_header, empty_state, section_title
from utils import image as img_utils
from utils.image import bgr_to_rgb

_ZONE_COLORS = {
    "table": "#00B894",
    "queue": "#FF7675",
    "staff": "#FDCB6E",
    "roi": "#6C5CE7",
    "line": "#00CEC9",
}


def _thresholds(cfg, db) -> None:
    section_title("Detection & Tracking", "🎛️")
    c1, c2, c3 = st.columns(3)
    conf = c1.slider("Confidence", 0.1, 0.9, float(cfg.detection.confidence), 0.05)
    iou = c2.slider("IoU", 0.1, 0.9, float(cfg.detection.iou), 0.05)
    track_act = c3.slider("Track activation", 0.1, 0.9,
                          float(cfg.tracking.track_activation_threshold), 0.05)

    section_title("Analytics Thresholds", "📐")
    c4, c5, c6 = st.columns(3)
    capacity = c4.number_input("Max capacity", 1, 500, int(cfg.analytics.max_capacity))
    queue_thr = c5.number_input("Queue alert length", 1, 50, int(cfg.analytics.queue_length_alert))
    opacity = c6.slider("Heatmap opacity", 0.1, 1.0, float(cfg.analytics.heatmap_opacity), 0.05)
    c7, c8, c9 = st.columns(3)
    table_dwell = c7.number_input("Table dwell (s)", 0.0, 60.0,
                                  float(cfg.analytics.table_occupied_seconds), 0.5)
    wait_alert = c8.number_input("Queue wait alert (s)", 10, 3600,
                                 int(cfg.analytics.queue_wait_alert_seconds))
    mode = c9.selectbox("Staff strategy", ["zone", "color", "manual"],
                        index=["zone", "color", "manual"].index(cfg.analytics.staff_classification))

    if st.button("💾 Apply & save settings"):
        updates = {
            "detection.confidence": conf,
            "detection.iou": iou,
            "tracking.track_activation_threshold": track_act,
            "analytics.max_capacity": int(capacity),
            "analytics.queue_length_alert": int(queue_thr),
            "analytics.heatmap_opacity": opacity,
            "analytics.table_occupied_seconds": float(table_dwell),
            "analytics.queue_wait_alert_seconds": float(wait_alert),
            "analytics.staff_classification": mode,
        }
        for key, value in updates.items():
            sect, attr = key.split(".")
            setattr(getattr(cfg, sect), attr, value)
            db.set_setting(key, value)
        st.success("Settings applied to the live pipeline and saved.")


def _get_base_frame() -> Tuple[np.ndarray, str]:
    result = state.get_latest_frame_result()
    if result is not None:
        return result.raw_frame.copy(), "current live frame"

    upload = st.file_uploader(
        "Reference image (optional — else a blank canvas is used)",
        type=["jpg", "jpeg", "png"], key="zone_ref_img",
    )
    if upload is not None:
        import cv2

        arr = np.frombuffer(upload.getbuffer(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img, "uploaded image"

    c1, c2 = st.columns(2)
    w = int(c1.number_input("Canvas width", 320, 3840, 1280, 20))
    h = int(c2.number_input("Canvas height", 240, 2160, 720, 20))
    blank = np.full((h, w, 3), 32, dtype=np.uint8)
    return blank, "blank canvas"


def _draw_all_zones(frame: np.ndarray, db, edit_points: List[Tuple[int, int]],
                    edit_type: str) -> np.ndarray:
    canvas = frame.copy()
    for z in db.get_zones(enabled_only=False):
        color = img_utils.hex_to_bgr(_ZONE_COLORS.get(z.zone_type, "#6C5CE7"))
        pts = [(float(x), float(y)) for x, y in z.points]
        if z.zone_type == "line" and len(pts) >= 2:
            img_utils.draw_line(canvas, (pts[0], pts[1]), color=color, label=z.name)
        elif len(pts) >= 2:
            img_utils.draw_polygon(canvas, pts, color=color, fill_alpha=0.15, label=z.name)

    if len(edit_points) >= 2:
        color = img_utils.hex_to_bgr(_ZONE_COLORS.get(edit_type, "#FFFFFF"))
        if edit_type == "line":
            img_utils.draw_line(canvas, (edit_points[0], edit_points[1]), color=(255, 255, 255),
                                thickness=3)
        else:
            img_utils.draw_polygon(canvas, edit_points, color=(255, 255, 255), thickness=3)
    for p in edit_points:
        img_utils.draw_point(canvas, p, color=(255, 255, 255), radius=6)
    return canvas


def _clean_points(df: pd.DataFrame) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    for _, row in df.iterrows():
        x, y = row.get("x"), row.get("y")
        if pd.notna(x) and pd.notna(y):
            points.append((int(x), int(y)))
    return points


def _zone_editor(db) -> None:
    section_title("Visual Zone Editor", "🗺️")
    st.caption(
        "Enter polygon points (or two points for a counting line) — the "
        "preview updates live. Tables become occupancy zones; queues track "
        "waits; lines can count entries/exits."
    )

    frame, source_desc = _get_base_frame()
    h, w = frame.shape[:2]

    c1, c2, c3 = st.columns([2, 1, 1])
    name = c1.text_input("Zone name", placeholder="e.g. Table 3 / Cashier")
    ztype = c2.selectbox("Type", ["table", "queue", "staff", "roi", "line"])
    kind = "counter"
    reserved = False
    if ztype == "queue":
        kind = c3.selectbox("Queue kind", ["cashier", "pickup", "counter"])
    elif ztype == "table":
        reserved = c3.checkbox("Reserved")

    default = (
        [[int(w * 0.4), int(h * 0.4)], [int(w * 0.6), int(h * 0.6)]]
        if ztype == "line"
        else [[int(w * 0.35), int(h * 0.35)], [int(w * 0.65), int(h * 0.35)],
              [int(w * 0.65), int(h * 0.65)], [int(w * 0.35), int(h * 0.65)]]
    )
    edited = st.data_editor(
        pd.DataFrame(default, columns=["x", "y"]),
        num_rows="dynamic", use_container_width=True, key=f"pts_{ztype}",
        column_config={
            "x": st.column_config.NumberColumn("x", min_value=0, max_value=w),
            "y": st.column_config.NumberColumn("y", min_value=0, max_value=h),
        },
    )
    points = _clean_points(edited)

    preview = _draw_all_zones(frame, db, points, ztype)
    st.image(bgr_to_rgb(preview), use_container_width=True,
             caption=f"Preview over {source_desc}")

    min_pts = 2 if ztype == "line" else 3
    can_save = bool(name) and len(points) >= min_pts
    if st.button("💾 Save zone", disabled=not can_save):
        meta = {"kind": kind} if ztype == "queue" else None
        db.upsert_zone(name, ztype, [[x, y] for x, y in points], reserved=reserved, meta=meta)
        orch = state.get_orchestrator(create=False)
        if orch:
            orch.reload_zones()
        st.success(f"Saved {ztype} zone '{name}' with {len(points)} points.")
    elif not can_save:
        st.caption(f"Enter a name and at least {min_pts} points to save.")

    _zone_list(db)


def _zone_list(db) -> None:
    section_title("Existing Zones", "📋")
    zones = db.get_zones(enabled_only=False)
    if not zones:
        empty_state("No zones defined yet", "🗺️")
        return
    for z in zones:
        cols = st.columns([2, 1, 1, 1, 1])
        cols[0].markdown(f"**{z.name}**")
        cols[1].caption(f"{z.zone_type} · {len(z.points)} pts")
        cols[2].caption("🔒" if z.reserved else "—")
        if cols[3].button("Reserve", key=f"res_{z.id}"):
            db.upsert_zone(z.name, z.zone_type, z.points, reserved=not z.reserved,
                           meta=z.meta)
            st.rerun()
        if cols[4].button("🗑", key=f"del_{z.id}"):
            db.delete_zone(z.id)
            orch = state.get_orchestrator(create=False)
            if orch:
                orch.reload_zones()
            st.rerun()


def _camera(cfg, db) -> None:
    section_title("Camera & Calibration", "📷")
    c1, c2 = st.columns(2)
    device = c1.selectbox(
        "Inference device", ["auto", "cpu", "cuda:0", "mps"],
        index=["auto", "cpu", "cuda:0", "mps"].index(cfg.detection.device)
        if cfg.detection.device in ["auto", "cpu", "cuda:0", "mps"] else 0,
    )
    ppm = c2.number_input(
        "Pixels per metre (speed calibration; 0 = disabled)",
        0.0, 2000.0, float(cfg.analytics.pixels_per_meter or 0.0), 1.0,
    )
    default_source = st.text_input(
        "Default source (webcam index or stream URL)",
        value=str(db.get_setting("camera.default_source", "0")),
    )
    st.caption(f"Model weights: `{cfg.detection.model_path}` · Output dir: `{cfg.paths.outputs}`")

    if st.button("💾 Save camera settings"):
        cfg.detection.device = device
        cfg.analytics.pixels_per_meter = ppm or None
        db.set_setting("detection.device", device)
        db.set_setting("analytics.pixels_per_meter", ppm or None)
        db.set_setting("camera.default_source", default_source)
        st.success("Camera settings saved.")


def _data(db) -> None:
    section_title("Database", "🗄️")
    counts = db.table_counts()
    cols = st.columns(len(counts))
    for col, (name, n) in zip(cols, counts.items()):
        col.metric(name.title(), n)

    section_title("Reset", "🔄")
    st.caption("Zero all KPIs and clear history so the owner can start fresh. "
               "Your zones and settings are kept.")
    confirm = st.checkbox("I understand this permanently deletes analytics data")
    c1, c2 = st.columns(2)
    if c1.button("🔄 Reset all KPIs to zero", disabled=not confirm,
                 use_container_width=True,
                 help="Stops any source, clears visits/snapshots/alerts, resets live counters."):
        removed = state.reset_all()
        st.success(f"Reset done — cleared {sum(removed.values())} records. All KPIs are back to zero.")
        st.rerun()
    if c2.button("♻️ Full database reset", disabled=not confirm,
                 use_container_width=True,
                 help="Also drops zones and settings (recreates all tables)."):
        state.stop_source()
        db.drop_all()
        db.create_all()
        st.session_state["orchestrator"] = None
        st.success("Database fully reset.")
        st.rerun()


from dashboard.components.llm_config_card import render_llm_config_card


def render() -> None:
    app_header("Settings", "Tune thresholds, configure AI Vision LLMs, draw zones, and manage data")
    cfg = state.get_config_cached()
    db = state.get_database()

    tab_llm, tab_thr, tab_zones, tab_cam, tab_data = st.tabs(
        ["🤖 AI Vision LLM", "🎛️ Thresholds", "🗺️ Zones", "📷 Camera", "🗄️ Data"]
    )
    with tab_llm:
        render_llm_config_card(cfg, db, expanded=True)
    with tab_thr:
        _thresholds(cfg, db)
    with tab_zones:
        _zone_editor(db)
    with tab_cam:
        _camera(cfg, db)
    with tab_data:
        _data(db)
