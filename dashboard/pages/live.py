from __future__ import annotations

import streamlit as st

from dashboard import state
from dashboard.components.cards import app_header, empty_state, kpi_row, section_title, status_pill
from dashboard.theme import PALETTE
from utils.image import bgr_to_rgb


from pathlib import Path
from dashboard.components.llm_config_card import render_llm_config_card


def _source_controls() -> None:
    cfg = state.get_config_cached()
    db = state.get_database()

    # Dynamic LLM Provider Configuration Section
    render_llm_config_card(cfg, db, expanded=False)

    section_title("Video Input Source", "🎥")
    kind = st.radio(
        "Input type",
        ["Upload Video File", "Local File Path", "Webcam", "CCTV / RTSP Stream"],
        horizontal=True,
        label_visibility="collapsed",
    )

    source: object = None
    label: str = ""

    col_a, col_b = st.columns([3, 1])
    with col_a:
        if kind == "Upload Video File":
            uploaded = st.file_uploader(
                "Upload Flat Visit Video (Up to 4 GB)",
                type=["mp4", "avi", "mov", "mkv", "m4v"],
                help="Upload recorded CCTV or camera footage from flat visits. Supports large files up to 4 GB.",
            )
            if uploaded is not None:
                upload_dir = cfg.paths.outputs / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                save_path = upload_dir / uploaded.name
                
                # Stream large files in 16MB chunks to avoid memory bottlenecks
                file_size_mb = uploaded.size / (1024 * 1024)
                if not save_path.exists() or save_path.stat().st_size != uploaded.size:
                    prog_bar = st.progress(0, text=f"Writing video file to disk ({file_size_mb:.1f} MB)...")
                    written = 0
                    with open(save_path, "wb") as f:
                        while chunk := uploaded.read(16 * 1024 * 1024):
                            f.write(chunk)
                            written += len(chunk)
                            prog_bar.progress(min(1.0, written / uploaded.size), text=f"Saved {written / (1024*1024):.1f} / {file_size_mb:.1f} MB")
                    prog_bar.empty()

                source = str(save_path)
                label = f"File: {uploaded.name}"
                size_str = f"{file_size_mb / 1024:.2f} GB" if file_size_mb >= 1000 else f"{file_size_mb:.1f} MB"
                st.success(f"✅ Ready: `{uploaded.name}` ({size_str})")
        elif kind == "Local File Path":
            # Auto-discover videos in data/ and outputs/uploads/
            discovered = []
            for scan_dir in [cfg.paths.outputs / "uploads", cfg.paths.data]:
                if scan_dir.exists():
                    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v"):
                        discovered.extend(list(scan_dir.glob(ext)))

            options = ["Enter Custom Path..."] + [str(p) for p in discovered]
            selected_opt = st.selectbox(
                "Select Existing Video or Enter Path",
                options,
                help="Directly process videos on your disk without uploading through browser.",
            )
            if selected_opt == "Enter Custom Path...":
                path_input = st.text_input(
                    "Full Video File Path on Disk",
                    placeholder="e.g. C:/videos/flat_visit_4k.mp4 or data/sample.mp4",
                )
            else:
                path_input = selected_opt

            if path_input and Path(path_input).exists():
                source = path_input
                sz = Path(path_input).stat().st_size / (1024 * 1024)
                sz_str = f"{sz / 1024:.2f} GB" if sz >= 1000 else f"{sz:.1f} MB"
                label = f"File: {Path(path_input).name}"
                st.success(f"✅ Found file: `{Path(path_input).name}` ({sz_str})")
            elif path_input:
                st.warning(f"File not found at: `{path_input}`")
                source = path_input
                label = "Video File"
        elif kind == "Webcam":
            cam_idx = st.number_input("Webcam index", min_value=0, value=0, step=1)
            source = int(cam_idx)
            label = f"Webcam {cam_idx}"
        else:
            stream_url = st.text_input("RTSP Stream URL", placeholder="rtsp://user:pass@host:554/stream")
            source = stream_url
            label = stream_url or "Stream"

    with col_b:
        record = st.toggle("Record Output", value=False, help="Save an annotated video to outputs/recordings")

    c1, c2, c3 = st.columns(3)
    running = state.is_running()
    can_start = (source is not None and str(source).strip() != "") and not running

    if c1.button("▶ Start Processing", use_container_width=True, disabled=not can_start, type="primary"):
        rec_path = None
        if record:
            from datetime import datetime

            rec_path = cfg.paths.recordings / (
                datetime.now().strftime("live_%Y%m%d_%H%M%S.mp4")
            )
        # Realtime = False if processing a pre-recorded video file so it runs smoothly
        is_file = isinstance(source, str) and not source.startswith("rtsp://") and not source.startswith("http://")
        state.start_source(source, label=str(label), realtime=not is_file, record_path=rec_path)
        st.rerun()

    if c2.button("⏸ Pause / Resume", use_container_width=True, disabled=not running):
        runner = st.session_state.get("runner")
        if runner:
            runner.pause() if runner.state.value == "running" else runner.resume()

    if c3.button("⏹ Stop", use_container_width=True, disabled=not running):
        state.stop_source()
        st.rerun()


@st.fragment(run_every=1.0)
def _live_view() -> None:
    result = state.get_latest_frame_result()
    orch = state.get_orchestrator(create=False)

    if result is None:
        empty_state("Waiting for frames…", "⏳", "The stream is starting up.")
        return

    left, right = st.columns([3, 2])
    with left:
        st.image(bgr_to_rgb(result.annotated_frame), use_container_width=True,
                 caption=f"Frame {result.frame_index} · {result.fps:.1f} FPS")
    with right:
        gallery = getattr(orch.engine.manager, "gallery", None) if orch else None
        tracker = getattr(orch, "flat_visits", None) if orch else None

        active_v = gallery.active_visitors_count() if gallery else result.person_count
        uniq_v = gallery.total_unique_visitors_count() if gallery else 0
        active_s = gallery.active_sales_count() if gallery else 0
        active_tours = len(tracker.get_active_sessions()) if tracker else 0

        kpi_row(
            [
                {"label": "Visitors in Flat", "value": active_v, "icon": "🧑‍🤝‍🧑", "accent": PALETTE["primary"]},
                {"label": "Total Unique", "value": uniq_v, "icon": "👤", "accent": PALETTE["accent"]},
                {"label": "Active Tours", "value": active_tours, "icon": "🏢", "accent": PALETTE["success"]},
                {"label": "Sales Agents", "value": active_s, "icon": "👔", "accent": PALETTE["warning"]},
            ],
            columns=2,
        )

        # Active Tour Card if in progress
        if tracker and tracker.get_active_sessions():
            active_s_list = tracker.get_active_sessions()
            curr = active_s_list[0]
            with st.container(border=True):
                st.markdown(f"**🔴 Active Flat Tour:** `{curr.session_id}`")
                st.caption(f"Sales Agent: {curr.sales_person_name} · Duration: {curr.format_duration()}")
                st.caption(f"Accompanying Visitors ({curr.visitor_count}): {', '.join(curr.accompanying_visitor_ids) if curr.accompanying_visitor_ids else 'None'}")

        section_title("Tracked Persons & Dwell Time", "🎯")
        tr = result.track_result
        if tr and tr.active_tracks:
            rows = []
            for t in tr.active_tracks:
                gid = t.global_id or f"P-{t.track_id:03d}"
                gp = t.global_person
                name = gp.display_name if gp else gid
                role = "👔 Sales Agent" if t.role == "sales_person" else "🧑 Visitor"
                sec = int(t.duration_seconds)
                m, rsec = divmod(sec, 60)
                stay_str = f"{m}m {rsec}s" if m > 0 else f"{rsec}s"
                
                # LLM Persona summary
                persona_str = gp.semantic_profile.persona_summary if (gp and gp.semantic_profile) else "Analyzing..."
                
                rows.append({
                    "Global ID": gid,
                    "Name": name,
                    "Role": role,
                    "Stay Time": stay_str,
                    "🤖 AI Persona": persona_str,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True, height=220)
        else:
            st.caption("No persons currently detected in frame.")

    # Live AI / LLM Decisions Stream
    if gallery and gallery.get_llm_decisions():
        st.divider()
        section_title("🤖 Live AI / LLM Re-ID Reasoning & Decision Stream", "🧠")
        st.caption(f"Real-time visual arbitration by {cfg.llm.provider.upper()} ({cfg.llm.model_name}) preventing duplicate visitor counting.")
        
        dec_rows = [d.to_dict() for d in reversed(gallery.get_llm_decisions()[-15:])]
        st.dataframe(
            dec_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "time": st.column_config.TextColumn("Time", width="small"),
                "track_id": st.column_config.NumberColumn("Track ID", width="small"),
                "decision": st.column_config.TextColumn("LLM Decision", width="medium"),
                "matched_id": st.column_config.TextColumn("Assigned ID", width="small"),
                "confidence": st.column_config.TextColumn("Confidence", width="small"),
                "reasoning": st.column_config.TextColumn("AI Reasoning & Visual Proof", width="large"),
                "persona": st.column_config.TextColumn("Extracted Persona", width="medium"),
                "engine": st.column_config.TextColumn("Model", width="small"),
            },
        )

    st.divider()
    c_snap, c_role = st.columns([1, 2])
    with c_snap:
        if st.button("📸 Capture Snapshot", use_container_width=True):
            runner = st.session_state.get("runner")
            if runner:
                path = runner.snapshot(prefix="flat_live")
                if path:
                    st.success(f"Snapshot saved: {path.name}")
    with c_role:
        if gallery and gallery.get_all_persons():
            persons_list = gallery.get_all_persons()
            selected_gid = st.selectbox(
                "Designate Sales Agent",
                [p.global_id for p in persons_list],
                format_func=lambda x: f"{x} - {gallery.get_person(x).display_name} ({gallery.get_person(x).role})",
            )
            if st.button("Toggle Sales / Visitor Role", use_container_width=True):
                p = gallery.get_person(selected_gid)
                if p:
                    new_role = (p.role != "sales_person")
                    gallery.designate_sales_person(selected_gid, is_sales=new_role)
                    st.success(f"Updated {selected_gid} to {'Sales Agent' if new_role else 'Visitor'}!")
                    st.rerun()


def render() -> None:
    app_header("Live Flat Monitor", "Real-time AI person detection, anti-double-counting, and stay time tracking")

    running = state.is_running()
    st_state = state.runner_state()
    pill = status_pill("● Live", "ok") if running else status_pill("Stopped", "idle")
    err = ""
    runner = st.session_state.get("runner")
    if runner and runner.stats.last_error:
        pill = status_pill("Error", "err")
        err = runner.stats.last_error
    st.markdown(pill, unsafe_allow_html=True)
    if err:
        st.error(f"Source error: {err}")

    _source_controls()
    st.divider()

    if running:
        _live_view()
    else:
        empty_state(
            "No active source", "📷",
            "Pick a webcam or stream above and press Start. "
            "On a headless server, use a stream URL or the Video Upload page.",
        )
