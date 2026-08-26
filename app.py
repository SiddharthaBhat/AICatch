"""Streamlit dashboard for AICatch CCTV intelligence."""
from __future__ import annotations

import logging
import shutil
import sys
import uuid
import inspect
from pathlib import Path

import streamlit as st
from PIL import Image

from config import OUTPUT_DIR, THUMBNAIL_DIR, UPLOAD_DIR, ensure_directories

logging.basicConfig(level=logging.INFO)
ensure_directories()
st.set_page_config(page_title="AICatch | CCTV Intelligence", page_icon="🎯", layout="wide")

st.markdown("""<style>
    .stApp {background:#07111f;color:#e8edf6}.block-container {padding-top:2rem}
    [data-testid="stSidebar"] {background:#0b1728}.stButton>button {background:#00a6a6;color:white;border:0;border-radius:7px;font-weight:600}
    .metric-card {background:#101f34;border:1px solid #1e3858;border-radius:10px;padding:12px;margin:5px 0}
</style>""", unsafe_allow_html=True)

if "cameras" not in st.session_state: st.session_state.cameras = {}
if "matches" not in st.session_state: st.session_state.matches = []
if "journey" not in st.session_state: st.session_state.journey = []
if "context_clip" not in st.session_state: st.session_state.context_clip = None

def _save_upload(upload) -> str:
    """Persist Streamlit's uploaded stream with its original video suffix for OpenCV."""
    suffix = Path(upload.name).suffix.lower() or ".mp4"
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    upload.seek(0)
    with path.open("wb") as temporary_video:
        shutil.copyfileobj(upload, temporary_video)
    upload.seek(0)
    return str(path)

def _thumbnail(match: dict) -> Path:
    return THUMBNAIL_DIR / "".join(char if char.isalnum() or char in "-_" else "_" for char in match["metadata"]["camera_id"]) / f"{match['id']}.jpg"


def _available_camera_name(preferred_name: str) -> str:
    """Return a camera label that does not overwrite an existing upload."""
    base_name = preferred_name.strip() or "Camera"
    if base_name not in st.session_state.cameras:
        return base_name
    number = 2
    while f"{base_name} {number}" in st.session_state.cameras:
        number += 1
    return f"{base_name} {number}"


with st.sidebar:
    st.title("🎯 AICatch")
    st.caption("Enterprise CCTV Search & Tracking")
    st.divider(); st.subheader("Camera Manager")
    st.caption("Use the upload control below for CCTV video files. The upload in the main panel is for search photos only.")
    upload = st.file_uploader(
        "CCTV video file (all extensions allowed)",
        help="All file extensions are allowed so any video container can be selected.",
    )
    suggested_camera_name = Path(upload.name).stem if upload else f"Camera {len(st.session_state.cameras) + 1}"
    camera_name = st.text_input("Camera name", value=suggested_camera_name)
    if upload:
        st.video(upload, format=upload.type or "video/mp4")
        upload_token = f"{upload.name}:{upload.size}"
        if st.session_state.get("saved_upload_token") != upload_token:
            automatic_name = _available_camera_name(Path(upload.name).stem)
            st.session_state.cameras[automatic_name] = _save_upload(upload)
            st.session_state.saved_upload_token = upload_token
            st.success(f"Added {automatic_name}. Now click Index All Cameras.")
    source_url = st.text_input("Or Cloud S3 / RTSP URL", placeholder="rtsp://... or s3 URL")
    if st.button("Add uploaded video as camera", use_container_width=True):
        source = _save_upload(upload) if upload else source_url.strip()
        if camera_name.strip() and source:
            st.session_state.cameras[camera_name.strip()] = source
            st.success(f"Added {camera_name.strip()}")
        else: st.error("Provide a camera name and a video upload or URL.")
    for camera, source in list(st.session_state.cameras.items()):
        left, right = st.columns([4, 1]); left.caption(f"📹 {camera}")
        if right.button("×", key=f"remove_{camera}"):
            del st.session_state.cameras[camera]; st.rerun()
    if st.button("🚀 Index All Cameras", use_container_width=True, disabled=not st.session_state.cameras):
        progress = st.progress(0, text="Loading AI models and starting fast indexing…")
        summaries = []
        try:
            # Loading the AI stack is expensive, so defer it until indexing is requested.
            from video_engine import process_video_stream

            for index, (camera, source) in enumerate(st.session_state.cameras.items(), 1):
                def update_camera_progress(done: int, total: int, camera_index: int = index, camera_name: str = camera) -> None:
                    # Streams often do not report a total length; keep the UI alive in that case.
                    within_camera = (done / total) if total else 0.0
                    overall = ((camera_index - 1) + within_camera) / len(st.session_state.cameras)
                    progress.progress(min(overall, 0.99), text=f"Indexing {camera_name}: {done / (total or 1):.0%}" if total else f"Indexing {camera_name}…")
                try:
                    # A hot-reloaded Streamlit session can retain an older module
                    # object; fall back cleanly until its next full restart.
                    if "progress_callback" in inspect.signature(process_video_stream).parameters:
                        summaries.append(process_video_stream(source, camera, progress_callback=update_camera_progress))
                    else:
                        summaries.append(process_video_stream(source, camera))
                except Exception as error: st.error(f"{camera}: {error}")
                progress.progress(index / len(st.session_state.cameras), text=f"Indexed {camera}")
        except Exception as error:
            st.error(f"Could not start video indexing: {error}")
        indexed_subjects = sum(summary["subjects_indexed"] for summary in summaries)
        st.session_state.index_summary = summaries
        if indexed_subjects:
            st.success(f"Indexing complete: {indexed_subjects} subjects added to search.")
        else:
            st.warning("No people or vehicles were indexed. Check the error message above.")

st.title("CCTV Intelligence Command Center")
st.caption(f"{len(st.session_state.cameras)} connected camera(s) • YOLO subject crops • CLIP multimodal retrieval")
search_tab, stitch_tab, log_tab = st.tabs(["💬 Multimodal Search & Chat", "🎬 Multi-Camera Journey Stitcher", "📊 Journey Log Details"])

with search_tab:
    query_col, preview_col = st.columns([2, 1])
    with query_col:
        text_query = st.text_input("Ask AICatch", placeholder="person in red jacket, white sedan, person with backpack")
        image_query = st.file_uploader("Target photo only (JPG or PNG)", type=["jpg", "jpeg", "png"], key="query_image")
        count = st.slider("Matches", 1, 50, 12)
        if st.button("Search all indexed footage", type="primary"):
            try:
                from db_engine import get_image_embedding, get_text_embedding, query_vectors

                embedding = get_image_embedding(Image.open(image_query)) if image_query else get_text_embedding(text_query)
                st.session_state.matches = query_vectors(embedding, count, query_text=None if image_query else text_query)
                st.session_state.journey = []
                st.success(f"Found {len(st.session_state.matches)} candidate observations.")
            except Exception as error: st.error(f"Search failed: {error}")
    with preview_col:
        if image_query: st.image(image_query, caption="Query image", use_container_width=True)
        elif st.session_state.cameras: st.info("Add a text or image query, then search across all indexed cameras.")
    if st.session_state.matches:
        st.subheader("Ranked observations")
        columns = st.columns(4)
        for index, match in enumerate(st.session_state.matches):
            metadata = match["metadata"]
            with columns[index % 4]:
                thumb = _thumbnail(match)
                if thumb.exists(): st.image(str(thumb), use_container_width=True)
                st.markdown(f"**{metadata['camera_id']}** · {float(metadata['timestamp']):.2f}s")
                st.caption(f"Similarity: {match['similarity']:.1%}")
                primary_plate = metadata.get("primary_plate", "")
                plate_candidates = metadata.get("plate_candidates", "")
                if primary_plate:
                    candidates_list = [c.strip() for c in plate_candidates.split(",") if c.strip() and c.strip() != primary_plate]
                    options_str = f" | Options: [{', '.join(candidates_list)}]" if candidates_list else ""
                    st.caption(f"Plate: {primary_plate}{options_str}")
                if st.button("Play full-scene clip", key=f"clip_{match['id']}"):
                    try:
                        from video_engine import create_context_clip
                        with st.spinner("Extracting 10-second full-scene clip..."):
                            st.session_state.context_clip = create_context_clip(match)
                    except Exception as error:
                        st.error(f"Clip extraction failed: {error}")
        if st.session_state.context_clip and Path(st.session_state.context_clip).exists():
            st.subheader("Selected full-scene clip")
            st.video(st.session_state.context_clip)

with stitch_tab:
    st.subheader("Automated camera-to-camera journey")
    st.write("Uses the currently ranked search observations, ordered by their original timestamps.")
    if st.button("Create stitched journey video", type="primary", disabled=not st.session_state.matches):
        try:
            with st.spinner("Extracting and joining 5-second clips…"):
                from video_engine import stitch_reid_clips

                result = stitch_reid_clips(st.session_state.matches, str(OUTPUT_DIR / "stitched_output.mp4"))
            st.session_state.journey = result["journey_log"]
            st.session_state.stitched_video = result["output_path"]
            st.success(f"Created a {result['segments']}-segment master journey.")
        except Exception as error: st.error(f"Stitching failed: {error}")
    if st.session_state.get("stitched_video") and Path(st.session_state.stitched_video).exists():
        st.video(st.session_state.stitched_video)

with log_tab:
    st.subheader("Detailed Journey Log")
    if st.session_state.journey:
        for entry in st.session_state.journey: st.markdown(f"- {entry}")
    else: st.info("Run a search, then create a stitched journey to populate this timeline.")
