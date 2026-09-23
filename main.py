"""FastAPI Web Application Entry Point for AICatch.

Exposes REST routes for Job Management, Search Workspace, Active Queries,
along with SSE streaming routes for live alerts and performance telemetry.
Serves crop thumbnails, clips, and frames securely from disk.
"""

import sqlite3
import logging
import json
import asyncio
import queue
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
import uuid
import shutil
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps
import mimetypes
import uvicorn

from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR, MEDIA_DIR, UPLOAD_DIR, SEARCH_DEBUG
from storage import SQLiteDatabase, VectorDatabase, MediaStorage, SearchLayer
from storage.search import VisualIndexEmptyError
from pipeline import JobManager, GPUScheduler, ActiveQueryMatcher, ResultDispatcher, JourneyEngine
from utils.monitor import PerformanceMonitorInstance

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
LOGGER = logging.getLogger("main")

# ==============================================================================
# 1. Pipeline and Storage Singletons Initialization
# ==============================================================================
sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
media_storage = MediaStorage()
search_layer = SearchLayer(sqlite_db, vector_db, media_storage)
journey_engine = JourneyEngine(sqlite_db, vector_db)

matcher = ActiveQueryMatcher()
dispatcher = ResultDispatcher(sqlite_db, vector_db, matcher)
job_manager = JobManager(sqlite_db)
gpu_scheduler = GPUScheduler(job_manager, media_storage, dispatcher)

# ==============================================================================
# 2. FastAPI Application Setup
# ==============================================================================
app = FastAPI(
    title="AICatch API",
    description="Production CCTV Analytics Backend with Sequential GPU Scheduler",
    version="1.0.0"
)

# Enable CORS for React Frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve cropped entities and video assets from media folder
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

# ==============================================================================
# 3. Application Lifecycle Events
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    # Automatically clean up any stale jobs stuck in indexing/starting status on server startup
    try:
        for job in sqlite_db.get_jobs():
            if job["status"] in ("indexing", "starting"):
                sqlite_db.update_job_status(job["id"], "failed", "Interrupted by server restart")
                LOGGER.info(f"Cleaned up stale job {job['id']} on server startup.")
    except Exception as e:
        LOGGER.error(f"Failed to clean up stale jobs during startup: {e}")

    LOGGER.info("Starting GPU Inference Scheduler thread...")
    gpu_scheduler.start()
    LOGGER.info("AICatch Server ready.")

@app.on_event("shutdown")
async def shutdown_event():
    LOGGER.info("Stopping GPU Scheduler...")
    gpu_scheduler.stop_scheduler()
    if gpu_scheduler.is_alive():
        gpu_scheduler.join(timeout=3.0)

    LOGGER.info("Stopping Result Dispatcher...")
    dispatcher.stop_dispatcher()

    LOGGER.info("Stopping active camera streams...")
    for job_id in list(job_manager.active_jobs.keys()):
        job_manager.cleanup_job_resources(job_id)

    LOGGER.info("Flushing pending media storage writes...")
    media_storage.stop()

    PerformanceMonitorInstance.stop()
    LOGGER.info("AICatch Server offline.")

# ==============================================================================
# 4. Request DTO Schemas
# ==============================================================================
class CameraCreate(BaseModel):
    name: str
    path: str
    is_live: bool = False

class JobCreate(BaseModel):
    name: str
    profile: str  # "FAST" | "BALANCED" | "FORENSIC"
    cameras: List[CameraCreate]
    investigation_type: Optional[str] = "single_video"

class CameraAddIn(BaseModel):
    name: str
    path: Optional[str] = ""
    is_live: Optional[bool] = False

class CameraUpdateIn(BaseModel):
    name: Optional[str] = None
    path: Optional[str] = None

# ==============================================================================
# 5. REST Route Handlers: Job Management
# ==============================================================================
@app.post("/api/jobs")
def create_job(job_in: JobCreate):
    try:
        camera_sources = [
            {"name": cam.name, "path": cam.path, "is_live": cam.is_live}
            for cam in job_in.cameras
        ]
        job_id = job_manager.create_job(
            name=job_in.name,
            profile=job_in.profile,
            camera_sources=camera_sources,
            investigation_type=job_in.investigation_type
        )
        return {"job_id": job_id, "status": "queued"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {e}")

@app.post("/api/jobs/{job_id}/start")
def start_job(job_id: str, background_tasks: BackgroundTasks):
    try:
        # Start job asynchronously in threadpool to avoid API timeout
        background_tasks.add_task(job_manager.start_job, job_id)
        return {"status": "starting", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs/{job_id}/start-all")
def start_all_cameras(job_id: str, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(job_manager.start_all_ready_cameras, job_id)
        return {"status": "starting", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs/{job_id}/cameras/{camera_id}/start")
def start_camera(job_id: str, camera_id: str, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(job_manager.start_camera, job_id, camera_id)
        return {"status": "starting", "job_id": job_id, "camera_id": camera_id}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs/{job_id}/cameras")
def add_job_camera(job_id: str, cam_in: CameraAddIn):
    try:
        job_record = sqlite_db.get_job(job_id)
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found.")

        from pipeline.job_manager import get_file_sha256
        cam_id = f"cam_{uuid.uuid4().hex[:8]}"
        checksum = get_file_sha256(cam_in.path) if cam_in.path and Path(cam_in.path).exists() else None
        sqlite_db.add_camera(
            camera_id=cam_id,
            job_id=job_id,
            name=cam_in.name,
            source_type="live" if cam_in.is_live else "file",
            source_path=cam_in.path or "",
            checksum=checksum
        )
        status = "ready" if cam_in.path and Path(cam_in.path).exists() else "not_configured"
        return {
            "id": cam_id,
            "job_id": job_id,
            "name": cam_in.name,
            "source_path": cam_in.path or "",
            "status": status,
            "progress": 0.0,
            "tracks_count": 0
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/jobs/{job_id}/cameras/{camera_id}")
def update_job_camera(job_id: str, camera_id: str, cam_update: CameraUpdateIn):
    try:
        job_record = sqlite_db.get_job(job_id)
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found.")

        cameras = sqlite_db.get_cameras_by_job(job_id)
        cam = next((c for c in cameras if c["id"] == camera_id), None)
        if not cam:
            raise HTTPException(status_code=404, detail="Camera not found.")

        from pipeline.job_manager import get_file_sha256
        new_path = cam_update.path if cam_update.path is not None else cam["source_path"]
        new_name = cam_update.name if cam_update.name is not None else cam["name"]
        checksum = get_file_sha256(new_path) if new_path and Path(new_path).exists() else None
        sqlite_db.update_camera_source(camera_id, new_path, new_name, checksum)
        status = "ready" if new_path and Path(new_path).exists() else "not_configured"
        return {
            "id": camera_id,
            "job_id": job_id,
            "name": new_name,
            "source_path": new_path,
            "status": status,
            "progress": 0.0
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs/{job_id}/pause")
def pause_job(job_id: str):
    job_manager.pause_job(job_id)
    return {"status": "paused", "job_id": job_id}

@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: str):
    job_manager.resume_job(job_id)
    return {"status": "indexing", "job_id": job_id}

@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job_manager.cancel_job(job_id)
    return {"status": "cancelled", "job_id": job_id}


@app.post("/api/jobs/{job_id}/reindex")
def reindex_job(job_id: str):
    """Clears derived detections/tracks/vectors and re-indexes footage with current upgraded AI models."""
    try:
        job_record = sqlite_db.get_job(job_id)
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found.")

        # 1. Stop any running decoders
        job_manager.cleanup_job_resources(job_id)

        # 2. Clean derived SQLite tracks and detections
        with sqlite_db.connection() as conn:
            conn.execute("DELETE FROM journey_events WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
            conn.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
            conn.execute("UPDATE jobs SET status = 'starting', processed_sec = 0.0, duration_sec = 0.0, fps = 0.0, error_msg = NULL WHERE id = ?", (job_id,))
            conn.execute("UPDATE cameras SET status = 'inactive', resolution = NULL, fps = NULL WHERE job_id = ?", (job_id,))
        # 3. Clean ChromaDB vectors for this job
        try:
            vector_db.delete_by_job(job_id)
        except Exception:
            pass

        # 4. Launch job with new models
        job_manager.start_job(job_id)
        LOGGER.info(f"Re-indexing launched for job {job_id} using upgraded AI models.")
        return {"status": "indexing", "job_id": job_id, "message": "Re-indexing started with upgraded AI models."}
    except Exception as e:
        LOGGER.error(f"Failed to re-index job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/reconcile")
def reconcile_investigations():
    """Scans all investigations, cross-checking SQLite status against ChromaDB counts and filesystem presence."""
    try:
        report = {
            "total_jobs_in_sqlite": 0,
            "healthy_jobs": [],
            "anomalies": [],
            "dangling_chroma_vectors": [],
            "dangling_crop_folders": []
        }

        all_jobs = sqlite_db.get_jobs()
        report["total_jobs_in_sqlite"] = len(all_jobs)
        known_job_ids = set()

        with sqlite_db.connection() as conn:
            conn.row_factory = sqlite3.Row
            for j in all_jobs:
                jid = j["id"]
                known_job_ids.add(jid)
                status = j["status"]

                cur_t = conn.execute("SELECT count(*) as count FROM tracks WHERE job_id = ?", (jid,))
                track_count = cur_t.fetchone()["count"]

                cur_d = conn.execute("SELECT count(*) as count FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (jid,))
                det_count = cur_d.fetchone()["count"]

                # Check ChromaDB counts
                vis_ids = vector_db.visual_coll.get(where={"job_id": jid})["ids"]
                reid_ids = vector_db.reid_coll.get(where={"job_id": jid})["ids"]
                face_ids = vector_db.face_coll.get(where={"job_id": jid})["ids"]

                vis_count = len(vis_ids)
                reid_count = len(reid_ids)
                face_count = len(face_ids)

                # Check crops on disk
                crop_dir = media_storage.get_crop_dir("person") / jid
                has_crops_on_disk = crop_dir.exists() and any(crop_dir.iterdir()) if crop_dir.exists() else False

                # Detect anomalies
                job_anomalies = []
                if status.upper() == "INDEXED" and vis_count == 0:
                    job_anomalies.append("STATUS_MISMATCH: Status is INDEXED but ChromaDB has 0 visual vectors")
                if status.upper() == "INDEXED" and det_count > 0 and vis_count == 0:
                    job_anomalies.append("DRIFT: SQLite has detections but ChromaDB has 0 vectors")
                if status.lower() == "failed" and track_count == 0 and det_count == 0:
                    job_anomalies.append("ORPHANED_FAILED_RUN: Test or pipeline run failed with 0 metadata")
                if status.upper() == "DELETE_FAILED":
                    job_anomalies.append(f"DELETE_FAILED: Investigation deletion previously failed: {j.get('error_msg')}")

                job_info = {
                    "job_id": jid,
                    "name": j["name"],
                    "status": status,
                    "created_at": j["created_at"],
                    "tracks": track_count,
                    "detections": det_count,
                    "chroma_vectors": {"visual": vis_count, "reid": reid_count, "face": face_count},
                    "has_crops_on_disk": has_crops_on_disk
                }

                if job_anomalies:
                    report["anomalies"].append({**job_info, "issues": job_anomalies})
                else:
                    report["healthy_jobs"].append(job_info)

        # Check for dangling ChromaDB vectors not belonging to any known SQLite job
        all_vis_metas = vector_db.visual_coll.get(include=["metadatas"])
        if all_vis_metas and all_vis_metas.get("metadatas"):
            orphaned_chroma_jobs = set()
            for meta in all_vis_metas["metadatas"]:
                cj_id = meta.get("job_id") or meta.get("investigation_id")
                if cj_id and cj_id not in known_job_ids:
                    orphaned_chroma_jobs.add(cj_id)
            report["dangling_chroma_vectors"] = list(orphaned_chroma_jobs)

        # Check for dangling crop folders on disk
        for crop_type in ["person", "vehicle", "face", "plate"]:
            cdir = media_storage.get_crop_dir(crop_type)
            if cdir.exists():
                for p in cdir.iterdir():
                    if p.is_dir() and p.name not in known_job_ids:
                        report["dangling_crop_folders"].append(str(p).replace("\\", "/"))

        return report
    except Exception as e:
        LOGGER.error(f"Failed to reconcile investigations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/reconcile/fix")
def fix_investigation_drift(purge_failed: bool = False, purge_orphaned_storage: bool = True):
    """Repairs mismatched statuses, cleans dangling vectors/media, and optionally purges failed test runs."""
    try:
        actions_taken = []
        all_jobs = sqlite_db.get_jobs()
        known_job_ids = set()

        jobs_to_purge = []
        with sqlite_db.connection() as conn:
            conn.row_factory = sqlite3.Row
            for j in all_jobs:
                jid = j["id"]
                known_job_ids.add(jid)
                status = j["status"]

                # Check ChromaDB visual vectors
                vis_ids = vector_db.visual_coll.get(where={"job_id": jid})["ids"]
                vis_count = len(vis_ids)

                # Fix false INDEXED status
                if status.upper() == "INDEXED" and vis_count == 0:
                    conn.execute(
                        "UPDATE jobs SET status = 'failed', error_msg = 'Reconciled: No visual vectors found in ChromaDB' WHERE id = ?",
                        (jid,)
                    )
                    actions_taken.append(f"Fixed status of {jid} from INDEXED to failed (0 Chroma vectors).")

                # Optionally check for failed zero-data runs
                if purge_failed and status.lower() == "failed":
                    cur_t = conn.execute("SELECT count(*) as count FROM tracks WHERE job_id = ?", (jid,))
                    if cur_t.fetchone()["count"] == 0:
                        jobs_to_purge.append(jid)

        if purge_failed and jobs_to_purge:
            for jid in jobs_to_purge:
                try:
                    delete_job(jid)
                    actions_taken.append(f"Purged failed empty test job {jid}.")
                except Exception as p_err:
                    actions_taken.append(f"Failed to purge {jid}: {p_err}")


        # Purge dangling Chroma vectors
        if purge_orphaned_storage:
            all_vis_metas = vector_db.visual_coll.get(include=["metadatas"])
            if all_vis_metas and all_vis_metas.get("metadatas"):
                for meta in all_vis_metas["metadatas"]:
                    cj_id = meta.get("job_id") or meta.get("investigation_id")
                    if cj_id and cj_id not in known_job_ids:
                        vector_db.delete_by_job(cj_id)
                        actions_taken.append(f"Deleted dangling vectors for unknown job {cj_id}.")

            # Purge dangling crop folders
            for crop_type in ["person", "vehicle", "face", "plate"]:
                cdir = media_storage.get_crop_dir(crop_type)
                if cdir.exists():
                    for p in list(cdir.iterdir()):
                        if p.is_dir() and p.name not in known_job_ids:
                            shutil.rmtree(p)
                            actions_taken.append(f"Deleted dangling disk crop folder {p.name}.")

        return {"status": "reconciled", "actions_count": len(actions_taken), "actions": actions_taken}
    except Exception as e:
        LOGGER.error(f"Failed to fix investigation drift: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    """Atomically cascades investigation deletion across SQLite, ChromaDB, and disk media."""
    job_record = sqlite_db.get_job(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    try:
        LOGGER.info(f"Permanently deleting investigation {job_id}...")
        # 1. Stop job decoders and threads if active
        with job_manager.lock:
            is_active = job_id in job_manager.active_jobs
        if is_active:
            try:
                job_manager.cleanup_job_resources(job_id)
            except Exception as je:
                LOGGER.warning(f"Cleanup job resources warning for {job_id}: {je}")
            with job_manager.lock:
                job_manager.active_jobs.pop(job_id, None)

        if hasattr(dispatcher, "job_stats") and job_id in dispatcher.job_stats:
            try:
                dispatcher.job_stats.pop(job_id, None)
            except Exception:
                pass

        # 2. Delete vectors from Vector DB across visual, reid, face, scene, and region collections
        try:
            vector_db.delete_by_job(job_id)
        except Exception as ve:
            LOGGER.warning(f"Vector DB cleanup warning for {job_id}: {ve}")

        # 3. Delete media folders from disk (crops, thumbnails, clips, keyframes, regions) safely
        for crop_type in ["person", "vehicle", "face", "plate"]:
            try:
                crop_dir = media_storage.get_crop_dir(crop_type)
                job_crop_path = crop_dir / job_id
                if job_crop_path.exists():
                    shutil.rmtree(job_crop_path, ignore_errors=True)
            except Exception as me:
                LOGGER.warning(f"Failed to delete crop dir for {crop_type}: {me}")

        from config import THUMBNAILS_DIR, CLIPS_DIR, KEYFRAMES_DIR, REGIONS_DIR
        for folder in [THUMBNAILS_DIR / job_id, CLIPS_DIR / job_id, KEYFRAMES_DIR / job_id, REGIONS_DIR / job_id]:
            try:
                if folder.exists():
                    shutil.rmtree(folder, ignore_errors=True)
            except Exception as fe:
                LOGGER.warning(f"Failed to delete media folder {folder}: {fe}")

        # 4. Atomic cascade delete across all SQLite tables
        with sqlite_db.connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")

            # Delete evidence items & folders
            conn.execute("DELETE FROM notes WHERE parent_type = 'job' AND parent_id = ?", (job_id,))
            conn.execute("DELETE FROM notes WHERE parent_type = 'folder' AND parent_id IN (SELECT id FROM folders WHERE job_id = ?)", (job_id,))
            conn.execute("DELETE FROM notes WHERE parent_type = 'track' AND parent_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
            conn.execute("DELETE FROM folder_items WHERE folder_id IN (SELECT id FROM folders WHERE job_id = ?)", (job_id,))
            conn.execute("DELETE FROM folder_items WHERE item_type = 'track' AND item_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
            conn.execute("DELETE FROM folder_items WHERE item_type = 'detection' AND item_id IN (SELECT d.id FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?)", (job_id,))
            conn.execute("DELETE FROM folders WHERE job_id = ?", (job_id,))

            # Delete saved persons & active query alerts
            conn.execute("DELETE FROM saved_persons WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM active_queries WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM journey_events WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM perf_stats WHERE job_id = ?", (job_id,))

            # Delete detections, tracks, frames, cameras
            conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
            conn.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM frames WHERE camera_id IN (SELECT id FROM cameras WHERE job_id = ?)", (job_id,))
            conn.execute("DELETE FROM cameras WHERE job_id = ?", (job_id,))

            # Delete primary job record
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

        LOGGER.info(f"Investigation {job_id} deleted successfully.")
        return {"job_id": job_id, "status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Failed to delete job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_video_file(file: UploadFile = File(...)):
    """Receives a raw video file from client laptop and saves it in UPLOAD_DIR for analysis."""
    try:
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (".", "_", "-")).strip()
        if not safe_filename:
            safe_filename = f"upload_{int(time.time())}.mp4"

        save_path = UPLOAD_DIR / safe_filename
        LOGGER.info(f"Saving uploaded video file to: {save_path}")

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"filename": safe_filename, "path": str(save_path).replace("\\", "/")}
    except Exception as e:
        LOGGER.error(f"Failed to save video upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save video upload: {e}")

@app.get("/api/videos")
def list_available_videos():
    """Returns list of existing video files available in media/uploads."""
    videos = []
    seen = set()
    for directory in [UPLOAD_DIR, MEDIA_DIR]:
        if directory.exists():
            for f in sorted(directory.glob("*.*")):
                if f.is_file() and f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"] and f.name not in seen:
                    seen.add(f.name)
                    videos.append({
                        "filename": f.name,
                        "path": str(f).replace("\\", "/"),
                        "size_bytes": f.stat().st_size
                    })
    return {"videos": sorted(videos, key=lambda x: x["filename"])}

@app.api_route("/api/video/stream", methods=["GET", "HEAD"])
async def stream_video(request: Request, path: str = Query(...)):
    """Streams video from local storage with HTTP Range support for HTML5 video playback and timeline seeking."""
    try:
        clean_path = path.replace("\\", "/").strip()

        video_path = Path(clean_path)

        # If relative or filename only, check inside MEDIA_DIR / uploads / clips
        if not video_path.is_absolute() or not video_path.exists():
            if (MEDIA_DIR / clean_path).exists():
                video_path = MEDIA_DIR / clean_path
            elif (UPLOAD_DIR / clean_path).exists():
                video_path = UPLOAD_DIR / clean_path
            elif (UPLOAD_DIR / Path(clean_path).name).exists():
                video_path = UPLOAD_DIR / Path(clean_path).name
            elif (MEDIA_DIR / Path(clean_path).name).exists():
                video_path = MEDIA_DIR / Path(clean_path).name

        if not video_path.exists() or not video_path.is_file():
            LOGGER.warning(f"Video stream request file not found: {path} (resolved: {video_path})")
            raise HTTPException(status_code=404, detail=f"Video file not found at: {path}")

        file_size = video_path.stat().st_size
        content_type, _ = mimetypes.guess_type(str(video_path))
        if not content_type:
            content_type = "video/mp4"

        # Handle HTTP 206 Partial Content Range requests for video seeking
        range_header = request.headers.get("range")
        if range_header:
            try:
                byte_range = range_header.replace("bytes=", "").split("-")
                start = int(byte_range[0]) if byte_range[0] else 0
                end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
                start = max(0, min(start, file_size - 1))
                end = max(start, min(end, file_size - 1))
                chunk_size = (end - start) + 1

                def iterfile():
                    with open(video_path, "rb") as f:
                        f.seek(start)
                        remaining = chunk_size
                        while remaining > 0:
                            read_size = min(128 * 1024, remaining)
                            data = f.read(read_size)
                            if not data:
                                break
                            remaining -= len(data)
                            yield data

                headers = {
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(chunk_size),
                    "Content-Type": content_type,
                }
                return StreamingResponse(iterfile(), status_code=206, headers=headers)
            except Exception as range_err:
                LOGGER.warning(f"Error handling Range header for {video_path}: {range_err}")

        # Full file response if no Range requested
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }
        return FileResponse(path=str(video_path), media_type=content_type, headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Failed to stream video {path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs")
def list_jobs():
    jobs = sqlite_db.get_jobs()
    enriched = []
    for j in jobs:
        cams = sqlite_db.get_cameras_by_job(j["id"])
        source_names = ", ".join(c["name"] for c in cams) if cams else "No source"
        enriched.append({**j, "sources": source_names})
    return enriched

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    job_record = sqlite_db.get_job(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found.")
    progress_info = job_manager.get_job_progress(job_id)
    details = {**job_record, **progress_info}
    if str(details.get("status", "")).lower() in ["completed", "indexed"]:
        details["progress"] = 1.0
        details["status"] = "completed"
    return details

@app.get("/api/jobs/{job_id}/diagnostics")
def get_job_diagnostics(job_id: str):
    job_record = sqlite_db.get_job(job_id)
    if not job_record:
        raise HTTPException(status_code=404, detail="Job not found.")

    diag_str = job_record.get("diagnostics")
    parsed_diag = {}
    if diag_str:
        try:
            parsed_diag = json.loads(diag_str)
        except Exception:
            parsed_diag = {"raw": diag_str}

    stats = {}
    if hasattr(dispatcher, "job_stats") and job_id in dispatcher.job_stats:
        stats = dispatcher.job_stats[job_id]

    with sqlite_db.connection() as conn:
        conn.row_factory = sqlite3.Row
        cur_t = conn.execute("SELECT count(*) as count FROM tracks WHERE job_id = ?", (job_id,))
        track_count = cur_t.fetchone()["count"]
        cur_d = conn.execute("SELECT count(*) as count FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (job_id,))
        det_count = cur_d.fetchone()["count"]

    vector_count = 0
    try:
        vis_res = vector_db.visual_coll.get(where={"job_id": job_id})
        vector_count = len(vis_res.get("ids", []))
    except Exception:
        pass

    return {
        "job_id": job_id,
        "status": job_record["status"],
        "profile": job_record.get("profile"),
        "tracks_count": track_count,
        "detections_count": det_count,
        "vectors_count": vector_count,
        "stored_diagnostics": parsed_diag,
        "runtime_stats": stats
    }

@app.get("/api/jobs/{job_id}/cameras")
def get_job_cameras(job_id: str):
    cameras = sqlite_db.get_cameras_by_job(job_id)
    job = job_manager.active_jobs.get(job_id)
    track_counts = {}
    try:
        with sqlite_db.connection() as conn:
            cur = conn.execute("SELECT camera_id, COUNT(*) FROM tracks WHERE job_id = ? GROUP BY camera_id", (job_id,))
            track_counts = dict(cur.fetchall())
    except Exception:
        pass

    for cam in cameras:
        cid = cam["id"]
        cam["tracks_count"] = track_counts.get(cid, 0)
        if job and cid in job.decoders:
            dec = job.decoders[cid]
            if dec.is_alive():
                cam["status"] = "indexing"
            elif getattr(dec, "has_finished", False):
                cam["status"] = "completed"
            elif getattr(dec, "failed", False):
                cam["status"] = "failed"
            fps = dec.target_fps or 25.0
            qsize = dec.frame_queue.qsize()
            processed = max(0, dec.frames_decoded - qsize) / fps
            dur = dec.duration_sec or 1.0
            cam["processed_sec"] = round(processed, 2)
            cam["duration_sec"] = round(dec.duration_sec, 2)
            cam["progress"] = 1.0 if cam["status"] == "completed" else min(0.99, processed / dur if dur > 0 else 0.0)
        else:
            raw_status = str(cam.get("status", "")).lower()
            if raw_status == "completed":
                cam["status"] = "completed"
                cam["progress"] = 1.0
            elif raw_status == "failed":
                cam["status"] = "failed"
                cam["progress"] = 0.0
            elif raw_status == "not_configured" or not cam.get("source_path") or not Path(cam.get("source_path", "")).exists():
                cam["status"] = "not_configured"
                cam["progress"] = 0.0
            else:
                cam["status"] = "ready"
                cam["progress"] = 0.0
    return cameras

@app.get("/api/jobs/{job_id}/tracks")
def get_job_tracks(job_id: str):
    with sqlite_db.connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM tracks WHERE job_id = ? ORDER BY start_time DESC", (job_id,))
        return [dict(row) for row in cursor.fetchall()]

@app.get("/api/tracks/{track_uuid}/detections")
def get_track_detections(track_uuid: str):
    return sqlite_db.get_detections_by_track(track_uuid)

# ==============================================================================
# 6. REST Route Handlers: Forensic Unified Search Layer
# ==============================================================================
@app.post("/api/search")
async def execute_search(
    qtype: str = Form(...),  # "text" | "image" | "face" | "plate" | "saved_person"
    query_text: Optional[str] = Form(None),
    query_file: Optional[UploadFile] = File(None),
    saved_person_id: Optional[str] = Form(None),
    qcategory: Optional[str] = Form(None),  # "person" | "object" | "scene" | "region" | None
    camera_id: Optional[str] = Form(None),
    object_type: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None),
    scope: str = Form("current_investigation"),  # "current_investigation" | "current_video" | "global"
    top_k: int = Form(20),
    strict_gate: bool = Form(True)
):
    filters = {}
    if scope == "current_investigation":
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required for investigation-scoped search.")
        filters["job_id"] = job_id
        if camera_id:
            filters["camera_id"] = camera_id
    elif scope == "current_video":
        if not job_id or not camera_id:
            raise HTTPException(status_code=400, detail="Both job_id and camera_id are required for video-scoped search.")
        filters["job_id"] = job_id
        filters["camera_id"] = camera_id
    elif scope == "global":
        pass  # No job_id/camera_id filters applied to ChromaDB visual/face query
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported search scope: {scope}")
    if object_type:
        normalized_type = object_type.lower().strip()
        if normalized_type not in {"person", "vehicle"}:
            raise HTTPException(status_code=400, detail="Unsupported object_type filter.")
        filters["entity_type"] = normalized_type
    if not strict_gate and not SEARCH_DEBUG:
        raise HTTPException(status_code=403, detail="Diagnostic gate bypass requires AICATCH_SEARCH_DEBUG=true.")

    qtype = qtype.lower().strip()

    try:
        matches = []
        # 1. Text Query Search (multimodal CLIP text search)
        if qtype == "text" and query_text:
            emb = gpu_scheduler.embed_text(query_text)
            matches = search_layer.search_by_text(emb, filters, top_k, query_text=query_text)

        # 2. Image Query Search (CLIP visual matching with intent-aware routing & face fallback)
        elif qtype in ["image", "face"] and query_file:
            img = ImageOps.exif_transpose(Image.open(query_file.file)).convert("RGB")
            if img.width <= 0 or img.height <= 0:
                raise HTTPException(status_code=400, detail="The uploaded image has invalid dimensions.")

            # Validate explicit query intent category if provided
            normalized_qcat = None
            if qcategory:
                normalized_qcat = qcategory.strip().lower()
                if normalized_qcat not in ["person", "object", "scene", "region"]:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid qcategory '{qcategory}'. Must be one of: person, object, scene, region."
                    )

            entity_type = None
            query_crop = img
            emb_reid = None
            emb_face = None
            face_status = "NO_FACE"
            intent_source = "explicit_intent" if normalized_qcat else "automatic_classification"

            if normalized_qcat == "person":
                # Explicit Person Intent: YOLO refinement for person box, extract CLIP + Re-ID + Face
                entity_type = "person"
                try:
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    with gpu_scheduler.gpu_lock:
                        yolo_results = gpu_scheduler.yolo.predict(cv_img, verbose=False, conf=0.30)
                    if yolo_results and len(yolo_results[0].boxes) > 0:
                        person_boxes = [b for b in yolo_results[0].boxes if int(b.cls[0]) == 0]
                        if person_boxes:
                            best_box = max(person_boxes, key=lambda b: float(b.conf[0]))
                            bx1, by1, bx2, by2 = [float(v) for v in best_box.xyxy[0]]
                            box_area = (bx2 - bx1) * (by2 - by1)
                            img_area = float(img.width * img.height)
                            if box_area < 0.80 * img_area:
                                w, h = img.width, img.height
                                pad_x = (bx2 - bx1) * 0.05
                                pad_y = (by2 - by1) * 0.05
                                cx1 = max(0, int(bx1 - pad_x))
                                cy1 = max(0, int(by1 - pad_y))
                                cx2 = min(w, int(bx2 + pad_x))
                                cy2 = min(h, int(by2 + pad_y))
                                if (cx2 - cx1) >= 32 and (cy2 - cy1) >= 32:
                                    query_crop = img.crop((cx1, cy1, cx2, cy2))
                except Exception as yolo_err:
                    LOGGER.debug(f"Person query crop refinement note: {yolo_err}")
                    query_crop = img

                emb = gpu_scheduler.embed_image(query_crop)
                try:
                    emb_reid = gpu_scheduler.embed_reid(query_crop)
                except Exception as re_err:
                    LOGGER.debug(f"Query Re-ID extraction note: {re_err}")
                    emb_reid = None

                # Attempt face detection on query image for person search
                try:
                    await query_file.seek(0)
                    face_img = Image.open(query_file.file)
                    res_face = gpu_scheduler.embed_face_with_diagnostics(face_img)
                    emb_face = res_face["embedding"]
                    face_status = res_face["face_status"]
                except Exception as face_err:
                    LOGGER.debug(f"Internal face detection failure: {face_err}")
                    emb_face = None
                    face_status = "NO_FACE"

            elif normalized_qcat == "object":
                # Explicit Object Intent: Treat as non-person object/vehicle; suppress face and Re-ID extraction
                entity_type = "vehicle"
                try:
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    with gpu_scheduler.gpu_lock:
                        yolo_results = gpu_scheduler.yolo.predict(cv_img, verbose=False, conf=0.30)
                    if yolo_results and len(yolo_results[0].boxes) > 0:
                        veh_boxes = [b for b in yolo_results[0].boxes if int(b.cls[0]) in [2, 3, 5, 7]]
                        if veh_boxes:
                            best_box = max(veh_boxes, key=lambda b: float(b.conf[0]))
                            bx1, by1, bx2, by2 = [float(v) for v in best_box.xyxy[0]]
                            box_area = (bx2 - bx1) * (by2 - by1)
                            img_area = float(img.width * img.height)
                            if box_area < 0.80 * img_area:
                                w, h = img.width, img.height
                                pad_x = (bx2 - bx1) * 0.05
                                pad_y = (by2 - by1) * 0.05
                                cx1 = max(0, int(bx1 - pad_x))
                                cy1 = max(0, int(by1 - pad_y))
                                cx2 = min(w, int(bx2 + pad_x))
                                cy2 = min(h, int(by2 + pad_y))
                                if (cx2 - cx1) >= 32 and (cy2 - cy1) >= 32:
                                    query_crop = img.crop((cx1, cy1, cx2, cy2))
                except Exception as yolo_err:
                    LOGGER.debug(f"Object query crop refinement note: {yolo_err}")
                    query_crop = img

                emb = gpu_scheduler.embed_image(query_crop)
                # Explicitly suppress face and Re-ID extraction for object queries to prevent misrouting
                emb_reid = None
                emb_face = None
                face_status = "NO_FACE"

            elif normalized_qcat in ["scene", "region"]:
                # Scene or Region search: use full frame or spatial region crop without entity filtering
                entity_type = None
                query_crop = img
                emb = gpu_scheduler.embed_image(query_crop)
                emb_reid = None
                emb_face = None
                face_status = "NO_FACE"

            else:
                # Backward-compatible auto-classification branch
                try:
                    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    with gpu_scheduler.gpu_lock:
                        yolo_results = gpu_scheduler.yolo.predict(cv_img, verbose=False, conf=0.30)
                    if yolo_results and len(yolo_results[0].boxes) > 0:
                        target_boxes = [b for b in yolo_results[0].boxes if int(b.cls[0]) in [0, 2, 3, 5, 7]]
                        if target_boxes:
                            best_box = max(target_boxes, key=lambda b: float(b.conf[0]))
                            cls_idx = int(best_box.cls[0])
                            if cls_idx == 0:
                                entity_type = "person"
                            elif cls_idx in [2, 3, 5, 7]:
                                entity_type = "vehicle"

                            bx1, by1, bx2, by2 = [float(v) for v in best_box.xyxy[0]]
                            box_area = (bx2 - bx1) * (by2 - by1)
                            img_area = float(img.width * img.height)
                            if box_area < 0.80 * img_area:
                                w, h = img.width, img.height
                                pad_x = (bx2 - bx1) * 0.05
                                pad_y = (by2 - by1) * 0.05
                                cx1 = max(0, int(bx1 - pad_x))
                                cy1 = max(0, int(by1 - pad_y))
                                cx2 = min(w, int(bx2 + pad_x))
                                cy2 = min(h, int(by2 + pad_y))
                                if (cx2 - cx1) >= 32 and (cy2 - cy1) >= 32:
                                    query_crop = img.crop((cx1, cy1, cx2, cy2))
                except Exception as yolo_err:
                    LOGGER.debug(f"Failed to detect query image entity type: {yolo_err}")
                    query_crop = img

                emb = gpu_scheduler.embed_image(query_crop)

                # If person entity, extract Re-ID embedding from the query crop
                if entity_type == "person":
                    try:
                        emb_reid = gpu_scheduler.embed_reid(query_crop)
                    except Exception as re_err:
                        LOGGER.debug(f"Query Re-ID extraction note: {re_err}")
                        emb_reid = None

                # Attempt face detection internally on query image
                try:
                    await query_file.seek(0)
                    face_img = Image.open(query_file.file)
                    res_face = gpu_scheduler.embed_face_with_diagnostics(face_img)
                    emb_face = res_face["embedding"]
                    face_status = res_face["face_status"]
                except Exception as face_err:
                    LOGGER.debug(f"Internal face detection failure: {face_err}")
                    emb_face = None
                    face_status = "NO_FACE"

            matches = search_layer.search_by_image(
                emb, filters, top_k,
                entity_type=entity_type,
                face_embedding=emb_face,
                face_status=face_status,
                strict_gate=strict_gate,
                reid_embedding=emb_reid,
                enable_rrf=True if emb_reid is not None else False
            )

            # Inject provenance and diagnostic labels
            effective_intent = normalized_qcat or (entity_type if entity_type else "scene")
            for match in matches:
                match["query_intent"] = effective_intent
                match["intent_source"] = intent_source
                if match.get("match_type") == "face_assisted":
                    match["diagnostic_label"] = "Face signal used"
                elif match.get("match_type") == "scene":
                    match["diagnostic_label"] = "Scene keyframe match"
                elif match.get("match_type") == "region":
                    match["diagnostic_label"] = "Region tile match"
                else:
                    match["diagnostic_label"] = "Visual similarity match"

        # 4. Plate OCR Query Search (SQLite text pattern lookup)
        elif qtype == "plate" and query_text:
            p_job_id = filters.get("job_id")
            p_cam_id = filters.get("camera_id")
            matches = search_layer.search_by_plate(query_text, p_cam_id, p_job_id)

        # 5. Saved Person re-ID Search (Face collection query)
        elif qtype == "saved_person" and saved_person_id:
            matches = search_layer.search_saved_person(saved_person_id, filters, top_k)

        else:
            raise HTTPException(status_code=400, detail=f"Malformed query payload. Type: {qtype}")

        # Enrich matches with display labels: Case/Job Name and Camera Name
        enriched_matches = []
        with sqlite_db.connection() as conn:
            conn.row_factory = sqlite3.Row

            job_names = {}
            cam_names = {}

            for match in matches:
                m_job_id = match.get("job_id")
                m_cam_id = match.get("camera_id")

                if m_job_id and m_job_id not in job_names:
                    cur = conn.execute("SELECT name FROM jobs WHERE id = ?", (m_job_id,))
                    row = cur.fetchone()
                    job_names[m_job_id] = row[0] if row else "Unknown Case"

                if m_cam_id and m_cam_id not in cam_names:
                    cur = conn.execute("SELECT name FROM cameras WHERE id = ?", (m_cam_id,))
                    row = cur.fetchone()
                    cam_names[m_cam_id] = row[0] if row else "Unknown Camera"

                match["job_name"] = job_names.get(m_job_id, "Unknown Case")
                match["camera_name"] = cam_names.get(m_cam_id, "Unknown Camera")
                enriched_matches.append(match)

        if SEARCH_DEBUG and qtype == "image":
            LOGGER.info("IMAGE_SEARCH_TRACE_FINAL %s", getattr(search_layer, "last_image_search_diagnostics", {}))
        return enriched_matches

    except VisualIndexEmptyError as e:
        LOGGER.info("IMAGE_SEARCH_INDEX_EMPTY %s", e.diagnostics)
        raise HTTPException(status_code=409, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Search query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"SYSTEM_ERROR: Vector search engine failure: {e}")

# ==============================================================================
# 7. REST Route Handlers: Saved Persons Profile Manager
# ==============================================================================
@app.post("/api/saved-persons")
async def save_person_profile(
    name: str = Form(...),
    face_image: UploadFile = File(...),
    job_id: Optional[str] = Form(None)
):
    try:
        # 1. Load image
        img = Image.open(face_image.file)

        # 2. Extract facial embedding using MTCNN + FaceNet (GPU safe)
        emb = gpu_scheduler.embed_face(img)
        if not emb:
            raise HTTPException(status_code=400, detail="Could not detect face in profile image.")

        # 3. Save profile image to media crops folder
        person_id = f"sp_{uuid.uuid4().hex[:12]}"

        # Convert PIL to openCV numpy BGR for save_crop
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        saved_path = media_storage.save_crop(
            cv_img, "face", "saved_profiles", "profile", person_id, 0
        )

        # 4. Save face embedding vector to ChromaDB
        face_emb_id = f"vector_{person_id}"
        vector_db.add_face_embeddings(
            ids=[face_emb_id],
            embeddings=[emb],
            metadatas=[{"person_id": person_id, "name": name, "job_id": job_id or ""}]
        )

        # 5. Save metadata profile to SQLite
        sqlite_db.add_saved_person(person_id, name, str(saved_path), face_emb_id, job_id)

        return {"person_id": person_id, "name": name, "status": "saved"}

    except Exception as e:
        LOGGER.error(f"Failed to save profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/saved-persons")
def list_saved_persons(job_id: Optional[str] = None):
    return sqlite_db.get_saved_persons(job_id)

@app.delete("/api/saved-persons/{person_id}")
def delete_saved_person(person_id: str):
    """Deletes a saved person profile, including database records, facial vector embeddings, and disk images."""
    try:
        person = sqlite_db.delete_saved_person(person_id)
        if not person:
            raise HTTPException(status_code=404, detail=f"Saved person {person_id} not found.")

        # Remove vector from ChromaDB
        try:
            vector_db.face_coll.delete(where={"person_id": person_id})
        except Exception:
            pass
        if person.get("face_embedding_id"):
            try:
                vector_db.delete_face_embeddings([person["face_embedding_id"]])
            except Exception:
                pass

        # Remove face image from disk
        face_path_str = person.get("face_image_path")
        if face_path_str:
            try:
                fp = Path(face_path_str)
                if fp.exists():
                    fp.unlink(missing_ok=True)
            except Exception as fe:
                LOGGER.warning(f"Could not delete face image file {face_path_str}: {fe}")

        LOGGER.info(f"Saved person {person_id} deleted successfully.")
        return {"person_id": person_id, "status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Failed to delete saved person {person_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# 8. REST Route Handlers: Journey Reconstruction Engine
# ==============================================================================
@app.get("/api/jobs/{job_id}/journey/{track_uuid}")
def get_track_journey(job_id: str, track_uuid: str, similarity: float = 0.70):
    try:
        return journey_engine.reconstruct_journey(
            target_track_uuid=track_uuid,
            similarity_threshold=similarity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# 9. REST Route Handlers: Active Query (Alert Rule) Coordinator
# ==============================================================================
@app.post("/api/active-queries")
async def register_alert_rule(
    session_id: str = Form(...),
    qtype: str = Form(...),  # "text" | "image" | "face" | "plate"
    query_text: Optional[str] = Form(None),
    query_file: Optional[UploadFile] = File(None),
    threshold: float = Form(0.70),
    job_id: Optional[str] = Form(None)
):
    query_id = f"aq_{uuid.uuid4().hex[:12]}"
    query_emb = None
    query_image_path = None

    try:
        # Load and embed visual rules
        if qtype in ["text"] and query_text:
            query_emb = gpu_scheduler.embed_text(query_text)

        elif qtype in ["image"] and query_file:
            img = Image.open(query_file.file)
            query_emb = gpu_scheduler.embed_image(img)
            # Save file to media folder
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            saved_path = media_storage.save_crop(
                cv_img, "person", "active_rules", "queries", query_id, 0
            )
            query_image_path = str(saved_path)

        elif qtype in ["face"] and query_file:
            img = Image.open(query_file.file)
            query_emb = gpu_scheduler.embed_face(img)
            if not query_emb:
                raise HTTPException(status_code=400, detail="No face detected in query file.")
            cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            saved_path = media_storage.save_crop(
                cv_img, "face", "active_rules", "queries", query_id, 0
            )
            query_image_path = str(saved_path)

        elif qtype == "plate" and query_text:
            pass # Plate matching runs string evaluation, no vector needed

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported active query configuration: {qtype}")

        # Register in SQLite active query registry
        sqlite_db.add_active_query(
            query_id=query_id,
            session_id=session_id,
            qtype=qtype,
            query_text=query_text,
            query_image_path=query_image_path,
            query_embedding_id=f"vector_{query_id}" if query_emb else None,
            threshold=threshold,
            job_id=job_id
        )

        # Register rule inside matcher engine
        matcher.register_query(
            query_id=query_id,
            session_id=session_id,
            qtype=qtype,
            query_text=query_text,
            query_embedding=query_emb,
            threshold=threshold,
            job_id=job_id
        )

        return {"query_id": query_id, "status": "active"}

    except Exception as e:
        LOGGER.error(f"Failed to register active query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/active-queries/{query_id}")
def delete_alert_rule(query_id: str):
    """Deletes/removes an active alert query rule from matcher and SQLite."""
    try:
        matcher.remove_query(query_id)
        sqlite_db.delete_active_query(query_id)
        return {"query_id": query_id, "status": "deleted"}
    except Exception as e:
        LOGGER.error(f"Failed to delete active query {query_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/active-queries")
def list_alert_rules(job_id: Optional[str] = None):
    return sqlite_db.get_active_queries(job_id)

# ==============================================================================
# 10. Server-Sent Events (SSE) Live Telemetry and Notification Channels
# ==============================================================================
@app.get("/api/stream/alerts")
async def stream_live_alerts(request: Request, session_id: Optional[str] = None):
    """Native SSE endpoint streaming real-time triggered alert notifications to connected clients."""
    alert_queue = queue.Queue(maxsize=200)

    # Listener callback
    def alert_listener(alert_payload: Dict[str, Any]) -> None:
        # If session filter specified, only push alerts matching that session
        if session_id and alert_payload.get("session_id") != session_id:
            return
        try:
            alert_queue.put_nowait(alert_payload)
        except queue.Full:
            pass

    dispatcher.add_alert_listener(alert_listener)

    async def sse_generator():
        try:
            # Yield initial connect event
            yield f"event: connect\ndata: Connected to AICatch Alert Stream\n\n"
            while True:
                if await request.is_disconnected():
                    LOGGER.info("Alert SSE client disconnected.")
                    break

                try:
                    # Non-blocking fetch from queue
                    alert = alert_queue.get_nowait()
                    yield f"event: alert\ndata: {json.dumps(alert)}\n\n"
                except queue.Empty:
                    # Periodic keep-alive ping (required by browser SSE)
                    yield "event: ping\ndata: keep-alive\n\n"

                await asyncio.sleep(0.5)

        finally:
            dispatcher.remove_alert_listener(alert_listener)

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.get("/api/stream/metrics/{job_id}")
async def stream_perf_metrics(request: Request, job_id: str):
    """Native SSE endpoint streaming rolling performance specs (FPS, VRAM, RAM) and indexing progress."""
    async def sse_generator():
        try:
            yield f"event: connect\ndata: Connected to Job Metrics Stream {job_id}\n\n"
            while True:
                if await request.is_disconnected():
                    LOGGER.info(f"Metrics SSE client disconnected for job {job_id}.")
                    break

                # 1. Fetch hardware telemetry metrics
                telemetry = PerformanceMonitorInstance.get_metrics()

                # 2. Fetch current job progress percentage
                progress_info = job_manager.get_job_progress(job_id)

                # Combine payload
                payload = {
                    "timestamp": time.time(),
                    "progress": progress_info.get("progress", 0.0),
                    "status": progress_info.get("status", "unknown"),
                    "hardware": telemetry
                }

                # Save telemetry stats periodically to DB (persistence path)
                sqlite_db.add_perf_stat(
                    job_id=job_id,
                    fps=telemetry["processing_fps"],
                    q_depth=telemetry["queue_depth"],
                    cpu=telemetry["cpu_util"],
                    gpu=telemetry["gpu_util"],
                    vram=telemetry["vram_used_mb"],
                    yolo_l=telemetry["latency_yolo"],
                    clip_l=telemetry["latency_clip"],
                    ocr_l=telemetry["latency_ocr"],
                    face_l=telemetry["latency_face"],
                    db_l=telemetry["latency_db"]
                )

                yield f"event: metrics\ndata: {json.dumps(payload)}\n\n"
                await asyncio.sleep(1.0)

        except Exception as e:
            LOGGER.error(f"Error in metrics generator: {e}")

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

# Helper to generate unique uuid
import uuid
def uuid_hex():
    return uuid.uuid4().hex[:12]

# ==============================================================================
# 11. Folder, Notes, Labels REST Endpoint Handlers
# ==============================================================================
class FolderCreate(BaseModel):
    name: str
    description: Optional[str] = None
    job_id: str

class FolderItemCreate(BaseModel):
    item_type: str  # 'track' | 'detection' | 'clip' | 'note'
    item_id: str
    metadata: Optional[str] = None

class NoteCreate(BaseModel):
    parent_type: str  # 'folder' | 'track'
    parent_id: str
    content: str

class TrackLabelUpdate(BaseModel):
    label: str

@app.post("/api/folders")
def create_folder(folder: FolderCreate):
    fid = f"fold_{uuid.uuid4().hex[:12]}"
    try:
        sqlite_db.create_folder(fid, folder.name, folder.description, folder.job_id)
        return {"id": fid, "name": folder.name, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/folders")
def list_folders(job_id: Optional[str] = None):
    try:
        if not job_id:
            raise HTTPException(status_code=400, detail="job_id is required to list folders.")
        return sqlite_db.get_folders(job_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/folders/{folder_id}")
def delete_folder(folder_id: str):
    try:
        sqlite_db.delete_folder(folder_id)
        return {"status": "deleted", "id": folder_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/folders/{folder_id}/items")
def add_folder_item(folder_id: str, item: FolderItemCreate):
    # Enforce strict cross-investigation validation check
    with sqlite_db.connection() as conn:
        conn.row_factory = sqlite3.Row

        # 1. Fetch folder to find its job_id
        f_row = conn.execute("SELECT job_id FROM folders WHERE id = ?", (folder_id,)).fetchone()
        if not f_row:
            raise HTTPException(status_code=404, detail="Folder not found.")
        folder_job_id = f_row["job_id"]

        # 2. Check the item's job_id
        item_job_id = None
        if item.item_type == "track":
            t_row = conn.execute("SELECT job_id FROM tracks WHERE id = ?", (item.item_id,)).fetchone()
            if t_row:
                item_job_id = t_row["job_id"]
        elif item.item_type == "detection":
            d_row = conn.execute("SELECT t.job_id FROM detections d JOIN tracks t ON d.track_id = t.id WHERE d.id = ?", (item.item_id,)).fetchone()
            if d_row:
                item_job_id = d_row["job_id"]

        # 3. Reject if they don't match (unless one is NULL/unassigned legacy, in which case we permit to bootstrap)
        if item_job_id and folder_job_id and item_job_id != folder_job_id:
            raise HTTPException(status_code=403, detail="Forbidden: Cannot add items from a different investigation.")

    iid = f"item_{uuid.uuid4().hex[:12]}"
    try:
        sqlite_db.add_folder_item(iid, folder_id, item.item_type, item.item_id, item.metadata)
        return {"status": "added", "id": iid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/folders/{folder_id}/items/{item_id}")
def delete_folder_item(folder_id: str, item_id: str):
    try:
        sqlite_db.delete_folder_item(folder_id, item_id)
        return {"status": "removed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/folders/{folder_id}/items")
def get_folder_items(folder_id: str):
    try:
        return sqlite_db.get_folder_items(folder_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/notes")
def add_note(note: NoteCreate):
    nid = f"note_{uuid.uuid4().hex[:12]}"
    try:
        sqlite_db.add_note(nid, note.parent_type, note.parent_id, note.content)
        return {"status": "saved", "id": nid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/notes/{note_id}")
def delete_note(note_id: str):
    try:
        sqlite_db.delete_note(note_id)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/notes")
def get_notes(parent_type: str, parent_id: str):
    try:
        return sqlite_db.get_notes(parent_type, parent_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tracks/{track_uuid}/label")
def update_track_label(track_uuid: str, body: TrackLabelUpdate):
    try:
        sqlite_db.update_track_label(track_uuid, body.label)
        return {"status": "updated", "track_uuid": track_uuid, "label": body.label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class CameraAppend(BaseModel):
    name: str
    path: str
    is_live: bool = False

@app.post("/api/jobs/{job_id}/append-video")
def append_video_to_job(job_id: str, cam: CameraAppend):
    try:
        # Check if job exists
        job_record = sqlite_db.get_job(job_id)
        if not job_record:
            raise HTTPException(status_code=404, detail="Job not found.")

        # Register new camera source in SQLite
        cam_id = f"cam_{uuid.uuid4().hex[:8]}"
        from pipeline.job_manager import get_file_sha256
        checksum = get_file_sha256(cam.path)
        sqlite_db.add_camera(
            camera_id=cam_id,
            job_id=job_id,
            name=cam.name,
            source_type="live" if cam.is_live else "file",
            source_path=cam.path,
            checksum=checksum
        )

        # Reset job status back to queued so it can be restarted
        sqlite_db.update_job_status(job_id, "queued")

        return {"status": "queued", "camera_id": cam_id}
    except Exception as e:
        LOGGER.error(f"Failed to append video to job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard/stats")
def get_dashboard_stats(job_id: Optional[str] = None):
    try:
        with sqlite_db.connection() as conn:
            # 1. Total hours indexed (completed or indexing jobs)
            if job_id:
                cur = conn.execute("SELECT SUM(duration_sec) FROM jobs WHERE status = 'completed' AND id = ?", (job_id,))
            else:
                cur = conn.execute("SELECT SUM(duration_sec) FROM jobs WHERE status = 'completed'")
            row = cur.fetchone()
            total_sec = row[0] if row and row[0] is not None else 0.0
            total_hours = total_sec / 3600.0

            # 2. Objects detected
            if job_id:
                cur = conn.execute("SELECT COUNT(d.id) FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?", (job_id,))
            else:
                cur = conn.execute("SELECT COUNT(*) FROM detections")
            total_detections = cur.fetchone()[0]

            # Category breakdowns
            if job_id:
                cur = conn.execute("SELECT COUNT(d.id) FROM detections d JOIN tracks t ON d.track_id = t.id WHERE d.entity_type = 'person' AND t.job_id = ?", (job_id,))
            else:
                cur = conn.execute("SELECT COUNT(*) FROM detections WHERE entity_type = 'person'")
            count_persons = cur.fetchone()[0]

            if job_id:
                cur = conn.execute("SELECT COUNT(d.id) FROM detections d JOIN tracks t ON d.track_id = t.id WHERE d.entity_type = 'vehicle' AND t.job_id = ?", (job_id,))
            else:
                cur = conn.execute("SELECT COUNT(*) FROM detections WHERE entity_type = 'vehicle'")
            count_vehicles = cur.fetchone()[0]

            if job_id:
                cur = conn.execute("SELECT COUNT(*) FROM tracks WHERE plate_text IS NOT NULL AND plate_text != '' AND job_id = ?", (job_id,))
            else:
                cur = conn.execute("SELECT COUNT(*) FROM tracks WHERE plate_text IS NOT NULL AND plate_text != ''")
            count_plates = cur.fetchone()[0]

            # 3. Active jobs
            if job_id:
                cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('indexing', 'starting') AND id = ?", (job_id,))
            else:
                cur = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('indexing', 'starting')")
            active_jobs_count = cur.fetchone()[0]

            # 4. Active cameras
            if job_id:
                cur = conn.execute("SELECT COUNT(DISTINCT id) FROM cameras WHERE job_id = ?", (job_id,))
            else:
                cur = conn.execute("SELECT COUNT(DISTINCT id) FROM cameras")
            active_cams = cur.fetchone()[0]

            # 5. Activity Feed (last 10 tracks with their camera and start time)
            if job_id:
                cur = conn.execute("""
                    SELECT t.id, t.entity_type, t.start_time, c.name as camera_name, j.name as job_name
                    FROM tracks t
                    JOIN cameras c ON t.camera_id = c.id
                    JOIN jobs j ON t.job_id = j.id
                    WHERE t.job_id = ?
                    ORDER BY t.start_time DESC LIMIT 10
                """, (job_id,))
            else:
                cur = conn.execute("""
                    SELECT t.id, t.entity_type, t.start_time, c.name as camera_name, j.name as job_name
                    FROM tracks t
                    JOIN cameras c ON t.camera_id = c.id
                    JOIN jobs j ON t.job_id = j.id
                    ORDER BY t.start_time DESC LIMIT 10
                """)
            feed = []
            for r in cur.fetchall():
                feed.append({
                    "track_id": r[0],
                    "entity_type": r[1],
                    "timestamp": r[2],
                    "camera_name": r[3],
                    "job_name": r[4]
                })

            return {
                "total_hours_indexed": round(total_hours, 2),
                "objects_detected": total_detections,
                "persons_count": count_persons,
                "vehicles_count": count_vehicles,
                "plates_count": count_plates,
                "active_jobs_count": active_jobs_count,
                "active_cameras": active_cams,
                "activity_feed": feed
            }
    except Exception as e:
        LOGGER.error(f"Failed to get dashboard stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search/quick")
def execute_quick_search(q: str):
    try:
        q = q.lower().strip()
        results = []
        with sqlite_db.connection() as conn:
            # 1. Search cases/jobs
            cur = conn.execute("SELECT id, name FROM jobs WHERE lower(name) LIKE ?", (f"%{q}%",))
            for r in cur.fetchall():
                results.append({"type": "case", "name": r[1], "id": r[0]})

            # 2. Search folders
            cur = conn.execute("SELECT id, name FROM folders WHERE lower(name) LIKE ?", (f"%{q}%",))
            for r in cur.fetchall():
                results.append({"type": "folder", "name": r[1], "id": r[0]})

            # 3. Search labeled tracks
            cur = conn.execute("SELECT t.id, t.label, t.camera_id, t.start_time, t.job_id FROM tracks t WHERE t.label IS NOT NULL AND t.label != '' AND lower(t.label) LIKE ?", (f"%{q}%",))
            for r in cur.fetchall():
                results.append({
                    "type": "label",
                    "name": r[1],
                    "id": r[0],
                    "camera_id": r[2],
                    "timestamp": r[3],
                    "job_id": r[4]
                })
        return results
    except Exception as e:
        LOGGER.error(f"Quick search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/system/telemetry")
def get_system_telemetry():
    try:
        # 1. Storage stats
        sqlite_size = SQLITE_DB_PATH.stat().st_size if SQLITE_DB_PATH.exists() else 0
        chroma_size = sum(f.stat().st_size for f in CHROMA_PERSIST_DIR.glob("**/*") if f.is_file()) if CHROMA_PERSIST_DIR.exists() else 0
        crops_size = sum(f.stat().st_size for f in MEDIA_DIR.glob("**/*") if f.is_file()) if MEDIA_DIR.exists() else 0

        total, used, free = shutil.disk_usage(".")

        # 2. Performance stats
        telemetry = PerformanceMonitorInstance.get_metrics()

        # 3. Model configurations
        from config import DEVICE, YOLO_DEVICE, CLIP_DEVICE, FACE_DEVICE, OCR_DEVICE
        models = [
            {"name": "YOLOv8 Object Detection", "device": str(YOLO_DEVICE), "status": "loaded" if gpu_scheduler.yolo else "unloaded"},
            {"name": "CLIP Multi-Modal Embeddings", "device": str(CLIP_DEVICE), "status": "loaded" if gpu_scheduler.clip_model else "unloaded"},
            {"name": "FaceNet InceptionResnetV1", "device": str(FACE_DEVICE), "status": "loaded" if gpu_scheduler.face_embedder else "unloaded"},
            {"name": "EasyOCR Plate Recognition", "device": str(OCR_DEVICE), "status": "loaded" if gpu_scheduler.ocr_reader else "unloaded"}
        ]

        return {
            "storage": {
                "sqlite_size_bytes": sqlite_size,
                "chroma_size_bytes": chroma_size,
                "crops_size_bytes": crops_size,
                "disk_total_bytes": total,
                "disk_used_bytes": used,
                "disk_free_bytes": free
            },
            "performance": telemetry,
            "models": models
        }
    except Exception as e:
        LOGGER.error(f"Failed to get system telemetry: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# 11b. Forensic Video Clip & Journey Merging Endpoints
# ==============================================================================
class MergeClipsRequest(BaseModel):
    observations: List[Dict[str, Any]]
    buffer_sec: float = 3.0
    folder_name: Optional[str] = "Merged Evidence Clips"

@app.post("/api/jobs/{job_id}/merge-clips")
def merge_selected_clips(job_id: str, req: MergeClipsRequest):
    """Extracts subclips for selected observations and concatenates them chronologically."""
    try:
        from pipeline.video_merger import video_merger
        result = video_merger.merge_observation_clips(
            job_id=job_id,
            observations=req.observations,
            context_buffer_sec=req.buffer_sec
        )
        if not result:
            raise HTTPException(status_code=400, detail="Could not generate merged video from selected observations.")

        # Automatically register in evidence folder
        try:
            folders = sqlite_db.get_folders(job_id)
            target_folder = next((f for f in folders if f["name"] == req.folder_name), None)
            if not target_folder:
                fid = f"fold_{uuid.uuid4().hex[:12]}"
                sqlite_db.create_folder(fid, req.folder_name, "Automatically compiled merged clips", job_id)
                target_folder_id = fid
            else:
                target_folder_id = target_folder["id"]

            item_id = f"clip_{uuid.uuid4().hex[:12]}"
            sqlite_db.add_folder_item(
                item_id=item_id,
                folder_id=target_folder_id,
                item_type="clip",
                source_id=result["path"],
                metadata=json.dumps(result)
            )
        except Exception as fe:
            LOGGER.warning(f"Note: Could not auto-bookmark merged clip in folder: {fe}")

        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        LOGGER.error(f"Failed to merge selected clips: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class MergeJourneyRequest(BaseModel):
    track_uuid: str
    similarity: float = 0.65
    buffer_sec: float = 3.0

@app.post("/api/jobs/{job_id}/merge-journey")
def merge_journey_video(job_id: str, req: MergeJourneyRequest):
    """Reconstructs cross-camera journey for a target track and stitches the video segments."""
    try:
        from pipeline.video_merger import video_merger
        journey_data = journey_engine.reconstruct_journey(req.track_uuid, similarity_threshold=req.similarity)

        events = journey_data.get("confirmed_journey", []) if isinstance(journey_data, dict) else journey_data
        if not events:
            raise HTTPException(status_code=400, detail="No validated journey transitions available to merge.")

        result = video_merger.merge_observation_clips(
            job_id=job_id,
            observations=events,
            context_buffer_sec=req.buffer_sec,
            output_filename=f"journey_{req.track_uuid[:16]}_{int(time.time())}.mp4"
        )
        if not result:
            raise HTTPException(status_code=400, detail="Could not generate merged journey video.")

        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        LOGGER.error(f"Failed to merge journey video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tracks/{track_uuid}/summary")
def get_track_summary(track_uuid: str):
    try:
        # 1. Fetch track details
        track = sqlite_db.get_track(track_uuid)
        if not track:
            raise HTTPException(status_code=404, detail="Track not found.")

        # 2. Get camera info
        cams = sqlite_db.get_cameras_by_job(track["job_id"])
        camera_map = {c["id"]: c["name"] for c in cams}
        camera_name = camera_map.get(track["camera_id"], track["camera_id"])

        # 3. Get label if any
        label_str = f"User Label: {track['label']}" if track.get("label") else "No User Label"

        # 4. Get journey details
        events = []
        try:
            journey_result = journey_engine.reconstruct_journey(track_uuid, similarity_threshold=0.5)
            if isinstance(journey_result, dict):
                events = journey_result.get("confirmed_journey", [])
            elif isinstance(journey_result, list):
                events = journey_result
        except Exception as je:
            LOGGER.warning(f"Journey reconstruction failed for track {track_uuid}: {je}")

        # 5. Build chronological narrative
        sentences = []
        if not events:
            # Fallback to single camera occurrence
            dur = track["end_time"] - track["start_time"]
            sentences.append(f"Subject was observed on the '{camera_name}' camera starting at time index {track['start_time']:.1f}s for a duration of {dur:.1f}s.")
        else:
            first_event = events[0]
            first_cam = camera_map.get(first_event.get("camera_id"), first_event.get("camera_id"))
            sentences.append(f"The subject was first detected on the '{first_cam}' camera at time index {first_event.get('timestamp', 0.0):.1f}s.")

            for i in range(1, len(events)):
                evt = events[i]
                cam = camera_map.get(evt.get("camera_id"), evt.get("camera_id"))
                sim = evt.get("confidence", evt.get("similarity", 0.85))
                transit = evt.get("transit_time_sec", 0.0)

                # Check certainty
                if sim >= 0.85:
                    match_desc = f"with a strong track association (similarity: {sim:.2f})"
                elif sim >= 0.70:
                    match_desc = f"with a possible visual match (similarity: {sim:.2f})"
                else:
                    match_desc = f"with a weak similarity correlation (similarity: {sim:.2f})"

                sentences.append(f"Subsequently, the subject was observed on the '{cam}' camera at time index {evt.get('timestamp', 0.0):.1f}s {match_desc}, representing a transit time of {transit:.1f}s from the prior camera field.")

        # 6. Retrieve track notes
        notes = sqlite_db.get_notes("track", track_uuid)
        notes_str = "\n".join([f"- [Note] {n['content']} (Saved on {time.strftime('%Y-%m-%d %H:%M', time.localtime(n['created_at']))})" for n in notes]) if notes else "No notes added."

        # 7. Check source footage characteristics (infrared/night monochrome vs color)
        detections = sqlite_db.get_detections_by_track(track_uuid)
        is_ir_mono = False
        if detections:
            conf_values = [d.get("color_confidence", 0.0) for d in detections if d.get("color_confidence") is not None]
            if conf_values and max(conf_values) < 0.05:
                is_ir_mono = True

        footage_characteristic = "**Footage Characteristic:** Infrared / Night-Vision Monochromatic (Color-based interpretation unavailable)\n" if is_ir_mono else ""

        # Compile report
        markdown = f"""### FORENSIC ANALYSIS SUMMARY: TRACK #{track['track_id']}

**Target Subject Entity Type:** {track['entity_type'].upper()}
**Initial Detection Location:** {camera_name}
**Time Interval:** {track['start_time']:.1f}s to {track['end_time']:.1f}s
**Identity Label:** {track.get('label') or 'Unidentified Subject'}
{footage_characteristic}
#### Chronological Trajectory Analysis
{chr(10).join(sentences)}

#### Investigation Notes
{notes_str}

*Report compiled on 2026-08-23 by AICatch Forensic Engine. All match correlations are based on visual embeddings and temporal constraints.*
"""
        return {"summary": markdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/jobs/{job_id}/summary")
def get_job_summary(job_id: str):
    try:
        # Fetch job details
        job = sqlite_db.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Investigation not found.")

        cams = sqlite_db.get_cameras_by_job(job_id)

        with sqlite_db.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM tracks WHERE job_id = ? ORDER BY start_time DESC", (job_id,))
            tracks = [dict(row) for row in cursor.fetchall()]

        # Camera list
        cam_list_str = "\n".join([f"- Camera: **{c['name']}** (Source: `{c['source_path']}`)" for c in cams]) if cams else "No cameras registered."

        # Group tracks
        persons = [t for t in tracks if t["entity_type"] == "person"]
        vehicles = [t for t in tracks if t["entity_type"] == "vehicle"]
        plates = [t for t in tracks if t.get("plate_text")]
        labeled = [t for t in tracks if t.get("label")]

        # Labeled/Plate tracks log
        interest_str = ""
        if labeled or plates:
            interest_lines = []
            for t in tracks:
                if t.get("label") or t.get("plate_text"):
                    cam_name = next((c["name"] for c in cams if c["id"] == t["camera_id"]), t["camera_id"])
                    label_desc = f"Label: **{t['label']}**" if t.get("label") else ""
                    plate_desc = f"Plate: **{t['plate_text']}**" if t.get("plate_text") else ""
                    desc = " | ".join(filter(None, [label_desc, plate_desc]))
                    interest_lines.append(f"- Track #{t['track_id']} ({t['entity_type'].upper()}) on camera **{cam_name}** at {t['start_time']:.1f}s to {t['end_time']:.1f}s [{desc}]")
            interest_str = "\n".join(interest_lines)
        else:
            interest_str = "No specific targets of interest labeled or license plates recognized."

        # Notes list (all notes saved under tracks of this job - single indexed query)
        all_notes = []
        try:
            with sqlite_db.connection() as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT n.content, n.created_at, t.track_id
                    FROM notes n
                    JOIN tracks t ON n.parent_id = t.id
                    WHERE n.parent_type = 'track' AND t.job_id = ?
                    ORDER BY n.created_at DESC
                """, (job_id,))
                all_notes = [(row["track_id"], row["content"], row["created_at"]) for row in cursor.fetchall()]
        except Exception as ne:
            LOGGER.warning("Could not load case notes for summary: %s", ne)

        notes_str = ""
        if all_notes:
            notes_str = "\n".join([f"- Track #{t_id}: \"{content}\" (Saved at {time.strftime('%Y-%m-%d %H:%M', time.localtime(c_at))})" for t_id, content, c_at in all_notes])
        else:
            notes_str = "No investigator notes recorded for this case."

        markdown = f"""# AICATCH CASE SUMMARY: {job['name'].upper()}

**Investigation Mode:** {'MULTI-CAMERA STITCH' if job.get('investigation_type') == 'multi_camera' else 'SINGLE VIDEO'}
**Case Status:** {job['status'].upper()}
**Duration Processed:** {job['processed_sec']:.1f}s of {job['duration_sec']:.1f}s

### Registered Camera Feeds
{cam_list_str}

### Forensic Statistics Overview
- **Total Tracks Extracted:** {len(tracks)}
- **Persons Detected:** {len(persons)}
- **Vehicles Detected:** {len(vehicles)}
- **Recognized License Plates:** {len(plates)}

### Key Targets of Interest
{interest_str}

### Investigator Notes Log
{notes_str}

*Report generated automatically from live forensic indexing indexes. All logs are persistent and verifiable.*
"""
        return {"summary": markdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/folders/{folder_id}/summary")
@app.get("/api/folders/{folder_id}/summary")
def get_folder_summary(folder_id: str):
    try:
        # Fetch folder info
        folders = sqlite_db.get_folders()
        folder = next((f for f in folders if f["id"] == folder_id), None)
        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found.")

        items = sqlite_db.get_folder_items(folder_id)
        notes = sqlite_db.get_notes("folder", folder_id)

        # Group items by timestamp to sort chronologically
        evidence_entries = []
        for item in items:
            metadata_str = item.get("metadata")
            # Parse metadata
            if metadata_str:
                import json
                try:
                    data = json.loads(metadata_str)
                    timestamp = data.get("timestamp", 0.0)
                    camera_id = data.get("camera_id", "Unknown")
                    track_id = data.get("track_id", "N/A")
                    entity_type = data.get("entity_type", "Unknown")
                    label = data.get("label", "")

                    desc = f"Track #{track_id.split('_').pop() if '_' in track_id else track_id} ({entity_type.upper()})"
                    if label:
                        desc += f" labeled as **{label}**"

                    evidence_entries.append({
                        "timestamp": timestamp,
                        "camera_id": camera_id,
                        "description": desc,
                        "created_at": item["created_at"]
                    })
                except Exception:
                    # Generic item
                    evidence_entries.append({
                        "timestamp": 0.0,
                        "camera_id": "Unknown",
                        "description": f"Item {item['item_type']} (ID: {item['item_id']})",
                        "created_at": item["created_at"]
                    })
            else:
                evidence_entries.append({
                    "timestamp": 0.0,
                    "camera_id": "Unknown",
                    "description": f"Item {item['item_type']} (ID: {item['item_id']})",
                    "created_at": item["created_at"]
                })

        # Sort chronologically by timestamp
        evidence_entries.sort(key=lambda x: x["timestamp"])

        chronological_lines = []
        for i, entry in enumerate(evidence_entries, 1):
            chronological_lines.append(f"{i}. **[{entry['camera_id']} @ {entry['timestamp']:.1f}s]** {entry['description']}")

        chronology_str = "\n".join(chronological_lines) if chronological_lines else "No observation evidence bookmark items added to this folder."

        # Notes list
        notes_lines = []
        for n in notes:
            notes_lines.append(f"- \"{n['content']}\" (Added on {time.strftime('%Y-%m-%d %H:%M', time.localtime(n['created_at']))})")
        notes_str = "\n".join(notes_lines) if notes_lines else "No notes recorded in this case folder."

        markdown = f"""# FORENSIC EVIDENCE DOSSIER: {folder['name'].upper()}

**Description / Scope:** {folder['description'] or 'General evidence collection'}
**Total Observations Gathered:** {len(items)}
**Notes Count:** {len(notes)}

### Chronological Evidence Timeline
{chronology_str}

### Folder Notes & Hypotheses
{notes_str}

*This report compiles chronological bookmarks, clips, and annotations gathered by investigators.*
"""
        return {"summary": markdown}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# 12. Main Entrypoint Runner
# ==============================================================================
if __name__ == "__main__":
    LOGGER.info("Starting AICatch FastAPI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
