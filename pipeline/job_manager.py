"""Multi-Job Manager for AICatch.

Manages video analysis job life cycles (queued, starting, indexing, paused,
completed, failed, cancelled), spawner threads, and decoder collections.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional
import threading
import hashlib
from pathlib import Path

import sqlite3
import json
from config import PROCESSING_PROFILES
from storage.sqlite_db import SQLiteDatabase
from pipeline.decoder import VideoStreamDecoder

LOGGER = logging.getLogger(__name__)

def get_file_sha256(filepath: str) -> Optional[str]:
    """Calculate the SHA-256 hash of a file to check for duplicate footage."""
    try:
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            return None
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256.update(byte_block)
        return sha256.hexdigest()
    except Exception as e:
        LOGGER.error(f"Failed to calculate file hash for {filepath}: {e}")
        return None

class Job:
    """Represents an active video analysis job containing multiple camera streams."""
    def __init__(self, job_id: str, name: str, profile: str):
        self.job_id = job_id
        self.name = name
        self.profile = profile
        self.status = "queued"
        self.decoders: Dict[str, VideoStreamDecoder] = {}
        self.error_msg: Optional[str] = None
        self.lock = threading.Lock()

class JobManager:
    def __init__(self, sqlite_db: SQLiteDatabase):
        self.sqlite = sqlite_db
        # In-memory mapping of active jobs: job_id -> Job
        self.active_jobs: Dict[str, Job] = {}
        self.lock = threading.RLock()
        
        # Recover status of interrupted jobs on startup
        self._recover_jobs()

    def _recover_jobs(self) -> None:
        """Sets any interrupted 'indexing' or 'starting' jobs to 'failed' on startup."""
        jobs = self.sqlite.get_jobs()
        for job in jobs:
            if job["status"] in ["indexing", "starting"]:
                LOGGER.info(f"Marking interrupted job {job['id']} as failed.")
                self.sqlite.update_job_status(
                    job["id"],
                    "failed",
                    "Job interrupted due to system shutdown or restart."
                )



    def create_job(
        self,
        name: str,
        profile: str,
        camera_sources: List[Dict[str, Any]],
        investigation_type: str = "single_video"
    ) -> str:
        """Registers a new job and its cameras in SQLite and memory."""
        if profile not in PROCESSING_PROFILES:
            raise ValueError(f"Invalid processing profile: {profile}. Choose FAST, BALANCED, or FORENSIC.")
            
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        
        # Save to SQLite
        self.sqlite.add_job(job_id, name, profile, investigation_type)
        
        # Register cameras in SQLite
        for cam in camera_sources:
            cam_id = f"cam_{uuid.uuid4().hex[:8]}"
            checksum = get_file_sha256(cam["path"])
            self.sqlite.add_camera(
                camera_id=cam_id,
                job_id=job_id,
                name=cam["name"],
                source_type="live" if cam.get("is_live", False) else "file",
                source_path=cam["path"],
                checksum=checksum
            )
            
        LOGGER.info(f"Created job: {name} ({job_id}) with {len(camera_sources)} cameras.")
        return job_id

    def start_job(self, job_id: str) -> None:
        """Prepares and launches decoder threads for all camera sources in the job."""
        with self.lock:
            # Check if job already active in memory
            if job_id in self.active_jobs:
                LOGGER.warning(f"Job {job_id} is already running.")
                return
                
            job_record = self.sqlite.get_job(job_id)
            if not job_record:
                raise ValueError(f"Job {job_id} not found in database.")
                
            cameras = self.sqlite.get_cameras_by_job(job_id)
            if not cameras:
                raise ValueError(f"Job {job_id} has no cameras registered.")

            # Create in-memory Job container
            job = Job(job_id, job_record["name"], job_record["profile"])
            job.status = "starting"
            self.active_jobs[job_id] = job
            self.sqlite.update_job_status(job_id, "starting")

            try:
                # Get profile settings to determine target_fps
                profile_name = job.profile
                profile_cfg = PROCESSING_PROFILES.get(profile_name, PROCESSING_PROFILES["BALANCED"])
                sampling_rate = profile_cfg.get("sampling_rate_sec", 2.0)
                target_fps = 1.0 / sampling_rate
                LOGGER.info(f"Job {job_id} profile: {profile_name}. Sampling rate: {sampling_rate}s ({target_fps} FPS).")

                # Initialize decoders
                for cam in cameras:
                    if cam.get("resolution") is not None and cam.get("fps") is not None:
                        LOGGER.info(f"Skipping camera {cam['name']} ({cam['id']}) - already indexed.")
                        continue
                    decoder = VideoStreamDecoder(
                        camera_id=cam["id"],
                        source_path=cam["source_path"],
                        is_live=(cam["source_type"] == "live"),
                        target_fps=target_fps
                    )
                    
                    # Update camera status in SQLite
                    resolution = f"{decoder.width}x{decoder.height}"
                    self.sqlite.update_camera_metadata(
                        camera_id=cam["id"],
                        status="active",
                        resolution=resolution,
                        fps=decoder.fps
                    )
                    
                    job.decoders[cam["id"]] = decoder
                    
                # Start all decoders
                for decoder in job.decoders.values():
                    decoder.start()
                    
                job.status = "indexing"
                self.sqlite.update_job_status(job_id, "indexing")
                LOGGER.info(f"Started video analytics indexing for job {job_id}.")
                
            except Exception as e:
                LOGGER.error(f"Failed to start job {job_id}: {e}", exc_info=True)
                job.status = "failed"
                job.error_msg = str(e)
                self.sqlite.update_job_status(job_id, "failed", str(e))
                self.cleanup_job_resources(job_id)
                raise e

    def pause_job(self, job_id: str) -> None:
        """Pauses frame decoding threads for the job."""
        with self.lock:
            job = self.active_jobs.get(job_id)
            if not job:
                LOGGER.warning(f"Cannot pause inactive job {job_id}.")
                return
                
            with job.lock:
                if job.status != "indexing":
                    return
                for decoder in job.decoders.values():
                    decoder.pause_stream()
                job.status = "paused"
                self.sqlite.update_job_status(job_id, "paused")
                LOGGER.info(f"Paused job {job_id}.")

    def resume_job(self, job_id: str) -> None:
        """Resumes frame decoding threads for the job."""
        with self.lock:
            job = self.active_jobs.get(job_id)
            if not job:
                LOGGER.warning(f"Cannot resume inactive job {job_id}.")
                return
                
            with job.lock:
                if job.status != "paused":
                    return
                for decoder in job.decoders.values():
                    decoder.resume_stream()
                job.status = "indexing"
                self.sqlite.update_job_status(job_id, "indexing")
                LOGGER.info(f"Resumed job {job_id}.")

    def cancel_job(self, job_id: str) -> None:
        """Stops and cleans up all stream decoders for the job."""
        with self.lock:
            job = self.active_jobs.get(job_id)
            if job:
                with job.lock:
                    job.status = "cancelled"
                    self.sqlite.update_job_status(job_id, "cancelled")
                    LOGGER.info(f"Cancelling job {job_id}...")
                self.cleanup_job_resources(job_id)
            else:
                # Update status in SQLite if it wasn't running in memory
                self.sqlite.update_job_status(job_id, "cancelled")

    def update_job_stage(self, job_id: str, stage: str, error_msg: Optional[str] = None, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        """Updates the fine-grained job state machine stage."""
        with self.lock:
            job = self.active_jobs.get(job_id)
            if job:
                with job.lock:
                    job.status = stage
                    if error_msg:
                        job.error_msg = error_msg
            diag_json = json.dumps(diagnostics) if diagnostics else None
            self.sqlite.update_job_status(job_id, stage, error_msg, diag_json)
            LOGGER.info("JOB_STAGE_TRANSITION job_id=%s stage=%s", job_id, stage)

    def complete_job(
        self,
        job_id: str,
        vector_db: Optional[Any] = None,
        dispatcher: Optional[Any] = None
    ) -> bool:
        """Transitions job through completion stages with vector write and read-back verification."""
        with self.lock:
            job = self.active_jobs.get(job_id)
            
            # 1. Transition to DETECTION_COMPLETE & TRACKING_COMPLETE
            self.update_job_stage(job_id, "TRACKING_COMPLETE")
            
            # 2. Wait for dispatcher to flush all queued items to SQLite & Chroma
            self.update_job_stage(job_id, "EMBEDDING_COMPLETE")
            if dispatcher:
                dispatcher.flush_and_wait(timeout=15.0)
            self.update_job_stage(job_id, "PERSISTENCE_COMPLETE")
            
            # 3. Perform Vector write, read-back, and Lineage verification
            verification_errors = []
            tracks = []
            detections = []
            with self.sqlite.connection() as conn:
                conn.row_factory = sqlite3.Row
                cur_t = conn.execute("SELECT * FROM tracks WHERE job_id = ?", (job_id,))
                tracks = [dict(r) for r in cur_t.fetchall()]
                
                cur_d = conn.execute(
                    "SELECT d.* FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ?",
                    (job_id,)
                )
                detections = [dict(r) for r in cur_d.fetchall()]

            # Lineage verification
            for det in detections[:10]:
                if not det.get("track_id"):
                    verification_errors.append(f"Detection {det.get('id')} has no track_id")
                if not det.get("crop_path"):
                    verification_errors.append(f"Detection {det.get('id')} has no crop_path")
                elif not Path(det["crop_path"]).exists():
                    LOGGER.warning("Crop file not found on disk: %s", det["crop_path"])

            # Vector DB verification
            indexed_vectors_count = 0
            reid_vectors_count = 0
            face_vectors_count = 0
            
            if vector_db is None:
                try:
                    from storage.vector_db import VectorDatabase
                    from config import CHROMA_PERSIST_DIR
                    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
                except Exception as db_init_err:
                    LOGGER.debug(f"Could not initialize VectorDatabase fallback: {db_init_err}")

            if vector_db is not None:
                try:
                    vis_res = vector_db.visual_coll.get(where={"job_id": job_id}, include=["metadatas", "embeddings"])
                    indexed_vectors_count = len(vis_res.get("ids", []))
                    
                    try:
                        reid_res = vector_db.reid_coll.get(where={"job_id": job_id})
                        reid_vectors_count = len(reid_res.get("ids", []))
                    except Exception:
                        reid_vectors_count = 0
                        
                    try:
                        face_res = vector_db.face_coll.get(where={"job_id": job_id})
                        face_vectors_count = len(face_res.get("ids", []))
                    except Exception:
                        face_vectors_count = 0
                    
                    if indexed_vectors_count == 0 and len(detections) > 0:
                        verification_errors.append("Zero visual vector embeddings persisted in ChromaDB despite detections present")
                    elif indexed_vectors_count == 0 and len(tracks) == 0:
                        verification_errors.append("No tracks or visual vectors were generated during indexing")
                    elif indexed_vectors_count > 0:
                        # Verify read-back retrieval for a sample
                        sample_id = vis_res["ids"][0]
                        sample_meta = vis_res["metadatas"][0]
                        if sample_meta.get("job_id") != job_id:
                            verification_errors.append(f"Vector metadata mismatch for sample {sample_id}")
                except Exception as ve:
                    verification_errors.append(f"Chroma verification query failed: {ve}")
            else:
                verification_errors.append("Vector database instance unavailable for verification")

            # Collect comprehensive Ingestion Diagnostics
            stats = {}
            if dispatcher and hasattr(dispatcher, "job_stats"):
                stats = dispatcher.job_stats.get(job_id, {})
                
            total_duration = 0.0
            if job:
                with job.lock:
                    for dec in job.decoders.values():
                        total_duration += dec.duration_sec
                        
            diagnostics = {
                "job_id": job_id,
                "status": "INDEXED" if not verification_errors else "FAILED",
                "tracks_count": len(tracks),
                "detections_count": len(detections),
                "crops_count": stats.get("crops", len(detections)),
                "embedding_attempts": stats.get("embedding_attempts", indexed_vectors_count),
                "successful_embeddings": stats.get("successful_embeddings", indexed_vectors_count),
                "failed_embeddings": stats.get("failed_embeddings", 0),
                "vectors_inserted": indexed_vectors_count,
                "vectors_reid": reid_vectors_count,
                "vectors_face": face_vectors_count,
                "vectors_verified": indexed_vectors_count if not verification_errors else 0,
                "collection_name": vector_db.visual_coll.name if vector_db else "N/A",
                "collection_count_after": vector_db.visual_coll.count() if vector_db else 0,
                "duration_sec": total_duration,
                "verification_errors": verification_errors
            }

            if verification_errors:
                err_msg = "; ".join(verification_errors)
                LOGGER.error("Job %s failed verification: %s", job_id, err_msg)
                self.fail_job(job_id, err_msg, diagnostics=diagnostics)
                return False
                
            # Set state to INDEXED
            self.update_job_stage(job_id, "INDEXED", diagnostics=diagnostics)
            LOGGER.info("INGESTION_DIAGNOSTICS %s", json.dumps(diagnostics))
            self.cleanup_job_resources(job_id)
            return True


    def fail_job(self, job_id: str, error_msg: str, diagnostics: Optional[Dict[str, Any]] = None) -> None:
        """Marks a job as failed and cleans up its decoders."""
        with self.lock:
            job = self.active_jobs.get(job_id)
            if job:
                with job.lock:
                    job.status = "FAILED"
                    job.error_msg = error_msg
            diag_json = json.dumps(diagnostics) if diagnostics else None
            self.sqlite.update_job_status(job_id, "FAILED", error_msg, diag_json)
            LOGGER.error(f"Job {job_id} failed: {error_msg}")
            self.cleanup_job_resources(job_id)

    def cleanup_job_resources(self, job_id: str) -> None:
        """Terminates and joins all decoder threads and cleans memory maps."""
        job = self.active_jobs.get(job_id)
        if not job:
            return
            
        LOGGER.info(f"Cleaning up resources for job {job_id}...")
        
        # Stop all decoders
        for cam_id, decoder in list(job.decoders.items()):
            try:
                decoder.stop_stream()
                if decoder.is_alive():
                    decoder.join(timeout=3.0)
                # Mark camera as inactive in SQLite
                self.sqlite.update_camera_metadata(cam_id, "inactive", f"{decoder.width}x{decoder.height}", decoder.fps)
            except Exception as e:
                LOGGER.error(f"Error stopping decoder for camera {cam_id}: {e}")
                
        # Remove from active jobs mapping
        with self.lock:
            self.active_jobs.pop(job_id, None)
            
        LOGGER.info(f"Finished resource cleanup for job {job_id}.")

                
        # Remove from active jobs mapping
        with self.lock:
            self.active_jobs.pop(job_id, None)
            
        LOGGER.info(f"Finished resource cleanup for job {job_id}.")

    def get_active_decoders(self) -> List[VideoStreamDecoder]:
        """Gathers all decoder instances that are currently decoding frames."""
        decoders = []
        with self.lock:
            for job in self.active_jobs.values():
                with job.lock:
                    if job.status == "indexing":
                        decoders.extend(job.decoders.values())
        return decoders

    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Calculates progress percentages and processing stats for a job."""
        job_record = self.sqlite.get_job(job_id)
        if not job_record:
            return {"status": "unknown", "progress": 0.0}
            
        status = job_record["status"]
        
        # If active, calculate dynamic progress based on decoder stats
        job = self.active_jobs.get(job_id)
        if job and status == "indexing":
            total_duration = 0.0
            total_processed = 0.0
            
            with job.lock:
                for decoder in job.decoders.values():
                    if decoder.is_live:
                        # Live feeds are infinite, show 100% active state
                        return {"status": "indexing", "progress": 1.0, "is_live": True}
                    total_duration += decoder.duration_sec
                    fps = decoder.target_fps or 25.0
                    frames_q = decoder.frame_queue.qsize()
                    # Approx processed video seconds = frames decoded minus queue backlog
                    processed_frames = max(0, decoder.frames_decoded - frames_q)
                    total_processed += processed_frames / fps
                    
            progress = (total_processed / total_duration) if total_duration > 0 else 0.0
            return {
                "status": "indexing",
                "progress": min(0.99, progress),
                "processed_sec": total_processed,
                "duration_sec": total_duration,
                "is_live": False
            }
            
        # For non-active states, return database static metrics
        if status == "completed":
            return {"status": "completed", "progress": 1.0}
            
        duration = job_record.get("duration_sec", 0.0)
        processed = job_record.get("processed_sec", 0.0)
        progress = (processed / duration) if duration > 0 else 0.0
        
        return {
            "status": status,
            "progress": progress,
            "error_msg": job_record.get("error_msg")
        }
