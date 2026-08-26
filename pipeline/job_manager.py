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

    def complete_job(self, job_id: str) -> None:
        """Marks a job as successfully completed and cleans up its decoders."""
        with self.lock:
            job = self.active_jobs.get(job_id)
            if job:
                with job.lock:
                    job.status = "completed"
                    self.sqlite.update_job_status(job_id, "completed")
                    LOGGER.info(f"Completed job {job_id} successfully.")
                self.cleanup_job_resources(job_id)

    def fail_job(self, job_id: str, error_msg: str) -> None:
        """Marks a job as failed and cleans up its decoders."""
        with self.lock:
            job = self.active_jobs.get(job_id)
            if job:
                with job.lock:
                    job.status = "failed"
                    job.error_msg = error_msg
                    self.sqlite.update_job_status(job_id, "failed", error_msg)
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
