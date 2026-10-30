"""Media Storage Manager for AICatch.

Handles security verification, directory structure organization, image saving,
and disk space cleanup policies for raw and processed assets.
"""

from pathlib import Path
import shutil
import logging
import time
import os
import queue
import threading
from typing import Union, List, Optional, Dict
import cv2
import numpy as np

from config import (
    MEDIA_DIR, UPLOAD_DIR, CLIPS_DIR, THUMBNAILS_DIR,
    PERSON_CROPS_DIR, VEHICLE_CROPS_DIR, FACE_CROPS_DIR, PLATE_CROPS_DIR, KEYFRAMES_DIR, REGIONS_DIR
)

LOGGER = logging.getLogger(__name__)

# Default JPEG quality (preserved at 95)
JPEG_QUALITY = 95
THUMBNAIL_QUALITY = 95

class MediaStorage:
    _write_queue: Optional[queue.Queue] = None
    _worker_threads: List[threading.Thread] = []
    _writer_stop: bool = False
    _class_lock: threading.Lock = threading.Lock()
    _created_dirs: set = set()
    _write_errors: Dict[str, str] = {}
    _errors_lock: threading.Lock = threading.Lock()

    @classmethod
    def _ensure_dir_cached(cls, target_dir: Path) -> None:
        """Caches already-created directory paths to prevent repeated NTFS mkdir syscalls."""
        dir_key = str(target_dir)
        if dir_key not in cls._created_dirs:
            with cls._class_lock:
                if dir_key not in cls._created_dirs:
                    target_dir.mkdir(parents=True, exist_ok=True)
                    cls._created_dirs.add(dir_key)

    def __init__(self):
        self.base_dir = MEDIA_DIR.resolve()
        self.ensure_dirs()
        self._ensure_writer_started()

    @classmethod
    def _ensure_writer_started(cls, num_workers: int = 2) -> None:
        """Starts a bounded pool of background disk writer daemon threads."""
        with cls._class_lock:
            if cls._write_queue is None:
                cls._write_queue = queue.Queue(maxsize=2048)
                cls._writer_stop = False
                cls._worker_threads = []
                for i in range(num_workers):
                    t = threading.Thread(
                        target=cls._disk_writer_loop,
                        daemon=True,
                        name=f"MediaStorageDiskWorker-{i}"
                    )
                    t.start()
                    cls._worker_threads.append(t)
                LOGGER.info("MediaStorage initialized %d background disk I/O workers (queue maxsize=2048)", num_workers)

    @classmethod
    def _disk_writer_loop(cls) -> None:
        """Worker thread pulling image write tasks and executing cv2.imwrite off inference threads."""
        while not cls._writer_stop:
            try:
                task = cls._write_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if task is None:
                cls._write_queue.task_done()
                break
            dest_path, image_data, encode_params = task
            try:
                success = cv2.imwrite(str(dest_path), image_data, encode_params)
                if not success:
                    with cls._errors_lock:
                        cls._write_errors[str(dest_path)] = "cv2.imwrite returned False"
                    LOGGER.error(f"cv2.imwrite failed to write: {dest_path}")
            except Exception as e:
                with cls._errors_lock:
                    cls._write_errors[str(dest_path)] = str(e)
                LOGGER.error(f"Async disk write error for {dest_path}: {e}")
            finally:
                cls._write_queue.task_done()

    @classmethod
    def flush_and_wait(cls, timeout: float = 15.0) -> bool:
        """Waits until all pending image writes are flushed to disk."""
        if cls._write_queue is None:
            return True
        start = time.time()
        while cls._write_queue.unfinished_tasks > 0:
            if time.time() - start > timeout:
                LOGGER.warning(f"MediaStorage write queue flush timed out with {cls._write_queue.unfinished_tasks} tasks pending.")
                return False
            time.sleep(0.01)
        return True

    @classmethod
    def wait_for_paths(cls, paths: List[Union[str, Path]], timeout: float = 10.0) -> bool:
        """Ensures that specific image files are completely written to disk before database commits."""
        start = time.time()
        for p in paths:
            if not p:
                continue
            p_obj = Path(p)
            while not (p_obj.exists() and p_obj.stat().st_size > 0):
                if time.time() - start > timeout:
                    LOGGER.warning(f"wait_for_paths timed out waiting for: {p}")
                    return False
                time.sleep(0.01)
        return True

    @classmethod
    def get_and_clear_write_errors(cls) -> Dict[str, str]:
        """Returns any recent write errors and resets the error registry."""
        with cls._errors_lock:
            errs = cls._write_errors.copy()
            cls._write_errors.clear()
            return errs

    @classmethod
    def stop(cls) -> None:
        """Flushes remaining writes and stops all background writer threads."""
        cls.flush_and_wait(timeout=5.0)
        cls._writer_stop = True
        if cls._write_queue is not None:
            for _ in cls._worker_threads:
                try:
                    cls._write_queue.put_nowait(None)
                except Exception:
                    pass
        for t in cls._worker_threads:
            t.join(timeout=2.0)
        cls._worker_threads.clear()

    def ensure_dirs(self) -> None:
        """Create media directory layout."""
        for folder in [
            UPLOAD_DIR, CLIPS_DIR, THUMBNAILS_DIR,
            PERSON_CROPS_DIR, VEHICLE_CROPS_DIR, FACE_CROPS_DIR, PLATE_CROPS_DIR, KEYFRAMES_DIR, REGIONS_DIR
        ]:
            folder.mkdir(parents=True, exist_ok=True)

    def sanitize_path(self, target_path: Union[str, Path]) -> Path:
        """Enforces that all operations occur strictly inside the base media directory to prevent directory traversal attacks."""
        norm_str = os.path.abspath(os.path.normpath(str(target_path)))
        base_str = str(self.base_dir)
        if not norm_str.lower().startswith(base_str.lower()):
            raise ValueError(f"Path traversal detected! Path outside of media directory: {norm_str}")
        return Path(norm_str)

    def save_upload(self, temp_file_path: Union[str, Path], filename: str) -> Path:
        """Saves a uploaded video file into the upload folder."""
        # Sanitize filename to prevent directory traversal
        clean_filename = Path(filename).name
        dest = UPLOAD_DIR / clean_filename
        dest = self.sanitize_path(dest)
        
        # Copy file
        shutil.copy(str(temp_file_path), str(dest))
        return dest

    def get_crop_dir(self, entity_type: str) -> Path:
        """Get crop directory based on entity type."""
        t = entity_type.lower().strip()
        if t == "person":
            return PERSON_CROPS_DIR
        elif t == "vehicle":
            return VEHICLE_CROPS_DIR
        elif t == "face":
            return FACE_CROPS_DIR
        elif t == "plate":
            return PLATE_CROPS_DIR
        else:
            raise ValueError(f"Unsupported entity type for crop: {entity_type}")

    def save_crop(
        self,
        image_data: np.ndarray,
        entity_type: str,
        job_id: str,
        camera_id: str,
        track_id: Union[int, str],
        frame_idx: int,
        sync: bool = False
    ) -> Path:
        """Saves a cropped BGR numpy image to the filesystem with a predictable path structure."""
        crop_dir = self.get_crop_dir(entity_type)
        
        # Build clean directory structure: crops_dir / job_id / camera_id
        # Sanitize parts to prevent injection/traversal
        clean_job = Path(job_id).name
        clean_cam = Path(camera_id).name
        
        target_dir = crop_dir / clean_job / clean_cam
        self._ensure_dir_cached(target_dir)
        
        filename = f"track_{track_id}_frame_{frame_idx}.jpg"
        dest_path = target_dir / filename
        dest_path = self.sanitize_path(dest_path)
        
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        if sync or self._write_queue is None:
            success = cv2.imwrite(str(dest_path), image_data, encode_params)
            if not success:
                raise IOError(f"Failed to write image crop to: {dest_path}")
        else:
            try:
                self._write_queue.put((dest_path, image_data, encode_params), block=True, timeout=5.0)
            except queue.Full:
                LOGGER.warning(f"MediaStorage write queue full! Executing synchronous fallback for: {dest_path}")
                cv2.imwrite(str(dest_path), image_data, encode_params)
            
        return dest_path

    def save_thumbnail(
        self,
        image_data: np.ndarray,
        job_id: str,
        camera_id: str,
        frame_idx: int,
        sync: bool = False
    ) -> Path:
        """Saves a downsampled frame thumbnail."""
        clean_job = Path(job_id).name
        clean_cam = Path(camera_id).name
        
        target_dir = THUMBNAILS_DIR / clean_job / clean_cam
        self._ensure_dir_cached(target_dir)
        
        filename = f"frame_{frame_idx}.jpg"
        dest_path = target_dir / filename
        dest_path = self.sanitize_path(dest_path)
        
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), THUMBNAIL_QUALITY]
        if sync or self._write_queue is None:
            success = cv2.imwrite(str(dest_path), image_data, encode_params)
            if not success:
                raise IOError(f"Failed to write thumbnail to: {dest_path}")
        else:
            try:
                self._write_queue.put((dest_path, image_data, encode_params), block=True, timeout=5.0)
            except queue.Full:
                LOGGER.warning(f"MediaStorage write queue full! Executing synchronous fallback for thumbnail: {dest_path}")
                cv2.imwrite(str(dest_path), image_data, encode_params)
            
        return dest_path

    def save_keyframe(
        self,
        image_data: np.ndarray,
        job_id: str,
        camera_id: str,
        frame_idx: int,
        timestamp: float,
        sync: bool = False
    ) -> Path:
        """Saves a full-frame scene keyframe to the keyframes directory."""
        clean_job = Path(job_id).name
        clean_cam = Path(camera_id).name
        
        target_dir = KEYFRAMES_DIR / clean_job / clean_cam
        self._ensure_dir_cached(target_dir)
        
        filename = f"keyframe_f{frame_idx}_t{int(timestamp * 1000)}.jpg"
        dest_path = target_dir / filename
        dest_path = self.sanitize_path(dest_path)
        
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        if sync or self._write_queue is None:
            success = cv2.imwrite(str(dest_path), image_data, encode_params)
            if not success:
                raise IOError(f"Failed to write scene keyframe to: {dest_path}")
        else:
            try:
                self._write_queue.put((dest_path, image_data, encode_params), block=True, timeout=5.0)
            except queue.Full:
                LOGGER.warning(f"MediaStorage write queue full! Executing synchronous fallback for keyframe: {dest_path}")
                cv2.imwrite(str(dest_path), image_data, encode_params)
            
        return dest_path

    def save_region(
        self,
        image_data: np.ndarray,
        job_id: str,
        camera_id: str,
        frame_idx: int,
        tile_idx: int,
        sync: bool = False
    ) -> Path:
        """Saves a spatial region/tile crop to the regions directory."""
        clean_job = Path(job_id).name
        clean_cam = Path(camera_id).name
        
        target_dir = REGIONS_DIR / clean_job / clean_cam
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"region_f{frame_idx}_t{tile_idx}.jpg"
        dest_path = target_dir / filename
        dest_path = self.sanitize_path(dest_path)
        
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        if sync or self._write_queue is None:
            success = cv2.imwrite(str(dest_path), image_data, encode_params)
            if not success:
                raise IOError(f"Failed to write region crop to: {dest_path}")
        else:
            try:
                self._write_queue.put((dest_path, image_data, encode_params), block=True, timeout=5.0)
            except queue.Full:
                LOGGER.warning(f"MediaStorage write queue full! Executing synchronous fallback for region: {dest_path}")
                cv2.imwrite(str(dest_path), image_data, encode_params)
            
        return dest_path

    def get_clip_path(self, job_id: str, camera_id: str, track_id: Union[int, str]) -> Path:
        """Generate structured path for generated video clips."""
        clean_job = Path(job_id).name
        clean_cam = Path(camera_id).name
        
        target_dir = CLIPS_DIR / clean_job / clean_cam
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"track_{track_id}.mp4"
        return self.sanitize_path(target_dir / filename)

    def cleanup_old_files(self, max_age_days: int = 15) -> List[Path]:
        """Deletes files older than a specified number of days."""
        deleted_files = []
        now = time.time()
        max_age_sec = max_age_days * 24 * 60 * 60
        
        for root, dirs, files in os.walk(str(self.base_dir)):
            for file in files:
                filepath = Path(root) / file
                # Skip original uploads unless explicitly requested
                if str(filepath).startswith(str(UPLOAD_DIR)):
                    continue
                try:
                    mtime = filepath.stat().st_mtime
                    if (now - mtime) > max_age_sec:
                        filepath.unlink()
                        deleted_files.append(filepath)
                except Exception as e:
                    LOGGER.error(f"Error deleting file {filepath}: {e}")
                    
        return deleted_files

    def cleanup_disk_space(self, target_free_percent: float = 15.0) -> List[Path]:
        """Automatically cleans up crops and thumbnails if free disk space falls below a percentage threshold."""
        deleted_files = []
        total, used, free = shutil.disk_usage(str(self.base_dir))
        free_percent = (free / total) * 100.0
        
        if free_percent >= target_free_percent:
            return deleted_files
            
        LOGGER.warning(f"Disk free space ({free_percent:.1f}%) is below target ({target_free_percent}%). Initiating cleanup.")
        
        # Find all files with their mtime, sorting oldest first
        candidates = []
        for root, dirs, files in os.walk(str(self.base_dir)):
            for file in files:
                filepath = Path(root) / file
                # Protect original uploads
                if str(filepath).startswith(str(UPLOAD_DIR)):
                    continue
                try:
                    candidates.append((filepath, filepath.stat().st_mtime))
                except Exception:
                    continue
                    
        candidates.sort(key=lambda x: x[1])
        
        for filepath, _ in candidates:
            try:
                filepath.unlink()
                deleted_files.append(filepath)
                
                # Check space again
                total, used, free = shutil.disk_usage(str(self.base_dir))
                free_percent = (free / total) * 100.0
                if free_percent >= target_free_percent:
                    LOGGER.info(f"Disk space recovered to {free_percent:.1f}%. Stopping cleanup.")
                    break
            except Exception as e:
                LOGGER.error(f"Error removing {filepath} during space recovery: {e}")
                
        return deleted_files

    def delete_job_media(self, job_id: str) -> None:
        """Deletes all crop, thumbnail, and clip folders associated with a specific job ID."""
        clean_job = Path(job_id).name
        
        # Crops
        for crop_type in ["person", "vehicle", "face", "plate"]:
            try:
                crop_dir = self.get_crop_dir(crop_type)
                job_crop_path = crop_dir / clean_job
                if job_crop_path.exists():
                    shutil.rmtree(str(job_crop_path))
            except Exception as me:
                LOGGER.warning(f"Failed to delete crop dir for {crop_type}: {me}")
                
        # Thumbnails, Clips, Keyframes & Regions
        for folder in [THUMBNAILS_DIR / clean_job, CLIPS_DIR / clean_job, KEYFRAMES_DIR / clean_job, REGIONS_DIR / clean_job]:
            try:
                if folder.exists():
                    shutil.rmtree(str(folder))
            except Exception as fe:
                LOGGER.warning(f"Failed to delete media folder {folder}: {fe}")
