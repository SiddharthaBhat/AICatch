"""Media Storage Manager for AICatch.

Handles security verification, directory structure organization, image saving,
and disk space cleanup policies for raw and processed assets.
"""

from pathlib import Path
import shutil
import logging
import time
import os
from typing import Union, List
import cv2
import numpy as np

from config import (
    MEDIA_DIR, UPLOAD_DIR, CLIPS_DIR, THUMBNAILS_DIR,
    PERSON_CROPS_DIR, VEHICLE_CROPS_DIR, FACE_CROPS_DIR, PLATE_CROPS_DIR, KEYFRAMES_DIR
)

LOGGER = logging.getLogger(__name__)

class MediaStorage:
    def __init__(self):
        self.base_dir = MEDIA_DIR.resolve()
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        """Create media directory layout."""
        for folder in [
            UPLOAD_DIR, CLIPS_DIR, THUMBNAILS_DIR,
            PERSON_CROPS_DIR, VEHICLE_CROPS_DIR, FACE_CROPS_DIR, PLATE_CROPS_DIR, KEYFRAMES_DIR
        ]:
            folder.mkdir(parents=True, exist_ok=True)

    def sanitize_path(self, target_path: Union[str, Path]) -> Path:
        """Enforces that all operations occur strictly inside the base media directory to prevent directory traversal attacks."""
        resolved = Path(target_path).resolve()
        if not str(resolved).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected! Path outside of media directory: {resolved}")
        return resolved

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
        frame_idx: int
    ) -> Path:
        """Saves a cropped BGR numpy image to the filesystem with a predictable path structure."""
        crop_dir = self.get_crop_dir(entity_type)
        
        # Build clean directory structure: crops_dir / job_id / camera_id
        # Sanitize parts to prevent injection/traversal
        clean_job = Path(job_id).name
        clean_cam = Path(camera_id).name
        
        target_dir = crop_dir / clean_job / clean_cam
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"track_{track_id}_frame_{frame_idx}.jpg"
        dest_path = target_dir / filename
        dest_path = self.sanitize_path(dest_path)
        
        # Save image using opencv
        success = cv2.imwrite(str(dest_path), image_data)
        if not success:
            raise IOError(f"Failed to write image crop to: {dest_path}")
            
        return dest_path

    def save_thumbnail(
        self,
        image_data: np.ndarray,
        job_id: str,
        camera_id: str,
        frame_idx: int
    ) -> Path:
        """Saves a downsampled frame thumbnail."""
        clean_job = Path(job_id).name
        clean_cam = Path(camera_id).name
        
        target_dir = THUMBNAILS_DIR / clean_job / clean_cam
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"frame_{frame_idx}.jpg"
        dest_path = target_dir / filename
        dest_path = self.sanitize_path(dest_path)
        
        # Save image
        success = cv2.imwrite(str(dest_path), image_data)
        if not success:
            raise IOError(f"Failed to write thumbnail to: {dest_path}")
            
        return dest_path

    def save_keyframe(
        self,
        image_data: np.ndarray,
        job_id: str,
        camera_id: str,
        frame_idx: int,
        timestamp: float
    ) -> Path:
        """Saves a full-frame scene keyframe to the keyframes directory."""
        clean_job = Path(job_id).name
        clean_cam = Path(camera_id).name
        
        target_dir = KEYFRAMES_DIR / clean_job / clean_cam
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"keyframe_f{frame_idx}_t{int(timestamp * 1000)}.jpg"
        dest_path = target_dir / filename
        dest_path = self.sanitize_path(dest_path)
        
        success = cv2.imwrite(str(dest_path), image_data)
        if not success:
            raise IOError(f"Failed to write scene keyframe to: {dest_path}")
            
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
                
        # Thumbnails & Clips
        for folder in [THUMBNAILS_DIR / clean_job, CLIPS_DIR / clean_job]:
            try:
                if folder.exists():
                    shutil.rmtree(str(folder))
            except Exception as fe:
                LOGGER.warning(f"Failed to delete media folder {folder}: {fe}")
