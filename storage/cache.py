"""Track and Scene Caching Module for AICatch.

Caches frame crops of active tracks, calculates frame sharpness,
and identifies static frames to prevent duplicate inference.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
import cv2

LOGGER = logging.getLogger(__name__)

class TrackData:
    """Stores temporary frame crops and sharpness scores for an active track."""
    def __init__(self, track_id: int, camera_id: str, start_time: float):
        self.track_id = track_id
        self.camera_id = camera_id
        self.start_time = start_time
        self.end_time = start_time
        
        # Chronological list of frame metadata and crops
        self.history: List[Dict[str, Any]] = []
        
        # Best crop (by sharpness)
        self.best_sharpness = -1.0
        self.best_crop: Optional[np.ndarray] = None
        self.best_frame_idx: int = -1
        self.best_timestamp: float = 0.0
        self.best_bbox: List[float] = []
        
        # Embedding caches (to prevent re-calculating similar views)
        self.clip_embedding: Optional[List[float]] = None
        self.face_embedding: Optional[List[float]] = None
        self.plate_text: Optional[str] = None
        
        # Counter for conditional inference throttling
        self.ocr_run_count = 0
        self.face_run_count = 0
        
        # Performance/Accuracy indexing markers
        self.last_embedded_sharpness = -1.0
        self.last_embedded_size = -1.0
        self.last_embedded_time = -1.0
        self.embedded_count = 0
        
        # Object-level attributes
        self.quality_score = 0.0
        self.dominant_upper_color: Optional[str] = None
        self.secondary_upper_color: Optional[str] = None
        self.dominant_lower_color: Optional[str] = None
        self.secondary_lower_color: Optional[str] = None
        self.color_confidence = 0.0
        self.is_monochrome = False

    def add_frame(
        self,
        frame_idx: int,
        timestamp: float,
        bbox: List[float],
        crop: np.ndarray,
        sharpness: float
    ) -> None:
        """Appends frame metadata and updates the keyframe based on visual sharpness."""
        self.end_time = timestamp
        
        # Record frame entry in track history (omit raw image data to conserve RAM)
        self.history.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "bbox": bbox,
            "sharpness": sharpness
        })
        
        # Maintain a maximum historical log depth to prevent memory growth
        if len(self.history) > 120:
            self.history.pop(0)

        # Update best keyframe if this frame is sharper
        if sharpness > self.best_sharpness or self.best_crop is None:
            self.best_sharpness = sharpness
            # Store a copy of the crop image (BGR numpy array)
            self.best_crop = crop.copy()
            self.best_frame_idx = frame_idx
            self.best_timestamp = timestamp
            self.best_bbox = bbox

class TrackCache:
    """Manages active track histories and metadata across multiple cameras."""
    def __init__(self):
        # camera_id -> {track_id -> TrackData}
        self.active_tracks: Dict[str, Dict[int, TrackData]] = {}
        
        # camera_id -> background gray image for static check
        self.background_refs: Dict[str, np.ndarray] = {}

    def get_or_create_track(
        self,
        camera_id: str,
        track_id: int,
        start_time: float
    ) -> TrackData:
        """Fetch or initialize a TrackData record."""
        if camera_id not in self.active_tracks:
            self.active_tracks[camera_id] = {}
            
        if track_id not in self.active_tracks[camera_id]:
            self.active_tracks[camera_id][track_id] = TrackData(track_id, camera_id, start_time)
            
        return self.active_tracks[camera_id][track_id]

    def add_to_track(
        self,
        camera_id: str,
        track_id: int,
        frame_idx: int,
        timestamp: float,
        bbox: List[float],
        crop: np.ndarray
    ) -> float:
        """Appends raw detection bounds to an active track, returning its calculated sharpness."""
        # Calculate Laplacian variance as a sharpness metric
        sharpness = self.calculate_sharpness(crop)
        
        track = self.get_or_create_track(camera_id, track_id, timestamp)
        track.add_frame(frame_idx, timestamp, bbox, crop, sharpness)
        return sharpness

    def remove_track(self, camera_id: str, track_id: int) -> Optional[TrackData]:
        """Cleans up and returns track metadata when a track terminates."""
        if camera_id in self.active_tracks and track_id in self.active_tracks[camera_id]:
            return self.active_tracks[camera_id].pop(track_id)
        return None

    def clear_camera_tracks(self, camera_id: str) -> List[TrackData]:
        """Clears and returns all remaining active tracks on a camera feed."""
        if camera_id in self.active_tracks:
            tracks = list(self.active_tracks[camera_id].values())
            self.active_tracks[camera_id] = {}
            return tracks
        return []

    # ==============================================================================
    # Static Scene Check Caching
    # ==============================================================================

    def update_background(self, camera_id: str, gray_frame: np.ndarray) -> None:
        """Downsamples and stores reference background frame for static scene filtering."""
        # Resize to 320x180 to keep calculation fast and low VRAM/RAM
        resized = cv2.resize(gray_frame, (320, 180))
        self.background_refs[camera_id] = cv2.GaussianBlur(resized, (5, 5), 0)

    def is_scene_static(self, camera_id: str, gray_frame: np.ndarray, threshold: float = 0.008) -> bool:
        """Compares current frame against cached background to check if the scene is static."""
        if camera_id not in self.background_refs:
            self.update_background(camera_id, gray_frame)
            return False
            
        resized = cv2.resize(gray_frame, (320, 180))
        current_blur = cv2.GaussianBlur(resized, (5, 5), 0)
        
        ref = self.background_refs[camera_id]
        diff = cv2.absdiff(ref, current_blur)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = thresh.size
        ratio = changed_pixels / total_pixels
        
        # Update background reference progressively to handle slow lighting changes
        cv2.addWeighted(ref, 0.95, current_blur, 0.05, 0, dst=ref)
        
        return ratio < threshold

    # ==============================================================================
    # Image Utility Metrics
    # ==============================================================================

    @staticmethod
    def calculate_sharpness(crop_image: np.ndarray) -> float:
        """Computes Laplacian variance of the crop to estimate blurriness."""
        if crop_image is None or crop_image.size == 0:
            return 0.0
        try:
            # Convert to grayscale if it is BGR
            if len(crop_image.shape) == 3:
                gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop_image
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0
