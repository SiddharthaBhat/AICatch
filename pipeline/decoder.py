"""FFmpeg CPU-based Video Decoder for AICatch.

Handles stream metadata extraction, high-speed asynchronous frame decoding,
backpressure, frame-dropping for live feeds, and reconnection with backoff.
"""

import subprocess
import threading
import queue
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np

from config import (
    MAX_RTSP_RECONNECT_ATTEMPTS, RTSP_RECONNECT_BACKOFF_FACTOR, GPU_QUEUE_LIMIT
)

LOGGER = logging.getLogger(__name__)

class VideoStreamDecoder(threading.Thread):
    def __init__(
        self,
        camera_id: str,
        source_path: str,
        is_live: bool = False,
        target_width: int = 960,
        target_fps: float = None
    ):
        super().__init__(name=f"Decoder-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source_path = source_path
        self.is_live = is_live
        self.target_width = target_width
        self.target_fps = target_fps
        
        # Bounded frame queue: holds (frame_idx, timestamp, frame_bgr)
        self.frame_queue = queue.Queue(maxsize=GPU_QUEUE_LIMIT)
        
        # Control flags
        self.stopped = False
        self.paused = False
        self._pause_cond = threading.Condition(threading.Lock())
        
        # Stream properties
        self.width = 0
        self.height = 0
        self.target_height = 0
        self.fps = 25.0
        self.total_frames = 0
        self.duration_sec = 0.0
        
        # Stats tracking
        self.frames_decoded = 0
        self.frames_dropped = 0
        self.is_connected = False
        self.failed = False
        self.error_msg = ""
        
        # FFmpeg executable resolve
        self.ffmpeg_exe = self._resolve_ffmpeg()
        
        # Extract initial properties using OpenCV
        self._probe_stream()

    def _resolve_ffmpeg(self) -> str:
        """Finds FFmpeg executable on the system or falls back to imageio-ffmpeg."""
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
            
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe:
                return exe
        except ImportError:
            pass
            
        raise RuntimeError("FFmpeg executable could not be found. Please install FFmpeg or imageio-ffmpeg.")

    def _probe_stream(self) -> None:
        """Inspects stream properties using OpenCV."""
        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Unable to open video source: {self.source_path}")
            
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if self.target_fps is None:
            self.target_fps = self.fps
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.total_frames < 0:
            self.total_frames = 0
            
        if self.fps > 0:
            self.duration_sec = self.total_frames / self.fps
        cap.release()
        
        # Calculate target height keeping aspect ratio, must be divisible by 2
        # Use native width if it is smaller than 960 (or target_width), otherwise downscale to target_width
        self.target_width = min(self.width, self.target_width)
        self.target_height = int((self.height / self.width) * self.target_width)
        if self.target_height % 2 != 0:
            self.target_height += 1
            
        LOGGER.info(
            f"Probed camera {self.camera_id}: {self.width}x{self.height} -> "
            f"{self.target_width}x{self.target_height} @ {self.fps:.2f} FPS. "
            f"Frames: {self.total_frames}"
        )

    def run(self) -> None:
        """Decoder loop."""
        reconnect_attempts = 0
        
        while not self.stopped:
            try:
                self._decode_loop()
                # If we exit the loop and it is not stopped, it means the stream ended or failed
                if self.stopped:
                    break
                    
                if not self.is_live:
                    LOGGER.info(f"Finished file ingestion for {self.camera_id}.")
                    break
                
                # RTSP stream disconnect recovery
                self.is_connected = False
                reconnect_attempts += 1
                if reconnect_attempts > MAX_RTSP_RECONNECT_ATTEMPTS:
                    LOGGER.error(f"Camera {self.camera_id} exceeded maximum RTSP reconnection attempts.")
                    break
                    
                backoff = RTSP_RECONNECT_BACKOFF_FACTOR ** reconnect_attempts
                LOGGER.warning(f"RTSP connection lost for {self.camera_id}. Reconnecting in {backoff:.1f}s...")
                
                # Check for stop signal while waiting to reconnect
                for _ in range(int(backoff * 10)):
                    if self.stopped:
                        break
                    time.sleep(0.1)
                    
            except Exception as e:
                LOGGER.error(f"Exception in decoder thread {self.camera_id}: {e}", exc_info=True)
                if not self.is_live:
                    break
                time.sleep(2.0)

    def _decode_loop(self) -> None:
        """Spawns FFmpeg and pipes BGR frames into the queue."""
        # Clean FFmpeg arguments
        cmd = [
            self.ffmpeg_exe,
            "-loglevel", "error",
            "-i", self.source_path,
            "-r", str(self.target_fps),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={self.target_width}:{self.target_height}",
            "-"
        ]
        
        # Configure process startup info to hide cmd window on Windows
        startupinfo = None
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 11  # SW_HIDE
            
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            bufsize=self.target_width * self.target_height * 3 * 2,  # 2 frames buffer
            startupinfo=startupinfo
        )
        
        self.is_connected = True
        frame_size = self.target_width * self.target_height * 3
        frame_idx = 0
        
        try:
            while not self.stopped:
                # Handle pause state (block decoder)
                with self._pause_cond:
                    while self.paused and not self.stopped:
                        self._pause_cond.wait(timeout=0.1)
                        if self.is_live:
                            # Flush RTSP stream buffer during pause by reading and discarding
                            # To keep layout clean we just break and reconnect on resume,
                            # or read and drop. Reading and dropping is safer.
                            proc.stdout.read(frame_size)
                            
                raw_bytes = proc.stdout.read(frame_size)
                if len(raw_bytes) != frame_size:
                    LOGGER.info(f"Stream or file EOF reached for camera {self.camera_id}.")
                    break
                    
                frame_idx += 1
                if frame_idx == 1:
                    LOGGER.info(f"MILESTONE: FIRST FRAME DECODED at {time.time()}")
                self.frames_decoded += 1
                timestamp = frame_idx / self.target_fps
                
                # Convert raw bytes to numpy image
                frame_bgr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
                    (self.target_height, self.target_width, 3)
                )
                
                # Dynamic Adaptive Sampling check
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (160, 90))
                resized_blur = cv2.GaussianBlur(resized, (5, 5), 0)
                
                if not hasattr(self, "last_gray_frame"):
                    self.last_gray_frame = resized_blur
                    self.last_pushed_time = 0.0
                    self.motion_cooldown_until = 0.0
                    
                is_static = True
                if self.last_gray_frame is not None:
                    diff = cv2.absdiff(self.last_gray_frame, resized_blur)
                    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
                    changed_ratio = cv2.countNonZero(thresh) / thresh.size
                    
                    # Slowly update the background reference frame
                    cv2.addWeighted(self.last_gray_frame, 0.95, resized_blur, 0.05, 0, dst=self.last_gray_frame)
                    
                    if changed_ratio >= 0.005:
                        is_static = False
                        self.motion_cooldown_until = timestamp + 4.0
                else:
                    self.last_gray_frame = resized_blur
                    is_static = False
                    
                should_push = True
                in_motion = (timestamp < self.motion_cooldown_until) or (not is_static)
                
                if not in_motion:
                    # In static scenes, limit pushes to once every 4.0s (heartbeat)
                    if (timestamp - self.last_pushed_time) < 4.0:
                        should_push = False
                        
                if should_push:
                    self.last_pushed_time = timestamp
                    data_packet = (frame_idx, timestamp, frame_bgr)
                    
                    # Apply backpressure / dropping strategy
                    if self.is_live:
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                                self.frames_dropped += 1
                            except queue.Empty:
                                pass
                        self.frame_queue.put(data_packet)
                    else:
                        self.frame_queue.put(data_packet, block=True)
                        if frame_idx == 1:
                            LOGGER.info(f"MILESTONE: FIRST FRAME ENTERS QUEUE at {time.time()}")
                else:
                    self.frames_dropped += 1
                    
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def pause_stream(self) -> None:
        """Pause frame decoding."""
        with self._pause_cond:
            self.paused = True
            LOGGER.info(f"Decoder {self.camera_id} paused.")

    def resume_stream(self) -> None:
        """Resume frame decoding."""
        with self._pause_cond:
            self.paused = False
            self._pause_cond.notify_all()
            LOGGER.info(f"Decoder {self.camera_id} resumed.")

    def stop_stream(self) -> None:
        """Stop frame decoding and join thread."""
        self.stopped = True
        with self._pause_cond:
            self.paused = False
            self._pause_cond.notify_all()
        # Flush queue to unblock any blocking puts
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        LOGGER.info(f"Decoder {self.camera_id} stop requested.")
