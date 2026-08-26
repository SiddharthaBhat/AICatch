"""Performance Monitor Utility for AICatch.

Tracks FPS, queue depths, model latency metrics, and hardware utilization
(CPU, RAM, GPU, VRAM) on Windows.
"""

import time
import os
import logging
from typing import Dict, Any, List
import threading
import psutil

LOGGER = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Telemetry metrics dictionary
        self.metrics: Dict[str, Any] = {
            "processing_fps": 0.0,
            "queue_depth": 0,
            "cpu_util": 0.0,
            "gpu_util": 0.0,
            "vram_used_mb": 0.0,
            "ram_used_gb": 0.0,
            "latency_yolo": 0.0,
            "latency_clip": 0.0,
            "latency_ocr": 0.0,
            "latency_face": 0.0,
            "latency_db": 0.0,
            "frames_decoded": 0,
            "frames_analyzed": 0,
            "detections_count": 0,
            "embeddings_count": 0
        }
        
        # In-memory history for smoothing metrics
        self._latencies: Dict[str, List[float]] = {
            "yolo": [], "clip": [], "ocr": [], "face": [], "db": []
        }
        self._fps_timestamps: List[float] = []
        self._monitor_thread_stop = False
        
        # Start background hardware utilization sampler
        self._sampler_thread = threading.Thread(target=self._hardware_sampler, daemon=True)
        self._sampler_thread.start()

    def record_frame_processed(self) -> None:
        """Records a frame process event to calculate rolling FPS."""
        with self.lock:
            now = time.time()
            self._fps_timestamps.append(now)
            self.metrics["frames_analyzed"] += 1
            
            # Keep timestamps from the last 10 seconds to compute average FPS
            cutoff = now - 10.0
            self._fps_timestamps = [t for t in self._fps_timestamps if t > cutoff]
            
            if len(self._fps_timestamps) > 1:
                duration = now - self._fps_timestamps[0]
                if duration > 0:
                    self.metrics["processing_fps"] = round(len(self._fps_timestamps) / duration, 2)
            else:
                self.metrics["processing_fps"] = 0.0

    def record_latency(self, model_name: str, duration_sec: float) -> None:
        """Records model execution latency, keeping a running average of the last 20 runs."""
        with self.lock:
            if model_name in self._latencies:
                history = self._latencies[model_name]
                history.append(duration_sec)
                if len(history) > 20:
                    history.pop(0)
                
                # Calculate smoothed average latency in milliseconds
                self.metrics[f"latency_{model_name}"] = round((sum(history) / len(history)) * 1000.0, 2)

    def update_counts(self, detections: int = 0, embeddings: int = 0, decoded: int = 0) -> None:
        """Updates cumulative ingestion and discovery counters."""
        with self.lock:
            self.metrics["detections_count"] += detections
            self.metrics["embeddings_count"] += embeddings
            self.metrics["frames_decoded"] += decoded

    def update_queue_depth(self, depth: int) -> None:
        """Records current frame queue depth."""
        with self.lock:
            self.metrics["queue_depth"] = depth

    def get_metrics(self) -> Dict[str, Any]:
        """Returns a snapshot of the current telemetry performance metrics."""
        with self.lock:
            return self.metrics.copy()

    def stop(self) -> None:
        """Stops the hardware utilization sampler thread."""
        self._monitor_thread_stop = True
        if self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=2.0)

    def _hardware_sampler(self) -> None:
        """Periodically samples CPU, RAM, and GPU utilization metrics on Windows."""
        # Optional: load GPUtil to sample NVIDIA GPU metrics on Windows
        gputil_available = False
        try:
            import GPUtil
            gputil_available = True
        except ImportError:
            pass

        process = psutil.Process(os.getpid())
        
        while not self._monitor_thread_stop:
            try:
                # 1. CPU & RAM utilization
                cpu_percent = psutil.cpu_percent(interval=None)
                # Read RAM used by the python process in GB
                ram_gb = process.memory_info().rss / (1024 ** 3)
                
                # 2. NVIDIA GPU metrics
                gpu_load = 0.0
                vram_used = 0.0
                
                if gputil_available:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            # Target first GPU (RTX 3050)
                            gpu = gpus[0]
                            gpu_load = gpu.load * 100.0
                            vram_used = gpu.memoryUsed  # MB
                    except Exception as ge:
                        LOGGER.debug(f"Failed to fetch GPUtil stats: {ge}")
                else:
                    # Fallback if GPUtil is not installed, parse nvidia-smi if available
                    # Since we want to keep it fast, we can fallback to 0 or check torch memory
                    try:
                        import torch
                        if torch.cuda.is_available():
                            vram_used = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    except Exception:
                        pass
                
                with self.lock:
                    self.metrics["cpu_util"] = round(cpu_percent, 1)
                    self.metrics["ram_used_gb"] = round(ram_gb, 2)
                    self.metrics["gpu_util"] = round(gpu_load, 1)
                    self.metrics["vram_used_mb"] = round(vram_used, 1)
                    
            except Exception as e:
                LOGGER.error(f"Error sampling hardware metrics: {e}")
                
            # Sample every 1.5 seconds
            time.sleep(1.5)
PerformanceMonitorInstance = PerformanceMonitor()
