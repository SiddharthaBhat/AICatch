# -*- coding: utf-8 -*-
"""AICatch Pipeline Benchmark Suite.

Accepts actual CCTV video paths and measures stage-by-stage latency,
throughput, hardware metrics, and database persistence.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

sys.path.append(os.path.abspath(str(Path(__file__).resolve().parents[1])))

from config import (
    CHROMA_PERSIST_DIR,
    CLIP_DEVICE,
    DEVICE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VERSION,
    FACE_DEVICE,
    OCR_DEVICE,
    REID_DEVICE,
    REID_MODEL_NAME,
    REID_VERSION,
    SQLITE_DB_PATH,
    YOLO_DEVICE,
    YOLO_MODEL_PATH,
)
from pipeline.decoder import VideoStreamDecoder
from pipeline.gpu_scheduler import GPUScheduler, calculate_iou
from pipeline.job_manager import JobManager
from storage.media_storage import MediaStorage
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase
from utils.color_analyzer import evaluate_crop_quality, extract_body_colors, extract_vehicle_colors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
LOGGER = logging.getLogger("PipelineBenchmark")


def measure_hardware() -> Dict[str, Any]:
    """Captures current system hardware metrics."""
    metrics = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": round(psutil.virtual_memory().used / (1024 * 1024), 2),
        "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024), 2),
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE",
        "gpu_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "gpu_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
    }
    return metrics


class PipelineBenchmarkRunner:
    def __init__(
        self,
        video_path: Path,
        duration: Optional[float] = None,
        sampling_fps: float = 2.0,
        batch_size: int = 8,
        warmup_iterations: int = 1,
        repetitions: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        gt_path: Optional[Path] = None,
        output_dir: Path = Path("benchmarks/results"),
        decoder_mode: str = "software",  # 'software' or 'cuvid'
    ):
        self.video_path = video_path
        self.duration = duration
        self.sampling_fps = sampling_fps
        self.batch_size = batch_size
        self.warmup_iterations = warmup_iterations
        self.repetitions = repetitions
        self.device = device
        self.gt_path = gt_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.decoder_mode = decoder_mode

        if not self.video_path.exists():
            raise FileNotFoundError(f"Source video not found: {self.video_path}")

        # Probe video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self.source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_duration = self.source_total_frames / self.source_fps if self.source_fps > 0 else 0.0
        cap.release()

        self.sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
        self.vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
        self.media = MediaStorage()
        self.job_manager = JobManager(self.sqlite_db)
        self.scheduler = GPUScheduler(self.job_manager, self.media, dispatcher=None)

    def warmup(self) -> None:
        """Warms up models and GPU contexts before timing."""
        print(f"[BENCHMARK] Warming up models on {self.device} (warmup_iterations={self.warmup_iterations})...", flush=True)
        self.scheduler.load_models()
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_frame)

        for _ in range(self.warmup_iterations):
            _ = self.scheduler.yolo.predict(dummy_frame, imgsz=640, device=YOLO_DEVICE, verbose=False)
            _ = self.scheduler.embed_image(dummy_pil)
            if self.scheduler.reid_model is not None:
                _ = self.scheduler.embed_reid(dummy_pil)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        print("[BENCHMARK] Warmup complete.", flush=True)

    def run_isolated_decode_benchmark(self) -> Dict[str, Any]:
        """Benchmarks isolated FFmpeg decoding performance."""
        print(f"[BENCHMARK] Running isolated FFmpeg decoder benchmark (mode={self.decoder_mode})...", flush=True)
        import shutil

        exe = shutil.which("ffmpeg")
        if not exe:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()

        target_w = min(self.source_width, 960)
        target_h = int((self.source_height / self.source_width) * target_w)
        if target_h % 2 != 0:
            target_h += 1

        cmd = [exe, "-loglevel", "error"]
        if self.decoder_mode == "cuvid":
            cmd.extend(["-hwaccel", "cuda", "-c:v", "h264_cuvid"])
        cmd.extend([
            "-i", str(self.video_path),
            "-r", str(self.sampling_fps),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={target_w}:{target_h}",
            "-"
        ])

        if self.duration:
            # -t before -i or as output limit
            cmd = [exe, "-loglevel", "error"]
            if self.decoder_mode == "cuvid":
                cmd.extend(["-hwaccel", "cuda", "-c:v", "h264_cuvid"])
            cmd.extend([
                "-t", str(self.duration),
                "-i", str(self.video_path),
                "-r", str(self.sampling_fps),
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-vf", f"scale={target_w}:{target_h}",
                "-"
            ])

        frame_size = target_w * target_h * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_size * 4)

        t0 = time.time()
        frames_read = 0
        cpu_start = psutil.cpu_percent(interval=None)

        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            frames_read += 1

        proc.stdout.close()
        proc.wait()
        total_time = max(0.0001, time.time() - t0)
        cpu_end = psutil.cpu_percent(interval=None)

        decode_fps = round(frames_read / total_time, 2)
        print(f"[BENCHMARK] Isolated Decode: mode={self.decoder_mode}, frames={frames_read}, time={total_time:.3f}s, decode_fps={decode_fps}", flush=True)

        return {
            "decoder_mode": self.decoder_mode,
            "frames_decoded": frames_read,
            "decode_time_sec": round(total_time, 3),
            "decode_fps": decode_fps,
            "cpu_util_approx": round((cpu_start + cpu_end) / 2.0, 1),
            "status": "PASS" if frames_read > 0 else "FAIL"
        }

    def run_pipeline_benchmark(self) -> Dict[str, Any]:
        """Runs the full end-to-end pipeline benchmark with fine-grained stage timings."""
        self.warmup()
        print(f"[BENCHMARK] Running full pipeline benchmark on {self.video_path.name} (fps={self.sampling_fps}, batch={self.batch_size})...", flush=True)

        # Isolated Decode first
        decode_bench = self.run_isolated_decode_benchmark()

        # Frame extraction for pipeline processing
        import shutil
        exe = shutil.which("ffmpeg")
        if not exe:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()

        target_w = min(self.source_width, 960)
        target_h = int((self.source_height / self.source_width) * target_w)
        if target_h % 2 != 0:
            target_h += 1

        cmd = [
            exe, "-loglevel", "error",
            "-i", str(self.video_path),
            "-r", str(self.sampling_fps),
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={target_w}:{target_h}",
            "-"
        ]
        if self.duration:
            cmd = [
                exe, "-loglevel", "error",
                "-t", str(self.duration),
                "-i", str(self.video_path),
                "-r", str(self.sampling_fps),
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-vf", f"scale={target_w}:{target_h}",
                "-"
            ]

        frame_size = target_w * target_h * 3
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_size * 4)

        decoded_frames: List[Tuple[int, float, np.ndarray]] = []
        frame_idx = 0
        t_decode_start = time.time()

        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            frame_idx += 1
            timestamp = frame_idx / self.sampling_fps
            frame_bgr = np.frombuffer(raw, dtype=np.uint8).reshape((target_h, target_w, 3))
            decoded_frames.append((frame_idx, timestamp, frame_bgr))

        proc.stdout.close()
        proc.wait()
        total_decode_time = time.time() - t_decode_start
        print(f"[BENCHMARK] Extracted {len(decoded_frames)} frames for pipeline analysis in {total_decode_time:.3f}s.", flush=True)

        if not decoded_frames:
            raise RuntimeError("No frames decoded from test video")

        stage_timings = {
            "decode": total_decode_time,
            "detection": 0.0,
            "tracking": 0.0,
            "cropping": 0.0,
            "clip": 0.0,
            "reid": 0.0,
            "face": 0.0,
            "ocr": 0.0,
            "sqlite": 0.0,
            "chroma_api": 0.0,
            "chroma_verify": 0.0,
        }

        counts = {
            "decoded_frames": len(decoded_frames),
            "processed_frames": 0,
            "detections": 0,
            "tracks": 0,
            "crops": 0,
            "clip_embeddings": 0,
            "reid_embeddings": 0,
            "face_embeddings": 0,
            "ocr_readings": 0,
            "sqlite_rows": 0,
            "chroma_vectors": 0,
            "chroma_verified": 0,
            "failed_frames": 0,
            "failed_embeddings": 0,
        }

        camera_id = "bench_cam_01"
        job_id = f"bench_job_{int(time.time())}"
        next_track_id = 1
        active_tracks: Dict[int, Any] = {}

        sqlite_records = []
        clip_ids, clip_embs, clip_metas = [], [], []
        reid_ids, reid_embs, reid_metas = [], [], []
        face_ids, face_embs, face_metas = [], [], []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t_pipeline_start = time.time()

        for b_idx in range(0, len(decoded_frames), self.batch_size):
            batch_slice = decoded_frames[b_idx : b_idx + self.batch_size]
            batch_images = [f[2] for f in batch_slice]

            # 1. Detection (YOLO)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            with self.scheduler.gpu_lock:
                yolo_results = self.scheduler.yolo.predict(
                    batch_images,
                    imgsz=640,
                    device=YOLO_DEVICE,
                    verbose=False,
                    conf=0.25
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_timings["detection"] += (time.time() - t0)

            for f_i, (f_idx, ts, img_bgr) in enumerate(batch_slice):
                counts["processed_frames"] += 1
                res = yolo_results[f_i]
                h_img, w_img = img_bgr.shape[:2]

                allowed_classes = {0, 2, 3, 5, 7}
                boxes_to_track = []
                if res.boxes is not None:
                    for box in res.boxes:
                        cls_id = int(box.cls[0].item())
                        if cls_id in allowed_classes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0].item())
                            boxes_to_track.append((x1, y1, x2, y2, conf, cls_id))
                            counts["detections"] += 1

                # 2. Tracking
                t0_tr = time.time()
                matched_dets = set()
                frame_tracked_items = []

                for t_id, tr_data in list(active_tracks.items()):
                    last_box = tr_data["last_box"]
                    best_iou, best_idx = 0.0, -1
                    for d_i, det in enumerate(boxes_to_track):
                        if d_i in matched_dets:
                            continue
                        iou = calculate_iou(last_box, [det[0], det[1], det[2], det[3]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = d_i

                    if best_iou > 0.30:
                        matched_dets.add(best_idx)
                        det = boxes_to_track[best_idx]
                        tr_data["last_box"] = [det[0], det[1], det[2], det[3]]
                        tr_data["end_time"] = ts
                        frame_tracked_items.append((t_id, det, "person" if det[5] == 0 else "vehicle"))

                for d_i, det in enumerate(boxes_to_track):
                    if d_i in matched_dets:
                        continue
                    t_id = next_track_id
                    next_track_id += 1
                    active_tracks[t_id] = {
                        "track_id": t_id,
                        "start_time": ts,
                        "end_time": ts,
                        "last_box": [det[0], det[1], det[2], det[3]],
                        "embedded": False
                    }
                    counts["tracks"] += 1
                    frame_tracked_items.append((t_id, det, "person" if det[5] == 0 else "vehicle"))

                stage_timings["tracking"] += (time.time() - t0_tr)

                # 3. Cropping & Specialists
                for t_id, det, entity_type in frame_tracked_items:
                    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                    w, h = x2 - x1, y2 - y1
                    x1_pad = max(0, int(x1 - 0.15 * w))
                    y1_pad = max(0, int(y1 - 0.15 * h))
                    x2_pad = min(w_img, int(x2 + 0.15 * w))
                    y2_pad = min(h_img, int(y2 + 0.15 * h))

                    t0_cr = time.time()
                    crop = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
                    stage_timings["cropping"] += (time.time() - t0_cr)

                    if crop.size == 0:
                        continue
                    counts["crops"] += 1
                    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                    # 4. CLIP Embedding
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0_clip = time.time()
                    clip_emb = self.scheduler.embed_image(pil_crop)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_timings["clip"] += (time.time() - t0_clip)
                    counts["clip_embeddings"] += 1

                    det_uuid = f"det_{job_id}_{camera_id}_{t_id}_{f_idx}"
                    clip_ids.append(det_uuid)
                    clip_embs.append(clip_emb)
                    clip_metas.append({
                        "job_id": job_id,
                        "investigation_id": job_id,
                        "camera_id": camera_id,
                        "video_id": camera_id,
                        "track_id": f"{job_id}_{camera_id}_{t_id}",
                        "detection_id": det_uuid,
                        "timestamp": ts,
                        "entity_type": entity_type,
                        "object_type": entity_type,
                        "embedding_model": EMBEDDING_MODEL_NAME,
                        "embedding_version": EMBEDDING_VERSION
                    })

                    # 5. MobileNetV3 Re-ID
                    if entity_type == "person" and self.scheduler.reid_model is not None:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t0_reid = time.time()
                        reid_emb = self.scheduler.embed_reid(pil_crop)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        stage_timings["reid"] += (time.time() - t0_reid)
                        if reid_emb:
                            counts["reid_embeddings"] += 1
                            reid_ids.append(f"reid_{det_uuid}")
                            reid_embs.append(reid_emb)
                            reid_metas.append({
                                "job_id": job_id,
                                "investigation_id": job_id,
                                "camera_id": camera_id,
                                "track_id": f"{job_id}_{camera_id}_{t_id}",
                                "detection_id": det_uuid,
                                "timestamp": ts,
                                "entity_type": entity_type,
                                "reid_model": REID_MODEL_NAME,
                                "reid_version": REID_VERSION
                            })

                    # 6. MTCNN & FaceNet (Only on select crops)
                    if entity_type == "person":
                        t0_fc = time.time()
                        face_diag = self.scheduler.embed_face_with_diagnostics(pil_crop)
                        stage_timings["face"] += (time.time() - t0_fc)
                        if face_diag.get("embedding"):
                            counts["face_embeddings"] += 1
                            face_ids.append(f"face_{det_uuid}")
                            face_embs.append(face_diag["embedding"])
                            face_metas.append({
                                "job_id": job_id,
                                "camera_id": camera_id,
                                "track_id": f"{job_id}_{camera_id}_{t_id}",
                                "timestamp": ts
                            })

                    # 7. EasyOCR
                    if entity_type == "vehicle" and self.scheduler.ocr_reader:
                        t0_ocr = time.time()
                        vh, vw = crop.shape[:2]
                        lower = crop[int(vh * 0.45):, :]
                        with self.scheduler.gpu_lock:
                            ocr_res = self.scheduler.ocr_reader.readtext(lower, decoder="greedy")
                        stage_timings["ocr"] += (time.time() - t0_ocr)
                        if ocr_res:
                            counts["ocr_readings"] += len(ocr_res)

                    sqlite_records.append((
                        det_uuid,
                        f"{job_id}_{camera_id}_{t_id}",
                        f_idx,
                        ts,
                        x1, y1, x2, y2,
                        "crops/mock.jpg",
                        det[4],
                        entity_type,
                        1.0,
                        "black", "white", "blue", "none",
                        0.9,
                        EMBEDDING_MODEL_NAME,
                        EMBEDDING_VERSION
                    ))

        # 8. SQLite Persistence Timing
        t0_sql = time.time()
        with self.sqlite_db.connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO jobs (id, name, status, created_at, updated_at, profile) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, "Benchmark Job", "indexing", time.time(), time.time(), "BALANCED")
            )
            conn.execute(
                "INSERT OR IGNORE INTO cameras (id, job_id, name, source_type, source_path, status) VALUES (?, ?, ?, ?, ?, ?)",
                (camera_id, job_id, "Benchmark Camera", "file", str(self.video_path), "active")
            )
            for t_id, tr in active_tracks.items():
                conn.execute(
                    "INSERT OR IGNORE INTO tracks (id, camera_id, job_id, track_id, entity_type, start_time, end_time) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (f"{job_id}_{camera_id}_{t_id}", camera_id, job_id, t_id, "person", tr["start_time"], tr["end_time"])
                )
            if sqlite_records:
                conn.executemany(
                    "INSERT OR REPLACE INTO detections (id, track_id, frame_idx, timestamp, bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, confidence, entity_type, quality_score, dominant_upper_color, secondary_upper_color, dominant_lower_color, secondary_lower_color, color_confidence, embedding_model, embedding_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    sqlite_records
                )
        stage_timings["sqlite"] = time.time() - t0_sql
        counts["sqlite_rows"] = len(sqlite_records)

        # 9. ChromaDB Vector Insert & Read-Back Verification Timing
        t0_chroma = time.time()
        if clip_ids:
            self.vector_db.add_visual_embeddings(ids=clip_ids, embeddings=clip_embs, metadatas=clip_metas)
            counts["chroma_vectors"] += len(clip_ids)
        if reid_ids:
            self.vector_db.add_reid_embeddings(ids=reid_ids, embeddings=reid_embs, metadatas=reid_metas)
            counts["chroma_vectors"] += len(reid_ids)
        stage_timings["chroma_api"] = time.time() - t0_chroma

        # Read-back verification
        t0_ver = time.time()
        if clip_ids:
            ver_res = self.vector_db.verify_vectors_written("visual", clip_ids, sample_size=10)
            if ver_res.get("verified"):
                counts["chroma_verified"] = len(clip_ids)
        stage_timings["chroma_verify"] = time.time() - t0_ver

        total_wall_clock = time.time() - t_pipeline_start
        stage_timings["total"] = total_wall_clock

        # Hardware metrics
        hw = measure_hardware()

        # Clean up benchmark vectors
        self.vector_db.delete_by_job(job_id)
        with self.sqlite_db.connection() as conn:
            conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
            conn.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM cameras WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

        fps_throughput = round(counts["processed_frames"] / max(0.001, total_wall_clock), 2)
        video_mins_per_min = round((counts["processed_frames"] / self.sampling_fps / 60.0) / (total_wall_clock / 60.0), 2)

        benchmark_result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "device": self.device,
                "gpu_name": hw["gpu_device"],
            },
            "video_metadata": {
                "file": self.video_path.name,
                "duration_sec": round(self.source_duration, 2),
                "resolution": f"{self.source_width}x{self.source_height}",
                "source_fps": round(self.source_fps, 2),
                "decoded_frames": counts["decoded_frames"],
                "processed_frames": counts["processed_frames"],
            },
            "configuration": {
                "sampling_fps": self.sampling_fps,
                "batch_size": self.batch_size,
                "decoder_mode": self.decoder_mode,
                "models": {
                    "detection": YOLO_MODEL_PATH,
                    "visual": f"{EMBEDDING_MODEL_NAME}:{EMBEDDING_VERSION}",
                    "reid": f"{REID_MODEL_NAME}:{REID_VERSION}",
                },
            },
            "throughput": {
                "frames_per_sec": fps_throughput,
                "video_minutes_per_minute": video_mins_per_min,
                "total_wall_clock_sec": round(total_wall_clock, 3),
            },
            "stage_latencies_sec": {k: round(v, 4) for k, v in stage_timings.items()},
            "counts": counts,
            "hardware": hw,
            "ground_truth_accuracy": "NOT AVAILABLE — NO LABELLED GROUND TRUTH DATASET",
        }

        # Save JSON & CSV
        out_json = self.output_dir / f"pipeline_bench_fps{self.sampling_fps}_b{self.batch_size}_{int(time.time())}.json"
        out_json.write_text(json.dumps(benchmark_result, indent=2), encoding="utf-8")

        out_csv = self.output_dir / "pipeline_benchmark_summary.csv"
        csv_exists = out_csv.exists()
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow([
                    "Timestamp", "Video", "FPS", "Batch", "Throughput_FPS", "Total_Sec",
                    "Decode_Sec", "YOLO_Sec", "Track_Sec", "CLIP_Sec", "ReID_Sec",
                    "Face_Sec", "OCR_Sec", "SQLite_Sec", "Chroma_Sec", "Peak_VRAM_MB", "RAM_MB"
                ])
            writer.writerow([
                benchmark_result["timestamp"],
                benchmark_result["video_metadata"]["file"],
                self.sampling_fps,
                self.batch_size,
                fps_throughput,
                round(total_wall_clock, 3),
                round(stage_timings["decode"], 3),
                round(stage_timings["detection"], 3),
                round(stage_timings["tracking"], 3),
                round(stage_timings["clip"], 3),
                round(stage_timings["reid"], 3),
                round(stage_timings["face"], 3),
                round(stage_timings["ocr"], 3),
                round(stage_timings["sqlite"], 3),
                round(stage_timings["chroma_api"] + stage_timings["chroma_verify"], 3),
                hw["gpu_peak_mb"],
                hw["ram_used_mb"]
            ])

        print(f"[BENCHMARK] Pipeline benchmark complete. Saved to {out_json}", flush=True)
        return benchmark_result


def main():
    parser = argparse.ArgumentParser(description="AICatch Pipeline Benchmark")
    parser.add_argument("--video", type=str, default="data/media/uploads/inside.mp4", help="Path to CCTV video file")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds to benchmark")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling FPS rate")
    parser.add_argument("--batch-size", type=int, default=8, help="GPU batch size")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--repetitions", type=int, default=1, help="Repetitions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device target")
    parser.add_argument("--gt-path", type=str, default=None, help="Ground-truth dataset path if available")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory")
    parser.add_argument("--decoder-mode", type=str, default="software", choices=["software", "cuvid"], help="Decoder mode")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = Path("data/media/uploads/outside.mp4")

    runner = PipelineBenchmarkRunner(
        video_path=video_path,
        duration=args.duration,
        sampling_fps=args.fps,
        batch_size=args.batch_size,
        warmup_iterations=args.warmup,
        repetitions=args.repetitions,
        device=args.device,
        gt_path=Path(args.gt_path) if args.gt_path else None,
        output_dir=Path(args.output_dir),
        decoder_mode=args.decoder_mode,
    )

    result = runner.run_pipeline_benchmark()
    print("\n================== PIPELINE BENCHMARK SUMMARY ==================")
    print(json.dumps(result, indent=2))
    print("===============================================================\n")


if __name__ == "__main__":
    main()
