"""Bounded stage-level telemetry benchmark for AICatch.

Instruments GPUScheduler to measure 17 distinct execution stages:
1. frame acquisition
2. batch formation
3. YOLO preprocessing
4. YOLO H2D transfer
5. YOLO inference
6. YOLO output materialization
7. tracking
8. crop preparation
9. CLIP preprocessing
10. CLIP H2D
11. CLIP inference
12. CLIP output materialization
13. Re-ID preprocessing
14. Re-ID H2D
15. Re-ID inference
16. Re-ID output materialization
17. scheduler bookkeeping

Also captures batch size histogram, queue depths, GPU utilization, VRAM,
low-GPU (<=5%) samples, and lock occupancy across 2, 4, and 5 cameras.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import (
    BATCH_SIZE,
    CHROMA_PERSIST_DIR,
    CLIP_DEVICE,
    FACE_DEVICE,
    OCR_DEVICE,
    PROCESSING_PROFILES,
    REID_DEVICE,
    SQLITE_DB_PATH,
    YOLO_DEVICE,
)
from pipeline import ActiveQueryMatcher, GPUScheduler, JobManager, ResultDispatcher
from pipeline.gpu_scheduler import evaluate_crop_quality
from storage import MediaStorage, SQLiteDatabase, VectorDatabase
from utils.monitor import PerformanceMonitorInstance

LOGGER = logging.getLogger("StageTelemetry")


class StageTimer:
    def __init__(self):
        self.total_sec: float = 0.0
        self.count: int = 0
        self.max_sec: float = 0.0

    def record(self, duration_sec: float):
        self.total_sec += duration_sec
        self.count += 1
        if duration_sec > self.max_sec:
            self.max_sec = duration_sec

    def to_dict(self, total_wall_sec: float = 0.0) -> Dict[str, Any]:
        avg_ms = (self.total_sec / self.count * 1000.0) if self.count > 0 else 0.0
        pct = (self.total_sec / total_wall_sec * 100.0) if total_wall_sec > 0 else 0.0
        return {
            "calls": self.count,
            "total_ms": round(self.total_sec * 1000.0, 2),
            "avg_ms": round(avg_ms, 2),
            "max_ms": round(self.max_sec * 1000.0, 2),
            "pct_runtime": round(pct, 2),
        }


class StageMetricsCollector:
    def __init__(self):
        self.stages: Dict[str, StageTimer] = {
            "1_frame_acquisition": StageTimer(),
            "2_batch_formation": StageTimer(),
            "3_yolo_preprocessing": StageTimer(),
            "4_yolo_h2d": StageTimer(),
            "5_yolo_inference": StageTimer(),
            "6_yolo_output_materialization": StageTimer(),
            "7_tracking": StageTimer(),
            "8_crop_preparation": StageTimer(),
            "9_clip_preprocessing": StageTimer(),
            "10_clip_h2d": StageTimer(),
            "11_clip_inference": StageTimer(),
            "12_clip_output_materialization": StageTimer(),
            "13_reid_preprocessing": StageTimer(),
            "14_reid_h2d": StageTimer(),
            "15_reid_inference": StageTimer(),
            "16_reid_output_materialization": StageTimer(),
            "17_scheduler_bookkeeping": StageTimer(),
        }
        self.batch_sizes: collections.Counter = collections.Counter()
        self.total_batches: int = 0
        self.total_frames_in_batches: int = 0
        self.total_candidates_evaluated: int = 0
        self.total_clip_crops: int = 0
        self.total_reid_crops: int = 0

    def get_summary(self, wall_clock_sec: float) -> Dict[str, Any]:
        return {
            name: timer.to_dict(wall_clock_sec)
            for name, timer in self.stages.items()
        }


class InstrumentedGPUScheduler(GPUScheduler):
    def __init__(self, job_manager: JobManager, media: MediaStorage, dispatcher: ResultDispatcher, collector: StageMetricsCollector):
        super().__init__(job_manager, media, dispatcher)
        self.collector = collector

    def run(self) -> None:
        try:
            self.load_models()
        except Exception as e:
            LOGGER.critical(f"Failed to load GPU models: {e}", exc_info=True)
            self.stopped = True
            return

        while not self.stopped:
            try:
                active_jobs = list(self.job_manager.active_jobs.items())
                for job_id, job in active_jobs:
                    job_failed = False
                    failure_msg = ""
                    with job.lock:
                        for dec in job.decoders.values():
                            if getattr(dec, "failed", False):
                                job_failed = True
                                failure_msg = getattr(dec, "error_msg", "Decoder error")
                                break
                    if job_failed:
                        self.job_manager.fail_job(job_id, failure_msg)
                        continue

                decoders = self.job_manager.get_active_decoders()
                if not decoders:
                    time.sleep(0.05)
                    continue

                t_form_start = time.perf_counter()
                batch_packets = []
                while len(batch_packets) < BATCH_SIZE:
                    pulled_in_round = 0
                    for dec in decoders:
                        t_acq_0 = time.perf_counter()
                        try:
                            packet = dec.frame_queue.get_nowait()
                            batch_packets.append((dec.camera_id, dec.is_live, packet))
                            pulled_in_round += 1
                            if len(batch_packets) >= BATCH_SIZE:
                                self.collector.stages["1_frame_acquisition"].record(time.perf_counter() - t_acq_0)
                                break
                        except queue.Empty:
                            pass
                        self.collector.stages["1_frame_acquisition"].record(time.perf_counter() - t_acq_0)
                    if pulled_in_round == 0:
                        break

                t_form_elapsed = time.perf_counter() - t_form_start
                self.collector.stages["2_batch_formation"].record(t_form_elapsed)

                if not batch_packets:
                    time.sleep(0.01)
                    continue

                b_size = len(batch_packets)
                self.collector.batch_sizes[b_size] += 1
                self.collector.total_batches += 1
                self.collector.total_frames_in_batches += b_size

                PerformanceMonitorInstance.update_queue_depth(
                    sum(d.frame_queue.qsize() for d in decoders)
                )

                self._instrumented_process_batch(batch_packets)

            except RuntimeError as re:
                if "out of memory" in str(re).lower():
                    torch.cuda.empty_cache()
                    time.sleep(1.0)
                else:
                    LOGGER.error(f"Inference error: {re}", exc_info=True)
            except Exception as e:
                LOGGER.error(f"Unexpected GPUScheduler error: {e}", exc_info=True)
                time.sleep(0.5)

    def _instrumented_process_batch(self, batch: List[Tuple[str, bool, Tuple[int, float, np.ndarray]]]) -> None:
        t3_0 = time.perf_counter()
        filtered_batch = []
        for camera_id, is_live, packet in batch:
            frame_idx, timestamp, frame_bgr = packet
            profile_name = "FORENSIC"
            job_id = None
            target_decoder = None
            for job in list(self.job_manager.active_jobs.values()):
                if camera_id in job.decoders:
                    profile_name = job.profile
                    job_id = job.job_id
                    target_decoder = job.decoders[camera_id]
                    break
            if not job_id:
                continue
            profile = PROCESSING_PROFILES.get(profile_name, PROCESSING_PROFILES["FORENSIC"])
            is_prefiltered = getattr(target_decoder, "static_filtered", False)
            if not is_prefiltered:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 and frame_bgr.shape[2] == 3 else frame_bgr
                if self.track_cache.is_scene_static(camera_id, gray, profile["skip_static_threshold"]):
                    PerformanceMonitorInstance.update_counts(decoded=1)
                    continue
            filtered_batch.append((camera_id, is_live, packet, job_id, profile))

        if not filtered_batch:
            self.collector.stages["3_yolo_preprocessing"].record(time.perf_counter() - t3_0)
            return

        frames = [
            cv2.cvtColor(item[2][2], cv2.COLOR_GRAY2BGR) if len(item[2][2].shape) == 2 or item[2][2].shape[2] == 1 else item[2][2]
            for item in filtered_batch
        ]
        self.collector.stages["3_yolo_preprocessing"].record(time.perf_counter() - t3_0)

        t_yolo_0 = time.perf_counter()
        with self.gpu_lock:
            results = self.yolo.predict(
                source=frames,
                imgsz=640,
                device=YOLO_DEVICE,
                verbose=False,
                conf=0.25
            )
        t_yolo_wall = time.perf_counter() - t_yolo_0

        if results and hasattr(results[0], "speed"):
            total_prep_ms = sum(r.speed.get("preprocess", 0.0) for r in results)
            total_infer_ms = sum(r.speed.get("inference", 0.0) for r in results)
            self.collector.stages["4_yolo_h2d"].record(total_prep_ms / 1000.0)
            self.collector.stages["5_yolo_inference"].record(total_infer_ms / 1000.0)
        else:
            self.collector.stages["4_yolo_h2d"].record(t_yolo_wall * 0.15)
            self.collector.stages["5_yolo_inference"].record(t_yolo_wall * 0.85)

        PerformanceMonitorInstance.record_latency("yolo", t_yolo_wall)
        PerformanceMonitorInstance.update_counts(decoded=len(filtered_batch))

        t6_0 = time.perf_counter()
        camera_groups = {}
        allowed_classes = {0, 2, 3, 5, 7}
        for idx, (camera_id, is_live, (frame_idx, timestamp, frame_bgr), job_id, profile) in enumerate(filtered_batch):
            res = results[idx]
            PerformanceMonitorInstance.record_frame_processed()
            boxes_to_track = []
            if res.boxes is not None and len(res.boxes) > 0:
                boxes_data = res.boxes.data.cpu().numpy()
                for row in boxes_data:
                    cls_id = int(row[5])
                    if cls_id in allowed_classes:
                        boxes_to_track.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), cls_id))
            if camera_id not in camera_groups:
                camera_groups[camera_id] = []
            camera_groups[camera_id].append((frame_idx, timestamp, frame_bgr, job_id, profile, boxes_to_track))
        self.collector.stages["6_yolo_output_materialization"].record(time.perf_counter() - t6_0)

        t7_0 = time.perf_counter()
        candidates = []
        for camera_id, frame_items in camera_groups.items():
            frame_items.sort(key=lambda x: (x[0], x[1]))
            for frame_idx, timestamp, frame_bgr, job_id, profile, boxes_to_track in frame_items:
                tracked_entities = self._update_camera_tracker(camera_id, job_id, frame_idx, timestamp, frame_bgr, boxes_to_track)
                for track_data, crop, entity_type, bbox in tracked_entities:
                    candidates.append((job_id, camera_id, track_data, crop, entity_type, bbox, frame_idx, timestamp, profile))
        self.collector.stages["7_tracking"].record(time.perf_counter() - t7_0)
        self.collector.total_candidates_evaluated += len(candidates)

        if not candidates:
            return

        t8_0 = time.perf_counter()
        clip_batch_items = []
        cand_keyframe_decisions: Dict[int, bool] = {}
        track_sim_state: Dict[Tuple[str, int], Dict[str, Any]] = {}

        for cand_idx, cand in enumerate(candidates):
            job_id, camera_id, track, crop, entity_type, bbox, frame_idx, timestamp, profile = cand
            track_key = (camera_id, track.track_id)
            if track_key not in track_sim_state:
                track_sim_state[track_key] = {
                    "has_embedding": track.clip_embedding is not None,
                    "last_sharpness": track.last_embedded_sharpness,
                    "last_size": track.last_embedded_size,
                    "last_time": track.last_embedded_time,
                }
            state = track_sim_state[track_key]
            is_valid, quality = evaluate_crop_quality(track.best_crop, track.best_sharpness)
            cand_run_clip = False
            if is_valid:
                crop_size = float(track.best_crop.shape[0] * track.best_crop.shape[1])
                if not state["has_embedding"]:
                    cand_run_clip = True
                elif track.best_sharpness > (state["last_sharpness"] * 1.20):
                    cand_run_clip = True
                elif crop_size > (state["last_size"] * 1.30):
                    cand_run_clip = True
                elif (timestamp - state["last_time"]) > 5.0:
                    cand_run_clip = True

            cand_keyframe_decisions[cand_idx] = cand_run_clip
            if cand_run_clip:
                state["has_embedding"] = True
                state["last_sharpness"] = track.best_sharpness
                state["last_size"] = float(track.best_crop.shape[0] * track.best_crop.shape[1])
                state["last_time"] = timestamp
                if len(track.best_crop.shape) == 2 or track.best_crop.shape[2] == 1:
                    pil_img = Image.fromarray(track.best_crop).convert("RGB")
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(track.best_crop, cv2.COLOR_BGR2RGB))
                clip_batch_items.append({
                    "cand_idx": cand_idx,
                    "cand": cand,
                    "pil_img": pil_img,
                    "quality": quality,
                    "entity_type": entity_type,
                    "timestamp": timestamp,
                    "track": track,
                })
        self.collector.stages["8_crop_preparation"].record(time.perf_counter() - t8_0)
        self.collector.total_clip_crops += len(clip_batch_items)

        clip_embeddings = []
        if clip_batch_items:
            clip_images = [item["pil_img"] for item in clip_batch_items]
            clip_embeddings = self._instrumented_embed_images_batch(clip_images, batch_size=16)

            reid_batch_items = [
                item for item in clip_batch_items
                if item["entity_type"] == "person" and self.reid_model is not None and self.reid_transform is not None
            ]
            self.collector.total_reid_crops += len(reid_batch_items)
            reid_embeddings = []
            if reid_batch_items:
                reid_images = [item["pil_img"] for item in reid_batch_items]
                reid_embeddings = self._instrumented_embed_reid_batch(reid_images, batch_size=16)

            for idx_c, item in enumerate(clip_batch_items):
                tr = item["track"]
                tr.clip_embedding = clip_embeddings[idx_c]
                tr.last_embedded_sharpness = tr.best_sharpness
                tr.last_embedded_size = float(tr.best_crop.shape[0] * tr.best_crop.shape[1])
                tr.last_embedded_time = item["timestamp"]
                tr.embedded_count += 1
                tr.quality_score = item["quality"]

            for idx_r, item in enumerate(reid_batch_items):
                if idx_r < len(reid_embeddings) and reid_embeddings[idx_r] is not None:
                    item["track"].reid_embedding = reid_embeddings[idx_r]

        t17_0 = time.perf_counter()
        for cand_idx, cand in enumerate(candidates):
            job_id, camera_id, track_data, crop, entity_type, bbox, frame_idx, timestamp, profile = cand
            run_clip_flag = cand_keyframe_decisions.get(cand_idx, False)
            run_face = False
            if entity_type == "person" and profile.get("run_face_analysis", True):
                if track_data.face_embedding is None and track_data.face_run_count < 3:
                    run_face = True
                    track_data.face_run_count += 1
            run_ocr = False
            if entity_type == "vehicle" and profile.get("run_ocr_analysis", True):
                if track_data.plate_text is None and track_data.ocr_run_count < 3:
                    run_ocr = True
                    track_data.ocr_run_count += 1

            best_crop = track_data.best_crop.copy() if track_data.best_crop is not None else crop.copy()
            raw_crop = crop.copy() if crop is not None else best_crop.copy()

            task = {
                "job_id": job_id,
                "camera_id": camera_id,
                "track_id": track_data.track_id,
                "track_uuid": f"{job_id}_{camera_id}_{track_data.track_id}",
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "bbox": list(bbox),
                "best_crop": best_crop,
                "raw_crop": raw_crop,
                "entity_type": entity_type,
                "profile": profile,
                "run_clip": run_clip_flag,
                "clip_embedding": track_data.clip_embedding,
                "reid_embedding": getattr(track_data, "reid_embedding", None),
                "quality_score": track_data.quality_score,
                "run_face": run_face,
                "run_ocr": run_ocr,
                "track_ref": track_data,
            }
            self.enrichment_queue.put(task, block=True)
        self.collector.stages["17_scheduler_bookkeeping"].record(time.perf_counter() - t17_0)

    def _instrumented_embed_images_batch(self, images: List[Image.Image], batch_size: int = 16) -> List[List[float]]:
        if not images:
            return []
        all_embeddings: List[List[float]] = []
        with self.gpu_lock:
            if self.clip_model is None or self.clip_processor is None:
                self.load_models()
            clip_model = self.clip_model
            clip_processor = self.clip_processor

        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            t9_0 = time.perf_counter()
            rgb_chunk = [img if img.mode == "RGB" else img.convert("RGB") for img in chunk]
            inputs = clip_processor(images=rgb_chunk, return_tensors="pt")
            self.collector.stages["9_clip_preprocessing"].record(time.perf_counter() - t9_0)

            t10_0 = time.perf_counter()
            inputs = inputs.to(CLIP_DEVICE)
            if CLIP_DEVICE == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            self.collector.stages["10_clip_h2d"].record(time.perf_counter() - t10_0)

            t11_0 = time.perf_counter()
            with self.gpu_lock:
                with torch.no_grad():
                    res = clip_model.get_image_features(**inputs)
                    if hasattr(res, "pooler_output"):
                        res = res.pooler_output
                    res = res / res.norm(p=2, dim=-1, keepdim=True)
            self.collector.stages["11_clip_inference"].record(time.perf_counter() - t11_0)

            t12_0 = time.perf_counter()
            vecs = res.cpu().float().numpy().tolist()
            all_embeddings.extend(vecs)
            self.collector.stages["12_clip_output_materialization"].record(time.perf_counter() - t12_0)

        return all_embeddings

    def _instrumented_embed_reid_batch(self, images: List[Image.Image], batch_size: int = 16) -> List[Optional[List[float]]]:
        if not images:
            return []
        with self.gpu_lock:
            if self.reid_model is None or self.reid_transform is None:
                self.load_models()
            if self.reid_model is None or self.reid_transform is None:
                return [None] * len(images)
            reid_model = self.reid_model
            reid_transform = self.reid_transform

        all_results: List[Optional[List[float]]] = []
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size]
            try:
                t13_0 = time.perf_counter()
                tensor_crops = torch.stack([reid_transform(img if img.mode == "RGB" else img.convert("RGB")) for img in chunk])
                self.collector.stages["13_reid_preprocessing"].record(time.perf_counter() - t13_0)

                t14_0 = time.perf_counter()
                tensor_crops = tensor_crops.to(REID_DEVICE)
                if REID_DEVICE == "cuda":
                    tensor_crops = tensor_crops.half()
                self.collector.stages["14_reid_h2d"].record(time.perf_counter() - t14_0)

                t15_0 = time.perf_counter()
                with self.gpu_lock:
                    with torch.no_grad():
                        reid_features = reid_model(tensor_crops)
                        reid_features = reid_features / reid_features.norm(p=2, dim=-1, keepdim=True)
                self.collector.stages["15_reid_inference"].record(time.perf_counter() - t15_0)

                t16_0 = time.perf_counter()
                vecs = reid_features.cpu().float().numpy().tolist()
                all_results.extend(vecs)
                self.collector.stages["16_reid_output_materialization"].record(time.perf_counter() - t16_0)

            except Exception as ex:
                LOGGER.warning(f"Batch Re-ID extraction error: {ex}")
                for img in chunk:
                    try:
                        rgb_img = img if img.mode == "RGB" else img.convert("RGB")
                        t = reid_transform(rgb_img).unsqueeze(0).to(REID_DEVICE)
                        if REID_DEVICE == "cuda":
                            t = t.half()
                        with self.gpu_lock:
                            with torch.no_grad():
                                rf = reid_model(t)
                                rf = rf / rf.norm(p=2, dim=-1, keepdim=True)
                                all_results.append(rf[0].cpu().float().numpy().tolist())
                    except Exception:
                        all_results.append(None)
        return all_results


def _number(value: str) -> Optional[float]:
    try:
        return float(value.strip().replace("[N/A]", ""))
    except (TypeError, ValueError):
        return None


class TelemetrySampler:
    def __init__(self, scheduler: GPUScheduler, dispatcher: ResultDispatcher, job_manager: JobManager, interval: float):
        self.scheduler = scheduler
        self.dispatcher = dispatcher
        self.job_manager = job_manager
        self.interval = interval
        self.samples: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._process = psutil.Process(os.getpid())
        self._process.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, name="TelemetrySampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval * 3))

    @staticmethod
    def _nvidia_smi() -> Dict[str, Optional[float]]:
        fields = "utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm"
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.0, check=True,
            )
            parts = result.stdout.strip().splitlines()[0].split(",")
            return dict(zip(("gpu_util_percent", "gpu_memory_used_mb", "gpu_temp_c", "gpu_power_w", "gpu_clock_mhz"), (_number(x) for x in parts)))
        except Exception:
            return {key: None for key in ("gpu_util_percent", "gpu_memory_used_mb", "gpu_temp_c", "gpu_power_w", "gpu_clock_mhz")}

    def _gpu_path_busy(self) -> bool:
        acquired = self.scheduler.gpu_lock.acquire(blocking=False)
        if acquired:
            self.scheduler.gpu_lock.release()
        return not acquired

    def _snapshot(self) -> Dict[str, Any]:
        decoders = self.job_manager.get_active_decoders()
        gpu = self._nvidia_smi()
        return {
            "timestamp": round(time.time(), 3),
            **gpu,
            "torch_allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 2) if torch.cuda.is_available() else 0.0,
            "torch_reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 2) if torch.cuda.is_available() else 0.0,
            "process_cpu_percent": round(self._process.cpu_percent(None), 1),
            "decoder_queue_depth": sum(dec.frame_queue.qsize() for dec in decoders),
            "decoder_queue_depths": {dec.camera_id: dec.frame_queue.qsize() for dec in decoders},
            "enrichment_queue_depth": self.scheduler.enrichment_queue.qsize(),
            "persistence_queue_depth": self.dispatcher.db_write_queue.qsize(),
            "gpu_path_busy": self._gpu_path_busy(),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._snapshot())
            self._stop.wait(self.interval)


def _summarize_telemetry(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    def values(key: str) -> List[float]:
        return [float(s[key]) for s in samples if s.get(key) is not None]

    def stats(key: str) -> Dict[str, Optional[float]]:
        vals = values(key)
        return {"average": round(mean(vals), 2) if vals else None, "peak": round(max(vals), 2) if vals else None}

    util = values("gpu_util_percent")
    return {
        "sample_count": len(samples),
        "gpu_util_percent": stats("gpu_util_percent"),
        "gpu_idle_percent_at_or_below_5": round(100.0 * sum(v <= 5.0 for v in util) / len(util), 2) if util else None,
        "gpu_memory_used_mb": stats("gpu_memory_used_mb"),
        "torch_allocated_mb": stats("torch_allocated_mb"),
        "decoder_queue_depth": stats("decoder_queue_depth"),
        "enrichment_queue_depth": stats("enrichment_queue_depth"),
        "persistence_queue_depth": stats("persistence_queue_depth"),
        "lock_occupancy_percent": round(100.0 * sum(bool(s["gpu_path_busy"]) for s in samples) / len(samples), 2) if samples else 0.0,
    }


def _cleanup(sqlite_db: SQLiteDatabase, vector_db: VectorDatabase, job_manager: JobManager, job_id: str) -> None:
    job_manager.cleanup_job_resources(job_id)
    vector_db.delete_by_job(job_id)
    with sqlite_db.connection() as conn:
        conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
        conn.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM cameras WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def run_stage_case(video: Path, cameras: int, seconds: float, interval: float, profile: str = "FORENSIC") -> Dict[str, Any]:
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    media = MediaStorage()
    dispatcher = ResultDispatcher(sqlite_db, vector_db, ActiveQueryMatcher())
    manager = JobManager(sqlite_db)
    collector = StageMetricsCollector()
    scheduler = InstrumentedGPUScheduler(manager, media, dispatcher, collector)
    scheduler.load_models()
    scheduler.start()

    sources = [{"name": f"Stage_Cam_{i + 1}", "type": "file", "path": str(video.resolve())} for i in range(cameras)]
    job_id = manager.create_job(f"StageTelemetry {cameras} cams", profile, sources, investigation_type="benchmark")
    before = PerformanceMonitorInstance.get_metrics()["frames_analyzed"]
    sampler = TelemetrySampler(scheduler, dispatcher, manager, interval)

    started = time.monotonic()
    sampler.start()
    try:
        manager.start_job(job_id)
        deadline = started + seconds
        while time.monotonic() < deadline:
            time.sleep(0.1)

        benchmark_job = manager.active_jobs.get(job_id)
        decoder_counts = {
            dec.camera_id: dec.frames_decoded
            for dec in (benchmark_job.decoders.values() if benchmark_job else [])
        }
        manager.cleanup_job_resources(job_id)
        time.sleep(1.0)
        enrichment_drained = scheduler.flush_enrichment_pool(timeout=20.0)
        dispatcher.flush_and_wait(timeout=20.0)
        dispatcher.stop_dispatcher()
        scheduler.stop_scheduler()
        elapsed = time.monotonic() - started

        with sqlite_db.connection() as conn:
            total_detections = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,)
            ).fetchone()[0]
            per_camera = dict(conn.execute(
                "SELECT t.camera_id, COUNT(*) FROM detections d JOIN tracks t ON d.track_id = t.id WHERE t.job_id = ? GROUP BY t.camera_id", (job_id,)
            ).fetchall())

        after = PerformanceMonitorInstance.get_metrics()["frames_analyzed"]
        frames_analyzed = max(0, after - before)
        effective_fps = round(frames_analyzed / max(elapsed, 0.001), 3)

        stage_breakdown = collector.get_summary(elapsed)
        batch_hist = dict(sorted(collector.batch_sizes.items()))
        avg_batch = round(collector.total_frames_in_batches / max(1, collector.total_batches), 2)
        max_batch = max(collector.batch_sizes.keys()) if collector.batch_sizes else 0

        telemetry_summary = _summarize_telemetry(sampler.samples)

        return {
            "camera_count": cameras,
            "profile": profile,
            "wall_clock_sec": round(elapsed, 3),
            "frames_analyzed": frames_analyzed,
            "effective_fps": effective_fps,
            "detections_persisted": total_detections,
            "detections_per_camera": per_camera,
            "decoder_frames": decoder_counts,
            "persistence_drained": enrichment_drained and scheduler.enrichment_queue.unfinished_tasks == 0,
            "stage_breakdown": stage_breakdown,
            "batch_stats": {
                "histogram": batch_hist,
                "total_batches": collector.total_batches,
                "total_frames": collector.total_frames_in_batches,
                "average_batch_size": avg_batch,
                "max_batch_size": max_batch,
            },
            "workload_counts": {
                "candidates_evaluated": collector.total_candidates_evaluated,
                "clip_crops": collector.total_clip_crops,
                "reid_crops": collector.total_reid_crops,
            },
            "telemetry": telemetry_summary,
        }
    finally:
        sampler.stop()
        _cleanup(sqlite_db, vector_db, manager, job_id)
        scheduler.stop_scheduler()
        dispatcher.stop_dispatcher()


def main():
    parser = argparse.ArgumentParser(description="Stage-level GPU bottleneck telemetry benchmark")
    parser.add_argument("--video", default="data/media/uploads/inside.mp4")
    parser.add_argument("--cameras", default="2,4,5")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--profile", default="FORENSIC")
    parser.add_argument("--output", default="benchmarks/results/stage_telemetry_multicamera.json")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(video)

    results = []
    for c_str in args.cameras.split(","):
        c = int(c_str.strip())
        print(f"\n>>> Running Stage Telemetry for {c} cameras ({args.seconds}s)...", flush=True)
        res = run_stage_case(video, c, args.seconds, args.interval, args.profile)
        results.append(res)
        print(f"    Completed {c} cams: {res['frames_analyzed']} frames in {res['wall_clock_sec']}s ({res['effective_fps']} FPS)", flush=True)

    report = {"video": str(video), "cases": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSuccessfully wrote report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
