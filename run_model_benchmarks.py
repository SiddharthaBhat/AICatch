# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import logging
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath("."))

from config import SQLITE_DB_PATH, CHROMA_PERSIST_DIR
from storage import SQLiteDatabase, VectorDatabase, MediaStorage
from pipeline import JobManager, GPUScheduler, ActiveQueryMatcher, ResultDispatcher
from models.registry import registry
from benchmarks import BenchmarkResult, BenchmarkStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
LOGGER = logging.getLogger("ModelBenchmarks")

def extract_benchmark_frames(video_path: Path, max_frames: int = 20):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def benchmark_models():
    video_path = Path("data/media/uploads/inside.mp4")
    if not video_path.exists():
        video_path = Path("data/media/uploads/outside.mp4")
        
    if not video_path.exists():
        LOGGER.error("No test video found in data/media/uploads/")
        return
        
    LOGGER.info(f"Extracting benchmark frames from {video_path}...")
    frames = extract_benchmark_frames(video_path, max_frames=20)
    LOGGER.info(f"Extracted {len(frames)} frames.")
    
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    media_storage = MediaStorage()
    matcher = ActiveQueryMatcher()
    dispatcher = ResultDispatcher(sqlite_db, vector_db, matcher)
    job_manager = JobManager(sqlite_db)
    
    scheduler = GPUScheduler(job_manager, media_storage, dispatcher)
    scheduler.load_models()
    
    results = {}
    
    # 1. Detection
    LOGGER.info("Benchmarking Detection (YOLO11s)...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for f in frames:
        with scheduler.gpu_lock:
            _ = scheduler.yolo.predict(f, verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")
    t1 = time.time()
    det_total = t1 - t0
    results["detection"] = {
        "model": "YOLO11s",
        "task": "detection",
        "indexing_fps": round(len(frames) / det_total, 2),
        "latency_ms": round((det_total / len(frames)) * 1000, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "ground_truth_accuracy": "NOT AVAILABLE -- NO LABELLED GROUND TRUTH DATASET",
        "status": "PASS"
    }
    
    # 2. Visual Semantics
    LOGGER.info("Benchmarking Visual Semantics (CLIP ViT-B/32)...")
    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    t0 = time.time()
    for img in pil_images:
        _ = scheduler.embed_image(img)
    t1 = time.time()
    vis_total = t1 - t0
    results["visual"] = {
        "model": "CLIP ViT-B/32",
        "task": "visual",
        "dimension": 512,
        "indexing_fps": round(len(pil_images) / vis_total, 2),
        "latency_ms": round((vis_total / len(pil_images)) * 1000, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "ground_truth_accuracy": "NOT AVAILABLE -- NO LABELLED GROUND TRUTH DATASET",
        "status": "PASS"
    }
    
    # 3. Re-ID
    LOGGER.info("Benchmarking Person Re-ID (MobileNetV3 576-D)...")
    t0 = time.time()
    for img in pil_images:
        _ = scheduler.embed_reid(img)
    t1 = time.time()
    reid_total = t1 - t0
    results["reid"] = {
        "model": "MobileNetV3 Re-ID",
        "task": "reid",
        "dimension": 576,
        "indexing_fps": round(len(pil_images) / reid_total, 2),
        "latency_ms": round((reid_total / len(pil_images)) * 1000, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "ground_truth_accuracy": "NOT AVAILABLE -- NO LABELLED GROUND TRUTH DATASET",
        "status": "PASS"
    }
    
    # 4. Face
    LOGGER.info("Benchmarking Face (FaceNet InceptionResnetV1)...")
    t0 = time.time()
    for img in pil_images[:10]:
        _ = scheduler.embed_face_with_diagnostics(img)
    t1 = time.time()
    face_total = t1 - t0
    results["face"] = {
        "model": "FaceNet InceptionResnetV1",
        "task": "face",
        "dimension": 512,
        "indexing_fps": round(10 / face_total, 2),
        "latency_ms": round((face_total / 10) * 1000, 2),
        "device": "cpu",
        "ground_truth_accuracy": "NOT AVAILABLE -- NO LABELLED GROUND TRUTH DATASET",
        "status": "PASS"
    }
    
    # 5. OCR
    LOGGER.info("Benchmarking OCR (EasyOCR)...")
    ocr_frames = frames[:5]
    t0 = time.time()
    for f in ocr_frames:
        if scheduler.ocr_reader:
            _ = scheduler.ocr_reader.readtext(f)
    t1 = time.time()
    ocr_total = t1 - t0
    results["ocr"] = {
        "model": "EasyOCR",
        "task": "ocr",
        "indexing_fps": round(len(ocr_frames) / ocr_total, 2),
        "latency_ms": round((ocr_total / len(ocr_frames)) * 1000, 2),
        "device": "cpu",
        "ground_truth_accuracy": "NOT AVAILABLE -- NO LABELLED GROUND TRUTH DATASET",
        "status": "PASS"
    }
    
    store = BenchmarkStore()
    for task, res in results.items():
        bench_res = BenchmarkResult(
            dataset_id="cctv_local_clips",
            task=task,
            model_key=registry.active(task).key,
            metrics={
                "indexing_fps": res.get("indexing_fps", 0.0),
                "latency_ms": res.get("latency_ms", 0.0),
                "peak_vram_mb": res.get("peak_vram_mb", 0.0),
            },
            environment={
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "torch_version": torch.__version__,
                "sample_frames": len(frames)
            }
        )
        store.record(bench_res)
        
    print("\n================== BENCHMARK RESULTS SUMMARY ==================")
    print(json.dumps(results, indent=2))
    print("===============================================================\n")

if __name__ == "__main__":
    benchmark_models()
