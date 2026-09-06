# -*- coding: utf-8 -*-
"""Master Benchmark Runner for AICatch.

Systematically evaluates:
1. Tier-0 Decoder (Software vs Hardware CUVID)
2. Batch Size Candidates (1, 2, 4, 8, 16)
3. Sampling FPS Candidates (1, 2, 5, 10 FPS)
4. ChromaDB Persistence Batch Candidates (25, 50, 100, 250, 500, 1000)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_benchmark import PipelineBenchmarkRunner, measure_hardware
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_VERSION
from storage.vector_db import VectorDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
LOGGER = logging.getLogger("MasterBenchmark")


def run_decoder_benchmark(video_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    LOGGER.info("=== 1. TIER-0 FFmpeg DECODER BENCHMARK ===")
    modes = ["software", "cuvid"]
    results = []

    for mode in modes:
        try:
            runner = PipelineBenchmarkRunner(
                video_path=video_path,
                duration=15.0,
                sampling_fps=5.0,
                batch_size=8,
                warmup_iterations=1,
                output_dir=output_dir,
                decoder_mode=mode,
            )
            res = runner.run_isolated_decode_benchmark()
            # Also run a short end-to-end pipeline run to see downstream throughput
            pipe_res = runner.run_pipeline_benchmark()
            res["downstream_throughput_fps"] = pipe_res["throughput"]["frames_per_sec"]
            res["total_wall_clock_sec"] = pipe_res["throughput"]["total_wall_clock_sec"]
            results.append(res)
        except Exception as e:
            LOGGER.warning(f"Decoder mode {mode} encountered note: {e}")
            results.append({
                "decoder_mode": mode,
                "status": "NOT AVAILABLE" if mode == "cuvid" else "FAIL",
                "error": str(e)
            })

    return results


def run_batch_size_benchmark(video_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    LOGGER.info("=== 2. BATCH SIZE BENCHMARK (1, 2, 4, 8, 16) ===")
    candidate_batches = [1, 2, 4, 8, 16]
    results = []

    for b in candidate_batches:
        LOGGER.info(f"Evaluating Batch Size = {b}...")
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            runner = PipelineBenchmarkRunner(
                video_path=video_path,
                duration=10.0,
                sampling_fps=2.0,
                batch_size=b,
                warmup_iterations=1,
                output_dir=output_dir,
            )
            pipe_res = runner.run_pipeline_benchmark()
            results.append({
                "batch_size": b,
                "throughput_fps": pipe_res["throughput"]["frames_per_sec"],
                "total_sec": pipe_res["throughput"]["total_wall_clock_sec"],
                "yolo_latency_sec": pipe_res["stage_latencies_sec"]["detection"],
                "clip_latency_sec": pipe_res["stage_latencies_sec"]["clip"],
                "peak_vram_mb": pipe_res["hardware"]["gpu_peak_mb"],
                "processed_frames": pipe_res["counts"]["processed_frames"],
                "failures": pipe_res["counts"]["failed_frames"],
                "status": "PASS"
            })
        except Exception as e:
            LOGGER.error(f"Batch size {b} failed: {e}")
            results.append({
                "batch_size": b,
                "status": "FAIL / OOM",
                "error": str(e)
            })

    return results


def run_fps_benchmark(video_path: Path, output_dir: Path) -> List[Dict[str, Any]]:
    LOGGER.info("=== 3. SAMPLING FPS BENCHMARK (1, 2, 5, 10 FPS) ===")
    candidate_fps = [1.0, 2.0, 5.0, 10.0]
    results = []
    baseline_time = None

    for fps in candidate_fps:
        LOGGER.info(f"Evaluating Sampling Rate = {fps} FPS...")
        try:
            runner = PipelineBenchmarkRunner(
                video_path=video_path,
                duration=10.0,
                sampling_fps=fps,
                batch_size=8,
                warmup_iterations=1,
                output_dir=output_dir,
            )
            pipe_res = runner.run_pipeline_benchmark()
            wall_time = pipe_res["throughput"]["total_wall_clock_sec"]
            if baseline_time is None:
                baseline_time = wall_time

            time_saved = round(baseline_time - wall_time, 2) if baseline_time else 0.0
            results.append({
                "sampling_fps": fps,
                "frames_processed": pipe_res["counts"]["processed_frames"],
                "total_sec": wall_time,
                "throughput_fps": pipe_res["throughput"]["frames_per_sec"],
                "detections": pipe_res["counts"]["detections"],
                "tracks": pipe_res["counts"]["tracks"],
                "clip_embeddings": pipe_res["counts"]["clip_embeddings"],
                "time_saved_vs_1fps": time_saved,
                "ground_truth_accuracy": "NOT AVAILABLE — NO LABELLED GROUND TRUTH DATASET",
                "status": "PASS"
            })
        except Exception as e:
            LOGGER.error(f"FPS {fps} failed: {e}")
            results.append({
                "sampling_fps": fps,
                "status": "FAIL",
                "error": str(e)
            })

    return results


def run_chroma_batch_benchmark(output_dir: Path) -> List[Dict[str, Any]]:
    LOGGER.info("=== 4. CHROMADB WRITE & PERSISTENCE BENCHMARK (25, 50, 100, 250, 500, 1000) ===")
    candidate_batches = [25, 50, 100, 250, 500, 1000]
    results = []

    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    dim = 512

    for b in candidate_batches:
        LOGGER.info(f"Testing ChromaDB write batch size = {b}...")
        # Generate dummy normalized vectors
        rng = np.random.RandomState(42)
        raw_vecs = rng.randn(b, dim).astype(np.float32)
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        norm_vecs = (raw_vecs / norms).tolist()

        ids = [f"bench_vec_{b}_{i}" for i in range(b)]
        metas = [{
            "job_id": f"bench_chroma_job_{b}",
            "investigation_id": f"bench_chroma_job_{b}",
            "camera_id": "bench_cam",
            "track_id": f"bench_track_{i}",
            "detection_id": ids[i],
            "timestamp": float(i),
            "entity_type": "person",
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_version": EMBEDDING_VERSION
        } for i in range(b)]

        try:
            t0 = time.time()
            vector_db.add_visual_embeddings(ids=ids, embeddings=norm_vecs, metadatas=metas)
            api_latency = time.time() - t0

            # Read-back verification timing
            t0_v = time.time()
            ver_res = vector_db.verify_vectors_written("visual", ids, sample_size=min(10, b))
            verify_latency = time.time() - t0_v

            total_write_time = api_latency + verify_latency
            vecs_per_sec = round(b / max(0.0001, total_write_time), 2)

            results.append({
                "batch_size": b,
                "api_latency_sec": round(api_latency, 4),
                "verify_latency_sec": round(verify_latency, 4),
                "end_to_end_latency_sec": round(total_write_time, 4),
                "vectors_per_sec": vecs_per_sec,
                "readback_verified": ver_res.get("verified", False),
                "status": "PASS"
            })

            # Cleanup
            vector_db.delete_by_job(f"bench_chroma_job_{b}")
        except Exception as e:
            LOGGER.error(f"Chroma batch {b} failed: {e}")
            results.append({
                "batch_size": b,
                "status": "FAIL",
                "error": str(e)
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Master AICatch Benchmark Suite")
    parser.add_argument("--video", type=str, default="data/media/uploads/inside.mp4")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        video_path = Path("data/media/uploads/outside.mp4")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": measure_hardware(),
        "video": video_path.name,
        "decoder_benchmark": run_decoder_benchmark(video_path, out_dir),
        "batch_size_benchmark": run_batch_size_benchmark(video_path, out_dir),
        "fps_benchmark": run_fps_benchmark(video_path, out_dir),
        "chroma_persistence_benchmark": run_chroma_batch_benchmark(out_dir),
    }

    report_path = out_dir / "master_benchmark_report.json"
    report_path.write_text(json.dumps(master_report, indent=2), encoding="utf-8")
    LOGGER.info(f"Master benchmark report saved to {report_path}")

    print("\n==================== MASTER BENCHMARK REPORT ====================")
    print(json.dumps(master_report, indent=2))
    print("=================================================================\n")


if __name__ == "__main__":
    main()
