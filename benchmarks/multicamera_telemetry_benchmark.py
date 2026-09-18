"""Bounded real-pipeline telemetry benchmark for AICatch.

Runs temporary 2/4/5-camera jobs against a supplied local fixture, samples
hardware and bounded queue state at a fixed interval, drains persistence, then
removes only the benchmark job's derived records and vectors.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CHROMA_PERSIST_DIR, SQLITE_DB_PATH
from pipeline import ActiveQueryMatcher, GPUScheduler, JobManager, ResultDispatcher
from storage import MediaStorage, SQLiteDatabase, VectorDatabase
from utils.monitor import PerformanceMonitorInstance

LOGGER = logging.getLogger("MultiCameraTelemetry")


def _number(value: str) -> Optional[float]:
    try:
        return float(value.strip().replace("[N/A]", ""))
    except (TypeError, ValueError):
        return None


class TelemetrySampler:
    """Low-overhead sampler; nvidia-smi is optional and never blocks a run."""

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
        except (FileNotFoundError, subprocess.SubprocessError, IndexError):
            return {key: None for key in ("gpu_util_percent", "gpu_memory_used_mb", "gpu_temp_c", "gpu_power_w", "gpu_clock_mhz")}

    def _gpu_path_busy(self) -> bool:
        # The lock covers scheduler GPU inference. A nonblocking probe does not
        # synchronize CUDA and distinguishes host-side scheduler serialization.
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
            # Frames feed the scheduler directly from decoder queues; there is
            # intentionally no separate unbounded scheduler queue.
            "scheduler_queue_depth": 0,
            "gpu_path_busy": self._gpu_path_busy(),
            "enrichment_queue_depth": self.scheduler.enrichment_queue.qsize(),
            "persistence_queue_depth": self.dispatcher.db_write_queue.qsize(),
            # Media writes occur inside the bounded enrichment task; no distinct
            # media-write queue exists in the current architecture.
            "media_queue_depth": None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._snapshot())
            self._stop.wait(self.interval)


def _summarize(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        "gpu_temperature_c": stats("gpu_temp_c"),
        "gpu_power_w": stats("gpu_power_w"),
        "gpu_clock_mhz": stats("gpu_clock_mhz"),
        "process_cpu_percent": stats("process_cpu_percent"),
        "torch_allocated_mb": stats("torch_allocated_mb"),
        "torch_reserved_mb": stats("torch_reserved_mb"),
        "decoder_queue_depth": stats("decoder_queue_depth"),
        "enrichment_queue_depth": stats("enrichment_queue_depth"),
        "persistence_queue_depth": stats("persistence_queue_depth"),
        "gpu_path_busy_percent": round(100.0 * sum(bool(s["gpu_path_busy"]) for s in samples) / len(samples), 2) if samples else 0.0,
    }


def _cleanup(sqlite_db: SQLiteDatabase, vector_db: VectorDatabase, job_manager: JobManager, job_id: str) -> None:
    job_manager.cleanup_job_resources(job_id)
    vector_db.delete_by_job(job_id)
    with sqlite_db.connection() as conn:
        conn.execute("DELETE FROM detections WHERE track_id IN (SELECT id FROM tracks WHERE job_id = ?)", (job_id,))
        conn.execute("DELETE FROM tracks WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM cameras WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def run_case(video: Path, cameras: int, seconds: float, interval: float, profile: str) -> Dict[str, Any]:
    sqlite_db = SQLiteDatabase(SQLITE_DB_PATH)
    vector_db = VectorDatabase(CHROMA_PERSIST_DIR)
    media = MediaStorage()
    dispatcher = ResultDispatcher(sqlite_db, vector_db, ActiveQueryMatcher())
    manager = JobManager(sqlite_db)
    scheduler = GPUScheduler(manager, media, dispatcher)
    scheduler.load_models()
    scheduler.start()

    sources = [{"name": f"Telemetry_Cam_{i + 1}", "type": "file", "path": str(video.resolve())} for i in range(cameras)]
    job_id = manager.create_job(f"Telemetry {cameras} cameras", profile, sources, investigation_type="benchmark")
    before = PerformanceMonitorInstance.get_metrics()["frames_analyzed"]
    sampler = TelemetrySampler(scheduler, dispatcher, manager, interval)
    started = time.monotonic()
    sampler.start()
    try:
        manager.start_job(job_id)
        deadline = started + seconds
        while time.monotonic() < deadline:
            time.sleep(0.1)

        # Stop only benchmark decoders, then prove all asynchronous work drains.
        benchmark_job = manager.active_jobs.get(job_id)
        decoder_counts = {
            dec.camera_id: dec.frames_decoded
            for dec in (benchmark_job.decoders.values() if benchmark_job else [])
        }
        manager.cleanup_job_resources(job_id)
        # Let a scheduler iteration that already pulled a decoder batch finish
        # enqueueing before measuring the enrichment drain.
        time.sleep(1.0)
        enrichment_drained = scheduler.flush_enrichment_pool(timeout=20.0)
        dispatcher.flush_and_wait(timeout=20.0)
        # A queue can be empty while the writer still owns a partial local
        # batch. Stop/join it before examining or deleting benchmark records.
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
        return {
            "camera_count": cameras,
            "profile": profile,
            "wall_clock_sec": round(elapsed, 3),
            "frames_analyzed": max(0, after - before),
            "effective_fps": round(max(0, after - before) / max(elapsed, 0.001), 3),
            "detections_persisted": total_detections,
            "detections_per_camera": per_camera,
            "decoder_frames": decoder_counts,
            "persistence_drained": enrichment_drained and scheduler.enrichment_queue.unfinished_tasks == 0 and not dispatcher.write_thread.is_alive(),
            "telemetry": _summarize(sampler.samples),
            "samples": sampler.samples,
        }
    finally:
        sampler.stop()
        _cleanup(sqlite_db, vector_db, manager, job_id)
        scheduler.stop_scheduler()
        dispatcher.stop_dispatcher()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded AICatch real-pipeline GPU telemetry benchmark")
    parser.add_argument("--video", default="data/media/uploads/inside.mp4")
    parser.add_argument("--cameras", default="2,4,5", help="Comma-separated camera counts")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--profile", choices=("FAST", "BALANCED", "FORENSIC"), default="FORENSIC")
    parser.add_argument("--output", default="benchmarks/results/multicamera_telemetry.json")
    args = parser.parse_args()
    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(video)
    if args.seconds <= 0 or args.interval < 0.2:
        raise ValueError("--seconds must be positive and --interval must be at least 0.2")

    results = [run_case(video, int(value), args.seconds, args.interval, args.profile) for value in args.cameras.split(",")]
    report = {"video": str(video), "sample_interval_sec": args.interval, "cases": results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": [{k: v for k, v in case.items() if k != "samples"} for case in results]}, indent=2))


if __name__ == "__main__":
    main()
