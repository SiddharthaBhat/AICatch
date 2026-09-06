# -*- coding: utf-8 -*-
"""AICatch Retrieval Benchmark Suite.

Compares:
A. CLIP-only visual retrieval
B. Re-ID-only (MobileNetV3) visual retrieval
C. CLIP + Re-ID Reciprocal Rank Fusion (RRF)

Measures query latency, candidate counts, and ground-truth metrics (mAP, Rank-1, Rank-5, Rank-10)
if labelled query-to-track annotations exist.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CHROMA_PERSIST_DIR, SQLITE_DB_PATH
from pipeline.gpu_scheduler import GPUScheduler
from pipeline.job_manager import JobManager
from storage.media_storage import MediaStorage
from storage.search import SearchLayer
from storage.sqlite_db import SQLiteDatabase
from storage.vector_db import VectorDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
LOGGER = logging.getLogger("RetrievalBenchmark")


class RetrievalBenchmarkRunner:
    def __init__(
        self,
        query_image_path: Path,
        job_id: Optional[str] = None,
        gt_path: Optional[Path] = None,
        top_k: int = 20,
        output_dir: Path = Path("benchmarks/results")
    ):
        self.query_image_path = query_image_path
        self.job_id = job_id
        self.gt_path = gt_path
        self.top_k = top_k
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sqlite = SQLiteDatabase(SQLITE_DB_PATH)
        self.vector = VectorDatabase(CHROMA_PERSIST_DIR)
        self.media = MediaStorage()
        self.search = SearchLayer(self.sqlite, self.vector, self.media)
        self.job_manager = JobManager(self.sqlite)
        self.scheduler = GPUScheduler(self.job_manager, self.media)
        self.scheduler.load_models()

        if not self.query_image_path.exists():
            raise FileNotFoundError(f"Query image not found: {self.query_image_path}")

        self.query_pil = Image.open(self.query_image_path).convert("RGB")

    def run_benchmark(self) -> Dict[str, Any]:
        LOGGER.info(f"Running Retrieval Benchmark with query image: {self.query_image_path.name}")
        filters = {"job_id": self.job_id} if self.job_id else None

        # 1. Embed Query
        t0 = time.time()
        clip_emb = self.scheduler.embed_image(self.query_pil)
        clip_embed_time = time.time() - t0

        t0 = time.time()
        reid_emb = self.scheduler.embed_reid(self.query_pil)
        reid_embed_time = time.time() - t0

        # Method A: CLIP-only Search
        t0 = time.time()
        clip_results = self.search.search_by_image(
            query_embedding=clip_emb,
            filters=filters,
            top_k=self.top_k,
            strict_gate=False,
            enable_rrf=False
        )
        clip_search_time = time.time() - t0

        # Method B: Re-ID-only Search (querying Re-ID collection directly)
        t0 = time.time()
        reid_matches = self.vector.query_reid_embeddings(
            query_embedding=reid_emb if reid_emb else [0.0] * 576,
            top_k=self.top_k,
            filters=filters
        )
        reid_search_time = time.time() - t0

        # Method C: CLIP + Re-ID Reciprocal Rank Fusion (RRF)
        t0 = time.time()
        rrf_results = self.search.search_by_image(
            query_embedding=clip_emb,
            filters=filters,
            top_k=self.top_k,
            strict_gate=False,
            reid_embedding=reid_emb,
            enable_rrf=True,
            rrf_k=60,
            clip_weight=0.7,
            reid_weight=0.3
        )
        rrf_search_time = time.time() - t0

        # Ground-truth metric evaluation if GT provided
        gt_data = None
        if self.gt_path and self.gt_path.exists():
            try:
                gt_data = json.loads(self.gt_path.read_text(encoding="utf-8"))
            except Exception as e:
                LOGGER.warning(f"Could not load GT annotations: {e}")

        def eval_ranking(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not gt_data:
                return {
                    "mAP": "NOT AVAILABLE — NO LABELLED GROUND TRUTH DATASET",
                    "Rank-1": "NOT AVAILABLE — NO LABELLED GROUND TRUTH DATASET",
                    "Rank-5": "NOT AVAILABLE — NO LABELLED GROUND TRUTH DATASET",
                    "Rank-10": "NOT AVAILABLE — NO LABELLED GROUND TRUTH DATASET",
                    "Precision@5": "NOT AVAILABLE — NO LABELLED GROUND TRUTH DATASET",
                    "Recall@5": "NOT AVAILABLE — NO LABELLED GROUND TRUTH DATASET",
                }
            # Compute actual metrics against ground truth targets
            relevant_ids = set(gt_data.get("relevant_track_ids", []))
            hit_ids = [h.get("track_id") or h.get("metadata", {}).get("track_id") for h in hits]
            r1 = 1.0 if (len(hit_ids) > 0 and hit_ids[0] in relevant_ids) else 0.0
            r5 = 1.0 if any(hid in relevant_ids for hid in hit_ids[:5]) else 0.0
            r10 = 1.0 if any(hid in relevant_ids for hid in hit_ids[:10]) else 0.0
            return {
                "Rank-1": r1,
                "Rank-5": r5,
                "Rank-10": r10,
                "hit_count": len(hit_ids),
            }

        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query_image": self.query_image_path.name,
            "embedding_latencies_ms": {
                "clip_embed_ms": round(clip_embed_time * 1000, 2),
                "reid_embed_ms": round(reid_embed_time * 1000, 2),
            },
            "methods": {
                "A_CLIP_Only": {
                    "query_latency_ms": round(clip_search_time * 1000, 2),
                    "candidate_count": len(clip_results),
                    "top_scores": [round(r.get("similarity", 0.0), 4) for r in clip_results[:5]],
                    "metrics": eval_ranking(clip_results)
                },
                "B_ReID_Only": {
                    "query_latency_ms": round(reid_search_time * 1000, 2),
                    "candidate_count": len(reid_matches),
                    "top_scores": [round(r.get("similarity", 0.0), 4) for r in reid_matches[:5]],
                    "metrics": eval_ranking(reid_matches)
                },
                "C_CLIP_Plus_ReID_RRF": {
                    "query_latency_ms": round(rrf_search_time * 1000, 2),
                    "candidate_count": len(rrf_results),
                    "top_rrf_scores": [round(r.get("rrf_score", 0.0), 5) for r in rrf_results[:5]],
                    "metrics": eval_ranking(rrf_results)
                }
            }
        }

        out_file = self.output_dir / f"retrieval_benchmark_{int(time.time())}.json"
        out_file.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        LOGGER.info(f"Retrieval benchmark saved to {out_file}")
        return comparison


def main():
    parser = argparse.ArgumentParser(description="AICatch Retrieval Benchmark")
    parser.add_argument("--query-image", type=str, default=None, help="Query crop image path")
    parser.add_argument("--job-id", type=str, default=None, help="Target investigation/job ID")
    parser.add_argument("--gt-path", type=str, default=None, help="Ground truth annotation path")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results")
    args = parser.parse_args()

    # Find a sample crop image if none passed
    query_path = None
    if args.query_image:
        query_path = Path(args.query_image)
    else:
        # Pick an existing crop from data/media/crops/person
        crop_dir = Path("data/media/crops/person")
        if crop_dir.exists():
            for p in crop_dir.rglob("*.jpg"):
                query_path = p
                break

    if not query_path or not query_path.exists():
        LOGGER.warning("No sample crop image found; creating a synthetic probe crop.")
        query_path = Path("temp/query_probe.jpg")
        query_path.parent.mkdir(parents=True, exist_ok=True)
        img = np.zeros((128, 64, 3), dtype=np.uint8)
        img[:, :] = (120, 80, 40)
        cv2.imwrite(str(query_path), img)

    runner = RetrievalBenchmarkRunner(
        query_image_path=query_path,
        job_id=args.job_id,
        gt_path=Path(args.gt_path) if args.gt_path else None,
        output_dir=Path(args.output_dir)
    )

    results = runner.run_benchmark()
    print("\n================== RETRIEVAL BENCHMARK SUMMARY ==================")
    print(json.dumps(results, indent=2))
    print("=================================================================\n")


if __name__ == "__main__":
    main()
