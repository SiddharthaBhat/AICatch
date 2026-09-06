"""Benchmark evidence store and promotion guard.

This is intentionally model-agnostic: inference runners add measured metrics;
the guard prevents a model becoming active on unlabelled or losing results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from models.registry import registry


@dataclass
class BenchmarkResult:
    dataset_id: str
    task: str
    model_key: str
    metrics: dict[str, float]
    environment: dict[str, Any]

    def passes_against(self, baseline: "BenchmarkResult") -> bool:
        if self.task != baseline.task or self.dataset_id != baseline.dataset_id:
            return False
        # Accuracy must not regress. Cost limits avoid an unusable 4 GB deployment.
        for key in ("precision", "recall", "precision_at_5", "recall_at_5"):
            if key in baseline.metrics and self.metrics.get(key, float("-inf")) < baseline.metrics[key]:
                return False
        if self.metrics.get("false_positive_rate", 0.0) > baseline.metrics.get("false_positive_rate", float("inf")):
            return False
        return self.metrics.get("peak_vram_mb", 0.0) <= 3600 and self.metrics.get("indexing_fps", 0.0) > 0


class BenchmarkStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(__file__).resolve().parent / "data" / "benchmarks.jsonl"

    def record(self, result: BenchmarkResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(result)) + "\n")

    def promote_if_winner(self, result: BenchmarkResult, baseline: BenchmarkResult) -> bool:
        passed = result.passes_against(baseline)
        self.record(result)
        if passed:
            registry.promote(result.task, result.model_key, {"passed": True, "dataset_id": result.dataset_id, "metrics": result.metrics, "environment": result.environment})
        return passed
