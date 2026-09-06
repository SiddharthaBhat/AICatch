"""Single source of truth for AICatch model configuration.

Candidate models deliberately remain candidates until a recorded local benchmark
promotes them.  This module never downloads weights and never changes the
last-known-good selection by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    version: str
    task: str
    source: str
    default_device: str
    dtype: str = "float32"
    candidate: bool = False


class ModelRegistry:
    """Stores active and candidate choices without spreading paths across code."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[1]
        self.state_path = self.root / "data" / "model_registry.json"
        self.specs = {
            "yolov8n": ModelSpec("yolov8n", "YOLOv8n", "8", "detection", "yolov8n.pt", "cuda"),
            "yolo11s": ModelSpec("yolo11s", "YOLO11s", "11", "detection", "yolo11s.pt", "cuda"),
            "yolo26n": ModelSpec("yolo26n", "YOLO26n", "candidate", "detection", "yolo26n.pt", "cuda", candidate=True),
            "yolo26s": ModelSpec("yolo26s", "YOLO26s", "candidate", "detection", "yolo26s.pt", "cuda", candidate=True),
            "clip_vit_b32": ModelSpec("clip_vit_b32", "CLIP ViT-B/32", "v1", "visual", "openai/clip-vit-base-patch32", "cuda"),
            "siglip2_base": ModelSpec("siglip2_base", "SigLIP2 Base", "v2", "visual", "google/siglip2-base-patch16-256", "cuda", candidate=True),
            "mobilenet_v3_reid_576": ModelSpec("mobilenet_v3_reid_576", "MobileNetV3 Re-ID", "v1", "reid", "torchvision", "cuda"),
            "osnet_x0_5": ModelSpec("osnet_x0_5", "OSNet x0.5", "candidate", "reid", "osnet_x0_5", "cuda", candidate=True),
            "facenet": ModelSpec("facenet", "FaceNet InceptionResnetV1", "vggface2", "face", "facenet_pytorch", "cpu"),
            "adaface": ModelSpec("adaface", "AdaFace", "candidate", "face", "adaface", "cpu", candidate=True),
            "easyocr": ModelSpec("easyocr", "EasyOCR", "v1", "ocr", "easyocr", "cpu"),
            "ppocrv6": ModelSpec("ppocrv6", "PP-OCRv6", "candidate", "ocr", "ppocrv6", "cpu", candidate=True),
        }
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        defaults = {"active": {"detection": "yolo11s", "visual": "clip_vit_b32", "reid": "mobilenet_v3_reid_576", "face": "facenet", "ocr": "easyocr"}, "fallback": {"detection": "yolov8n"}, "promotions": []}
        if not self.state_path.exists():
            return defaults
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
            return {**defaults, **saved}
        except (OSError, json.JSONDecodeError):
            return defaults

    def active(self, task: str) -> ModelSpec:
        key = os.getenv(f"AICATCH_{task.upper()}_MODEL", self._state["active"][task])
        return self.specs[key]

    def fallback(self, task: str) -> ModelSpec | None:
        key = self._state.get("fallback", {}).get(task)
        return self.specs.get(key) if key else None

    def metadata(self, task: str, device: str | None = None, memory_mb: float | None = None) -> dict[str, Any]:
        spec = self.active(task)
        return {**asdict(spec), "device": device or spec.default_device, "memory_mb": memory_mb, "status": "configured"}

    def promote(self, task: str, candidate: str, benchmark: dict[str, Any]) -> None:
        """Promote only a candidate with an explicitly passing local benchmark."""
        if candidate not in self.specs or self.specs[candidate].task != task:
            raise ValueError(f"{candidate!r} is not a {task} model")
        if not benchmark.get("passed") or not benchmark.get("dataset_id"):
            raise ValueError("Promotion requires a passing benchmark with a dataset_id")
        previous = self._state["active"][task]
        self._state["active"][task] = candidate
        self._state.setdefault("fallback", {})[task] = previous
        self._state.setdefault("promotions", []).append({"task": task, "from": previous, "to": candidate, "benchmark": benchmark})
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")


registry = ModelRegistry()
