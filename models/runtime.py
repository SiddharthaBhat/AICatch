"""Lazy, VRAM-aware model runtime shared by indexing and search.

The runtime owns one instance of each model and deliberately keeps only one
GPU-heavy feature model resident at a time on 4 GB devices.  Callers should
obtain models through this class rather than importing/loading weights inside
per-frame functions.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from typing import Any

import torch

from .registry import ModelSpec, registry

LOGGER = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any
    processor: Any | None
    device: str
    dtype: str
    memory_mb: float | None


class ModelRuntime:
    def __init__(self) -> None:
        self.loaded: dict[str, LoadedModel] = {}

    @staticmethod
    def _device(spec: ModelSpec) -> str:
        return "cuda" if spec.default_device == "cuda" and torch.cuda.is_available() else "cpu"

    @staticmethod
    def _memory_mb() -> float | None:
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.memory_allocated() / 1024**2, 1)

    def release(self, task: str) -> None:
        loaded = self.loaded.pop(task, None)
        if loaded:
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def status(self) -> list[dict[str, Any]]:
        result = []
        for task in ("detection", "visual", "reid", "face", "ocr"):
            active = registry.active(task)
            live = self.loaded.get(task)
            result.append({**registry.metadata(task, live.device if live else None, live.memory_mb if live else None), "loaded": live is not None})
        return result

    def visual(self) -> LoadedModel:
        spec = registry.active("visual")
        current = self.loaded.get("visual")
        if current and current.spec.key == spec.key:
            return current
        self.release("visual")
        # A visual encoder can coexist with YOLO on many 4 GB cards only in FP16;
        # we offload previously used feature models before loading it.
        for task in ("reid", "face"):
            self.release(task)
        from transformers import AutoModel, AutoProcessor
        device = self._device(spec)
        # Indexing must not stall a long CCTV job on a hub metadata request.
        # Model acquisition is explicit; inference uses verified cached weights.
        processor = AutoProcessor.from_pretrained(spec.source, local_files_only=True)
        model = AutoModel.from_pretrained(spec.source, local_files_only=True).to(device).eval()
        dtype = "float16" if device == "cuda" else "float32"
        if device == "cuda":
            model = model.half()
        loaded = LoadedModel(spec, model, processor, device, dtype, self._memory_mb())
        self.loaded["visual"] = loaded
        LOGGER.info("Loaded visual model %s on %s (%s)", spec.name, device, dtype)
        return loaded


runtime = ModelRuntime()
