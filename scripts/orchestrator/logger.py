"""
Structured logging module for AICatch Orchestrator.
Emits human-readable console output and writes structured JSONL telemetry.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional


class StructuredOrchestratorLogger:
    """Manages console formatting and persistent JSONL execution logs."""

    def __init__(self, log_dir: Path, run_id: Optional[str] = None):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = (
            run_id
            or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        )
        self.jsonl_path = self.log_dir / "orchestrator.jsonl"
        self._setup_standard_logger()

    def _setup_standard_logger(self) -> None:
        self.logger = logging.getLogger(f"AICatchOrchestrator.{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # Console handler with clean formatting
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def _write_jsonl(self, event_type: str, data: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "run_id": self.run_id,
            "event_type": event_type,
            **data,
        }
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to append to jsonl log: {e}")

    def info(self, msg: str, **kwargs: Any) -> None:
        self.logger.info(msg)
        if kwargs:
            self._write_jsonl("INFO", {"message": msg, **kwargs})

    def warning(self, msg: str, **kwargs: Any) -> None:
        self.logger.warning(msg)
        self._write_jsonl("WARNING", {"message": msg, **kwargs})

    def error(self, msg: str, **kwargs: Any) -> None:
        self.logger.error(msg)
        self._write_jsonl("ERROR", {"message": msg, **kwargs})

    def log_task_start(self, task_id: str, objective: str, mode: str) -> None:
        self.info(
            f"--> Task Started: [{task_id}] (Mode: {mode})",
            task_id=task_id,
            objective=objective,
            mode=mode,
            event_name="TASK_START",
        )

    def log_task_complete(
        self,
        task_id: str,
        status: str,
        elapsed_sec: float,
        conversation_id: Optional[str] = None,
    ) -> None:
        self.info(
            f"<-- Task Completed: [{task_id}] Status: {status} (Elapsed: {elapsed_sec:.2f}s, Conv: {conversation_id})",
            task_id=task_id,
            status=status,
            elapsed_seconds=elapsed_sec,
            conversation_id=conversation_id,
            event_name="TASK_COMPLETE",
        )

    def save_task_summary(self, task_id: str, task_result_dict: Dict[str, Any]) -> Path:
        """Save full task result payload to logs/orchestrator/task_<id>.json."""
        target = self.log_dir / f"task_{task_id}.json"
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(task_result_dict, f, indent=2)
        except Exception as e:
            self.error(f"Failed to save task summary {target}: {e}")
        return target
