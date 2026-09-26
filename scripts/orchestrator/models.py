"""
Data models and state definitions for AICatch Orchestrator (V0.2).
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    """Explicit deterministic state machine statuses for tasks."""

    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    AGENT_COMPLETE = "AGENT_COMPLETE"
    VALIDATING = "VALIDATING"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    DISCARDED = "DISCARDED"
    VALIDATED = "VALIDATED"
    READY_FOR_MERGE = "READY_FOR_MERGE"
    MERGED = "MERGED"
    BLOCKED = "BLOCKED"
    STOPPED = "STOPPED"


class ExecutionMode(str, Enum):
    """Orchestrator execution modes."""

    DRY_RUN = "DRY_RUN"
    SAFE = "SAFE"
    RUN = "RUN"
    NOTION_LOOP = "NOTION_LOOP"


class AgentExitStatus(str, Enum):
    """Exit status of the Antigravity agent execution."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    CRASHED = "CRASHED"
    UNKNOWN = "UNKNOWN"


@dataclass
class Task:
    """Represents a structured engineering task to be orchestrated."""

    task_id: str
    objective: str
    description: str
    baseline: str = ""
    priority: str = "NORMAL"
    acceptance_criteria: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    allowed_paths: List[str] = field(default_factory=list)
    forbidden_paths: List[str] = field(default_factory=list)
    validation_commands: List[str] = field(default_factory=list)
    timeout_seconds: int = 1800
    max_attempts: int = 1
    status: TaskStatus = TaskStatus.QUEUED
    worktree_path: Optional[str] = None
    conversation_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    result_summary: Optional[str] = None
    error: Optional[str] = None
    next_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Reconstruct task from dictionary."""
        data_copy = data.copy()
        if "status" in data_copy and isinstance(data_copy["status"], str):
            data_copy["status"] = TaskStatus(data_copy["status"])
        return cls(**data_copy)


@dataclass
class AgentResult:
    """Raw and structured results captured from an Antigravity agent execution."""

    conversation_id: str
    exit_status: AgentExitStatus
    raw_output: str
    structured_output: Optional[Dict[str, Any]] = None
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    tool_calls_count: int = 0
    transcript_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent result to dictionary."""
        data = asdict(self)
        data["exit_status"] = self.exit_status.value
        return data


@dataclass
class TaskResult:
    """Overall outcome of a task execution through the orchestrator."""

    task_id: str
    status: TaskStatus
    agent_result: Optional[AgentResult] = None
    validation_passed: bool = False
    validation_output: str = ""
    start_time: str = ""
    end_time: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    decision: str = "PENDING"
    worktree_path: Optional[str] = None
    git_status_before: str = ""
    git_status_after: str = ""
    aicatch_integrity: str = "PASS"

    def to_dict(self) -> Dict[str, Any]:
        """Convert task result to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        if self.agent_result:
            data["agent_result"] = self.agent_result.to_dict()
        return data


@dataclass
class OrchestratorBatchResult:
    """Summary of an entire orchestrator execution run."""

    mode: ExecutionMode
    start_time: str
    end_time: str
    total_elapsed_seconds: float
    tasks_total: int
    tasks_succeeded: int
    tasks_failed: int
    task_results: List[TaskResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert batch result to dictionary."""
        return {
            "mode": self.mode.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "tasks_total": self.tasks_total,
            "tasks_succeeded": self.tasks_succeeded,
            "tasks_failed": self.tasks_failed,
            "task_results": [tr.to_dict() for tr in self.task_results],
        }
