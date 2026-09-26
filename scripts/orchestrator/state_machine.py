"""
Deterministic State Machine for AICatch Orchestration Tasks.
Enforces explicit legal transitions and invariants.
"""

from datetime import datetime
from typing import Dict, List, Set
from .models import Task, TaskStatus


class InvalidStateTransitionError(Exception):
    """Raised when an illegal task state transition is attempted."""

    pass


class TaskStateMachine:
    """Validates and applies deterministic state machine transitions for tasks."""

    # Explicit map of legal forward and backward state transitions
    TRANSITIONS: Dict[TaskStatus, Set[TaskStatus]] = {
        TaskStatus.QUEUED: {
            TaskStatus.CLAIMED,
            TaskStatus.BLOCKED,
            TaskStatus.STOPPED,
        },
        TaskStatus.CLAIMED: {
            TaskStatus.RUNNING,
            TaskStatus.AGENT_COMPLETE,
            TaskStatus.COMPLETED,
            TaskStatus.QUEUED,
            TaskStatus.STOPPED,
            TaskStatus.FAILED,
        },
        TaskStatus.RUNNING: {
            TaskStatus.AGENT_COMPLETE,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.TIMEOUT,
            TaskStatus.STOPPED,
        },
        TaskStatus.AGENT_COMPLETE: {
            TaskStatus.COMPLETED,
            TaskStatus.VALIDATING,
            TaskStatus.REVIEW_REQUIRED,
            TaskStatus.APPROVED,
            TaskStatus.FAILED,
            TaskStatus.DISCARDED,
            TaskStatus.STOPPED,
        },
        TaskStatus.COMPLETED: {
            TaskStatus.VALIDATING,
            TaskStatus.REVIEW_REQUIRED,
            TaskStatus.APPROVED,
            TaskStatus.DISCARDED,
        },
        TaskStatus.VALIDATING: {
            TaskStatus.VALIDATED,
            TaskStatus.REVIEW_REQUIRED,
            TaskStatus.FAILED,
            TaskStatus.DISCARDED,
            TaskStatus.STOPPED,
        },
        TaskStatus.VALIDATED: {
            TaskStatus.REVIEW_REQUIRED,
            TaskStatus.APPROVED,
            TaskStatus.READY_FOR_MERGE,
            TaskStatus.DISCARDED,
        },
        TaskStatus.REVIEW_REQUIRED: {
            TaskStatus.APPROVED,
            TaskStatus.REJECTED,
            TaskStatus.FAILED,
            TaskStatus.DISCARDED,
            TaskStatus.STOPPED,
        },
        TaskStatus.APPROVED: {
            TaskStatus.READY_FOR_MERGE,
            TaskStatus.MERGED,
            TaskStatus.COMPLETED,
            TaskStatus.DISCARDED,
        },
        TaskStatus.READY_FOR_MERGE: {
            TaskStatus.MERGED,
            TaskStatus.FAILED,
            TaskStatus.DISCARDED,
            TaskStatus.STOPPED,
        },
        TaskStatus.REJECTED: {
            TaskStatus.DISCARDED,
            TaskStatus.QUEUED,
        },
        TaskStatus.FAILED: {
            TaskStatus.DISCARDED,
            TaskStatus.QUEUED,
            TaskStatus.STOPPED,
        },
        TaskStatus.TIMEOUT: {
            TaskStatus.DISCARDED,
            TaskStatus.QUEUED,
            TaskStatus.STOPPED,
        },
        TaskStatus.DISCARDED: set(),  # Terminal state
        TaskStatus.MERGED: set(),     # Terminal state
        TaskStatus.BLOCKED: {
            TaskStatus.QUEUED,
            TaskStatus.STOPPED,
        },
        TaskStatus.STOPPED: {
            TaskStatus.QUEUED,
        },
    }

    @classmethod
    def can_transition(cls, current: TaskStatus, target: TaskStatus) -> bool:
        """Check if a transition from current to target is allowed."""
        if current == target:
            return True
        allowed = cls.TRANSITIONS.get(current, set())
        return target in allowed

    @classmethod
    def transition(
        cls, task: Task, target: TaskStatus, reason: str = ""
    ) -> None:
        """Apply a validated state transition to a Task instance."""
        if not cls.can_transition(task.status, target):
            raise InvalidStateTransitionError(
                f"Illegal state transition for task '{task.task_id}': "
                f"Cannot move from {task.status.value} to {target.value}. Reason: {reason}"
            )
        task.status = target
        task.updated_at = datetime.utcnow().isoformat() + "Z"
