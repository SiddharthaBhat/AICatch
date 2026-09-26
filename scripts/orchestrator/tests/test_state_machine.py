"""
Unit tests for deterministic task state machine.
"""

import unittest
from scripts.orchestrator.models import Task, TaskStatus
from scripts.orchestrator.state_machine import (
    InvalidStateTransitionError,
    TaskStateMachine,
)


class TestStateMachine(unittest.TestCase):
    def test_valid_forward_transitions(self):
        task = Task(task_id="t1", objective="Obj", description="Desc")
        self.assertEqual(task.status, TaskStatus.QUEUED)

        TaskStateMachine.transition(task, TaskStatus.CLAIMED, "Claimed by runner")
        self.assertEqual(task.status, TaskStatus.CLAIMED)

        TaskStateMachine.transition(task, TaskStatus.RUNNING, "Started execution")
        self.assertEqual(task.status, TaskStatus.RUNNING)

        TaskStateMachine.transition(task, TaskStatus.AGENT_COMPLETE, "Agent finished")
        self.assertEqual(task.status, TaskStatus.AGENT_COMPLETE)

        TaskStateMachine.transition(task, TaskStatus.VALIDATING, "Validating output")
        self.assertEqual(task.status, TaskStatus.VALIDATING)

        TaskStateMachine.transition(task, TaskStatus.VALIDATED, "Validation passed")
        self.assertEqual(task.status, TaskStatus.VALIDATED)

        TaskStateMachine.transition(task, TaskStatus.APPROVED, "Engineering review approved")
        self.assertEqual(task.status, TaskStatus.APPROVED)

        TaskStateMachine.transition(task, TaskStatus.READY_FOR_MERGE, "Ready to merge")
        self.assertEqual(task.status, TaskStatus.READY_FOR_MERGE)

        TaskStateMachine.transition(task, TaskStatus.MERGED, "Merged to main")
        self.assertEqual(task.status, TaskStatus.MERGED)

    def test_invalid_transitions(self):
        task = Task(task_id="t2", objective="Obj", description="Desc")
        self.assertEqual(task.status, TaskStatus.QUEUED)

        with self.assertRaises(InvalidStateTransitionError):
            TaskStateMachine.transition(task, TaskStatus.MERGED, "Illegal jump")

        with self.assertRaises(InvalidStateTransitionError):
            TaskStateMachine.transition(task, TaskStatus.VALIDATED, "Illegal jump")

    def test_failure_transitions(self):
        task = Task(task_id="t3", objective="Obj", description="Desc")
        TaskStateMachine.transition(task, TaskStatus.CLAIMED)
        TaskStateMachine.transition(task, TaskStatus.RUNNING)
        TaskStateMachine.transition(task, TaskStatus.FAILED, "Agent crashed")
        self.assertEqual(task.status, TaskStatus.FAILED)

        TaskStateMachine.transition(task, TaskStatus.DISCARDED, "Worktree cleaned up")
        self.assertEqual(task.status, TaskStatus.DISCARDED)


if __name__ == "__main__":
    unittest.main()
