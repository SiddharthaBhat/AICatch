"""
Unit tests for data models and serialization.
"""

import unittest
from scripts.orchestrator.models import (
    AgentExitStatus,
    AgentResult,
    ExecutionMode,
    Task,
    TaskResult,
    TaskStatus,
)


class TestModels(unittest.TestCase):
    def test_task_serialization(self):
        task = Task(
            task_id="test_001",
            objective="Verify system architecture",
            description="Harmless inspection of system files.",
            acceptance_criteria=["Return SUCCESS"],
            timeout_seconds=300,
        )
        d = task.to_dict()
        self.assertEqual(d["task_id"], "test_001")
        self.assertEqual(d["status"], "QUEUED")
        self.assertEqual(d["timeout_seconds"], 300)

        reconstructed = Task.from_dict(d)
        self.assertEqual(reconstructed.task_id, "test_001")
        self.assertEqual(reconstructed.status, TaskStatus.QUEUED)
        self.assertEqual(reconstructed.acceptance_criteria, ["Return SUCCESS"])

    def test_agent_result_serialization(self):
        res = AgentResult(
            conversation_id="conv_123",
            exit_status=AgentExitStatus.SUCCESS,
            raw_output="AUTOMATION_TEST_SUCCESS",
            elapsed_seconds=4.5,
            tool_calls_count=2,
        )
        d = res.to_dict()
        self.assertEqual(d["conversation_id"], "conv_123")
        self.assertEqual(d["exit_status"], "SUCCESS")
        self.assertEqual(d["elapsed_seconds"], 4.5)
        self.assertEqual(d["tool_calls_count"], 2)

    def test_task_result_serialization(self):
        res = TaskResult(
            task_id="task_abc",
            status=TaskStatus.AGENT_COMPLETE,
            agent_result=AgentResult(
                conversation_id="c_1",
                exit_status=AgentExitStatus.SUCCESS,
                raw_output="OK",
            ),
            validation_passed=True,
        )
        d = res.to_dict()
        self.assertEqual(d["task_id"], "task_abc")
        self.assertEqual(d["status"], "AGENT_COMPLETE")
        self.assertTrue(d["validation_passed"])
        self.assertEqual(d["agent_result"]["conversation_id"], "c_1")


if __name__ == "__main__":
    unittest.main()
