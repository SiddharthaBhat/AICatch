"""
Unit and integration tests for OrchestratorRunner.
"""

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
from scripts.orchestrator.config import OrchestratorConfig
from scripts.orchestrator.models import (
    AgentExitStatus,
    AgentResult,
    ExecutionMode,
    Task,
    TaskResult,
    TaskStatus,
)
from scripts.orchestrator.runner import OrchestratorRunner


class TestRunner(unittest.TestCase):
    def test_dry_run_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = OrchestratorConfig(log_dir=tmp_path / "logs")
            runner = OrchestratorRunner(config=config, mode=ExecutionMode.DRY_RUN)

            tasks = [
                Task(
                    task_id="dry_task_1",
                    objective="Dry run task 1",
                    description="Inspection simulation",
                ),
                Task(
                    task_id="dry_task_2",
                    objective="Dry run task 2",
                    description="Inspection simulation 2",
                ),
            ]

            batch_result = runner.execute_tasks(tasks)

            self.assertEqual(batch_result.tasks_total, 2)
            self.assertEqual(batch_result.tasks_succeeded, 2)
            self.assertEqual(batch_result.tasks_failed, 0)
            self.assertEqual(len(batch_result.task_results), 2)
            self.assertEqual(batch_result.task_results[0].status, TaskStatus.AGENT_COMPLETE)
            self.assertEqual(batch_result.task_results[1].status, TaskStatus.AGENT_COMPLETE)

            # Verify log summary files were created
            self.assertTrue((tmp_path / "logs" / "task_dry_task_1.json").exists())
            self.assertTrue((tmp_path / "logs" / "task_dry_task_2.json").exists())
            self.assertTrue((tmp_path / "logs" / "orchestrator.jsonl").exists())

    @patch.object(OrchestratorRunner, "_execute_single_task")
    def test_sequential_execution_order(self, mock_single):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = OrchestratorConfig(log_dir=tmp_path / "logs")
            runner = OrchestratorRunner(config=config, mode=ExecutionMode.SAFE)

            t1 = Task(task_id="seq_1", objective="First", description="Desc")
            t2 = Task(task_id="seq_2", objective="Second", description="Desc")

            mock_single.side_effect = [
                TaskResult(task_id="seq_1", status=TaskStatus.AGENT_COMPLETE),
                TaskResult(task_id="seq_2", status=TaskStatus.AGENT_COMPLETE),
            ]

            res = runner.execute_tasks([t1, t2])
            self.assertEqual(res.tasks_succeeded, 2)
            self.assertEqual(mock_single.call_count, 2)

    def test_timeout_handling_in_runner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = OrchestratorConfig(
                log_dir=tmp_path / "logs",
                brain_dir=tmp_path / "brain",
            )
            runner = OrchestratorRunner(config=config, mode=ExecutionMode.SAFE)

            # Mock agent runner to return TIMEOUT
            mock_agent = MagicMock()
            mock_agent.run_task.return_value = AgentResult(
                conversation_id="conv_timeout",
                exit_status=AgentExitStatus.TIMEOUT,
                raw_output="",
                elapsed_seconds=1.5,
                error_message="Agent execution timed out",
            )
            runner.agent_runner = mock_agent

            task = Task(
                task_id="timeout_task",
                objective="Test timeout",
                description="Will time out",
                timeout_seconds=1,
            )

            res = runner.execute_tasks([task])
            self.assertEqual(res.tasks_failed, 1)
            self.assertEqual(res.task_results[0].status, TaskStatus.FAILED)
            self.assertEqual(res.task_results[0].agent_result.exit_status, AgentExitStatus.TIMEOUT)


if __name__ == "__main__":
    unittest.main()
