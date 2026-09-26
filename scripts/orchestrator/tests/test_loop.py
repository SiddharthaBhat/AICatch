"""
Unit tests for NotionOrchestratorLoop.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from scripts.orchestrator.config import OrchestratorConfig
from scripts.orchestrator.loop import NotionOrchestratorLoop
from scripts.orchestrator.models import (
    AgentExitStatus,
    AgentResult,
    Task,
    TaskStatus,
)


class TestLoop(unittest.TestCase):
    def test_loop_processes_single_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = OrchestratorConfig(
                log_dir=tmp_path / "logs",
                notion_poll_interval_sec=0.1,
            )
            loop = NotionOrchestratorLoop(config=config, max_iterations=1)

            # Mock agent runner
            mock_agent = MagicMock()
            mock_agent.run_task.return_value = AgentResult(
                conversation_id="conv_loop_test",
                exit_status=AgentExitStatus.SUCCESS,
                raw_output="AUTOMATION_TEST_SUCCESS",
                elapsed_seconds=2.0,
            )
            loop.agent_runner = mock_agent

            # Mock git status to return clean before and after
            loop.check_git_status = MagicMock(return_value="")

            # Run loop
            loop.run_loop()

            # Verify task was processed and marked completed
            self.assertEqual(loop.bridge.get_next_queued_task(), None)
            summary_path = tmp_path / "logs" / "task_notion_automation_test_001.json"
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
