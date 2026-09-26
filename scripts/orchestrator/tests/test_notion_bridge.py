"""
Unit tests for NotionBridge markdown parsing, task claiming, and completion updates.
"""

from pathlib import Path
import tempfile
import unittest
from scripts.orchestrator.config import OrchestratorConfig
from scripts.orchestrator.models import (
    AgentExitStatus,
    AgentResult,
    Task,
    TaskResult,
    TaskStatus,
)
from scripts.orchestrator.notion_bridge import NotionBridge, NotionBridgeError


class TestNotionBridge(unittest.TestCase):
    def test_parse_markdown_tasks(self):
        sample_md = (
            "# AICatch Autonomous Engineering Control\n\n"
            "**Status:** RUNNING  \n"
            "**Last Heartbeat:** 2026-09-22T14:45:00Z  \n"
            "**Stop Requested:** False  \n\n"
            "---\n\n"
            "## Task Queue\n\n"
            "### Task: `test_task_001`\n"
            "- **Status:** `QUEUED`\n"
            "- **Priority:** `HIGH`\n"
            "- **Objective:** Verify markdown parsing.\n"
            "- **Description:** Harmless read-only inspection.\n"
            "- **Constraints:** READ-ONLY ONLY.\n"
            "- **Acceptance Criteria:** Return success.\n"
            "- **Timeout Seconds:** 600\n"
        )
        config = OrchestratorConfig()
        bridge = NotionBridge(config)

        header, tasks = bridge.parse_markdown_tasks(sample_md)
        self.assertEqual(header["status"], "RUNNING")
        self.assertFalse(header["stop_requested"])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "test_task_001")
        self.assertEqual(tasks[0].status, TaskStatus.QUEUED)
        self.assertEqual(tasks[0].priority, "HIGH")
        self.assertEqual(tasks[0].timeout_seconds, 600)

    def test_task_claiming_and_completion(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = OrchestratorConfig(log_dir=tmp_path / "logs")
            bridge = NotionBridge(config)

            # 1. Get queued task
            task = bridge.get_next_queued_task()
            self.assertIsNotNone(task)
            self.assertEqual(task.task_id, "notion_automation_test_001")
            self.assertEqual(task.status, TaskStatus.QUEUED)

            # 2. Claim task
            claimed = bridge.claim_task("notion_automation_test_001")
            self.assertEqual(claimed.status, TaskStatus.CLAIMED)

            # 3. Start task
            bridge.start_task("notion_automation_test_001", "conv_mock_999")

            # 4. Complete task
            res = TaskResult(
                task_id="notion_automation_test_001",
                status=TaskStatus.COMPLETED,
                agent_result=AgentResult(
                    conversation_id="conv_mock_999",
                    exit_status=AgentExitStatus.SUCCESS,
                    raw_output="AUTOMATION_SUCCESS",
                    elapsed_seconds=5.2,
                ),
                validation_passed=True,
                elapsed_seconds=5.2,
            )
            bridge.complete_task("notion_automation_test_001", res, git_before="", git_after="")

            # 5. Verify no more queued tasks
            next_task = bridge.get_next_queued_task()
            self.assertIsNone(next_task)


if __name__ == "__main__":
    unittest.main()
