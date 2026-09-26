"""
Unit tests for AntigravityAgentRunner prompt formatting and transcript parsing.
"""

import json
from pathlib import Path
import tempfile
import time
import unittest
from scripts.orchestrator.agent_runner import AntigravityAgentRunner
from scripts.orchestrator.config import OrchestratorConfig
from scripts.orchestrator.models import AgentExitStatus, Task


class TestAgentRunner(unittest.TestCase):
    def test_prompt_formatting(self):
        config = OrchestratorConfig()
        runner = AntigravityAgentRunner(config)
        task = Task(
            task_id="t_format_01",
            objective="Verify retrieval accuracy",
            description="Run test suite without modifying database.",
            baseline="10 FPS, 90% recall",
            acceptance_criteria=["FPS >= 10", "Recall >= 0.90"],
            constraints=["READ_ONLY"],
            validation_commands=["pytest test_retrieval.py"],
        )

        prompt = runner.format_task_prompt(task, "C:/fake/worktree")
        self.assertIn("C:/fake/worktree", prompt)
        self.assertIn("t_format_01", prompt)
        self.assertIn("Verify retrieval accuracy", prompt)
        self.assertIn("10 FPS, 90% recall", prompt)
        self.assertIn("FPS >= 10", prompt)
        self.assertIn("READ_ONLY", prompt)
        self.assertIn("pytest test_retrieval.py", prompt)

    def test_poll_transcript_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = OrchestratorConfig(brain_dir=tmp_path)
            runner = AntigravityAgentRunner(config)

            conv_id = "mock_conv_123"
            log_dir = tmp_path / conv_id / ".system_generated" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            transcript_file = log_dir / "transcript.jsonl"

            # Write sample transcript lines
            records = [
                {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE"},
                {
                    "step_index": 1,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "content": "AUTOMATION_V01_TASK_1_SUCCESS\nSummary here.",
                    "tool_calls": [{"name": "list_dir", "args": {}}],
                },
            ]
            with open(transcript_file, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            output, status, tool_count, tpath = runner._poll_transcript(
                conv_id, start_time=time.time(), timeout_seconds=10
            )
            self.assertEqual(status, AgentExitStatus.SUCCESS)
            self.assertIn("AUTOMATION_V01_TASK_1_SUCCESS", output)
            self.assertEqual(tool_count, 1)
            self.assertEqual(tpath, transcript_file)


if __name__ == "__main__":
    unittest.main()
