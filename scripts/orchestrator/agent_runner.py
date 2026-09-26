"""
Antigravity Headless Agent Runner for AICatch Orchestrator.
Spawns and monitors headless agent tasks via the native language_server agentapi interface.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Optional, Tuple
from .config import OrchestratorConfig
from .models import AgentExitStatus, AgentResult, Task


class AntigravityAgentRunner:
    """Manages spawning, monitoring, and result capture for Antigravity headless agents."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config

    def format_task_prompt(self, task: Task, workspace_path: str) -> str:
        """Format a structured task prompt for the Antigravity agent."""
        criteria_str = "\n".join(f"- {c}" for c in task.acceptance_criteria) or "- None specified"
        constraints_str = "\n".join(f"- {c}" for c in task.constraints) or "- Standard safety invariants"
        validation_str = "\n".join(f"- {v}" for v in task.validation_commands) or "- None specified"

        prompt = (
            f"You are an autonomous AICatch engineering agent.\n\n"
            f"WORKSPACE:\n{workspace_path}\n\n"
            f"TASK ID:\n{task.task_id}\n\n"
            f"OBJECTIVE:\n{task.objective}\n\n"
            f"DESCRIPTION:\n{task.description}\n\n"
            f"BASELINE:\n{task.baseline or 'None'}\n\n"
            f"ACCEPTANCE CRITERIA:\n{criteria_str}\n\n"
            f"CONSTRAINTS:\n{constraints_str}\n\n"
            f"REQUIRED VALIDATION:\n{validation_str}\n\n"
            f"SAFETY INVARIANTS:\n"
            f"- Inspect before modifying.\n"
            f"- Do NOT modify files outside the assigned workspace.\n"
            f"- Do NOT push to GitHub or modify Git remotes.\n"
            f"- Do NOT delete production databases or Chroma collections.\n\n"
            f"COMPLETION PROTOCOL:\n"
            f"When finished, return a structured markdown summary reporting status, "
            f"exact measured results, files changed, and decisions."
        )
        return prompt

    def run_task(
        self,
        task: Task,
        workspace_path: str,
        timeout_seconds: Optional[int] = None,
    ) -> AgentResult:
        """Spawn a headless agent conversation, wait for completion, and capture results."""
        timeout = timeout_seconds or task.timeout_seconds or self.config.default_timeout_sec
        start_time = time.time()

        if not self.config.language_server_path.exists():
            return AgentResult(
                conversation_id="",
                exit_status=AgentExitStatus.FAILED,
                raw_output="",
                elapsed_seconds=0.0,
                error_message=f"Antigravity executable not found at: {self.config.language_server_path}",
            )

        prompt = self.format_task_prompt(task, workspace_path)

        cmd = [
            str(self.config.language_server_path),
            "agentapi",
            "new-conversation",
            f"--model={self.config.model_tier}",
            f"--title={task.task_id}: {task.objective[:40]}",
            prompt,
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(self.config.aicatch_root),
                timeout=30,  # 30s timeout to create the conversation
            )
            data = json.loads(proc.stdout)
            conv_id = data["response"]["newConversation"]["conversationId"]
        except subprocess.TimeoutExpired:
            return AgentResult(
                conversation_id="",
                exit_status=AgentExitStatus.TIMEOUT,
                raw_output="",
                elapsed_seconds=time.time() - start_time,
                error_message="Timeout while executing agentapi new-conversation command.",
            )
        except Exception as e:
            return AgentResult(
                conversation_id="",
                exit_status=AgentExitStatus.FAILED,
                raw_output="",
                elapsed_seconds=time.time() - start_time,
                error_message=f"Failed to start Antigravity agent: {e}",
            )

        task.conversation_id = conv_id

        # Monitor transcript.jsonl for task completion
        raw_output, exit_status, tool_count, transcript_path = self._poll_transcript(
            conv_id, start_time, timeout
        )

        elapsed = time.time() - start_time
        return AgentResult(
            conversation_id=conv_id,
            exit_status=exit_status,
            raw_output=raw_output,
            elapsed_seconds=elapsed,
            tool_calls_count=tool_count,
            transcript_path=str(transcript_path) if transcript_path else None,
        )

    def _poll_transcript(
        self,
        conversation_id: str,
        start_time: float,
        timeout_seconds: int,
    ) -> Tuple[str, AgentExitStatus, int, Optional[Path]]:
        """Poll the transcript log file for completion status and output text."""
        transcript_path = (
            self.config.brain_dir
            / conversation_id
            / ".system_generated"
            / "logs"
            / "transcript.jsonl"
        )

        captured_output = ""
        exit_status = AgentExitStatus.TIMEOUT
        tool_count = 0

        while (time.time() - start_time) < timeout_seconds:
            if transcript_path.exists():
                try:
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]

                    current_tool_count = 0
                    latest_model_content = None
                    turn_done = False

                    for line in lines:
                        try:
                            record = json.loads(line)
                            if record.get("source") == "MODEL":
                                if record.get("tool_calls"):
                                    current_tool_count += len(record["tool_calls"])
                                if record.get("type") == "PLANNER_RESPONSE":
                                    if "content" in record and record["content"]:
                                        latest_model_content = record["content"]
                                        if record.get("status") == "DONE":
                                            turn_done = True
                            elif record.get("source") == "SYSTEM" and record.get("type") == "ERROR_MESSAGE":
                                # If there's an unrecoverable system error
                                pass
                        except json.JSONDecodeError:
                            continue

                    tool_count = current_tool_count

                    if latest_model_content is not None and turn_done:
                        captured_output = latest_model_content
                        exit_status = AgentExitStatus.SUCCESS
                        return captured_output, exit_status, tool_count, transcript_path

                except Exception:
                    pass

            time.sleep(self.config.poll_interval_sec)

        # If timeout reached
        return captured_output, exit_status, tool_count, transcript_path
