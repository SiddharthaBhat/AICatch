"""
Continuous Notion-Orchestrator-Antigravity Execution Loop for AICatch (V0.2).
Polls Notion for QUEUED tasks, executes them headlessly via Antigravity, and writes results back to Notion.
"""

from datetime import datetime
import os
from pathlib import Path
import subprocess
import time
from typing import Optional
from .agent_runner import AntigravityAgentRunner
from .config import OrchestratorConfig
from .logger import StructuredOrchestratorLogger
from .models import (
    AgentExitStatus,
    AgentResult,
    ExecutionMode,
    Task,
    TaskResult,
    TaskStatus,
)
from .notion_bridge import NotionBridge
from .state_machine import TaskStateMachine


class NotionOrchestratorLoop:
    """Manages the autonomous polling, claiming, execution, and reporting loop."""

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        max_iterations: Optional[int] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.config.ensure_directories()
        self.max_iterations = max_iterations
        self.run_id = datetime.utcnow().strftime("notion_loop_%Y%m%d_%H%M%S")
        self.logger = StructuredOrchestratorLogger(
            self.config.log_dir, run_id=self.run_id
        )
        self.bridge = NotionBridge(self.config)
        self.agent_runner = AntigravityAgentRunner(self.config)
        self._running = True

    def check_git_status(self) -> str:
        """Capture current short git status for integrity checking."""
        try:
            proc = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self.config.aicatch_root),
                capture_output=True,
                text=True,
                check=True,
            )
            return proc.stdout.strip()
        except Exception as e:
            self.logger.error(f"Failed to check git status: {e}")
            return "ERROR_CHECKING_GIT"

    def run_loop(self) -> None:
        """Run the continuous orchestration loop."""
        self.logger.info(
            f"=== Starting Notion Autonomous Loop [{self.run_id}] ==="
        )
        self.bridge.update_heartbeat(status="RUNNING")

        iteration = 0
        tasks_processed = 0

        try:
            while self._running:
                iteration += 1

                # 1. Check for external stop request
                if self.bridge.is_stop_requested():
                    self.logger.info("Stop requested via Notion control state. Exiting loop.")
                    self.bridge.update_heartbeat(status="STOPPED")
                    break

                # 2. Update heartbeat
                self.bridge.update_heartbeat(status="IDLE_POLLING")

                # 3. Poll for next QUEUED task
                self.logger.info(
                    f"[Loop #{iteration}] Polling Notion for QUEUED tasks..."
                )
                task = self.bridge.get_next_queued_task()

                if not task:
                    self.logger.info(
                        f"No queued tasks found. Sleeping for {self.config.notion_poll_interval_sec}s..."
                    )
                    time.sleep(self.config.notion_poll_interval_sec)
                    if self.max_iterations and iteration >= self.max_iterations:
                        break
                    continue

                # 4. Task found: Process task through full lifecycle
                self.logger.info(
                    f"--> Discovered Task: [{task.task_id}] Objective: {task.objective}"
                )
                self._process_task(task)
                tasks_processed += 1

                if self.max_iterations and tasks_processed >= self.max_iterations:
                    self.logger.info(
                        f"Reached maximum requested iterations ({self.max_iterations}). Terminating loop."
                    )
                    break

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Stopping loop cleanly...")
        finally:
            self.bridge.update_heartbeat(status="STOPPED")
            self.logger.info(
                f"=== Notion Autonomous Loop Terminated. Total Tasks Processed: {tasks_processed} ==="
            )

    def _process_task(self, task: Task) -> None:
        """Claim, execute, verify, and complete a single Notion task."""
        t0 = time.time()
        start_time_iso = datetime.utcnow().isoformat() + "Z"

        # Pre-execution AICatch repository integrity check
        git_before = self.check_git_status()
        self.logger.info(f"AICatch Git status before task: '{git_before}'")

        # 1. Claim task in Notion: QUEUED -> CLAIMED
        try:
            claimed_task = self.bridge.claim_task(task.task_id)
            TaskStateMachine.transition(task, TaskStatus.CLAIMED, reason="Claimed in Notion")
            self.logger.info(f"Task [{task.task_id}] successfully claimed in Notion.")
        except Exception as e:
            self.logger.error(f"Failed to claim task in Notion: {e}")
            return

        # 2. State: CLAIMED -> RUNNING
        TaskStateMachine.transition(task, TaskStatus.RUNNING, reason="Starting agent")
        self.bridge.start_task(task.task_id, conversation_id="pending")

        # 3. Spawn Antigravity Headless Agent
        self.logger.info(
            f"Spawning Antigravity agent for Task [{task.task_id}] (Timeout: {task.timeout_seconds}s)..."
        )
        agent_res = self.agent_runner.run_task(
            task,
            workspace_path=str(self.config.aicatch_root),
            timeout_seconds=task.timeout_seconds,
        )

        elapsed = time.time() - t0
        end_time_iso = datetime.utcnow().isoformat() + "Z"

        # 4. Post-execution AICatch repository integrity check
        git_after = self.check_git_status()
        self.logger.info(f"AICatch Git status after task: '{git_after}'")

        # Verify zero repository modification
        integrity_pass = (git_before == git_after)
        if not integrity_pass:
            self.logger.error(
                f"EMERGENCY INTEGRITY FAULT: Repository changed during read-only task! "
                f"Before: '{git_before}', After: '{git_after}'"
            )
            TaskStateMachine.transition(task, TaskStatus.FAILED, reason="Repository integrity fault")
            self.bridge.fail_task(
                task.task_id,
                error_message=f"Integrity Fault: Repo modified during read-only execution. (Diff: {git_after})",
                elapsed_sec=elapsed,
            )
            return

        # 5. Evaluate agent outcome
        if agent_res.exit_status == AgentExitStatus.SUCCESS:
            TaskStateMachine.transition(task, TaskStatus.COMPLETED, reason="Agent succeeded")
            final_status = TaskStatus.COMPLETED
        elif agent_res.exit_status == AgentExitStatus.TIMEOUT:
            TaskStateMachine.transition(task, TaskStatus.TIMEOUT, reason="Agent timed out")
            final_status = TaskStatus.TIMEOUT
        else:
            TaskStateMachine.transition(task, TaskStatus.FAILED, reason="Agent failed")
            final_status = TaskStatus.FAILED

        task_result = TaskResult(
            task_id=task.task_id,
            status=final_status,
            agent_result=agent_res,
            validation_passed=(final_status == TaskStatus.COMPLETED),
            start_time=start_time_iso,
            end_time=end_time_iso,
            elapsed_seconds=elapsed,
            error=agent_res.error_message,
            git_status_before=git_before,
            git_status_after=git_after,
            aicatch_integrity="PASS" if integrity_pass else "FAIL",
        )

        # 6. Write structured result back to Notion
        self.logger.info(
            f"Writing task [{task.task_id}] completion status ({final_status.value}) back to Notion..."
        )
        if final_status == TaskStatus.COMPLETED:
            self.bridge.complete_task(
                task.task_id,
                task_result,
                git_before=git_before,
                git_after=git_after,
            )
        else:
            self.bridge.fail_task(
                task.task_id,
                error_message=agent_res.error_message or "Agent execution failed",
                elapsed_sec=elapsed,
            )

        # 7. Save local summary
        self.logger.save_task_summary(task.task_id, task_result.to_dict())
        self.logger.log_task_complete(
            task.task_id,
            final_status.value,
            elapsed,
            agent_res.conversation_id,
        )
