"""
Core Execution Engine for AICatch Orchestrator (V0.2).
Coordinates task queue processing, state transitions, agent invocation, and logging.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import List, Optional
from .agent_runner import AntigravityAgentRunner
from .config import OrchestratorConfig
from .logger import StructuredOrchestratorLogger
from .models import (
    AgentExitStatus,
    AgentResult,
    ExecutionMode,
    OrchestratorBatchResult,
    Task,
    TaskResult,
    TaskStatus,
)
from .state_machine import TaskStateMachine
from .worktree import WorktreeManager


class OrchestratorRunner:
    """Manages sequential execution of tasks according to the specified execution mode."""

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        mode: ExecutionMode = ExecutionMode.SAFE,
        run_id: Optional[str] = None,
    ):
        self.config = config or OrchestratorConfig()
        self.config.ensure_directories()
        self.mode = mode
        self.run_id = (
            run_id or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        )
        self.logger = StructuredOrchestratorLogger(
            self.config.log_dir, run_id=self.run_id
        )
        self.agent_runner = AntigravityAgentRunner(self.config)
        self.worktree_manager = WorktreeManager(self.config)

    def execute_tasks(self, tasks: List[Task]) -> OrchestratorBatchResult:
        """Execute a sequence of tasks with deterministic state machine transitions."""
        batch_start_time = datetime.utcnow().isoformat() + "Z"
        start_ts = time.time()
        self.logger.info(
            f"=== Starting Orchestrator Run [{self.run_id}] (Mode: {self.mode.value}, Tasks: {len(tasks)}) ==="
        )

        task_results: List[TaskResult] = []
        succeeded_count = 0
        failed_count = 0

        for idx, task in enumerate(tasks, start=1):
            self.logger.info(
                f"\n--- Processing Task [{idx}/{len(tasks)}]: {task.task_id} ---"
            )
            result = self._execute_single_task(task)
            task_results.append(result)

            if result.status in (TaskStatus.AGENT_COMPLETE, TaskStatus.VALIDATED, TaskStatus.APPROVED):
                succeeded_count += 1
            else:
                failed_count += 1
                self.logger.warning(
                    f"Task {task.task_id} ended with status: {result.status.value}"
                )

        total_elapsed = time.time() - start_ts
        batch_end_time = datetime.utcnow().isoformat() + "Z"

        batch_result = OrchestratorBatchResult(
            mode=self.mode,
            start_time=batch_start_time,
            end_time=batch_end_time,
            total_elapsed_seconds=total_elapsed,
            tasks_total=len(tasks),
            tasks_succeeded=succeeded_count,
            tasks_failed=failed_count,
            task_results=task_results,
        )

        self.logger.info(
            f"\n=== Batch Run Finished: {succeeded_count}/{len(tasks)} tasks succeeded "
            f"in {total_elapsed:.2f}s ==="
        )
        return batch_result

    def _execute_single_task(self, task: Task) -> TaskResult:
        """Execute a single task through the full lifecycle."""
        task_start_time = datetime.utcnow().isoformat() + "Z"
        t0 = time.time()
        self.logger.log_task_start(task.task_id, task.objective, self.mode.value)

        # 1. State: QUEUED -> CLAIMED
        TaskStateMachine.transition(task, TaskStatus.CLAIMED, reason="Task claimed by runner")

        # 2. Handle DRY_RUN mode
        if self.mode == ExecutionMode.DRY_RUN:
            self.logger.info(
                f"[DRY-RUN] Simulated task execution: {task.task_id}"
            )
            self.logger.info(
                f"[DRY-RUN] Would invoke Antigravity with prompt objective: '{task.objective}'"
            )
            TaskStateMachine.transition(
                task, TaskStatus.AGENT_COMPLETE, reason="Dry-run simulation complete"
            )
            elapsed = time.time() - t0
            res = TaskResult(
                task_id=task.task_id,
                status=task.status,
                agent_result=AgentResult(
                    conversation_id="dry-run-conv-id",
                    exit_status=AgentExitStatus.SUCCESS,
                    raw_output=f"[DRY-RUN] Simulated execution for {task.task_id}",
                    elapsed_seconds=elapsed,
                ),
                validation_passed=True,
                start_time=task_start_time,
                end_time=datetime.utcnow().isoformat() + "Z",
                elapsed_seconds=elapsed,
                decision="DRY_RUN_PASSED",
            )
            self.logger.save_task_summary(task.task_id, res.to_dict())
            self.logger.log_task_complete(task.task_id, res.status.value, elapsed, "dry-run")
            return res

        # 3. Setup workspace / worktree
        workspace_path = str(self.config.aicatch_root)
        worktree_created = False
        worktree_path_obj = None
        branch_name = None

        if self.mode == ExecutionMode.RUN:
            try:
                exp_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{task.task_id[:8]}"
                worktree_path_obj, branch_name = self.worktree_manager.create_worktree(exp_id)
                workspace_path = str(worktree_path_obj)
                task.worktree_path = workspace_path
                worktree_created = True
                self.logger.info(f"Created isolated worktree at: {workspace_path}")
            except Exception as e:
                self.logger.error(f"Failed to create worktree: {e}")
                TaskStateMachine.transition(task, TaskStatus.FAILED, reason=str(e))
                elapsed = time.time() - t0
                res = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    start_time=task_start_time,
                    end_time=datetime.utcnow().isoformat() + "Z",
                    elapsed_seconds=elapsed,
                    error=f"Worktree creation failed: {e}",
                )
                self.logger.save_task_summary(task.task_id, res.to_dict())
                return res

        # 4. State: CLAIMED -> RUNNING
        TaskStateMachine.transition(task, TaskStatus.RUNNING, reason="Agent started")

        # 5. Dispatch task to Antigravity
        agent_result = self.agent_runner.run_task(
            task,
            workspace_path=workspace_path,
            timeout_seconds=task.timeout_seconds,
        )

        elapsed = time.time() - t0

        # 6. Evaluate Result and State Transition
        if agent_result.exit_status == AgentExitStatus.SUCCESS:
            TaskStateMachine.transition(
                task, TaskStatus.AGENT_COMPLETE, reason="Agent finished successfully"
            )
            task_status = TaskStatus.AGENT_COMPLETE
        elif agent_result.exit_status == AgentExitStatus.TIMEOUT:
            TaskStateMachine.transition(
                task, TaskStatus.FAILED, reason="Agent execution timed out"
            )
            task_status = TaskStatus.FAILED
        else:
            TaskStateMachine.transition(
                task, TaskStatus.FAILED, reason=agent_result.error_message or "Agent failed"
            )
            task_status = TaskStatus.FAILED

        # 7. Cleanup worktree if in RUN mode and configured
        if worktree_created and worktree_path_obj:
            self.logger.info(f"Cleaning up worktree at: {worktree_path_obj}")
            self.worktree_manager.remove_worktree(
                worktree_path_obj, delete_branch=True, branch_name=branch_name
            )

        task_end_time = datetime.utcnow().isoformat() + "Z"
        final_result = TaskResult(
            task_id=task.task_id,
            status=task_status,
            agent_result=agent_result,
            validation_passed=(task_status == TaskStatus.AGENT_COMPLETE),
            start_time=task_start_time,
            end_time=task_end_time,
            elapsed_seconds=elapsed,
            error=agent_result.error_message,
            worktree_path=workspace_path,
        )

        self.logger.save_task_summary(task.task_id, final_result.to_dict())
        self.logger.log_task_complete(
            task.task_id,
            final_result.status.value,
            elapsed,
            agent_result.conversation_id,
        )
        return final_result
