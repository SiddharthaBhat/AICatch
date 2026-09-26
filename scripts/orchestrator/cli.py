"""
Command Line Interface for AICatch Autonomous Orchestrator.
Usage:
    python -m scripts.orchestrator.cli [--dry-run | --safe | --run | --notion-loop] [--timeout SEC] [--model MODEL]
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List
from .config import OrchestratorConfig
from .loop import NotionOrchestratorLoop
from .models import ExecutionMode, Task
from .runner import OrchestratorRunner


def get_default_v01_tasks(timeout_sec: int) -> List[Task]:
    """Return the standard harmless tasks for V0.1 capability verification."""
    return [
        Task(
            task_id="v01_task_1_repo_inspection",
            objective="Read-only inspection of AICatch repository structure",
            description=(
                "Read-only inspection. Inspect the AICatch repository structure. "
                "Do not modify any file. Do not modify Git. "
                "Return exactly: AUTOMATION_V01_TASK_1_SUCCESS "
                "Then provide a brief repository summary."
            ),
            acceptance_criteria=[
                "Must return exact token AUTOMATION_V01_TASK_1_SUCCESS",
                "Must not modify any files",
                "Must not modify Git",
            ],
            constraints=["READ_ONLY", "NO_GIT_MODIFICATIONS"],
            timeout_seconds=timeout_sec,
        ),
        Task(
            task_id="v01_task_2_report_inspection",
            objective="Read-only inspection of existing AICatch engineering reports",
            description=(
                "Read-only inspection. Inspect the existing AICatch engineering reports available locally in scratch/. "
                "Do not modify any file. Do not modify Git. "
                "Return exactly: AUTOMATION_V01_TASK_2_SUCCESS "
                "Then identify the latest available engineering phase."
            ),
            acceptance_criteria=[
                "Must return exact token AUTOMATION_V01_TASK_2_SUCCESS",
                "Must identify the latest engineering phase in scratch/",
                "Must not modify any files",
            ],
            constraints=["READ_ONLY", "NO_GIT_MODIFICATIONS"],
            timeout_seconds=timeout_sec,
        ),
    ]


def load_tasks_from_file(file_path: Path) -> List[Task]:
    """Load tasks from an external JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [Task.from_dict(t) for t in data]
    elif isinstance(data, dict):
        return [Task.from_dict(data)]
    else:
        raise ValueError("Invalid task JSON structure: expected list or object.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AICatch Autonomous Engineering Orchestrator V0.2"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without touching files or spawning agents.",
    )
    mode_group.add_argument(
        "--safe",
        action="store_true",
        help="Execute harmless read-only tasks against the workspace (Default).",
    )
    mode_group.add_argument(
        "--run",
        action="store_true",
        help="Execute tasks inside isolated Git worktrees.",
    )
    mode_group.add_argument(
        "--notion-loop",
        action="store_true",
        help="Run continuous Notion task polling and execution loop.",
    )

    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="Optional path to a JSON file containing task definitions.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout in seconds per task (default: 1800).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum tasks to process before exiting the loop.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="flash",
        help="Antigravity model tier to use (flash, flash_lite, pro).",
    )

    args = parser.parse_args()

    config = OrchestratorConfig(
        default_timeout_sec=args.timeout,
        model_tier=args.model,
    )

    if args.notion_loop:
        loop = NotionOrchestratorLoop(
            config=config, max_iterations=args.max_iterations
        )
        loop.run_loop()
        sys.exit(0)

    if args.dry_run:
        mode = ExecutionMode.DRY_RUN
    elif args.run:
        mode = ExecutionMode.RUN
    else:
        mode = ExecutionMode.SAFE

    if args.task_file:
        task_path = Path(args.task_file)
        if not task_path.exists():
            print(f"Error: Task file not found: {task_path}", file=sys.stderr)
            sys.exit(1)
        tasks = load_tasks_from_file(task_path)
    else:
        tasks = get_default_v01_tasks(args.timeout)

    runner = OrchestratorRunner(config=config, mode=mode)
    batch_result = runner.execute_tasks(tasks)

    # Print clean execution summary table
    print("\n" + "=" * 80)
    print("AICATCH AUTONOMOUS ORCHESTRATOR — EXECUTION SUMMARY (V0.2)")
    print("=" * 80)
    print(f"Execution Mode:       {batch_result.mode.value}")
    print(f"Total Tasks:          {batch_result.tasks_total}")
    print(f"Succeeded:            {batch_result.tasks_succeeded}")
    print(f"Failed:               {batch_result.tasks_failed}")
    print(f"Total Elapsed Time:   {batch_result.total_elapsed_seconds:.2f}s")
    print("-" * 80)
    for tr in batch_result.task_results:
        agent_conv = tr.agent_result.conversation_id if tr.agent_result else "N/A"
        print(f"Task ID: {tr.task_id:<28} Status: {tr.status.value:<12} Elapsed: {tr.elapsed_seconds:>6.2f}s Conv: {agent_conv}")
    print("=" * 80 + "\n")

    if batch_result.tasks_failed > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
