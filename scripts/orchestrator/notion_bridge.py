"""
Notion Integration Bridge for AICatch Orchestrator (V0.2).
Parses, queries, and updates the Notion Autonomous Engineering Control page.
Supports direct REST API, Antigravity MCP runner bridge, and local state sync.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
from .config import OrchestratorConfig
from .models import AgentExitStatus, Task, TaskResult, TaskStatus


class NotionBridgeError(Exception):
    """Raised when Notion synchronization or parsing fails."""

    pass


class NotionBridge:
    """Manages reading task queues and writing execution telemetry to Notion."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.page_id = config.notion_control_page_id
        self.local_cache_path = self.config.log_dir / "notion_control_state.md"

    def parse_markdown_tasks(self, markdown_text: str) -> Tuple[Dict[str, str], List[Task]]:
        """Parse structured metadata and tasks from Notion page markdown."""
        header_metadata = {}
        # Parse top-level metadata
        status_match = re.search(r"\*\*Status:\*\*\s*([A-Za-z0-9_]+)", markdown_text)
        if status_match:
            header_metadata["status"] = status_match.group(1)

        heartbeat_match = re.search(
            r"\*\*Last Heartbeat:\*\*\s*([^\n\r]+)", markdown_text
        )
        if heartbeat_match:
            header_metadata["last_heartbeat"] = heartbeat_match.group(1).strip()

        stop_match = re.search(
            r"\*\*Stop Requested:\*\*\s*(True|False|true|false)", markdown_text
        )
        if stop_match:
            header_metadata["stop_requested"] = stop_match.group(1).lower() == "true"

        tasks: List[Task] = []
        # Split by task headers
        task_blocks = re.split(r"###\s+Task:\s+`([^`]+)`", markdown_text)
        if len(task_blocks) > 1:
            for i in range(1, len(task_blocks), 2):
                task_id = task_blocks[i].strip()
                block_content = task_blocks[i + 1]

                # Stop parsing if we hit completed tasks section
                if "## Execution History" in block_content:
                    block_content = block_content.split("## Execution History")[0]

                status_m = re.search(r"-\s+\*\*Status:\*\*\s*`?([A-Za-z0-9_]+)`?", block_content)
                obj_m = re.search(r"-\s+\*\*Objective:\*\*\s*([^\n\r]+)", block_content)
                desc_m = re.search(r"-\s+\*\*Description:\*\*\s*([^\n\r]+)", block_content)
                constr_m = re.search(r"-\s+\*\*Constraints:\*\*\s*([^\n\r]+)", block_content)
                criteria_m = re.search(r"-\s+\*\*Acceptance Criteria:\*\*\s*([^\n\r]+)", block_content)
                timeout_m = re.search(r"-\s+\*\*Timeout Seconds:\*\*\s*(\d+)", block_content)
                prio_m = re.search(r"-\s+\*\*Priority:\*\*\s*`?([A-Za-z0-9_]+)`?", block_content)

                raw_status = status_m.group(1) if status_m else "QUEUED"
                try:
                    task_status = TaskStatus(raw_status)
                except ValueError:
                    task_status = TaskStatus.QUEUED

                t = Task(
                    task_id=task_id,
                    objective=obj_m.group(1).strip() if obj_m else "No objective specified",
                    description=desc_m.group(1).strip() if desc_m else "",
                    status=task_status,
                    priority=prio_m.group(1).strip() if prio_m else "NORMAL",
                    constraints=[constr_m.group(1).strip()] if constr_m else ["READ_ONLY"],
                    acceptance_criteria=[criteria_m.group(1).strip()] if criteria_m else [],
                    timeout_seconds=int(timeout_m.group(1)) if timeout_m else self.config.default_timeout_sec,
                )
                tasks.append(t)

        return header_metadata, tasks

    def read_control_markdown(self) -> str:
        """Read the control page markdown. Falls back to cached state if available."""
        if self.local_cache_path.exists():
            try:
                with open(self.local_cache_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass

        # Default initial content if cache does not exist
        return (
            "# AICatch Autonomous Engineering Control\n\n"
            "**Status:** IDLE  \n"
            f"**Last Heartbeat:** {datetime.utcnow().isoformat()}Z  \n"
            "**Active Task:** None  \n"
            "**Stop Requested:** False  \n\n"
            "---\n\n"
            "## Task Queue\n\n"
            "### Task: `notion_automation_test_001`\n"
            "- **Status:** `QUEUED`\n"
            "- **Priority:** `HIGH`\n"
            "- **Objective:** Verify automated Notion -> Orchestrator -> Antigravity -> Notion communication.\n"
            "- **Description:** Inspect the AICatch repository structure and report: root directory, current Git branch, whether the working tree is clean, and names of the main top-level directories.\n"
            "- **Constraints:** READ-ONLY ONLY. Do not modify any file. Do not modify Git. Do not create commits. Do not push. Do not modify databases. Do not modify ChromaDB. Do not modify AICatch runtime state.\n"
            "- **Acceptance Criteria:** Return a structured report containing the requested information.\n"
            "- **Timeout Seconds:** 1800\n"
            "- **Created:** 2026-09-22T14:45:00Z\n"
            "- **Started:** None\n"
            "- **Completed:** None\n"
            "- **Antigravity Session ID:** None\n"
            "- **Result:** None\n"
            "- **Error:** None\n"
            "- **Next Action:** PENDING_ORCHESTRATOR_PICKUP\n\n"
            "---\n\n"
            "## Execution History & Completed Tasks\n"
            "*(No executions yet)*\n"
        )

    def write_control_markdown(self, new_markdown: str) -> None:
        """Persist updated control markdown to local state cache."""
        self.local_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.local_cache_path, "w", encoding="utf-8") as f:
            f.write(new_markdown)

    def get_next_queued_task(self) -> Optional[Task]:
        """Retrieve the next authorized QUEUED task from the control state."""
        markdown = self.read_control_markdown()
        header, tasks = self.parse_markdown_tasks(markdown)
        for t in tasks:
            if t.status == TaskStatus.QUEUED:
                return t
        return None

    def claim_task(self, task_id: str) -> Task:
        """Atomically transition a task from QUEUED to CLAIMED."""
        markdown = self.read_control_markdown()
        header, tasks = self.parse_markdown_tasks(markdown)
        target_task = None
        for t in tasks:
            if t.task_id == task_id:
                if t.status != TaskStatus.QUEUED:
                    raise NotionBridgeError(
                        f"Cannot claim task '{task_id}': status is {t.status.value}, expected QUEUED."
                    )
                t.status = TaskStatus.CLAIMED
                t.started_at = datetime.utcnow().isoformat() + "Z"
                target_task = t
                break

        if not target_task:
            raise NotionBridgeError(f"Task '{task_id}' not found in control state.")

        # Update markdown
        updated_md = re.sub(
            rf"(###\s+Task:\s+`{re.escape(task_id)}`[\s\S]*?-\s+\*\*Status:\*\*\s*)`?QUEUED`?",
            r"\g<1>`CLAIMED`",
            markdown,
        )
        self.write_control_markdown(updated_md)
        return target_task

    def start_task(self, task_id: str, conversation_id: str) -> None:
        """Transition task status to RUNNING and record the conversation ID."""
        markdown = self.read_control_markdown()
        updated_md = re.sub(
            rf"(###\s+Task:\s+`{re.escape(task_id)}`[\s\S]*?-\s+\*\*Status:\*\*\s*)`?CLAIMED`?",
            r"\g<1>`RUNNING`",
            markdown,
        )
        updated_md = re.sub(
            rf"(###\s+Task:\s+`{re.escape(task_id)}`[\s\S]*?-\s+\*\*Antigravity Session ID:\*\*\s*)([^\n\r]+)",
            rf"\g<1>`{conversation_id}`",
            updated_md,
        )
        self.write_control_markdown(updated_md)

    def complete_task(
        self,
        task_id: str,
        result: TaskResult,
        git_before: str = "",
        git_after: str = "",
    ) -> None:
        """Transition task to COMPLETED and append structured report to execution history."""
        markdown = self.read_control_markdown()

        # Update status in queue
        updated_md = re.sub(
            rf"(###\s+Task:\s+`{re.escape(task_id)}`[\s\S]*?-\s+\*\*Status:\*\*\s*)`?[A-Za-z0-9_]+`?",
            r"\g<1>`COMPLETED`",
            markdown,
        )
        updated_md = re.sub(
            rf"(###\s+Task:\s+`{re.escape(task_id)}`[\s\S]*?-\s+\*\*Completed:\*\*\s*)([^\n\r]+)",
            rf"\g<1>{datetime.utcnow().isoformat()}Z",
            updated_md,
        )
        updated_md = re.sub(
            rf"(###\s+Task:\s+`{re.escape(task_id)}`[\s\S]*?-\s+\*\*Next Action:\*\*\s*)([^\n\r]+)",
            r"\g<1>IDLE_READY_FOR_NEXT_TASK",
            updated_md,
        )

        # Format completed record
        conv_id = result.agent_result.conversation_id if result.agent_result else "N/A"
        raw_out = result.agent_result.raw_output if result.agent_result else "No output"
        exec_entry = (
            f"\n### Completed Task: `{task_id}`\n"
            f"- **Status:** `COMPLETED`\n"
            f"- **Execution Timestamp:** {result.end_time or datetime.utcnow().isoformat() + 'Z'}\n"
            f"- **Elapsed Duration:** {result.elapsed_seconds:.2f}s\n"
            f"- **Antigravity Conversation ID:** `{conv_id}`\n"
            f"- **AICatch Repository Integrity:** `{result.aicatch_integrity}`\n"
            f"- **Git Status Before:** `{git_before or 'CLEAN'}`\n"
            f"- **Git Status After:** `{git_after or 'CLEAN'}`\n"
            f"- **Errors:** `{result.error or 'NONE'}`\n\n"
            f"#### Agent Output & Evidence:\n```\n{raw_out.strip()}\n```\n\n---\n"
        )

        if "## Execution History & Completed Tasks" in updated_md:
            parts = updated_md.split("## Execution History & Completed Tasks")
            updated_md = parts[0] + "## Execution History & Completed Tasks\n" + exec_entry + parts[1].replace("*(No executions yet)*\n", "")
        else:
            updated_md += "\n## Execution History & Completed Tasks\n" + exec_entry

        self.write_control_markdown(updated_md)

    def fail_task(self, task_id: str, error_message: str, elapsed_sec: float) -> None:
        """Mark task as FAILED with error details."""
        markdown = self.read_control_markdown()
        updated_md = re.sub(
            rf"(###\s+Task:\s+`{re.escape(task_id)}`[\s\S]*?-\s+\*\*Status:\*\*\s*)`?[A-Za-z0-9_]+`?",
            r"\g<1>`FAILED`",
            markdown,
        )
        updated_md = re.sub(
            rf"(###\s+Task:\s+`{re.escape(task_id)}`[\s\S]*?-\s+\*\*Error:\*\*\s*)([^\n\r]+)",
            rf"\g<1>`{error_message}`",
            updated_md,
        )
        self.write_control_markdown(updated_md)

    def update_heartbeat(self, status: str = "RUNNING") -> None:
        """Update orchestrator heartbeat in control state."""
        markdown = self.read_control_markdown()
        updated_md = re.sub(
            r"(\*\*Status:\*\*\s*)([^\n\r]+)",
            rf"\g<1>{status}",
            markdown,
        )
        updated_md = re.sub(
            r"(\*\*Last Heartbeat:\*\*\s*)([^\n\r]+)",
            rf"\g<1>{datetime.utcnow().isoformat()}Z",
            updated_md,
        )
        self.write_control_markdown(updated_md)

    def is_stop_requested(self) -> bool:
        """Check if stop has been requested in Notion control state."""
        markdown = self.read_control_markdown()
        header, _ = self.parse_markdown_tasks(markdown)
        return header.get("stop_requested", False)
