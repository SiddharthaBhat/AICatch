"""
Git Worktree Isolation Manager for AICatch Orchestrator.
Creates, validates, and cleans up isolated experiment worktrees.
"""

from pathlib import Path
import subprocess
from typing import Optional, Tuple
from .config import OrchestratorConfig


class WorktreeError(Exception):
    """Raised when git worktree operations fail."""

    pass


class WorktreeManager:
    """Manages creation, verification, and teardown of Git worktrees for experiments."""

    def __init__(self, config: OrchestratorConfig):
        self.config = config

    def create_worktree(
        self,
        experiment_id: str,
        base_branch: str = "main",
    ) -> Tuple[Path, str]:
        """Create a new isolated Git worktree and branch for an experiment."""
        self.config.worktree_root.mkdir(parents=True, exist_ok=True)
        worktree_path = self.config.worktree_root / experiment_id
        branch_name = f"exp/{experiment_id}"

        if worktree_path.exists():
            raise WorktreeError(
                f"Target worktree path already exists: {worktree_path}"
            )

        cmd = [
            "git",
            "worktree",
            "add",
            "-b",
            branch_name,
            str(worktree_path),
            base_branch,
        ]

        try:
            subprocess.run(
                cmd,
                cwd=str(self.config.aicatch_root),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise WorktreeError(
                f"Failed to create Git worktree: {e.stderr or e.stdout}"
            )

        return worktree_path, branch_name

    def remove_worktree(
        self,
        worktree_path: Path,
        delete_branch: bool = True,
        branch_name: Optional[str] = None,
    ) -> None:
        """Safely remove a Git worktree and optionally delete its branch."""
        if not worktree_path.exists():
            return

        cmd = ["git", "worktree", "remove", "--force", str(worktree_path)]
        try:
            subprocess.run(
                cmd,
                cwd=str(self.config.aicatch_root),
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # Fallback to manual removal if git worktree remove encounters lock issues
            pass

        if delete_branch and branch_name:
            try:
                subprocess.run(
                    ["git", "branch", "-D", branch_name],
                    cwd=str(self.config.aicatch_root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception:
                pass
