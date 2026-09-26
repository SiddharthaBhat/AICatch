"""
Unit tests for Git WorktreeManager.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.orchestrator.config import OrchestratorConfig
from scripts.orchestrator.worktree import WorktreeManager


class TestWorktree(unittest.TestCase):
    def test_worktree_manager_initialization(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = OrchestratorConfig(worktree_root=tmp_path / "experiments")
            mgr = WorktreeManager(config)
            self.assertEqual(mgr.config.worktree_root, tmp_path / "experiments")

    @patch("subprocess.run")
    def test_create_worktree_mock(self, mock_run):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mock_run.return_value.returncode = 0
            config = OrchestratorConfig(
                aicatch_root=tmp_path / "repo",
                worktree_root=tmp_path / "experiments",
            )
            mgr = WorktreeManager(config)

            path, branch = mgr.create_worktree("exp_001", "main")
            self.assertEqual(branch, "exp/exp_001")
            self.assertEqual(path, tmp_path / "experiments" / "exp_001")
            self.assertTrue(mock_run.called)


if __name__ == "__main__":
    unittest.main()
