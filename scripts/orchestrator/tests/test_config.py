"""
Unit tests for OrchestratorConfig.
"""

from pathlib import Path
import tempfile
import unittest
from scripts.orchestrator.config import OrchestratorConfig


class TestOrchestratorConfig(unittest.TestCase):
    def test_default_config_paths(self):
        config = OrchestratorConfig()
        self.assertTrue(config.aicatch_root.exists())
        self.assertIsInstance(config.language_server_path, Path)
        self.assertEqual(config.default_timeout_sec, 1800)
        self.assertIn(config.model_tier, ("flash", "flash_lite", "pro"))
        self.assertEqual(config.max_concurrent_agents, 1)

    def test_custom_config_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            custom_log = Path(tmp_dir) / "custom_logs"
            config = OrchestratorConfig(
                log_dir=custom_log,
                default_timeout_sec=60,
                model_tier="flash_lite",
            )
            config.ensure_directories()
            self.assertTrue(custom_log.exists())
            self.assertEqual(config.default_timeout_sec, 60)
            self.assertEqual(config.model_tier, "flash_lite")


if __name__ == "__main__":
    unittest.main()
