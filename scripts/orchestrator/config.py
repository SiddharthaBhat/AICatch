"""
Configuration management for AICatch Orchestrator (V0.2).
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


@dataclass
class OrchestratorConfig:
    """Strongly-typed configuration for the AICatch autonomous orchestrator."""

    aicatch_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("AICATCH_ROOT", r"c:\Users\siddu\Downloads\AICatch")
        ).resolve()
    )
    language_server_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "ANTIGRAVITY_LANGUAGE_SERVER",
                r"C:\Users\siddu\AppData\Local\Programs\antigravity\resources\bin\language_server.exe",
            )
        )
    )
    agentapi_bat_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "ANTIGRAVITY_AGENTAPI_BAT",
                r"C:\Users\siddu\.gemini\antigravity\bin\agentapi.bat",
            )
        )
    )
    brain_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "ANTIGRAVITY_BRAIN_DIR",
                r"C:\Users\siddu\.gemini\antigravity\brain",
            )
        )
    )
    worktree_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("AICATCH_WORKTREE_ROOT", r"c:\Users\siddu\Downloads\aicatch-experiments")
        )
    )
    log_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("ORCHESTRATOR_LOG_DIR", r"c:\Users\siddu\Downloads\AICatch\logs\orchestrator")
        )
    )
    notion_control_page_id: str = field(
        default_factory=lambda: os.getenv(
            "NOTION_CONTROL_PAGE_ID", "3e30cc46-e52f-81e0-9443-c071ea712103"
        )
    )
    notion_poll_interval_sec: float = 10.0
    default_timeout_sec: int = field(
        default_factory=lambda: int(os.getenv("ORCHESTRATOR_TIMEOUT_SEC", "1800"))
    )
    poll_interval_sec: float = 0.5
    model_tier: str = "flash"
    max_concurrent_agents: int = 1

    def ensure_directories(self) -> None:
        """Ensure required log and worktree directories exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.worktree_root.parent.exists():
            self.worktree_root.mkdir(parents=True, exist_ok=True)

    def validate_antigravity_executable(self) -> bool:
        """Check whether the configured Antigravity language_server binary exists."""
        return self.language_server_path.exists()
