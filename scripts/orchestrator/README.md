# AICatch Autonomous Engineering Orchestrator V0.2
## Notion ↔ Antigravity Closed Engineering Loop

The **AICatch Autonomous Engineering Orchestrator** is an external Python coordination harness that automatically syncs with Notion, drives Antigravity headless agents to safely execute engineering tasks, validates AICatch repository integrity, and writes structured reports and execution telemetry back to Notion without manual intervention.

---

## V0.2 Closed-Loop Architecture

```
                      ┌────────────────────────────────────────┐
                      │          Notion Control Page           │
                      │  "AICatch Autonomous Engineering       │
                      │              Control"                  │
                      └──────────────────┬─────────────────────┘
                                         │
                                         │ 1. Poll & Claim QUEUED Task
                                         ▼
                      ┌────────────────────────────────────────┐
                      │       Python Orchestrator Loop         │
                      │        (scripts/orchestrator/)         │
                      └──────────────────┬─────────────────────┘
                                         │
                                         │ 2. Spawns Headless Agent
                                         ▼
                                ┌───────────────────┐
                                │ Antigravity Agent │
                                │ (agentapi CLI)    │
                                └─────────┬─────────┘
                                          │
                                          │ 3. Harmless Task Execution
                                          ▼
                                ┌───────────────────┐
                                │ AICatch Workspace │
                                │ (Strict Read-Only)│
                                └─────────┬─────────┘
                                          │
                                          │ 4. Output Text & Transcript
                                          ▼
                      ┌────────────────────────────────────────┐
                      │       Python Orchestrator Loop         │
                      │  • Validates Git working tree clean    │
                      │  • Verifies zero mutations (PASS/FAIL) │
                      │  • Marks Task State COMPLETED / FAILED │
                      └──────────────────┬─────────────────────┘
                                         │
                                         │ 5. Writes Execution History & Telemetry
                                         ▼
                      ┌────────────────────────────────────────┐
                      │          Notion Control Page           │
                      │  - Task marked COMPLETED / FAILED      │
                      │  - Full Agent Output Evidence          │
                      │  - Git Integrity Status & Metadata     │
                      └────────────────────────────────────────┘
```

---

## Core Principles & Safety Invariants

1. **Strict Repository Read-Only**: The AICatch core repository (`pipeline/`, `storage/`, `frontend/`, `main.py`, `app.py`, `config.py`, databases, media) remains strictly read-only during orchestrator verification tasks.
2. **Git Working Tree Verification**: After every agent task execution, the orchestrator inspects `git status` to verify zero unexpected mutations to tracked production files.
3. **Dedicated Orchestration Workspace**: All orchestrator logic resides under `scripts/orchestrator/`, decoupled from AICatch runtime logic.
4. **Direct Headless Agentapi Integration**: Spawns and manages Antigravity sessions via the native `agentapi` CLI subprocess.

---

## Live Notion Control Page

- **Page Title:** `AICatch Autonomous Engineering Control`
- **Page ID:** `3e30cc46-e52f-81e0-9443-c071ea712103`
- **Live URL:** [Notion Control Page](https://app.notion.com/p/AICatch-Autonomous-Engineering-Control-3e30cc46e52f81e09443c071ea712103)

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `NOTION_CONTROL_PAGE_ID` | `3e30cc46-e52f-81e0-9443-c071ea712103` | Notion Autonomous Engineering Control page ID |
| `ORCHESTRATOR_TIMEOUT_SEC` | `1800` | Default per-task timeout in seconds |
| `AICATCH_MODEL_TIER` | `flash` | Default Antigravity model tier |

---

## Running the Orchestrator

### Continuous Notion Loop
```powershell
.venv\Scripts\python.exe -m scripts.orchestrator.cli --notion-loop
```

### Single Iteration Loop Test
```powershell
.venv\Scripts\python.exe -m scripts.orchestrator.cli --notion-loop --max-iterations 1
```

### Safe Mode Local Task Execution
```powershell
.venv\Scripts\python.exe -m scripts.orchestrator.cli --safe
```

### Dry Run (Simulated)
```powershell
.venv\Scripts\python.exe -m scripts.orchestrator.cli --dry-run
```

---

## Running the Unit Test Suite (18 Tests)

```powershell
.venv\Scripts\python.exe -m unittest discover -s scripts/orchestrator/tests -v
```
