#!/usr/bin/env python3
"""Monitor fresh B200 GRPO + minority_answer production runs."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = (
    REPO_ROOT
    / "main/docs/probes/artifacts/b200_prod_monitor/state.json"
)

RUNS = {
    "grpo": {
        "modal_app": "ap-VBmgTVFefkECyZa0r52RMb",
        "wandb_id": "t11jct0t",
        "ckpt_dir": "/vol/checkpoints/train_real_b200",
        "launch": [
            "bash",
            "main/scripts/launch_train.sh",
            "--mode",
            "full",
            "--gpu-class",
            "b200",
            "--arm",
            "grpo",
            "--config",
            "main/configs/train_real_b200_fresh_grpo.yaml",
            "--no-resume",
            "--fresh-wandb",
        ],
    },
    "minority": {
        "modal_app": "ap-3Acz8FrtQY4D4ubqkzJ4jB",
        "wandb_id": "o5ypkzja",
        "ckpt_dir": "/vol/checkpoints/train_minority_answer_b200",
        "launch": [
            "bash",
            "main/scripts/launch_train.sh",
            "--mode",
            "full",
            "--gpu-class",
            "b200",
            "--arm",
            "minority_answer",
            "--config",
            "main/configs/train_real_b200_fresh_minority.yaml",
            "--no-resume",
            "--fresh-wandb",
        ],
    },
}

MILESTONES = (1, 10, 20)
CRASH_PATTERNS = re.compile(
    r"(Traceback|CUDA out of memory|Runner terminated|Application error|"
    r"free pointer not allocated|NCCL error)",
    re.I,
)
# Relaunch only on infra/container death, not training logic skips.
SIMPLE_RELAUNCH_HINTS = re.compile(
    r"(Runner terminated|Application error|Modal client|Connection reset|"
    r"Container exited|SIGKILL|preempted)",
    re.I,
)


@dataclass
class RunStatus:
    name: str
    modal_app: str
    modal_state: str
    wandb_id: str
    wandb_state: str
    wandb_step: int
    has_loss: bool
    log_tail_crash: bool
    simple_relaunch: bool

    @property
    def ok_running(self) -> bool:
        return self.modal_state.startswith("ephemeral") and self.wandb_state == "running"

    @property
    def crashed(self) -> bool:
        if self.wandb_state in ("failed", "crashed", "killed"):
            return True
        if self.modal_state == "stopped" and self.wandb_state != "finished":
            return True
        return self.log_tail_crash and not self.ok_running


def _modal_app_state(app_id: str) -> str:
    proc = subprocess.run(
        ["main/.venv/bin/modal", "app", "list", "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return f"modal_cli_error:{proc.stderr[:120]}"
    apps = json.loads(proc.stdout)
    for a in apps:
        if a.get("App ID") == app_id:
            return str(a.get("State", "unknown"))
    return "not_listed"


def _modal_log_tail(app_id: str, n: int = 400) -> str:
    proc = subprocess.run(
        ["main/.venv/bin/modal", "app", "logs", app_id, "--tail", str(n)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (proc.stdout or "") + (proc.stderr or "")


def _wandb_status(run_id: str) -> tuple[str, int, bool]:
    import wandb

    api = wandb.Api()
    run = api.run(f"224r-project/cs224r-minority-voting/{run_id}")
    hist = run.history(samples=500)
    if hist.empty or "_step" not in hist.columns:
        return run.state, -1, False
    step = int(hist["_step"].max())
    has_loss = "train/loss" in hist.columns and hist["train/loss"].notna().any()
    return run.state, step, has_loss


def _load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text())
    return {"milestones_done": {}, "relaunchs": []}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _next_sleep_seconds(max_step: int, milestones_done: dict[str, list[int]]) -> int:
    """Dynamic poll interval until step 20, then hourly."""
    done = set(milestones_done.get("both", []))
    for m in MILESTONES:
        if m not in done and max_step < m:
            # ~4–6 min/step early → poll every 3–5 min until milestone
            return 180 if m == 1 else 300
    return 3600


def main() -> int:
    import os

    sys.path.insert(0, str(REPO_ROOT / "main"))
    os.chdir(REPO_ROOT)

    state = _load_state()
    now = datetime.now(timezone.utc).isoformat()
    statuses: list[RunStatus] = []

    for name, meta in RUNS.items():
        logs = _modal_log_tail(meta["modal_app"])
        wb_state, wb_step, has_loss = _wandb_status(meta["wandb_id"])
        statuses.append(
            RunStatus(
                name=name,
                modal_app=meta["modal_app"],
                modal_state=_modal_app_state(meta["modal_app"]),
                wandb_id=meta["wandb_id"],
                wandb_state=wb_state,
                wandb_step=wb_step,
                has_loss=has_loss,
                log_tail_crash=bool(CRASH_PATTERNS.search(logs)),
                simple_relaunch=bool(SIMPLE_RELAUNCH_HINTS.search(logs)),
            )
        )

    max_step = max((s.wandb_step for s in statuses if s.wandb_step >= 0), default=-1)
    md = state.setdefault("milestones_done", {})
    both_done = md.setdefault("both", [])
    for m in MILESTONES:
        if max_step >= m and m not in both_done:
            both_done.append(m)
            print(f"MILESTONE step>={m} reached (max_step={max_step})")

    report = {
        "checked_at": now,
        "runs": [
            {
                "name": s.name,
                "modal_app": s.modal_app,
                "modal_state": s.modal_state,
                "wandb": f"https://wandb.ai/224r-project/cs224r-minority-voting/runs/{s.wandb_id}",
                "wandb_state": s.wandb_state,
                "wandb_step": s.wandb_step,
                "has_loss": bool(s.has_loss),
                "ok_running": bool(s.ok_running),
                "crashed": bool(s.crashed),
            }
            for s in statuses
        ],
        "max_step": max_step,
        "milestones_done": both_done,
        "next_poll_seconds": _next_sleep_seconds(max_step, md),
    }
    state["last_report"] = report
    _save_state(state)

    print(json.dumps(report, indent=2))

    relaunch: list[str] = []
    for s in statuses:
        if s.crashed and s.simple_relaunch:
            relaunch.append(s.name)

    if relaunch and "--relaunch" in sys.argv:
        for name in relaunch:
            cmd = RUNS[name]["launch"]
            print(f"RELAUNCH {name}: {' '.join(cmd)}")
            subprocess.Popen(cmd, cwd=REPO_ROOT)
            state["relaunchs"].append({"at": now, "arm": name})
        _save_state(state)

    if any(s.crashed for s in statuses):
        return 2
    if max_step < 0 and not all(s.ok_running for s in statuses):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
