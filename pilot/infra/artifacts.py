"""Artifact directory layout and run bootstrap files."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PILOT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PILOT_ROOT / "artifacts"

REQUIRED_ARTIFACTS = (
    "config.snapshot.yaml",
    "git_sha.txt",
    "raw_predictions.jsonl",
    "metrics.json",
    "train.log",
    "cost.json",
)


def artifact_dir(run_id: str, *, artifacts_root: Path | None = None) -> Path:
    root = artifacts_root or ARTIFACTS_ROOT
    return root / run_id


def new_timestamped_run_dir(
    run_id: str,
    *,
    artifacts_root: Path | None = None,
    when: datetime | None = None,
) -> Path:
    """``pilot/artifacts/<run_id>/<UTC-timestamp>/`` for one Modal/local run."""
    stamp = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    out = artifact_dir(run_id, artifacts_root=artifacts_root) / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def link_latest_run(
    run_id: str,
    run_dir: Path,
    *,
    artifacts_root: Path | None = None,
) -> Path:
    """Point ``pilot/artifacts/<run_id>/latest`` at *run_dir* (symlink)."""
    parent = artifact_dir(run_id, artifacts_root=artifacts_root)
    latest = parent / "latest"
    if latest.is_symlink() or latest.is_file():
        latest.unlink()
    elif latest.is_dir():
        raise RuntimeError(f"Refusing to replace non-symlink latest dir: {latest}")
    latest.symlink_to(run_dir.name, target_is_directory=True)
    return latest


def resolve_latest_run_dir(
    run_id: str,
    *,
    artifacts_root: Path | None = None,
) -> Path:
    """Best-effort path for gate scripts: ``latest`` symlink, else newest timestamp dir."""
    parent = artifact_dir(run_id, artifacts_root=artifacts_root)
    latest = parent / "latest"
    if latest.is_symlink():
        return latest.resolve()
    if (parent / "metrics.json").exists():
        return parent
    ts_dirs = sorted(
        (d for d in parent.iterdir() if d.is_dir() and d.name not in ("latest",)),
        key=lambda p: p.name,
        reverse=True,
    )
    return ts_dirs[0] if ts_dirs else parent


def git_sha(*, repo_root: Path | None = None) -> str:
    root = repo_root or PILOT_ROOT.parent
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse failed in {root}: {result.stderr.strip() or 'unknown error'}"
        )
    return result.stdout.strip()


def bootstrap_run_artifacts(
    config: dict[str, Any],
    *,
    artifacts_root: Path | None = None,
    repo_root: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Create artifact dir and write config.snapshot.yaml + git_sha.txt on start."""
    run_id = str(config["run_id"])
    out = out_dir or artifact_dir(run_id, artifacts_root=artifacts_root)
    out.mkdir(parents=True, exist_ok=True)

    snapshot_path = out / "config.snapshot.yaml"
    snapshot_path.write_text(yaml.safe_dump(config, sort_keys=False, default_flow_style=False))

    sha_path = out / "git_sha.txt"
    sha_path.write_text(git_sha(repo_root=repo_root) + "\n")

    return out
