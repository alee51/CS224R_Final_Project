"""Shared reproducibility metadata helpers for trainer + probes.

`git_metadata()` honors CS224R_GIT_SHA* env overrides so Modal workers can be
stamped from the launcher's git state (the worker may not have git installed
or a clean checkout).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any


def git_metadata() -> dict[str, Any]:
    env_sha = os.environ.get("CS224R_GIT_SHA")
    if env_sha:
        dirty_raw = os.environ.get("CS224R_GIT_DIRTY", "false").lower()
        return {
            "git_sha": env_sha,
            "git_dirty": dirty_raw in ("true", "1", "yes"),
            "git_sha_short": os.environ.get("CS224R_GIT_SHA_SHORT", env_sha[:7]),
        }

    def _run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        dirty = bool(_run(["git", "status", "--porcelain"]))
        short = _run(["git", "rev-parse", "--short", "HEAD"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, dirty, short = "unknown", False, "unknown"
    return {"git_sha": sha, "git_dirty": dirty, "git_sha_short": short}


def dep_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("vllm", "torch", "transformers", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions
