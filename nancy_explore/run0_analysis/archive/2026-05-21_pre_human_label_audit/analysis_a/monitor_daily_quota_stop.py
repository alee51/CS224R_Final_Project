#!/usr/bin/env python3
"""Watch Analysis A run; kill it when Gemini daily quota errors appear."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO / "pilot/artifacts/run0_proxy/20260519T190202Z/llm_clusters"
LOG_PATH = Path(__file__).resolve().parent / "analysis_a_full_run.log"
POLL_SEC = 15
PROC_MATCH = "analysis_a_llm_clusters.py"


def is_daily_quota_error(err: str) -> bool:
    e = err.lower()
    if "perday" in e or "per_day" in e or "requestsperday" in e:
        return True
    if "perminute" in e or "per_minute" in e:
        return False
    if "please retry in" in e and "perday" not in e:
        return False
    if "quota exceeded" in e and ("day" in e or "daily" in e):
        return True
    return False


def scan_caches() -> list[str]:
    hits: list[str] = []
    if not CACHE_DIR.is_dir():
        return hits
    for path in CACHE_DIR.glob("*.json"):
        try:
            j = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        err = j.get("error") or ""
        if err and is_daily_quota_error(err):
            hits.append(f"{path.name}: {err[:240]}")
    return hits


def scan_log() -> list[str]:
    if not LOG_PATH.is_file():
        return []
    text = LOG_PATH.read_text(errors="replace")
    if not is_daily_quota_error(text):
        return []
    return ["analysis_a_full_run.log contains daily quota text"]


def pids_running() -> list[int]:
    out = subprocess.run(
        ["pgrep", "-f", PROC_MATCH],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return []
    return [int(x) for x in out.stdout.split() if x.strip()]


def stop_run(pids: list[int], reason: str) -> None:
    print(f"STOP: {reason}", flush=True)
    for pid in pids:
        subprocess.run(["kill", str(pid)], check=False)
    time.sleep(2)
    still = pids_running()
    if still:
        for pid in still:
            subprocess.run(["kill", "-9", str(pid)], check=False)
    print(f"Stopped Analysis A (was pids {pids})", flush=True)


def main() -> int:
    print(
        f"Monitoring {PROC_MATCH} for daily quota (poll every {POLL_SEC}s)...",
        flush=True,
    )
    seen_daily: set[str] = set()
    while True:
        pids = pids_running()
        if not pids:
            print("Analysis A not running; monitor exiting.", flush=True)
            return 0

        hits = scan_caches() + scan_log()
        new = [h for h in hits if h not in seen_daily]
        if new:
            for h in new:
                print(f"daily quota detected: {h}", flush=True)
                seen_daily.add(h)
            stop_run(pids, new[0])
            return 0

        ok = 0
        if CACHE_DIR.is_dir():
            for p in CACHE_DIR.glob("*.json"):
                try:
                    j = json.loads(p.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if j.get("parse_ok") and j.get("prompt_format") == "poly_epo_paper_a1":
                    ok += 1
        # cheap progress line
        print(f"  ... running pids={pids} valid_caches≈{ok}", flush=True)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
