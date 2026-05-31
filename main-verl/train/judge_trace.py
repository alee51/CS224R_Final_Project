"""Verbose judge I/O trace for sanity-checking cluster assignments.

Enabled when ``CS224R_JUDGE_TRACE=1`` (set by the trace smoke probe). Logs one
prompt per ``assign_judge_clusters`` call — same decode + HTTP path as training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from judge.types import JudgeClusterResult


def trace_prompt_index() -> int | None:
    """Return the batch prompt index to dump, or None if tracing is off."""
    flag = os.environ.get("CS224R_JUDGE_TRACE", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return None
    return int(os.environ.get("CS224R_JUDGE_TRACE_PROMPT_IDX", "0"))


def _max_chars() -> int:
    """0 = no truncation."""
    return int(os.environ.get("CS224R_JUDGE_TRACE_MAX_CHARS", "12000"))


def _clip(text: str) -> str:
    limit = _max_chars()
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated, total {len(text)} chars]"


def _emit(section: str, body: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n[judge-trace] {section}\n{bar}\n{body}\n", flush=True)


def trace_artifact_path() -> Path | None:
    raw = os.environ.get("CS224R_JUDGE_TRACE_PATH", "").strip()
    return Path(raw) if raw else None


def step_log_path() -> Path | None:
    raw = os.environ.get("CS224R_JUDGE_STEP_LOG", "").strip()
    return Path(raw) if raw else None


def parse_fail_log_path() -> Path | None:
    raw = os.environ.get("CS224R_JUDGE_PARSE_FAIL_LOG", "").strip()
    return Path(raw) if raw else None


def append_step_log(record: dict[str, Any]) -> Path | None:
    """Append one JSON line per ``assign_judge_clusters`` call (survives Ray log routing)."""
    path = step_log_path()
    if path is None:
        return None
    import time

    record = {**record, "logged_at_unix": time.time(), "pid": os.getpid()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return path


def append_parse_failure(record: dict[str, Any]) -> Path | None:
    """Append one JSON line per parse-failed judge response for offline inspection."""
    path = parse_fail_log_path()
    if path is None:
        return None
    import time

    record = {**record, "logged_at_unix": time.time(), "pid": os.getpid()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
    return path


def write_trace_artifact(payload: dict[str, Any]) -> Path | None:
    """Persist trace JSON and print a grep-friendly marker on the driver process."""
    path = trace_artifact_path()
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"JUDGE_TRACE_ARTIFACT={path}", flush=True)
    return path


def build_trace_payload(
    *,
    prompt_index: int,
    n_prompts: int,
    problem_id: Any,
    prompt_text: str,
    rollouts: list[str],
    system: str,
    user: str,
    envelope_token_ct: int,
    judge_max_input_tokens: int,
    overflow: bool,
    result: JudgeClusterResult | None,
    cluster_ids_row: torch.Tensor,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "meta": {
            "prompt_index": prompt_index,
            "n_prompts_in_batch": n_prompts,
            "problem_id": problem_id,
            "n_rollouts": len(rollouts),
            "envelope_token_ct": envelope_token_ct,
            "judge_max_input_tokens": judge_max_input_tokens,
            "overflow_skipped": overflow,
        },
        "decoded_problem": prompt_text,
        "rollouts": rollouts,
        "judge_messages": {"system": system, "user": user},
        "final_cluster_ids": {
            "per_rollout": cluster_ids_row.tolist(),
            "distinct": sorted(set(cluster_ids_row.tolist())),
        },
    }
    if overflow:
        payload["judge_result"] = "SKIPPED (input overflow)"
        return payload
    if result is None:
        payload["judge_result"] = "NO RESULT"
        return payload
    payload["judge_parse"] = {
        "parse_ok": result.parse_ok,
        "degenerate_rollout_count": result.degenerate_count,
        "n_clusters_in_payload": len(result.clusters),
    }
    payload["judge_raw_response"] = result.raw_response
    if result.parse_ok:
        payload["judge_parsed_assignment"] = {
            "assignment_0idx": {str(k): v for k, v in result.assignment.items()},
            "clusters": result.clusters,
        }
    return payload


def dump_prompt_trace(
    *,
    prompt_index: int,
    n_prompts: int,
    problem_id: Any,
    prompt_text: str,
    rollouts: list[str],
    system: str,
    user: str,
    envelope_token_ct: int,
    judge_max_input_tokens: int,
    overflow: bool,
    result: JudgeClusterResult | None,
    cluster_ids_row: torch.Tensor,
) -> None:
    """Print rollouts, judge messages, raw output, and parsed clusters for one prompt."""
    payload = build_trace_payload(
        prompt_index=prompt_index,
        n_prompts=n_prompts,
        problem_id=problem_id,
        prompt_text=prompt_text,
        rollouts=rollouts,
        system=system,
        user=user,
        envelope_token_ct=envelope_token_ct,
        judge_max_input_tokens=judge_max_input_tokens,
        overflow=overflow,
        result=result,
        cluster_ids_row=cluster_ids_row,
    )
    write_trace_artifact(payload)

    _emit("meta", json.dumps(payload["meta"], indent=2, default=str))
    _emit("decoded_problem", _clip(prompt_text))
    for r, text in enumerate(rollouts):
        _emit(f"rollout_{r}", _clip(text))
    _emit(
        "judge_messages",
        json.dumps(payload["judge_messages"], indent=2, ensure_ascii=False),
    )
    if overflow:
        _emit("judge_result", "SKIPPED (input overflow)")
        return
    if result is None:
        _emit("judge_result", "NO RESULT (unexpected)")
        return
    _emit("judge_parse", json.dumps(payload["judge_parse"], indent=2))
    raw = result.raw_response
    _emit("judge_raw_response", _clip(raw) if raw else "<none>")
    if result.parse_ok and "judge_parsed_assignment" in payload:
        _emit(
            "judge_parsed_assignment",
            json.dumps(payload["judge_parsed_assignment"], indent=2, ensure_ascii=False),
        )
    _emit("final_cluster_ids", json.dumps(payload["final_cluster_ids"], indent=2))
