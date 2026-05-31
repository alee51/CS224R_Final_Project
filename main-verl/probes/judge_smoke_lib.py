"""Shared helpers for Stage 4 judge Modal smokes."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

from judge.client import JudgeClient, JudgeClientConfig
from judge.types import JudgeTask


def main_verl_dir() -> Path:
    here = Path(__file__).resolve()
    if len(here.parents) >= 2 and (here.parents[1] / "judge").is_dir():
        return here.parents[1]
    return Path("/root/main-verl")


def load_smoke_config(config_name: str) -> dict[str, Any]:
    path = main_verl_dir() / "configs" / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing smoke config: {path}")
    with path.open() as f:
        cfg = yaml.safe_load(f)

    judge = cfg.setdefault("judge", {})
    if base_url := os.environ.get("JUDGE_BASE_URL"):
        judge["base_url"] = base_url
    if model := os.environ.get("JUDGE_MODEL"):
        judge["model"] = model
    if conc := os.environ.get("JUDGE_CONCURRENCY"):
        judge["concurrency"] = int(conc)
    if max_tokens := os.environ.get("JUDGE_MAX_TOKENS"):
        judge["max_tokens"] = int(max_tokens)
    return cfg


def synthetic_tasks(cfg: dict[str, Any]) -> list[dict]:
    smoke = cfg["smoke"]
    n = int(smoke["n_tasks"])
    n_rollouts = int(smoke["n_rollouts"])
    seed = int(smoke.get("seed", 0))
    template = smoke.get("task_template", "addition")
    rng = random.Random(seed)
    tasks: list[dict] = []
    for i in range(n):
        if template == "product":
            problem = (
                f"Compute the product of {rng.randint(2, 12)} and {rng.randint(2, 12)}."
            )
            rollouts = [
                f"Multiply using distributive property variant {j}. "
                f"Result: {rng.randint(1, 200)}"
                for j in range(n_rollouts)
            ]
        elif template == "addition":
            problem = f"What is {rng.randint(2, 20)} + {rng.randint(2, 20)}?"
            rollouts = [
                f"Add the numbers step by step. Answer: {rng.randint(1, 100)}"
                for _ in range(n_rollouts)
            ]
        else:
            raise ValueError(f"unknown task_template: {template}")
        tasks.append({"problem": problem, "rollouts": rollouts, "problem_id": i})
    return tasks


def judge_client_from_config(cfg: dict[str, Any]) -> JudgeClient:
    judge = cfg["judge"]
    base_url = judge.get("base_url", "")
    if not base_url:
        raise ValueError("judge.base_url is required in smoke config")
    return JudgeClient(
        JudgeClientConfig(
            base_url=base_url,
            auth_token=os.environ.get("JUDGE_AUTH_TOKEN"),
            model=judge.get("model", "Qwen/Qwen2.5-7B-Instruct"),
            concurrency=int(judge.get("concurrency", 1)),
            timeout_s=float(judge.get("timeout_s", 120)),
            temperature=float(judge.get("temperature", 0.0)),
            max_tokens=int(judge.get("max_tokens", 4096)),
        )
    )


def judge_tasks_from_config(cfg: dict[str, Any]) -> list[JudgeTask]:
    return [
        JudgeTask(
            problem=t["problem"],
            rollouts=t["rollouts"],
            problem_id=t.get("problem_id"),
        )
        for t in synthetic_tasks(cfg)
    ]


def parse_failure_diagnostics(results: list) -> dict[str, Any]:
    """Summarize why parses failed (truncated JSON vs empty HTTP error)."""
    no_raw = 0
    truncated_json = 0
    invalid_json = 0
    samples: list[dict] = []
    for i, r in enumerate(results):
        if r.parse_ok:
            continue
        raw = r.raw_response
        if raw is None:
            no_raw += 1
            kind = "http_or_empty"
        elif not raw.strip().endswith("}"):
            truncated_json += 1
            kind = "truncated"
        else:
            invalid_json += 1
            kind = "invalid_json"
        if len(samples) < 3:
            samples.append(
                {
                    "task_index": i,
                    "kind": kind,
                    "raw_tail": (raw or "")[-200:],
                }
            )
    return {
        "parse_fail_http_or_empty": no_raw,
        "parse_fail_truncated": truncated_json,
        "parse_fail_invalid_json": invalid_json,
        "parse_fail_samples": samples,
    }
