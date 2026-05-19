#!/usr/bin/env python3
"""
Run0 proxy: base-model rollouts only (no training).

Writes `pilot/artifacts/run0_proxy/metrics.json` with `minority_correct_prompt_rate`.
Integration: `pilot/infra/modal_launch.py` should invoke this with `--config pilot/configs/run0_proxy.yaml`.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pilot.train.canonicalize import cluster_id
from pilot.eval.io import write_metrics


@dataclass
class RolloutRecord:
    prompt_id: str
    parsed_answer: str
    correct: bool
    cluster_id: int
    completion: str = ""


@dataclass
class PromptProxyResult:
    prompt_id: str
    gold_answer: str
    problem: str
    rollouts: list[RolloutRecord]
    n_distinct_clusters: int
    has_correct: bool
    has_minority_correct: bool


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _load_prompt_slice(data_path: Path, start: int, end: int) -> list[dict[str, str]]:
    """Rows [start, end) from train JSONL — no shuffle (Run0 uses train prefix)."""
    rows: list[dict[str, str]] = []
    with data_path.open() as f:
        for i, line in enumerate(f):
            if i >= end:
                break
            if i >= start:
                rows.append(json.loads(line))
    return rows


def _is_correct(parsed: str, gold: str) -> bool:
    return parsed.strip() == gold.strip()


def has_minority_correct_cluster(correct: list[bool], cluster_ids: list[int]) -> bool:
    """
    True when ≥1 correct cluster exists and some correct rollout lies in a
    minority correct cluster (freq strictly below the majority among correct).
    """
    correct_clusters = [cid for ok, cid in zip(correct, cluster_ids) if ok]
    if not correct_clusters:
        return False
    freq = Counter(correct_clusters)
    majority_freq = max(freq.values())
    return any(count < majority_freq for count in freq.values())


def minority_correct_prompt_rate(results: list[PromptProxyResult]) -> float:
    if not results:
        return 0.0
    n = sum(1 for r in results if r.has_minority_correct)
    return n / len(results)


def mock_rollouts_for_prompt(
    prompt_id: str,
    gold_answer: str,
    n: int,
    rng: random.Random,
) -> list[RolloutRecord]:
    """
    Deterministic CPU mock: synthesize diverse clusters without a model.

    Designed so proxy metrics are computable in CI; replace with real
    `PolicyModel` rollouts in Modal launch.
    """
    records: list[RolloutRecord] = []
    # Three answer modes: gold, alternate correct formatting, wrong
    modes = [
        gold_answer,
        f" {gold_answer} ",
        f"wrong_{rng.randint(0, 99)}",
    ]
    for i in range(n):
        mode = modes[i % len(modes)]
        parsed = mode if i % 3 != 2 else modes[2]
        cid = cluster_id(parsed)
        records.append(
            RolloutRecord(
                prompt_id=prompt_id,
                parsed_answer=parsed,
                correct=_is_correct(parsed, gold_answer),
                cluster_id=cid,
            )
        )
    return records


def run_proxy_on_prompts(
    prompts: list[dict[str, str]],
    *,
    n_rollouts: int,
    seed: int,
    mock: bool = True,
) -> list[PromptProxyResult]:
    rng = random.Random(seed)
    results: list[PromptProxyResult] = []
    for row in prompts:
        pid = row["prompt_id"]
        gold = str(row["answer"])
        if mock:
            rollouts = mock_rollouts_for_prompt(pid, gold, n_rollouts, rng)
        else:
            raise NotImplementedError(
                "GPU rollout path not wired in scaffold; use --mock or integrate via pilot/infra/modal_launch.py"
            )
        correct = [r.correct for r in rollouts]
        cluster_ids = [r.cluster_id for r in rollouts]
        results.append(
            PromptProxyResult(
                prompt_id=pid,
                gold_answer=gold,
                problem=str(row["problem"]),
                rollouts=rollouts,
                n_distinct_clusters=len(set(cluster_ids)),
                has_correct=any(correct),
                has_minority_correct=has_minority_correct_cluster(correct, cluster_ids),
            )
        )
    return results


def write_run0_artifacts(
    artifacts_dir: Path,
    results: list[PromptProxyResult],
    *,
    run_id: str = "run0_proxy",
) -> Path:
    out_dir = artifacts_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = out_dir / "prompt_inputs.jsonl"
    with inputs_path.open("w") as f:
        for pr in results:
            f.write(
                json.dumps(
                    {
                        "prompt_id": pr.prompt_id,
                        "problem": pr.problem,
                        "gold_answer": pr.gold_answer,
                    }
                )
                + "\n"
            )

    pred_path = out_dir / "raw_predictions.jsonl"
    with pred_path.open("w") as f:
        for pr in results:
            for r in pr.rollouts:
                row: dict[str, object] = {
                    "prompt_id": r.prompt_id,
                    "parsed_answer": r.parsed_answer,
                    "correct": r.correct,
                    "cluster_id": r.cluster_id,
                }
                if r.completion:
                    row["completion"] = r.completion
                f.write(json.dumps(row) + "\n")

    rate = minority_correct_prompt_rate(results)
    metrics = {
        "run_id": run_id,
        "minority_correct_prompt_rate": rate,
        "n_prompts": len(results),
        "n_rollouts_per_prompt": len(results[0].rollouts) if results else 0,
        "fraction_with_correct": sum(1 for r in results if r.has_correct) / max(len(results), 1),
        "mean_distinct_clusters": (
            sum(r.n_distinct_clusters for r in results) / max(len(results), 1)
        ),
    }
    metrics_path = out_dir / "metrics.json"
    write_metrics(metrics_path, metrics)
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run0 proxy rollouts (no training)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pilot/configs/run0_proxy.yaml"),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("pilot/artifacts"),
    )
    parser.add_argument("--mock", action="store_true", default=True)
    parser.add_argument("--no-mock", action="store_false", dest="mock")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    shared = _load_yaml(Path("pilot/configs/shared_train.yaml"))
    n_rollouts = int(shared.get("rollouts_per_prompt", 8))
    seed = int(cfg.get("seed", shared.get("seed", 42)))
    start = int(cfg.get("run0_slice_start", 0))
    end = int(cfg.get("run0_slice_end", 500))
    data_path = Path(shared.get("train_data", "pilot/data/dapo_slice_3k.jsonl"))
    run_id = str(cfg.get("run_id", "run0_proxy"))

    prompts = _load_prompt_slice(data_path, start, end)
    results = run_proxy_on_prompts(prompts, n_rollouts=n_rollouts, seed=seed, mock=args.mock)
    metrics_path = write_run0_artifacts(args.artifacts_dir, results, run_id=run_id)
    print(json.dumps({"metrics_path": str(metrics_path), "n_prompts": len(results)}))


if __name__ == "__main__":
    main()
