"""Dispatch pilot runs (Run0 proxy, future GRPO training)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml

from pilot.eval.io import write_metrics
from pilot.infra.artifacts import artifact_dir
from pilot.infra.budget_guard import record_cost
from pilot.infra.config_resolver import resolve_run_config
from pilot.train.answer_parse import extract_answer, is_correct
from pilot.train.canonicalize import cluster_id
from pilot.train.hf_grpo_train import GRPO_RUN_IDS, run_grpo_training
from pilot.train.run_proxy import (
    PromptProxyResult,
    RolloutRecord,
    has_minority_correct_cluster,
    minority_correct_prompt_rate,
    write_run0_artifacts,
    _load_prompt_slice,
)

logger = logging.getLogger(__name__)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_merged_config(run_id: str, config: dict[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return config
    return resolve_run_config(run_id)


def run0_proxy(
    config: dict[str, Any],
    *,
    repo_root: Path,
    artifacts_root: Path,
) -> Path:
    """GPU rollouts on train slice; writes Run0 artifacts."""
    from pilot.train.rollout_engine import HFRolloutEngine, RolloutEngineConfig

    shared_path = repo_root / "pilot" / "configs" / "shared_train.yaml"
    shared = yaml.safe_load(shared_path.read_text())

    run_id = str(config["run_id"])
    out_dir = artifact_dir(run_id, artifacts_root=artifacts_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    t0 = time.time()
    n_rollouts = int(shared.get("rollouts_per_prompt", 8))
    seed = int(config.get("seed", shared.get("seed", 42)))
    start = int(config.get("run0_slice_start", 0))
    end = int(config.get("run0_slice_end", 500))
    max_prompts = config.get("debug_max_prompts")
    data_path = repo_root / str(shared.get("train_data", "pilot/data/dapo_slice_3k.jsonl"))

    prompts = _load_prompt_slice(data_path, start, end)
    if max_prompts is not None:
        prompts = prompts[: int(max_prompts)]
        logger.info("debug_max_prompts=%s", max_prompts)

    logger.info("Run0: %s prompts, N=%s, model=%s", len(prompts), n_rollouts, shared["model_id"])

    engine = HFRolloutEngine(
        RolloutEngineConfig(
            model_id=str(shared["model_id"]),
            max_new_tokens=min(int(shared.get("max_new_tokens", 2048)), 1024),
            temperature=float(shared.get("temperature", 1.0)),
            top_p=float(shared.get("top_p", 0.95)),
        )
    )

    results: list[PromptProxyResult] = []
    for i, row in enumerate(prompts):
        pid = row["prompt_id"]
        gold = str(row["answer"])
        problem = row["problem"]
        texts = engine.sample_rollouts(problem, n_rollouts, seed=seed + i)
        rollouts: list[RolloutRecord] = []
        for text in texts:
            parsed = extract_answer(text)
            rollouts.append(
                RolloutRecord(
                    prompt_id=pid,
                    parsed_answer=parsed,
                    correct=is_correct(text, gold),
                    cluster_id=cluster_id(parsed),
                    completion=text,
                )
            )
        correct = [r.correct for r in rollouts]
        cluster_ids = [r.cluster_id for r in rollouts]
        results.append(
            PromptProxyResult(
                prompt_id=pid,
                gold_answer=gold,
                problem=problem,
                rollouts=rollouts,
                n_distinct_clusters=len(set(cluster_ids)),
                has_correct=any(correct),
                has_minority_correct=has_minority_correct_cluster(correct, cluster_ids),
            )
        )
        if (i + 1) % 25 == 0:
            logger.info("completed %s/%s prompts", i + 1, len(prompts))

    metrics_path = write_run0_artifacts(artifacts_root, results, run_id=run_id)
    gpu_seconds = time.time() - t0
    price = float(shared.get("modal_price_per_sec", 0.000694))
    record_cost(
        out_dir,
        gpu_seconds=gpu_seconds,
        price_per_sec=price,
        run_id=run_id,
    )
    logger.info(
        "Run0 done: minority_correct_prompt_rate=%.3f gpu_seconds=%.1f",
        minority_correct_prompt_rate(results),
        gpu_seconds,
    )
    return out_dir


def execute_run(
    config: dict[str, Any],
    *,
    repo_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> Path:
    root = repo_root or repo_root_from_here()
    art = artifacts_root or root / "pilot" / "artifacts"
    run_id = str(config["run_id"])
    mode = config.get("mode")

    if run_id == "run0_proxy" or mode == "proxy_rollout_only":
        return run0_proxy(config, repo_root=root, artifacts_root=art)

    raise NotImplementedError(
        f"execute_run does not yet implement run_id={run_id!r}. Run0 only."
    )
