"""Dispatch pilot runs (Run0 proxy, GRPO training + tier-1 eval)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import yaml

from pilot.eval.splits import pilot_eval_paths
from pilot.infra.artifacts import artifact_dir, bootstrap_run_artifacts
from pilot.infra.budget_guard import record_cost
from pilot.infra.config_resolver import resolve_run_config
from pilot.train.answer_parse import extract_answer, is_correct
from pilot.train.canonicalize import cluster_id
from pilot.train.eval_rollouts import run_tier1_eval
from pilot.train.rollout_engine import ROLLOUT_MICRO_BATCH_SIZE
from pilot.train.run_proxy import (
    PromptProxyResult,
    RolloutRecord,
    has_minority_correct_cluster,
    minority_correct_prompt_rate,
    write_run0_artifacts,
    _load_prompt_slice,
)

logger = logging.getLogger(__name__)

TRAINING_RUN_IDS = frozenset(
    {"smoke", "run1_grpo", "run1b_grpo", "run2_inverse_freq", "run3_f_grpo"}
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_merged_config(run_id: str, config: dict[str, Any] | None) -> dict[str, Any]:
    if config is not None:
        return config
    return resolve_run_config(run_id)


def _setup_run_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in root.handlers):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
            force=True,
        )


def _rollout_engine_config(config: dict[str, Any]) -> Any:
    from pilot.train.rollout_engine import RolloutEngineConfig

    return RolloutEngineConfig(
        model_id=str(config["model_id"]),
        max_new_tokens=min(int(config.get("max_new_tokens", 2048)), 1536),
        temperature=float(config.get("temperature", 1.0)),
        top_p=float(config.get("top_p", 0.95)),
        micro_batch_size=int(config.get("rollout_micro_batch_size", ROLLOUT_MICRO_BATCH_SIZE)),
        allow_seeded_prompt_batching=bool(
            config.get("allow_seeded_prompt_batching", False)
        ),
    )


def _engine_for_eval(config: dict[str, Any], train_dir: Path) -> Any:
    """Use in-memory engine when exposed; otherwise reload checkpoint or base weights."""
    from pilot.train.rollout_engine import HFRolloutEngine

    try:
        from pilot.train.hf_grpo_train import get_trained_rollout_engine

        engine = get_trained_rollout_engine()
        if engine is not None:
            return engine
    except ImportError:
        pass

    ckpt = train_dir / "checkpoint"
    cfg = _rollout_engine_config(config)
    if ckpt.is_dir():
        return HFRolloutEngine.from_checkpoint(ckpt, cfg)
    return HFRolloutEngine(cfg)


def _tier1_eval_paths(repo_root: Path | None = None) -> dict[str, Path]:
    root = repo_root or repo_root_from_here()
    paths = pilot_eval_paths()
    return {
        "aime25_eval_30": (root / paths["primary"]).resolve(),
        "hmmt_nov25_eval_30": (root / paths["secondary"]).resolve(),
    }


def _objective_overrides(config: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    inv = config.get("inverse_freq")
    if isinstance(inv, dict):
        if "gamma" in inv:
            overrides["inverse_gamma"] = float(inv["gamma"])
        if "w_max" in inv:
            overrides["w_max"] = float(inv["w_max"])
    fg = config.get("f_grpo")
    if isinstance(fg, dict) and "focal_gamma" in fg:
        overrides["focal_gamma"] = float(fg["focal_gamma"])
    return overrides


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
    _setup_run_logging(log_path)

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
            max_new_tokens=min(int(shared.get("max_new_tokens", 2048)), 1536),
            temperature=float(shared.get("temperature", 1.0)),
            top_p=float(shared.get("top_p", 0.95)),
            micro_batch_size=int(
                config.get("rollout_micro_batch_size", ROLLOUT_MICRO_BATCH_SIZE)
            ),
            allow_seeded_prompt_batching=bool(
                config.get("allow_seeded_prompt_batching", False)
            ),
        )
    )

    results: list[PromptProxyResult] = []
    mb = max(1, int(config.get("rollout_micro_batch_size", ROLLOUT_MICRO_BATCH_SIZE)))
    for mb_start in range(0, len(prompts), mb):
        chunk = prompts[mb_start : mb_start + mb]
        problems = [row["problem"] for row in chunk]
        chunk_seeds = [seed + mb_start + j for j in range(len(chunk))]
        done_after = mb_start + len(chunk)
        logger.info(
            "run0 chunk %s-%s/%s (rollouts=%s)",
            mb_start + 1,
            done_after,
            len(prompts),
            n_rollouts,
        )
        texts_batch = engine.sample_rollouts_batch(
            problems, n_rollouts, seeds=chunk_seeds
        )
        for j, row in enumerate(chunk):
            pid = row["prompt_id"]
            gold = str(row["answer"])
            problem = row["problem"]
            texts = texts_batch[j]
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
        done = done_after
        logger.info("completed %s/%s prompts", done, len(prompts))

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


def run_training_with_eval(
    config: dict[str, Any],
    *,
    repo_root: Path,
    artifacts_root: Path,
) -> Path:
    """GRPO training from base model, then tier-1 eval artifacts."""
    from pilot.train.hf_grpo_train import run_grpo_training

    run_id = str(config["run_id"])
    out_dir = artifact_dir(run_id, artifacts_root=artifacts_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_run_artifacts(
        config,
        artifacts_root=artifacts_root,
        repo_root=repo_root,
        out_dir=out_dir,
    )
    _setup_run_logging(out_dir / "train.log")

    overrides = _objective_overrides(config)
    train_cfg = dict(config)
    if overrides:
        train_cfg["objective_overrides"] = overrides
        logger.info("objective=%s overrides=%s", config.get("objective"), overrides)
    train_cfg["defer_cost_record"] = True

    seed = int(config.get("seed", 42))
    n_rollouts = int(config.get("rollouts_per_prompt", 8))
    max_prompts = config.get("debug_max_prompts")
    price = float(config.get("modal_price_per_sec", 0.000694))

    t0 = time.time()
    out_dir = run_grpo_training(
        train_cfg,
        repo_root=repo_root,
        artifacts_root=artifacts_root,
    )
    logger.info("training finished: %s", out_dir)

    if run_id == "smoke":
        logger.info(
            "smoke run: skipping tier-1 eval (§6 gate is training-only; "
            "run B6 eval separately with debug_max_prompts=4 if needed)"
        )
    else:
        engine = _engine_for_eval(config, out_dir)
        eval_paths = _tier1_eval_paths(repo_root)
        logger.info("starting tier-1 eval on %s", list(eval_paths.keys()))
        run_tier1_eval(
            engine,
            eval_paths,
            run_id=run_id,
            out_dir=out_dir,
            seed=seed,
            n_rollouts=n_rollouts,
            debug_max_prompts=max_prompts,
        )

    gpu_seconds = time.time() - t0
    record_cost(
        out_dir,
        gpu_seconds=gpu_seconds,
        price_per_sec=price,
        run_id=run_id,
    )
    logger.info("Run %s done: gpu_seconds=%.1f artifacts=%s", run_id, gpu_seconds, out_dir)
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

    if run_id in TRAINING_RUN_IDS:
        return run_training_with_eval(config, repo_root=root, artifacts_root=art)

    raise NotImplementedError(
        f"execute_run does not yet implement run_id={run_id!r}. "
        f"Known: run0_proxy, {', '.join(sorted(TRAINING_RUN_IDS))}."
    )
