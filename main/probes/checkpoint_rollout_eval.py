"""Fixed-slice rollout eval: base model vs training checkpoints.

Supports parallel Modal workers (one GPU per checkpoint) and multiple
datasets (Polaris train slice, DAPO HF sample, AIME-25 jsonl).
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
import yaml

_MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

from infra.modal_image import image
from infra.modal_volume import (
    ARTIFACTS_MOUNT,
    ARTIFACTS_VOLUME_NAME,
    HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME,
)
from train.prompts import format_problem
from train.repro import git_metadata as _git_metadata
from train.reward import compute_reward
from train.rollout import RolloutCfg, RolloutEngine
from train.trainer import TrainCfg, build_hf, load_ckpt, rollout_seed, set_seeds
from data.dataset import JsonlPromptDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

N_ROLLOUTS_DEFAULT = 8
K_PASS = 8

app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-checkpoint-eval"))

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)

_MODAL_FN_KWARGS = dict(
    gpu=os.environ.get("CS224R_GPU_CLASS", "H200"),
    timeout=14400,
    image=image,
    secrets=[modal.Secret.from_name("HUGGINGFACE")],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)


def _pass_at_k_unbiased(n_correct: int, n: int, k: int) -> float:
    if n - n_correct < k:
        return 1.0
    return 1.0 - math.comb(n - n_correct, k) / math.comb(n, k)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_hf_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        str(spec["hf_dataset"]),
        str(spec.get("hf_config", "default")),
        split=str(spec.get("hf_split", "train")),
    )
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        problem = row.get("problem") or row.get("question") or ""
        if not problem and "prompt" in row:
            prompt = row["prompt"]
            if isinstance(prompt, list) and prompt:
                problem = prompt[0].get("content", "")
            else:
                problem = str(prompt)
        gold = row.get("answer") or row.get("gold") or row.get("solution") or ""
        rows.append(
            {
                "problem_id": row.get("id", row.get("prompt_id", i)),
                "problem": str(problem).strip(),
                "gold": str(gold).strip(),
            }
        )
    return rows


def _sample_rows(
    rows: list[dict[str, Any]],
    *,
    n_prompts: int,
    seed: int,
) -> list[dict[str, Any]]:
    if len(rows) < n_prompts:
        raise ValueError(f"need {n_prompts} prompts, dataset has {len(rows)}")
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [rows[i] for i in order[:n_prompts]]


def _rows_to_slice(rows: list[dict[str, Any]]) -> tuple[list[str], list[str], list[int]]:
    problems = [str(r["problem"]).strip() for r in rows]
    golds = [str(r.get("gold", r.get("answer", ""))).strip() for r in rows]
    problem_ids = [
        int(r["problem_id"]) if "problem_id" in r else i for i, r in enumerate(rows)
    ]
    return problems, golds, problem_ids


def _score_rollouts(
    rollouts_by_prompt: dict[int, list[dict[str, Any]]],
    *,
    n_rollouts: int,
) -> dict[str, Any]:
    n_prompts = len(rollouts_by_prompt)
    rewards: list[float] = []
    correct_count_hist = [0] * (n_rollouts + 1)
    pass8_vals: list[float] = []
    n_mixed = 0

    for p_idx in sorted(rollouts_by_prompt.keys()):
        rows = rollouts_by_prompt[p_idx]
        n_correct = sum(1 for r in rows if r["reward"] > 0)
        k = min(n_correct, n_rollouts)
        correct_count_hist[k] += 1
        if 0 < n_correct < n_rollouts:
            n_mixed += 1
        pass8_vals.append(_pass_at_k_unbiased(n_correct, n_rollouts, K_PASS))
        for r in rows:
            rewards.append(float(r["reward"]))

    frac_hist = {
        f"frac_prompts_{i}_correct": c / n_prompts for i, c in enumerate(correct_count_hist)
    }
    return {
        "n_prompts": n_prompts,
        "n_rollouts": n_rollouts,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "prompt_coverage": sum(1 for h in correct_count_hist[1:] if h) / n_prompts,
        "mixed_reward_rate": n_mixed / n_prompts if n_prompts else 0.0,
        "pass_at_8_mean": sum(pass8_vals) / len(pass8_vals) if pass8_vals else 0.0,
        "pass_at_1_rollout": sum(rewards) / len(rewards) if rewards else 0.0,
        **frac_hist,
    }


def _train_cfg(cfg_raw: dict[str, Any], n_rollouts: int) -> TrainCfg:
    train_raw = {
        **cfg_raw,
        "arm": "grpo",
        "train": {**cfg_raw.get("train", {}), "n_rollouts": n_rollouts},
        "rollout": cfg_raw["rollout"],
        "loss": cfg_raw.get("loss", {}),
        "weight_sync": cfg_raw.get("weight_sync", {"every_n_steps": 1}),
        "wandb": cfg_raw.get("wandb", {}),
    }
    return TrainCfg.from_dict(train_raw)


def _generate_chunked(
    rollout_engine: RolloutEngine,
    formatted: list[str],
    golds: list[str],
    problem_ids: list[int],
    *,
    n_rollouts: int,
    global_seed: int,
    prompt_variant: str,
    chunk_prompts: int,
) -> dict[int, list[dict[str, Any]]]:
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    n = len(formatted)
    for start in range(0, n, chunk_prompts):
        end = min(start + chunk_prompts, n)
        chunk_fmt = formatted[start:end]
        chunk_golds = golds[start:end]
        chunk_pids = problem_ids[start:end]
        chunk_seeds = [
            rollout_seed(global_seed, pid, n_rollouts, r)
            for pid in chunk_pids
            for r in range(n_rollouts)
        ]
        logger.info(
            "  rollout chunk prompts [%s:%s) (%s seqs)",
            start,
            end,
            len(chunk_fmt) * n_rollouts,
        )
        rollouts = rollout_engine.generate(chunk_fmt, n_rollouts, chunk_seeds)
        for rr in rollouts:
            p_idx = start + rr.prompt_idx
            meta = compute_reward(
                rr.completion_text,
                chunk_golds[rr.prompt_idx],
                prompt_variant=prompt_variant,
            )
            by_prompt[p_idx].append(
                {
                    "rollout_idx": rr.rollout_idx,
                    "reward": float(meta["reward"]),
                    "parse_ok": bool(meta["parse_ok"]),
                    "extract_path": meta.get("extract_path"),
                    "finish_reason": rr.finish_reason,
                }
            )
    return by_prompt


def _generate_and_score(
    rollout_engine: RolloutEngine,
    *,
    label: str,
    ckpt_path: Path | None,
    cfg: TrainCfg,
    problems: list[str],
    golds: list[str],
    problem_ids: list[int],
    prompt_variant: str,
    global_seed: int,
    n_rollouts: int,
    data_path: Path,
    chunk_prompts: int,
) -> dict[str, Any]:
    import torch

    if ckpt_path is not None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint missing: {ckpt_path}")
        dataset = JsonlPromptDataset(str(data_path), global_seed)
        hf_model, opt = build_hf(cfg)
        load_ckpt(ckpt_path, hf_model, opt, dataset, restore_dataset=False)
        hf_model.eval()
        sync_stats = rollout_engine.update_weights(hf_model)
        logger.info(
            "%s: loaded %s, synced %.2f MB in %.2fs",
            label,
            ckpt_path.name,
            sync_stats.bytes_moved / (1024**2),
            sync_stats.wall_clock_s,
        )
        del hf_model, opt, dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    formatted = [format_problem(p, variant=prompt_variant) for p in problems]
    t0 = time.monotonic()
    logger.info(
        "%s: generating %s prompts x %s rollouts (chunk=%s)",
        label,
        len(problems),
        n_rollouts,
        chunk_prompts,
    )
    by_prompt = _generate_chunked(
        rollout_engine,
        formatted,
        golds,
        problem_ids,
        n_rollouts=n_rollouts,
        global_seed=global_seed,
        prompt_variant=prompt_variant,
        chunk_prompts=chunk_prompts,
    )
    gen_s = time.monotonic() - t0

    metrics = _score_rollouts(by_prompt, n_rollouts=n_rollouts)
    metrics["label"] = label
    metrics["checkpoint"] = str(ckpt_path) if ckpt_path else "base"
    metrics["wall_clock_s"] = gen_s
    return metrics


def _load_eval_slice(
    data_path: Path,
    *,
    global_seed: int,
    n_prompts: int,
) -> tuple[list[str], list[str], list[int]]:
    rows = _load_jsonl_rows(data_path)
    picked = _sample_rows(rows, n_prompts=n_prompts, seed=global_seed)
    return _rows_to_slice(picked)


def _print_summary(results: list[dict[str, Any]], *, title: str) -> None:
    if not results:
        return
    base = results[0]
    print(f"\n=== {title} ===")
    print(
        f"{'label':<12} {'mean_reward':>12} {'pass@8':>10} {'f0':>8} {'f1-3':>8} {'mixed':>8} {'cov':>8} {'sec':>8}"
    )
    for r in results:
        f13 = sum(r.get(f"frac_prompts_{k}_correct", 0.0) for k in range(1, 4))
        print(
            f"{r['label']:<12} {r['mean_reward']:12.4f} {r['pass_at_8_mean']:10.4f} "
            f"{r.get('frac_prompts_0_correct', 0):8.3f} {f13:8.3f} "
            f"{r.get('mixed_reward_rate', 0):8.3f} {r.get('prompt_coverage', 0):8.3f} "
            f"{r.get('wall_clock_s', 0):8.0f}"
        )
    print("\nDelta vs base (mean_reward / pass@8 / frac_0):")
    for r in results[1:]:
        dr = r["mean_reward"] - base["mean_reward"]
        dp = r["pass_at_8_mean"] - base["pass_at_8_mean"]
        df0 = r.get("frac_prompts_0_correct", 0) - base.get("frac_prompts_0_correct", 0)
        print(f"  {r['label']}: reward {dr:+.4f}  pass@8 {dp:+.4f}  frac_0 {df0:+.3f}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _safe_name(raw: str) -> str:
    return raw.replace("/", "_").replace(" ", "_")


def _resolve_jsonl_path(path: Path) -> Path:
    if path.is_file():
        return path
    bundled = Path("/root/main/data/eval") / path.name
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(path)


def _dataset_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    kind = str(spec.get("kind", "jsonl"))
    if kind == "jsonl":
        rows = _load_jsonl_rows(_resolve_jsonl_path(Path(str(spec["path"]))))
    elif kind == "hf":
        rows = _load_hf_rows(spec)
    else:
        raise ValueError(f"unknown dataset kind: {kind!r}")
    n_prompts = int(spec["n_prompts"])
    seed = int(spec.get("seed", 42))
    return _sample_rows(rows, n_prompts=n_prompts, seed=seed)


@app.function(**_MODAL_FN_KWARGS)
def prepare_dataset_manifest(config_path: str, dataset_key: str) -> str:
    """Write fixed prompt slice to volume; return manifest path."""
    with open(config_path) as f:
        cfg_raw = yaml.safe_load(f)
    spec = cfg_raw["eval"]["datasets"][dataset_key]
    rows = _dataset_rows(spec)
    manifest_dir = Path(ARTIFACTS_MOUNT) / "probes/checkpoint_eval_2k/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out_path = manifest_dir / (
        f"{dataset_key}_n{len(rows)}_seed{spec.get('seed', 42)}.jsonl"
    )
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    if dataset_key == "aime25":
        vol_aime = Path(ARTIFACTS_MOUNT) / "data/eval/aime25.jsonl"
        vol_aime.parent.mkdir(parents=True, exist_ok=True)
        if not vol_aime.exists() or vol_aime.stat().st_size == 0:
            vol_aime.write_text(out_path.read_text())
    artifacts_volume.commit()
    logger.info("Wrote manifest %s (%s rows)", out_path, len(rows))
    return str(out_path)


@app.function(**_MODAL_FN_KWARGS)
def eval_one_checkpoint(
    config_path: str,
    run_stamp: str,
    dataset_key: str,
    manifest_path: str,
    label: str,
    ckpt_path: str | None,
) -> dict[str, Any]:
    """Single checkpoint × single dataset on one GPU."""
    with open(config_path) as f:
        cfg_raw = yaml.safe_load(f)
    eval_cfg = cfg_raw["eval"]
    ds_spec = eval_cfg["datasets"][dataset_key]
    dataset_kind = str(ds_spec.get("kind", "jsonl"))
    if "polaris" in eval_cfg["datasets"]:
        data_path = Path(str(eval_cfg["datasets"]["polaris"]["path"]))
    else:
        # Checkpoint load only needs HF weights; eval metrics come from manifest_path.
        data_path = Path("/vol/data/polaris_train.jsonl")
        if not data_path.is_file():
            data_path = _MAIN_ROOT / "data/polaris_train.jsonl"

    rows = _load_jsonl_rows(Path(manifest_path))
    problems, golds, problem_ids = _rows_to_slice(rows)
    prompt_variant = str(ds_spec["prompt_variant"])
    global_seed = int(cfg_raw.get("global_seed", 42))
    n_rollouts = int(eval_cfg.get("n_rollouts", N_ROLLOUTS_DEFAULT))
    chunk_prompts = int(eval_cfg.get("rollout_chunk_prompts", 64))
    cfg = _train_cfg(cfg_raw, n_rollouts)

    set_seeds(global_seed)
    os.environ.setdefault("CS224R_VLLM_SLEEP", "0")

    ckpt = Path(ckpt_path) if ckpt_path else None
    if ckpt is not None and not ckpt.is_file():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")

    logger.info(
        "eval_one_checkpoint dataset=%s label=%s ckpt=%s n_prompts=%s",
        dataset_key,
        label,
        ckpt_path or "base",
        len(problems),
    )
    rollout_engine = RolloutEngine(cfg.rollout)
    metrics = _generate_and_score(
        rollout_engine,
        label=label,
        ckpt_path=ckpt,
        cfg=cfg,
        problems=problems,
        golds=golds,
        problem_ids=problem_ids,
        prompt_variant=prompt_variant,
        global_seed=global_seed,
        n_rollouts=n_rollouts,
        data_path=data_path,
        chunk_prompts=chunk_prompts,
    )
    metrics["dataset"] = dataset_key
    metrics["manifest_path"] = manifest_path
    metrics["prompt_variant"] = prompt_variant
    partial_path = (
        Path(str(eval_cfg.get("output_dir", f"{ARTIFACTS_MOUNT}/probes/checkpoint_eval_2k")))
        / run_stamp
        / "partials"
        / dataset_key
        / f"{_safe_name(label)}.json"
    )
    _write_json(partial_path, metrics)
    artifacts_volume.commit()
    logger.info("Wrote partial %s", partial_path)
    return metrics


@app.function(**_MODAL_FN_KWARGS)
def run_parallel_eval(config_path: str) -> str:
    """Prepare manifests, spawn 4×GPU per dataset, merge results."""
    with open(config_path) as f:
        cfg_raw = yaml.safe_load(f)
    eval_cfg = cfg_raw["eval"]
    ckpt_dir = Path(str(eval_cfg["checkpoint_dir"]))
    steps = [int(s) for s in eval_cfg.get("checkpoint_steps", [49, 99, 149])]

    variants: list[tuple[str, str | None]] = [("base", None)]
    for step in steps:
        p = ckpt_dir / f"step_{step:06d}.pt"
        if not p.is_file():
            raise FileNotFoundError(f"missing {p}")
        variants.append((f"step_{step}", str(p)))

    dataset_keys = list(eval_cfg["datasets"].keys())
    all_results: dict[str, list[dict[str, Any]]] = {}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(str(eval_cfg.get("output_dir", f"{ARTIFACTS_MOUNT}/probes/checkpoint_eval_2k")))
    run_dir = out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "run_meta.json",
        {
            "git": _git_metadata(),
            "parallel_gpus": len(variants),
            "variants": [v[0] for v in variants],
            "datasets": dataset_keys,
        },
    )
    artifacts_volume.commit()

    for dataset_key in dataset_keys:
        logger.info("=== Dataset %s ===", dataset_key)
        manifest_path = prepare_dataset_manifest.remote(config_path, dataset_key)
        handles = []
        for label, ckpt in variants:
            handles.append(
                eval_one_checkpoint.spawn(
                    config_path,
                    stamp,
                    dataset_key,
                    manifest_path,
                    label,
                    ckpt,
                )
            )
        results = [h.get() for h in handles]
        all_results[dataset_key] = sorted(results, key=lambda r: r["label"])
        _print_summary(results, title=f"{dataset_key} summary")
        _write_json(run_dir / f"{dataset_key}_summary.json", {"results": all_results[dataset_key]})
        artifacts_volume.commit()
    payload = {
        "git": _git_metadata(),
        "parallel_gpus": len(variants),
        "variants": [v[0] for v in variants],
        "datasets": all_results,
    }
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", out_path)
    artifacts_volume.commit()
    return str(out_path)


def run_checkpoint_eval(cfg_raw: dict[str, Any]) -> dict[str, Any]:
    """Sequential all-checkpoints on one GPU (legacy / small runs)."""
    eval_cfg = cfg_raw["eval"]
    data_path = Path(str(eval_cfg["data_path"]))
    ckpt_dir = Path(str(eval_cfg["checkpoint_dir"]))
    n_batches = int(eval_cfg.get("n_batches", 2))
    batch_size = int(eval_cfg.get("batch_size", 64))
    n_prompts = int(eval_cfg.get("n_prompts", n_batches * batch_size))
    steps: list[int] = [int(s) for s in eval_cfg.get("checkpoint_steps", [49, 99, 149])]

    set_seeds(int(cfg_raw.get("global_seed", 42)))
    if "n_prompts" in eval_cfg and "data_path" in eval_cfg:
        problems, golds, problem_ids = _load_eval_slice(
            data_path,
            global_seed=int(cfg_raw.get("global_seed", 42)),
            n_prompts=n_prompts,
        )
    else:
        problems, golds, problem_ids = _load_eval_slice(
            data_path,
            global_seed=int(cfg_raw.get("global_seed", 42)),
            n_prompts=n_batches * batch_size,
        )

    variants: list[tuple[str, Path | None]] = [("base", None)]
    for step in steps:
        variants.append((f"step_{step}", ckpt_dir / f"step_{step:06d}.pt"))

    missing = [str(p) for _, p in variants if p is not None and not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")

    n_rollouts = int(eval_cfg.get("n_rollouts", N_ROLLOUTS_DEFAULT))
    prompt_variant = str(cfg_raw.get("prompt_variant", "hybrid_answer_boxed"))
    global_seed = int(cfg_raw.get("global_seed", 42))
    chunk_prompts = int(eval_cfg.get("rollout_chunk_prompts", 64))
    cfg = _train_cfg(cfg_raw, n_rollouts)

    os.environ.setdefault("CS224R_VLLM_SLEEP", "0")
    rollout_engine = RolloutEngine(cfg.rollout)

    results: list[dict[str, Any]] = []
    for label, ckpt_path in variants:
        results.append(
            _generate_and_score(
                rollout_engine,
                label=label,
                ckpt_path=ckpt_path,
                cfg=cfg,
                problems=problems,
                golds=golds,
                problem_ids=problem_ids,
                prompt_variant=prompt_variant,
                global_seed=global_seed,
                n_rollouts=n_rollouts,
                data_path=data_path,
                chunk_prompts=chunk_prompts,
            )
        )

    out_dir = Path(str(eval_cfg.get("output_dir", f"{ARTIFACTS_MOUNT}/probes/checkpoint_eval")))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "git": _git_metadata(),
        "n_prompts": len(problems),
        "problem_ids": problem_ids,
        "results": results,
    }
    out_path = run_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    _print_summary(results, title="checkpoint eval summary")
    return payload


@app.function(**_MODAL_FN_KWARGS)
def run_eval(config_path: str) -> str:
    with open(config_path) as f:
        cfg_raw = yaml.safe_load(f)
    if cfg_raw.get("eval", {}).get("datasets"):
        return run_parallel_eval(config_path)
    run_checkpoint_eval(cfg_raw)
    artifacts_volume.commit()
    return "ok"


@app.local_entrypoint()
def main(config: str = "main/configs/checkpoint_eval.yaml") -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg_path = Path(config)
    if not cfg_path.is_file():
        cfg_path = repo / config
    with open(cfg_path) as f:
        cfg_raw = yaml.safe_load(f)
    if cfg_raw.get("eval", {}).get("datasets"):
        print(run_parallel_eval.remote(str(cfg_path)))
    else:
        print(run_eval.remote(str(cfg_path)))
