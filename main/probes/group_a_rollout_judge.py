"""Group A probe — Phase 1 rollouts (Phase 2 judge added in Phase D)."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
import yaml

# Local `modal run` resolves imports from repo root's `main/` package root.
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

POLARIS_DATASET_ID = "POLARIS-Project/Polaris-Dataset-53K"
POLARIS_CACHE_REL = "probes/05-24/group_a/polaris_cache.jsonl"
PROMPT_VARIANT = "dapo_answer_v1"

app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-probe-a-untagged"))

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def _load_yaml(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _git_metadata() -> dict[str, Any]:
    def _run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        dirty = bool(_run(["git", "status", "--porcelain"]))
        short = _run(["git", "rev-parse", "--short", "HEAD"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, dirty, short = "unknown", False, "unknown"
    return {"git_sha": sha, "git_dirty": dirty, "git_sha_short": short}


def _dep_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("vllm", "torch", "transformers", "bitsandbytes"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def _is_integer_gold(answer: Any) -> bool:
    s = str(answer).strip().replace(",", "")
    if not s:
        return False
    if s.startswith("-") and len(s) > 1:
        return s[1:].isdigit()
    return s.isdigit()


def _verify_difficulty_bands(
    counts: dict[str, int], configured_bands: list[str]
) -> list[str]:
    present = sorted(counts.keys())
    configured_set = set(configured_bands)
    present_set = set(present)
    if configured_set != present_set:
        logger.warning(
            "Polaris difficulty bands mismatch: configured=%s present=%s counts=%s",
            configured_bands,
            present,
            counts,
        )
        return [b for b in configured_bands if b in present_set] or present
    return configured_bands


def _load_or_build_polaris_cache(vol_root: Path) -> list[dict[str, Any]]:
    cache_path = vol_root / POLARIS_CACHE_REL
    if cache_path.is_file():
        logger.info("Reading Polaris cache from %s", cache_path)
        rows: list[dict[str, Any]] = []
        with cache_path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    from datasets import load_dataset

    logger.info("Downloading %s (first run)", POLARIS_DATASET_ID)
    ds = load_dataset(POLARIS_DATASET_ID, split="train")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with cache_path.open("w") as out:
        for hf_index, row in enumerate(ds):
            rec = {
                "problem": row["problem"],
                "answer": row["answer"],
                "difficulty": row["difficulty"],
                "hf_index": hf_index,
            }
            rows.append(rec)
            out.write(json.dumps(rec) + "\n")
    logger.info("Wrote Polaris cache (%s rows) to %s", len(rows), cache_path)
    return rows


def _clean_polaris_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for row in rows:
        problem = str(row.get("problem", "")).strip()
        gold = row.get("answer", row.get("gold", ""))
        if not problem:
            continue
        if not _is_integer_gold(gold):
            continue
        clean.append(
            {
                "problem": problem,
                "gold": str(gold).strip(),
                "difficulty": row["difficulty"],
                "hf_index": row.get("hf_index"),
            }
        )
    return clean


def _sample_manifest(
    rows: list[dict[str, Any]],
    bands: list[str],
    per_band: int,
    global_seed: int,
) -> list[dict[str, Any]]:
    by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_band[row["difficulty"]].append(row)

    rng = random.Random(global_seed)
    manifest: list[dict[str, Any]] = []
    problem_id = 0
    for band in bands:
        pool = by_band.get(band, [])
        if len(pool) < per_band:
            raise RuntimeError(
                f"Band {band} has only {len(pool)} clean rows; need {per_band}"
            )
        indices = list(range(len(pool)))
        rng.shuffle(indices)
        chosen = [pool[i] for i in indices[:per_band]]
        for row in chosen:
            manifest.append(
                {
                    "problem_id": problem_id,
                    "problem": row["problem"],
                    "gold": row["gold"],
                    "difficulty_band": band,
                    "hf_index": row.get("hf_index"),
                }
            )
            problem_id += 1
    return manifest


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _percentiles(values: list[float], ps: tuple[int, ...]) -> dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in ps}
    sorted_v = sorted(values)
    n = len(sorted_v)
    out: dict[str, float] = {}
    for p in ps:
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        out[f"p{p}"] = float(sorted_v[idx])
    return out


def _vram_gb_used() -> float:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**3)
    return 0.0


def _init_wandb(cfg: dict[str, Any], repro: dict[str, Any]) -> Any:
    import wandb

    operator = cfg["operator"]
    ts = datetime.now(timezone.utc).strftime("%m-%d-%H%M")
    run_name = f"probe-A_{operator}_{ts}"
    tags = ["probe", operator, cfg["gpu_class"], repro["git_sha_short"]]
    if cfg.get("smoke"):
        tags.append("smoke")

    run = wandb.init(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        group=cfg["wandb"]["group"],
        name=run_name,
        config=cfg,
        tags=tags,
    )
    wandb.log(
        {
            "prompt_variant": PROMPT_VARIANT,
            "global_seed": cfg["global_seed"],
            **repro,
            **_dep_versions(),
        }
    )
    return run


@app.function(
    gpu="H100",
    timeout=10800,
    image=image,
    secrets=[
        modal.Secret.from_name("HUGGINGFACE"),
        modal.Secret.from_name("WANDB_API_KEY"),
    ],
    volumes={
        ARTIFACTS_MOUNT: artifacts_volume,
        HF_CACHE_MOUNT: hf_cache_volume,
    },
)
def run_phase1(config: str) -> str:
    """Phase 1: Polaris sample, vLLM rollouts, volume artifacts, wandb logging."""
    from vllm import LLM, SamplingParams

    from train.prompts import format_problem
    from train.reward import compute_reward

    cfg = _load_yaml(config)
    repro = _git_metadata()
    vol_root = Path(ARTIFACTS_MOUNT)
    art = cfg["artifacts"]
    manifest_path = vol_root / art["manifest_path"]
    rollouts_path = vol_root / art["rollouts_path"]
    phase1_done_path = vol_root / art["phase1_done_path"]

    run = _init_wandb(cfg, repro)
    wandb_run_id = run.id

    smoke = bool(cfg.get("smoke", False))
    per_band = int(cfg["smoke_per_band"] if smoke else cfg["sampling"]["per_band"])
    rollouts_per_prompt = int(
        cfg["smoke_n_rollouts"] if smoke else cfg["sampling"]["rollouts_per_prompt"]
    )
    global_seed = int(cfg["global_seed"])
    temperature = float(cfg["sampling"]["temperature"])
    phase1_cfg = cfg["phase1"]

    random.seed(global_seed)
    import numpy as np

    np.random.seed(global_seed)
    import torch

    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)

    raw_rows = _load_or_build_polaris_cache(vol_root)
    clean_rows = _clean_polaris_rows(raw_rows)
    band_counts: dict[str, int] = defaultdict(int)
    for row in clean_rows:
        band_counts[row["difficulty"]] += 1
    logger.info("Polaris clean difficulty counts: %s", dict(band_counts))

    configured_bands = list(cfg["sampling"]["difficulty_bands"])
    bands = _verify_difficulty_bands(dict(band_counts), configured_bands)

    manifest = _sample_manifest(clean_rows, bands, per_band, global_seed)
    _write_jsonl(manifest_path, manifest)
    logger.info("Wrote manifest (%s prompts) to %s", len(manifest), manifest_path)

    if rollouts_path.exists():
        rollouts_path.unlink()

    llm = LLM(
        model=phase1_cfg["model"],
        max_model_len=int(phase1_cfg["max_model_len"]),
        gpu_memory_utilization=float(phase1_cfg["gpu_memory_utilization"]),
        max_num_seqs=int(phase1_cfg["max_num_seqs"]),
        enable_prefix_caching=bool(phase1_cfg.get("enable_prefix_caching", True)),
    )
    max_response_length = int(phase1_cfg["max_response_length"])
    batch_size = int(phase1_cfg["max_num_seqs"])

    import wandb

    table_columns = [
        "problem_id",
        "rollout_idx",
        "difficulty_band",
        "reward",
        "parse_ok",
        "parsed_answer",
        "parsed_is_int",
        "has_boxed",
        "has_answer_line",
        "strict_parse_ok",
        "length_tokens",
        "prompt_tokens",
        "finish_reason",
        "prompt_variant",
    ]
    table_rows: list[list[Any]] = []

    total_rollouts = len(manifest) * rollouts_per_prompt
    rollouts_done = 0
    run_t0 = time.monotonic()
    total_output_tokens = 0

    requests: list[tuple[str, int, int, int]] = []
    for entry in manifest:
        prompt = format_problem(entry["problem"])
        pid = int(entry["problem_id"])
        for rollout_idx in range(rollouts_per_prompt):
            seed = global_seed + pid * rollouts_per_prompt + rollout_idx
            requests.append((prompt, pid, rollout_idx, seed))

    for batch_start in range(0, len(requests), batch_size):
        batch = requests[batch_start : batch_start + batch_size]
        prompts = [p for p, _, _, _ in batch]
        params_list = [
            SamplingParams(
                temperature=temperature,
                max_tokens=max_response_length,
                seed=seed,
                n=1,
            )
            for _, _, _, seed in batch
        ]
        batch_t0 = time.monotonic()
        outputs = llm.generate(prompts, params_list)
        batch_elapsed = time.monotonic() - batch_t0
        batch_output_tokens = 0

        for (prompt, problem_id, rollout_idx, _seed), out in zip(batch, outputs):
            completion_out = out.outputs[0]
            completion = completion_out.text
            finish_reason = completion_out.finish_reason
            length_tokens = len(completion_out.token_ids)
            prompt_tokens = len(out.prompt_token_ids)
            batch_output_tokens += length_tokens

            entry = manifest[problem_id]
            reward_fields = compute_reward(completion, entry["gold"])
            record = {
                "problem_id": problem_id,
                "rollout_idx": rollout_idx,
                "completion": completion,
                "reward": reward_fields["reward"],
                "parse_ok": reward_fields["parse_ok"],
                "parsed_answer": reward_fields["parsed_answer"],
                "parsed_is_int": reward_fields["parsed_is_int"],
                "has_boxed": reward_fields["has_boxed"],
                "has_answer_line": reward_fields["has_answer_line"],
                "strict_parse_ok": reward_fields["strict_parse_ok"],
                "length_tokens": length_tokens,
                "prompt_tokens": prompt_tokens,
                "finish_reason": finish_reason,
            }
            _append_jsonl(rollouts_path, record)
            table_rows.append(
                [
                    problem_id,
                    rollout_idx,
                    entry["difficulty_band"],
                    reward_fields["reward"],
                    reward_fields["parse_ok"],
                    reward_fields.get("parsed_answer"),
                    reward_fields["parsed_is_int"],
                    reward_fields["has_boxed"],
                    reward_fields["has_answer_line"],
                    reward_fields["strict_parse_ok"],
                    length_tokens,
                    prompt_tokens,
                    finish_reason,
                    PROMPT_VARIANT,
                ]
            )
            rollouts_done += 1

        total_output_tokens += batch_output_tokens
        tokens_per_sec = (
            batch_output_tokens / batch_elapsed if batch_elapsed > 0 else 0.0
        )
        wandb.log(
            {
                "vllm_tokens_per_sec": tokens_per_sec,
                "wall_clock_s": batch_elapsed,
                "vram_gb_used": _vram_gb_used(),
            },
            step=rollouts_done,
        )
        logger.info(
            "Batch rollouts %s/%s (%.1f tok/s)",
            rollouts_done,
            total_rollouts,
            tokens_per_sec,
        )

    wall_clock_total = time.monotonic() - run_t0
    wandb.log(
        {
            "phase1_rollouts": wandb.Table(columns=table_columns, data=table_rows),
        }
    )

    by_band_rollouts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_prompt_rewards: dict[int, list[int]] = defaultdict(list)
    length_tokens_all: list[float] = []
    prompt_tokens_all: list[float] = []

    with rollouts_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            band = manifest[row["problem_id"]]["difficulty_band"]
            by_band_rollouts[band].append(row)
            per_prompt_rewards[row["problem_id"]].append(int(row["reward"]))
            length_tokens_all.append(float(row["length_tokens"]))
            prompt_tokens_all.append(float(row["prompt_tokens"]))

    final_log: dict[str, Any] = {
        "n_prompts": len(manifest),
        "n_rollouts": rollouts_done,
        "wall_clock_total_s": wall_clock_total,
        "tokens_per_sec_mean": (
            total_output_tokens / wall_clock_total if wall_clock_total > 0 else 0.0
        ),
    }
    for key, val in _percentiles(length_tokens_all, (50, 90, 95, 99)).items():
        final_log[f"length_tokens_{key}"] = val
    for key, val in _percentiles(prompt_tokens_all, (50, 90, 95, 99)).items():
        final_log[f"prompt_tokens_{key}"] = val

    all_rows = [r for rows in by_band_rollouts.values() for r in rows]
    if all_rows:
        final_log["parse_ok_rate"] = sum(1 for r in all_rows if r["parse_ok"]) / len(
            all_rows
        )

    for band, rows in sorted(by_band_rollouts.items()):
        if not rows:
            continue
        final_log[f"pass_rate/{band}"] = sum(r["reward"] for r in rows) / len(rows)
        final_log[f"parse_rate/{band}"] = sum(1 for r in rows if r["parse_ok"]) / len(
            rows
        )
        band_prompt_ids = [
            m["problem_id"] for m in manifest if m["difficulty_band"] == band
        ]
        mixed = sum(
            1
            for pid in band_prompt_ids
            if per_prompt_rewards[pid]
            and 0 < sum(per_prompt_rewards[pid]) < len(per_prompt_rewards[pid])
        )
        final_log[f"mixed_reward_rate/{band}"] = (
            mixed / len(band_prompt_ids) if band_prompt_ids else 0.0
        )

    wandb.log(final_log)

    artifacts_volume.commit()
    completed_at = datetime.now(timezone.utc).isoformat()
    done_record = {
        "n_prompts": len(manifest),
        "n_rollouts": rollouts_done,
        "wandb_run_id": wandb_run_id,
        "completed_at": completed_at,
    }
    with phase1_done_path.open("w") as f:
        json.dump(done_record, f)
    artifacts_volume.commit()

    logger.info(
        "Phase 1 complete: %s prompts, %s rollouts, wandb=%s",
        len(manifest),
        rollouts_done,
        wandb_run_id,
    )
    run.finish()
    return wandb_run_id


@app.local_entrypoint()
def run_full(config: str) -> str:
    wandb_run_id = run_phase1.remote(config=config)
    # Phase D will extend this to call run_phase2 after Phase 1.
    return wandb_run_id
