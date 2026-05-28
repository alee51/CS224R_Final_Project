"""Group A probe — Phase 1 rollouts and Phase 2 judge."""

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
from train.repro import dep_versions as _dep_versions
from train.repro import git_metadata as _git_metadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

POLARIS_DATASET_ID = "POLARIS-Project/Polaris-Dataset-53K"
POLARIS_CACHE_REL = "probes/05-24/group_a/polaris_cache.jsonl"
_PHASE1_GPU = os.environ.get("CS224R_GPU_CLASS", "H100")


def _phase2_step_offset(phase1_done: dict[str, Any]) -> int:
    """Phase 1 logs batch scalars at step=rollouts_done; Phase 2 must start above that."""
    n_rollouts = int(phase1_done.get("n_rollouts", 0))
    return n_rollouts + 1000

app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-probe-a-untagged"))

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def _load_yaml(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


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


def _reset_vram_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _vram_gb_used() -> float:
    import torch

    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        if peak_bytes > 0:
            return peak_bytes / (1024**3)

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        mibs = [float(line.strip()) for line in out.splitlines() if line.strip()]
        if mibs:
            return max(mibs) / 1024.0
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize_manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem_id": int(row["problem_id"]),
        "problem": str(row["problem"]).strip(),
        "gold": str(row.get("gold", row.get("answer", ""))).strip(),
        "difficulty_band": str(
            row.get("difficulty_band", row.get("difficulty", "unknown"))
        ),
        "hf_index": row.get("hf_index"),
    }


def _load_external_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows = [_normalize_manifest_row(r) for r in _read_jsonl(manifest_path)]
    rows.sort(key=lambda r: r["problem_id"])
    return rows


def _manifest_by_id(manifest: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(m["problem_id"]): m for m in manifest}


def _shard_params(cfg: dict[str, Any]) -> tuple[int, int]:
    shard_cfg = cfg.get("shard") or {}
    shard_index = int(os.environ.get("CS224R_SHARD_INDEX", shard_cfg.get("index", 0)))
    shard_count = int(os.environ.get("CS224R_NUM_SHARDS", shard_cfg.get("count", 1)))
    if shard_count < 1:
        raise ValueError(f"shard count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard index {shard_index} out of range for count {shard_count}"
        )
    return shard_index, shard_count


def _apply_shard(
    manifest: list[dict[str, Any]], shard_index: int, shard_count: int
) -> list[dict[str, Any]]:
    if shard_count == 1:
        return manifest
    return [
        m
        for m in manifest
        if int(m["problem_id"]) % shard_count == shard_index
    ]


def _resolve_phase1_artifacts(
    cfg: dict[str, Any], vol_root: Path
) -> tuple[Path, Path, Path, int, int, str]:
    art = cfg["artifacts"]
    shard_index, shard_count = _shard_params(cfg)
    stamp = os.environ.get("CS224R_RUN_STAMP") or art.get("run_stamp")
    if not stamp:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_dir = vol_root / str(art.get("base_dir", Path(art["manifest_path"]).parent))
    if art.get("rollouts_path") and art.get("phase1_done_path"):
        manifest_path = vol_root / art["manifest_path"]
        rollouts_path = vol_root / art["rollouts_path"]
        phase1_done_path = vol_root / art["phase1_done_path"]
    else:
        shard_dir = (
            base_dir
            / stamp
            / f"shard_{shard_index:02d}_of_{shard_count:02d}"
        )
        manifest_path = vol_root / art["manifest_path"]
        rollouts_path = shard_dir / "rollouts.jsonl"
        phase1_done_path = shard_dir / "phase1_done.json"
    return (
        manifest_path,
        rollouts_path,
        phase1_done_path,
        shard_index,
        shard_count,
        stamp,
    )


def _init_wandb(
    cfg: dict[str, Any],
    repro: dict[str, Any],
    prompt_variant: str,
    *,
    shard_index: int | None = None,
    shard_count: int | None = None,
    run_stamp: str | None = None,
) -> Any:
    import wandb

    operator = cfg["operator"]
    ts = datetime.now(timezone.utc).strftime("%m-%d-%H%M")
    if cfg.get("base_rollout_pass"):
        shard_tag = ""
        if shard_count and shard_count > 1 and shard_index is not None:
            shard_tag = f"-s{shard_index}of{shard_count}"
        run_name = f"base-rollout-51k{shard_tag}_{operator}_{ts}"
    else:
        run_name = f"probe-prompt-{prompt_variant}_{operator}_{ts}"
    tags = ["probe", operator, cfg["gpu_class"], repro["git_sha_short"], prompt_variant]
    if cfg.get("smoke"):
        tags.append("smoke")
    if cfg.get("base_rollout_pass"):
        tags.append("base_rollout_pass")

    group = cfg["wandb"]["group"]
    if cfg.get("base_rollout_pass") and run_stamp:
        group = f"{group}-{run_stamp}"

    run = wandb.init(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        group=group,
        name=run_name,
        config=cfg,
        tags=tags,
    )
    wandb.log(
        {
            "prompt_variant": prompt_variant,
            "global_seed": cfg["global_seed"],
            **repro,
            **_dep_versions(),
        }
    )
    return run


def _resume_wandb(cfg: dict[str, Any], wandb_run_id: str) -> Any:
    import wandb

    return wandb.init(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        group=cfg["wandb"]["group"],
        id=wandb_run_id,
        resume="must",
    )


@app.function(
    gpu=_PHASE1_GPU,
    timeout=86400,
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
    prompt_variant = cfg.get("prompt_variant", "dapo_answer_v1")
    reuse_manifest = bool(cfg.get("reuse_manifest", False))
    external_manifest = bool(cfg.get("external_manifest", False))
    vol_root = Path(ARTIFACTS_MOUNT)
    (
        manifest_path,
        rollouts_path,
        phase1_done_path,
        shard_index,
        shard_count,
        run_stamp,
    ) = _resolve_phase1_artifacts(cfg, vol_root)

    run = _init_wandb(
        cfg,
        repro,
        prompt_variant,
        shard_index=shard_index,
        shard_count=shard_count,
        run_stamp=run_stamp,
    )
    wandb_run_id = run.id

    smoke = bool(cfg.get("smoke", False))
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

    log_rollout_table = bool(cfg.get("log_rollout_table", True))
    wandb_log_every_batches = int(cfg.get("wandb_log_every_batches", 1))

    if reuse_manifest or external_manifest:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"reuse_manifest: missing manifest at {manifest_path}"
            )
        if external_manifest:
            manifest = _load_external_manifest(manifest_path)
        else:
            manifest = [_normalize_manifest_row(r) for r in _read_jsonl(manifest_path)]
        manifest = _apply_shard(manifest, shard_index, shard_count)
        logger.info(
            "Loaded manifest (%s prompts, shard %s/%s) from %s",
            len(manifest),
            shard_index,
            shard_count,
            manifest_path,
        )
    else:
        per_band = int(cfg["smoke_per_band"] if smoke else cfg["sampling"]["per_band"])
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
    _reset_vram_peak()
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
    manifest_by_id = _manifest_by_id(manifest)

    total_rollouts = len(manifest) * rollouts_per_prompt
    rollouts_done = 0
    run_t0 = time.monotonic()
    total_output_tokens = 0

    requests: list[tuple[str, int, int, int]] = []
    for entry in manifest:
        prompt = format_problem(entry["problem"], variant=prompt_variant)
        pid = int(entry["problem_id"])
        for rollout_idx in range(rollouts_per_prompt):
            seed = global_seed + pid * rollouts_per_prompt + rollout_idx
            requests.append((prompt, pid, rollout_idx, seed))

    batches_done = 0
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

            entry = manifest_by_id[problem_id]
            reward_fields = compute_reward(
                completion, entry["gold"], prompt_variant=prompt_variant
            )
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
            if log_rollout_table:
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
                        prompt_variant,
                    ]
                )
            rollouts_done += 1

        total_output_tokens += batch_output_tokens
        batches_done += 1
        tokens_per_sec = (
            batch_output_tokens / batch_elapsed if batch_elapsed > 0 else 0.0
        )
        if batches_done % wandb_log_every_batches == 0:
            wandb.log(
                {
                    "vllm_tokens_per_sec": tokens_per_sec,
                    "wall_clock_s": batch_elapsed,
                    "vram_gb_used": _vram_gb_used(),
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                },
                step=rollouts_done,
            )
        logger.info(
            "Batch rollouts %s/%s (%.1f tok/s)",
            rollouts_done,
            total_rollouts,
            tokens_per_sec,
        )
        if batches_done % 20 == 0:
            artifacts_volume.commit()

    wall_clock_total = time.monotonic() - run_t0
    if log_rollout_table and table_rows:
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
            band = manifest_by_id[row["problem_id"]]["difficulty_band"]
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
        "shard_index": shard_index,
        "shard_count": shard_count,
        "run_stamp": run_stamp,
        "rollouts_path": str(rollouts_path.relative_to(vol_root)),
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


@app.function(
    image=image,
    timeout=36000,
)
def run_pipeline(config: str) -> str:
    """Orchestrate Phase 1 → Phase 2 on Modal (safe to chain .remote() here)."""
    wandb_run_id = run_phase1.remote(config=config)
    return run_phase2.remote(config=config, wandb_run_id=wandb_run_id)


@app.function(
    gpu="H100",
    timeout=14400,
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
def run_phase2(config: str, wandb_run_id: str | None = None) -> str:
    """Phase 2: read Phase 1 artifacts, vLLM judge, wandb resume, pointer json."""
    from vllm import LLM, SamplingParams

    from judge.format import (
        _assignment_from_poly_epo_payload,
        _strip_json_fences,
        build_judge_messages,
    )

    cfg = _load_yaml(config)
    vol_root = Path(ARTIFACTS_MOUNT)
    art = cfg["artifacts"]
    manifest_path = vol_root / art["manifest_path"]
    rollouts_path = vol_root / art["rollouts_path"]
    phase1_done_path = vol_root / art["phase1_done_path"]
    pointer_path = vol_root / art["pointer_path"]

    if not phase1_done_path.is_file():
        raise FileNotFoundError(
            f"Phase 1 not complete: missing {phase1_done_path}. "
            "Run run_phase1 first or check volume artifacts."
        )

    with phase1_done_path.open() as f:
        phase1_done = json.load(f)

    phase2_step_offset = _phase2_step_offset(phase1_done)

    if wandb_run_id is None:
        wandb_run_id = phase1_done["wandb_run_id"]
    if not wandb_run_id:
        raise ValueError("wandb_run_id required (arg or phase1_done.json)")

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not rollouts_path.is_file():
        raise FileNotFoundError(f"Missing rollouts: {rollouts_path}")

    manifest = _read_jsonl(manifest_path)
    rollout_rows = _read_jsonl(rollouts_path)

    smoke = bool(cfg.get("smoke"))
    rollouts_per_prompt = int(
        cfg["smoke_n_rollouts"] if smoke else cfg["sampling"]["rollouts_per_prompt"]
    )
    expected_total = len(manifest) * rollouts_per_prompt
    phase1_total = int(phase1_done.get("n_rollouts", expected_total))
    if len(rollout_rows) != expected_total or len(rollout_rows) != phase1_total:
        raise RuntimeError(
            "Rollout count mismatch: rollouts_path has "
            f"{len(rollout_rows)}, manifest*rollouts_per_prompt={expected_total}, "
            f"phase1_done.n_rollouts={phase1_total}. Phase-1 file is truncated or "
            "rollouts_per_prompt drifted between phases."
        )

    rollouts_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        rollouts_by_pid[int(row["problem_id"])].append(row)
    for pid in rollouts_by_pid:
        rollouts_by_pid[pid].sort(key=lambda r: int(r["rollout_idx"]))

    missing_pids = [
        int(m["problem_id"])
        for m in manifest
        if len(rollouts_by_pid.get(int(m["problem_id"]), [])) != rollouts_per_prompt
    ]
    if missing_pids:
        raise RuntimeError(
            f"Per-prompt rollout count != {rollouts_per_prompt} for "
            f"{len(missing_pids)} problem_ids (first 5: {missing_pids[:5]})"
        )

    run = _resume_wandb(cfg, wandb_run_id)
    import wandb

    phase2_cfg = cfg["phase2"]
    max_model_len = int(phase2_cfg["max_model_len"])
    max_num_seqs = int(phase2_cfg["max_num_seqs"])
    modal_price = float(cfg["modal_price_per_sec"])
    apply_chat_template = bool(phase2_cfg.get("apply_chat_template", True))

    llm = LLM(
        model=phase2_cfg["model"],
        max_model_len=max_model_len,
        gpu_memory_utilization=float(phase2_cfg["gpu_memory_utilization"]),
        max_num_seqs=max_num_seqs,
    )
    _reset_vram_peak()
    tokenizer = llm.get_tokenizer()
    judge_params = SamplingParams(
        temperature=float(phase2_cfg["temperature"]),
        max_tokens=int(phase2_cfg["max_tokens"]),
    )

    judge_tasks: list[dict[str, Any]] = []
    for entry in manifest:
        problem_id = int(entry["problem_id"])
        rollouts = rollouts_by_pid.get(problem_id, [])
        if not rollouts:
            raise RuntimeError(f"No rollouts for problem_id={problem_id}")
        n_rollouts = len(rollouts)
        system, user = build_judge_messages(entry["problem"], rollouts)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if apply_chat_template:
            prompt_str = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt_str = f"{system}\n\n{user}"

        judge_input_tokens = len(tokenizer.encode(prompt_str))
        truncated = judge_input_tokens > max_model_len
        judge_tasks.append(
            {
                "problem_id": problem_id,
                "difficulty_band": entry.get("difficulty_band"),
                "prompt_str": prompt_str,
                "n_rollouts": n_rollouts,
                "judge_input_tokens": judge_input_tokens,
                "truncated": truncated,
            }
        )

    table_columns = [
        "problem_id",
        "difficulty_band",
        "judge_input_tokens",
        "output_tokens",
        "wall_clock_s",
        "json_parse_ok",
        "truncated",
        "cluster_count",
        "cluster_100_hits",
        "cost_per_call",
        "finish_reason",
    ]
    table_rows: list[list[Any]] = []

    wall_clocks: list[float] = []
    output_tokens_all: list[float] = []
    cluster_counts: list[float] = []
    costs_per_call: list[float] = []
    json_parse_ok_count = 0
    judged_count = 0

    runnable = [t for t in judge_tasks if not t["truncated"]]
    truncated_tasks = [t for t in judge_tasks if t["truncated"]]

    for batch_start in range(0, len(runnable), max_num_seqs):
        batch = runnable[batch_start : batch_start + max_num_seqs]
        prompts = [t["prompt_str"] for t in batch]
        batch_t0 = time.monotonic()
        outputs = llm.generate(prompts, judge_params)
        batch_elapsed = time.monotonic() - batch_t0
        per_call_wall = batch_elapsed / len(batch) if batch else 0.0

        for task, out in zip(batch, outputs):
            completion_out = out.outputs[0]
            completion_text = completion_out.text
            output_tokens = len(completion_out.token_ids)
            finish_reason = completion_out.finish_reason
            wall_clock_s = per_call_wall
            cost_per_call = wall_clock_s * modal_price

            json_parse_ok = False
            cluster_count: int | None = None
            cluster_100_hits: int | None = None

            try:
                payload = json.loads(_strip_json_fences(completion_text))
                assignment, clusters = _assignment_from_poly_epo_payload(
                    payload, task["n_rollouts"]
                )
                cluster_count = len(clusters)
                cluster_100_hits = sum(1 for cid in assignment.values() if cid == -1)
                json_parse_ok = True
                json_parse_ok_count += 1
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                logger.warning(
                    "Judge JSON parse failed problem_id=%s: %s",
                    task["problem_id"],
                    exc,
                )

            judged_count += 1
            wall_clocks.append(wall_clock_s)
            output_tokens_all.append(float(output_tokens))
            if cluster_count is not None:
                cluster_counts.append(float(cluster_count))
            costs_per_call.append(cost_per_call)

            wandb.log(
                {
                    "judge_wall_clock_s": wall_clock_s,
                    "judge_input_tokens": task["judge_input_tokens"],
                    "judge_output_tokens": output_tokens,
                    "judge_vram_gb_used": _vram_gb_used(),
                    "json_parse_ok": json_parse_ok,
                    "truncated": False,
                    "cluster_count": cluster_count,
                    "cluster_100_hits": cluster_100_hits,
                    "cost_per_call": cost_per_call,
                },
                step=phase2_step_offset + judged_count,
            )

            table_rows.append(
                [
                    task["problem_id"],
                    task["difficulty_band"],
                    task["judge_input_tokens"],
                    output_tokens,
                    wall_clock_s,
                    json_parse_ok,
                    False,
                    cluster_count,
                    cluster_100_hits,
                    cost_per_call,
                    finish_reason,
                ]
            )

    for task in truncated_tasks:
        table_rows.append(
            [
                task["problem_id"],
                task["difficulty_band"],
                task["judge_input_tokens"],
                None,
                None,
                False,
                True,
                None,
                None,
                None,
                None,
            ]
        )

    n_prompts = len(judge_tasks)
    truncated_count = len(truncated_tasks)

    final_log: dict[str, Any] = {
        "phase2_n_prompts": n_prompts,
        "phase2_judged_calls": judged_count,
        "phase2_truncated_count": truncated_count,
        "phase2_truncated_rate": truncated_count / n_prompts if n_prompts else 0.0,
        "phase2_json_parse_ok_rate": (
            json_parse_ok_count / judged_count if judged_count else 0.0
        ),
        "judge_vram_gb_used": _vram_gb_used(),
    }
    if wall_clocks:
        for key, val in _percentiles(wall_clocks, (50, 90, 95, 99)).items():
            final_log[f"judge_wall_clock_{key}"] = val
        final_log["phase2_judge_wall_clock_hist"] = wandb.Histogram(wall_clocks)
    if output_tokens_all:
        for key, val in _percentiles(output_tokens_all, (50, 90, 95, 99)).items():
            final_log[f"judge_output_tokens_{key}"] = val
        final_log["phase2_judge_output_tokens_hist"] = wandb.Histogram(output_tokens_all)
    if costs_per_call:
        final_log["phase2_cost_per_call_hist"] = wandb.Histogram(costs_per_call)
        final_log["phase2_cost_per_call_median"] = _percentiles(costs_per_call, (50,))[
            "p50"
        ]
    if cluster_counts:
        final_log["phase2_cluster_count_hist"] = wandb.Histogram(cluster_counts)

    wandb.log(
        {
            **final_log,
            "phase2_judge_results": wandb.Table(columns=table_columns, data=table_rows),
        }
    )

    group_dir = str(Path(art["manifest_path"]).parent)
    pointer_record = {
        "modal_volume": ARTIFACTS_VOLUME_NAME,
        "path": f"{group_dir}/",
        "wandb_run_id": wandb_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    with pointer_path.open("w") as f:
        json.dump(pointer_record, f, indent=2)
    artifacts_volume.commit()

    logger.info(
        "Phase 2 complete: %s prompts (%s judged, %s truncated), wandb=%s, pointer=%s",
        n_prompts,
        judged_count,
        truncated_count,
        wandb_run_id,
        pointer_path,
    )
    run.finish()
    return wandb_run_id


@app.local_entrypoint()
def run_phase1_only(config: str) -> None:
    """Launch Phase 1 only; survives laptop disconnect via spawn."""
    call = run_phase1.spawn(config=config)
    print(f"Spawned Phase 1 only: {call.object_id}")
    print("Track progress on Modal dashboard; wandb run appears after Phase 1 init.")


@app.local_entrypoint()
def run_full(config: str) -> None:
    """Launch full probe pipeline; survives laptop disconnect via spawn."""
    call = run_pipeline.spawn(config=config)
    print(f"Spawned Group A pipeline (Phase 1 → Phase 2): {call.object_id}")
    print("Track progress on Modal dashboard; wandb run appears after Phase 1 init.")
