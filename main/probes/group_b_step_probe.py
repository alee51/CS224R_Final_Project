"""Group B step probe — GRPO step timing, VRAM headroom, microbatch OOM sweep.

train_step_cache.pt structure:
  {"cache": {"tensors": {new_logprobs, old_logprobs, mask, advantages, keep_mask, n_kept}},
   "wandb_run_id": str, "prompt_variant": str}

Phase 1b re-runs a full timed step at max_microbatch_ok (fresh rollouts) so timings match
production microbatch; Phase 2 sweep only touches cached tensors with forward+backward.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
import torch
import yaml

def package_root() -> Path:
    """Directory containing train/, configs/, probes/ — from env or local layout."""
    if env := os.environ.get("CS224R_MAIN_ROOT"):
        return Path(env)
    probe = Path(__file__).resolve()
    for candidate in (probe.parents[1], probe.parents[2] / "main"):
        if (candidate / "train").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return probe.parents[1]


_MAIN_ROOT = package_root()
if str(_MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MAIN_ROOT))

_RUNTIME_CFG_NAME = ".probe_b_runtime.yaml"

from infra.modal_image import image
from infra.modal_volume import (
    ARTIFACTS_MOUNT,
    ARTIFACTS_VOLUME_NAME,
    HF_CACHE_MOUNT,
    HF_CACHE_VOLUME_NAME,
)
from train.repro import dep_versions as _dep_versions
from train.repro import git_metadata as _git_metadata
from train.rollout import RolloutEngine
from train.trainer import (
    StepBatch,
    TrainCfg,
    build_hf,
    run_microbatch_forward_backward,
    run_one_grpo_step,
    set_seeds,
    train_cfg_from_dict,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = modal.App(os.environ.get("CS224R_APP_NAME", "cs224r-probe-b-untagged"))

artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def _resolve_config_path(config_path: str) -> Path:
    """Resolve yaml path under package root (same contract as Group A `main/configs/...`)."""
    raw = Path(config_path)
    if raw.is_absolute():
        if not raw.is_file():
            raise FileNotFoundError(config_path)
        return raw
    # `main/configs/foo.yaml` from repo root → `{CS224R_MAIN_ROOT}/configs/foo.yaml`
    rel = Path(*raw.parts[1:]) if raw.parts[:1] == ("main",) else raw
    under_root = _MAIN_ROOT / rel
    if under_root.is_file():
        return under_root
    if raw.is_file():
        return raw.resolve()
    raise FileNotFoundError(
        f"Config not found: {config_path} (expected under {_MAIN_ROOT})"
    )


def _deep_merge(base_d: dict, override: dict) -> dict:
    out = dict(base_d)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_merged_config(
    config_path: str, _visited: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Merge `extends` chain locally (launcher only)."""
    path = _resolve_config_path(config_path)
    key = str(path.resolve())
    if key in _visited:
        chain = " -> ".join((*_visited, key))
        raise ValueError(f"Cycle in config `extends` chain: {chain}")
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    extends = cfg.pop("extends", None)
    if extends:
        base = load_merged_config(str(extends), _visited=(*_visited, key))
        cfg = _deep_merge(base, cfg)
    return cfg


def write_merged_runtime_config(config_path: str) -> str:
    """Merge yaml locally; return path for Modal (`main/configs/.probe_b_runtime.yaml`)."""
    cfg = load_merged_config(config_path)
    out = _MAIN_ROOT / "configs" / _RUNTIME_CFG_NAME
    with out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return f"main/configs/{_RUNTIME_CFG_NAME}"


def load_probe_config(config_path: str) -> dict[str, Any]:
    """Load flat yaml on Modal — no `extends` (merged by launch script)."""
    path = _resolve_config_path(config_path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("extends"):
        raise ValueError(
            "Remote probe expects a merged config; launcher must call write_merged_runtime_config"
        )
    return cfg


def _probe_train_cfg(cfg: dict[str, Any]) -> TrainCfg:
    merged = dict(cfg)
    if cfg.get("smoke"):
        merged["train"] = dict(cfg["train"])
        merged["train"]["n_rollouts"] = int(cfg.get("smoke_rollouts", 2))
    merged["train"]["microbatch"] = int(
        cfg["train"].get("starting_microbatch", 1)
    )
    return train_cfg_from_dict(merged)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _load_toy_batch(
    cfg: dict[str, Any], vol_root: Path
) -> tuple[list[str], list[str], list[int]]:
    toy = cfg["toy_batch"]
    manifest_path = vol_root / toy["source_manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing manifest {manifest_path}. Run Group A n800 probe first."
        )
    rows = _read_jsonl(manifest_path)
    by_pid = {int(r["problem_id"]): r for r in rows}
    if cfg.get("smoke"):
        problem_ids = list(range(int(cfg.get("smoke_prompts", 4))))
    else:
        problem_ids = [int(x) for x in toy["problem_ids"]]

    prompts: list[str] = []
    golds: list[str] = []
    pids: list[int] = []
    for pid in problem_ids:
        if pid not in by_pid:
            raise KeyError(f"problem_id {pid} not in manifest")
        row = by_pid[pid]
        prompts.append(str(row["problem"]))
        golds.append(str(row["gold"]))
        pids.append(pid)
    return prompts, golds, pids


def _artifact_paths(cfg: dict[str, Any], vol_root: Path) -> dict[str, Path]:
    art = cfg["artifacts"]
    return {
        "phase1_done": vol_root / art["phase1_done_path"],
        "cache": vol_root / art["train_step_cache_path"],
        "sweep": vol_root / art["microbatch_sweep_path"],
        "pointer": vol_root / art["pointer_path"],
    }


def _init_wandb(cfg: dict[str, Any], repro: dict[str, Any]) -> Any:
    import wandb

    operator = cfg["operator"]
    ts = datetime.now(timezone.utc).strftime("%m-%d-%H%M")
    run_name = f"probe-B_{operator}_{ts}"
    tags = [
        "probe",
        operator,
        cfg["gpu_class"],
        "arm=grpo",
        repro["git_sha_short"],
        f"prompt_variant={cfg.get('prompt_variant', 'dapo_answer_v1')}",
    ]
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
            "prompt_variant": cfg.get("prompt_variant"),
            "global_seed": cfg["global_seed"],
            "rollout.gpu_memory_utilization": cfg["rollout"][
                "gpu_memory_utilization"
            ],
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


def _log_step_result(
    result,
    cfg: dict[str, Any],
    *,
    prefix: str = "",
) -> None:
    import wandb

    log: dict[str, Any] = {}
    for phase, t_s in result.phase_times_s.items():
        log[f"{prefix}{phase}"] = t_s
        if phase in result.vram_peak_gb:
            log[f"{prefix}vram_peak_gb_{phase}"] = result.vram_peak_gb[phase]

    log.update(
        {
            f"{prefix}vram_peak_gb_step": result.diagnostics.get(
                "vram_peak_gb_step", 0.0
            ),
            "device_vram_total_gb": result.diagnostics.get(
                "device_vram_total_gb", 0.0
            ),
            f"{prefix}vram_headroom_gb_step": result.diagnostics.get(
                "vram_headroom_gb_step", 0.0
            ),
        }
    )
    if "vram_headroom_gb_after_rollout" in result.diagnostics:
        log[f"{prefix}vram_headroom_gb_after_rollout"] = result.diagnostics[
            "vram_headroom_gb_after_rollout"
        ]

    rollout_tokens = int(result.diagnostics.get("rollout_output_tokens", 0))
    t_rollout = result.phase_times_s.get("t_rollout", 0.0)
    if rollout_tokens and t_rollout > 0:
        log[f"{prefix}tokens_per_sec_collocated"] = rollout_tokens / t_rollout

    total_s = sum(result.phase_times_s.values())
    if total_s > 0:
        for phase, t_s in result.phase_times_s.items():
            log[f"{prefix}pct_{phase}"] = 100.0 * t_s / total_s
        log[f"{prefix}cost_per_step_usd"] = total_s * float(
            cfg["modal_price_per_sec"]
        )

    for key, val in result.diagnostics.items():
        if key.startswith("vram_"):
            continue
        log[f"{prefix}{key}"] = val

    log[f"{prefix}mean_reward"] = result.mean_reward
    log[f"{prefix}fraction_filtered"] = result.fraction_filtered
    log[f"{prefix}n_kept_sequences"] = result.n_kept_sequences
    wandb.log(log)


def _next_microbatch_candidates(
    last_ok: int, last_fail: int | None, smoke_cap: int | None
) -> list[int]:
    if last_fail is None:
        return []
    if last_fail - last_ok <= 1:
        return []
    mid = (last_ok + last_fail) // 2
    return [mid] if mid > last_ok else []


_MODAL_FN_KWARGS = dict(
    gpu="H200",
    timeout=3600,
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


@app.function(**_MODAL_FN_KWARGS)
def run_phase1(config: str) -> str:
    cfg_raw = load_probe_config(config)
    train_cfg = _probe_train_cfg(cfg_raw)
    repro = _git_metadata()
    vol_root = Path(ARTIFACTS_MOUNT)
    paths = _artifact_paths(cfg_raw, vol_root)
    set_seeds(train_cfg.global_seed)

    run = _init_wandb(cfg_raw, repro)
    wandb_run_id = run.id

    prompts, golds, pids = _load_toy_batch(cfg_raw, vol_root)
    batch = StepBatch(prompts=prompts, golds=golds, problem_ids=pids)
    rollout_engine = RolloutEngine(train_cfg.rollout)
    hf_model, opt = build_hf(train_cfg)

    starting_mb = int(cfg_raw["train"].get("starting_microbatch", 1))
    logger.info("Phase 1 warmup @ microbatch=1")
    run_one_grpo_step(
        train_cfg,
        rollout_engine,
        hf_model,
        opt,
        batch,
        instrument=False,
        microbatch=1,
    )
    torch.cuda.empty_cache()

    logger.info("Phase 1 timed step @ microbatch=%s", starting_mb)
    timed = run_one_grpo_step(
        train_cfg,
        rollout_engine,
        hf_model,
        opt,
        batch,
        instrument=True,
        microbatch=starting_mb,
    )
    _log_step_result(timed, cfg_raw, prefix="phase1_")

    cache_payload = {
        "cache": timed.step_cache,
        "prompt_variant": cfg_raw.get("prompt_variant"),
        "wandb_run_id": wandb_run_id,
    }
    if timed.step_cache is None:
        raise RuntimeError("timed step produced no step_cache")
    paths["cache"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_payload, paths["cache"])

    phase1_done = {
        "wandb_run_id": wandb_run_id,
        "prompt_variant": cfg_raw.get("prompt_variant"),
        "phase_times_s": timed.phase_times_s,
        "phase1b_times_s": {},
        "vram_peak_gb": timed.diagnostics.get("vram_peak_gb_step", 0.0),
        "vram_peak_gb_at_max_mb": 0.0,
        "max_microbatch_ok": 0,
        "n_kept_sequences": timed.n_kept_sequences,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with paths["phase1_done"].open("w") as f:
        json.dump(phase1_done, f, indent=2)

    artifacts_volume.commit()
    run.finish()
    logger.info("Phase 1 complete wandb=%s", wandb_run_id)
    return wandb_run_id


@app.function(**_MODAL_FN_KWARGS)
def run_phase2(config: str, wandb_run_id: str | None = None) -> int:
    cfg_raw = load_probe_config(config)
    train_cfg = _probe_train_cfg(cfg_raw)
    vol_root = Path(ARTIFACTS_MOUNT)
    paths = _artifact_paths(cfg_raw, vol_root)

    if not paths["phase1_done"].is_file():
        raise FileNotFoundError(f"Phase 1 not done: {paths['phase1_done']}")
    with paths["phase1_done"].open() as f:
        phase1_done = json.load(f)

    if wandb_run_id is None:
        wandb_run_id = phase1_done["wandb_run_id"]
    if not paths["cache"].is_file():
        raise FileNotFoundError(f"Missing cache: {paths['cache']}")

    cache_payload = torch.load(paths["cache"], map_location="cpu", weights_only=False)
    step_cache = cache_payload["cache"]
    n_kept = int(step_cache.get("n_kept") or len(step_cache.get("rollouts", [])))

    run = _resume_wandb(cfg_raw, wandb_run_id)
    import wandb

    if paths["sweep"].exists():
        paths["sweep"].unlink()

    rollout_engine = RolloutEngine(train_cfg.rollout)
    hf_model, _ = build_hf(train_cfg)

    sweep_cfg = cfg_raw.get("sweep", {})
    start_mb = int(sweep_cfg.get("start_microbatch", 1))
    max_attempts = int(sweep_cfg.get("max_attempts", 12))
    smoke_cap = (
        int(cfg_raw.get("smoke_max_microbatch", 4)) if cfg_raw.get("smoke") else None
    )

    last_ok = 0
    last_fail: int | None = None
    attempts = 0
    max_microbatch_ok = 0
    limited_by: str | None = None
    to_try: list[int] = [start_mb]

    while to_try and attempts < max_attempts:
        mb = to_try.pop(0)
        if smoke_cap and mb > smoke_cap:
            break
        # Stop early when requested mb meets/exceeds n_kept: _train_step_microbatched
        # silently clamps to n_kept, so further requests would re-run identical work
        # and falsely report the requested mb as "ok".
        if n_kept > 0 and mb >= n_kept:
            max_microbatch_ok = max(max_microbatch_ok, n_kept)
            limited_by = "n_kept"
            logger.info(
                "Sweep stop: requested mb=%s >= n_kept=%s; ceiling is n_kept",
                mb,
                n_kept,
            )
            break
        attempts += 1
        torch.cuda.empty_cache()
        try:
            ok, vram_peak, t_fwd_bwd = run_microbatch_forward_backward(
                train_cfg, hf_model, step_cache, mb
            )
        except torch.cuda.OutOfMemoryError:
            ok, vram_peak, t_fwd_bwd = False, 0.0, 0.0

        record: dict[str, Any] = {
            "microbatch": mb,
            "ok": ok,
            "vram_peak_gb": vram_peak if ok else None,
            "t_fwd_bwd_s": t_fwd_bwd if ok else None,
        }
        if not ok:
            record["error"] = "CUDA out of memory"
        _append_jsonl(paths["sweep"], record)
        wandb.log(
            {
                "sweep_microbatch": mb,
                "sweep_ok": ok,
                "sweep_vram_peak_gb": vram_peak if ok else None,
                "sweep_t_fwd_bwd_s": t_fwd_bwd if ok else None,
            },
            step=attempts,
        )
        logger.info("Sweep mb=%s ok=%s", mb, ok)

        if ok:
            last_ok = mb
            max_microbatch_ok = mb
            if smoke_cap and mb >= smoke_cap:
                break
            if last_fail is None:
                nxt = mb * 2
                to_try = [nxt] if nxt > mb else []
            else:
                to_try = _next_microbatch_candidates(last_ok, last_fail, smoke_cap)
        else:
            last_fail = mb
            limited_by = f"OOM_at_mb={mb}"
            if last_ok == 0:
                break
            to_try = _next_microbatch_candidates(last_ok, last_fail, smoke_cap)

    if limited_by is None and max_microbatch_ok > 0:
        limited_by = "max_attempts"

    phase1_done["max_microbatch_ok"] = max_microbatch_ok
    phase1_done["sweep_limited_by"] = limited_by
    phase1_done["sweep_n_kept"] = n_kept
    with paths["phase1_done"].open("w") as f:
        json.dump(phase1_done, f, indent=2)
    artifacts_volume.commit()
    run.finish()
    logger.info("Phase 2 complete max_microbatch_ok=%s", max_microbatch_ok)
    return max_microbatch_ok


@app.function(**_MODAL_FN_KWARGS)
def run_phase1b(config: str, wandb_run_id: str | None = None) -> str:
    cfg_raw = load_probe_config(config)
    train_cfg = _probe_train_cfg(cfg_raw)
    vol_root = Path(ARTIFACTS_MOUNT)
    paths = _artifact_paths(cfg_raw, vol_root)

    with paths["phase1_done"].open() as f:
        phase1_done = json.load(f)
    if wandb_run_id is None:
        wandb_run_id = phase1_done["wandb_run_id"]
    max_mb = int(phase1_done.get("max_microbatch_ok", 0))
    if max_mb < 1:
        raise ValueError("max_microbatch_ok missing; run Phase 2 first")

    run = _resume_wandb(cfg_raw, wandb_run_id)
    set_seeds(train_cfg.global_seed)

    prompts, golds, pids = _load_toy_batch(cfg_raw, vol_root)
    batch = StepBatch(prompts=prompts, golds=golds, problem_ids=pids)
    rollout_engine = RolloutEngine(train_cfg.rollout)
    hf_model, opt = build_hf(train_cfg)

    # Untimed warmup step so Phase 1b's timed step is "warm" like Phase 1 — same
    # code path (rollout + score + advantage + forward + backward + optim + sync),
    # but no instrument/timers/wandb/output.
    logger.info("Phase 1b warmup step @ microbatch=%s", max_mb)
    run_one_grpo_step(
        train_cfg,
        rollout_engine,
        hf_model,
        opt,
        batch,
        instrument=False,
        microbatch=max_mb,
    )
    torch.cuda.empty_cache()

    logger.info("Phase 1b full timed step @ microbatch=%s", max_mb)
    timed = run_one_grpo_step(
        train_cfg,
        rollout_engine,
        hf_model,
        opt,
        batch,
        instrument=True,
        microbatch=max_mb,
    )
    _log_step_result(timed, cfg_raw, prefix="phase1b_")

    phase1_done["phase1b_times_s"] = timed.phase_times_s
    phase1_done["vram_peak_gb_at_max_mb"] = timed.diagnostics.get(
        "vram_peak_gb_step", 0.0
    )
    phase1_done["completed_at"] = datetime.now(timezone.utc).isoformat()
    with paths["phase1_done"].open("w") as f:
        json.dump(phase1_done, f, indent=2)

    pointer_record = {
        "modal_volume": ARTIFACTS_VOLUME_NAME,
        "path": str(cfg_raw["artifacts"]["base_path"]),
        "wandb_run_id": wandb_run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["pointer"].parent.mkdir(parents=True, exist_ok=True)
    with paths["pointer"].open("w") as f:
        json.dump(pointer_record, f, indent=2)

    artifacts_volume.commit()
    run.finish()
    logger.info("Phase 1b complete")
    return wandb_run_id


@app.function(image=image, timeout=3600)
def run_pipeline(config: str) -> str:
    """Orchestrate Phase 1 → Phase 2 → Phase 1b (Modal-side chaining)."""
    wandb_run_id = run_phase1.remote(config=config)
    max_mb = run_phase2.remote(config=config, wandb_run_id=wandb_run_id)
    if max_mb < 1:
        logger.warning("No successful microbatch; skipping phase 1b")
        return wandb_run_id
    return run_phase1b.remote(config=config, wandb_run_id=wandb_run_id)


@app.local_entrypoint()
def run_full(config: str) -> None:
    """Detached launch (matches Group A `run_full`)."""
    call = run_pipeline.spawn(config=config)
    print(f"Spawned Group B pipeline: {call.object_id}")


@app.local_entrypoint()
def run_full_sync(config: str) -> str:
    """Foreground launch — blocks until pipeline finishes (smoke debugging)."""
    return run_pipeline.remote(config=config)
