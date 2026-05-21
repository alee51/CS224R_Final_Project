"""GPU GRPO training with HuggingFace Qwen (Run1–Run3)."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from pilot.eval.io import write_metrics
from pilot.infra.artifacts import artifact_dir, bootstrap_run_artifacts, git_sha
from pilot.infra.budget_guard import record_cost
from pilot.train.answer_parse import extract_answer, extract_boxed_answer, is_correct
from pilot.train.canonicalize import canonicalize_answer, cluster_id
from pilot.train.run_proxy import has_minority_correct_cluster
from pilot.train.grpo_trainer import (
    GRPOConfig,
    PromptRolloutGroup,
    TrainStepOutput,
    _kl_penalty,
)
from pilot.train.objectives import (
    ObjectiveName,
    _clip_surrogate_scalar,
    _clip_surrogate_tensor,
    advantage_l2,
    f_grpo_prompt_scale,
    grpo_advantages,
    inverse_freq_weights,
    per_trajectory_cluster_counts,
    weighted_advantages,
)
from pilot.train.rollout_engine import (
    HFRolloutEngine,
    PROMPT_TEMPLATE,
    ROLLOUT_MICRO_BATCH_SIZE,
    RolloutEngineConfig,
    batch_generate_rollouts,
)

logger = logging.getLogger(__name__)


class BudgetCapHit(Exception):
    """Raised when in-train budget polling hits budget_cap_usd."""


GRPO_RUN_IDS = frozenset(
    {"run1_grpo", "run1b_grpo", "run2_inverse_freq", "run3_f_grpo"}
)

# Micro-batch size for completion logprob forwards in _build_step_groups.
COMPLETION_LOGPROB_MICRO_BATCH_SIZE = 16

_trained_rollout_engine: HFRolloutEngine | None = None


def get_trained_rollout_engine() -> HFRolloutEngine | None:
    """In-process trained model for tier-1 eval (set by `run_grpo_training`)."""
    return _trained_rollout_engine


@dataclass
class _RolloutRecord:
    prompt_id: str
    problem: str
    completion: str
    reward: float
    cluster_id: int
    old_logprob: float
    ref_logprob: float
    rollout_idx: int = 0
    raw_advantage: float = 0.0
    weighted_advantage: float = 0.0
    cluster_key: str = ""
    cluster_size: int = 0
    is_minority_correct: bool = False
    completion_tokens: int = 0
    parser_clean: bool = False


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_step_diagnostic(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 1.0
    if len(xs) == 1:
        return 1.0 if abs(xs[0] - ys[0]) < 1e-9 else 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x < 1e-12 or den_y < 1e-12:
        return 1.0 if all(abs(x - y) < 1e-9 for x, y in zip(xs, ys)) else 0.0
    return num / (den_x * den_y)


def _expected_weighted_advantages(
    objective: ObjectiveName,
    rewards: list[float],
    cluster_ids: list[int],
    *,
    inverse_gamma: float,
    w_max: float,
    focal_gamma: float,
) -> list[float]:
    """Mirror of ``weighted_advantages`` for mechanism verification."""
    base = grpo_advantages(rewards)
    if objective == "grpo":
        return base
    if objective == "inverse_freq":
        counts = per_trajectory_cluster_counts(cluster_ids)
        weights = inverse_freq_weights(counts, gamma=inverse_gamma, w_max=w_max)
        return [a * w for a, w in zip(base, weights)]
    if objective == "f_grpo":
        mean_r = sum(rewards) / max(len(rewards), 1)
        scale = f_grpo_prompt_scale(mean_r, focal_gamma=focal_gamma)
        return [a * scale for a in base]
    raise ValueError(f"unknown objective: {objective!r}")


def _mechanism_signal_per_variant(
    objective: ObjectiveName,
    groups: list[PromptRolloutGroup],
    weighted_by_group: list[list[float]],
    *,
    inverse_gamma: float,
    w_max: float,
    focal_gamma: float,
) -> float:
    """Correlation of weighted vs expected advantages on minority-correct prompts."""
    actual: list[float] = []
    expected: list[float] = []
    for group, weighted in zip(groups, weighted_by_group):
        correct = [bool(r >= 0.5) for r in group.rewards]
        if not has_minority_correct_cluster(correct, group.cluster_ids):
            continue
        exp = _expected_weighted_advantages(
            objective,
            group.rewards,
            group.cluster_ids,
            inverse_gamma=inverse_gamma,
            w_max=w_max,
            focal_gamma=focal_gamma,
        )
        actual.extend(weighted)
        expected.extend(exp)
    if not actual:
        return 1.0
    return _pearson_corr(actual, expected)


def _is_minority_correct_rollout(
    correct: list[bool], cluster_ids: list[int], idx: int
) -> bool:
    if not correct[idx]:
        return False
    correct_clusters = [cid for ok, cid in zip(correct, cluster_ids) if ok]
    freq = Counter(correct_clusters)
    majority_freq = max(freq.values())
    return freq[cluster_ids[idx]] < majority_freq


def _completion_token_count(tokenizer: AutoTokenizer, completion: str) -> int:
    return len(tokenizer.encode(completion, add_special_tokens=False))


def _init_wandb(
    *,
    run_name: str,
    run_config: dict[str, Any],
    out_dir: Path,
) -> Any | None:
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping experiment tracking")
        return None
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    run = wandb.init(
        project="cs224r-minority-voting",
        name=run_name,
        config=run_config,
        mode=wandb_mode,
        dir=str(out_dir),
    )
    return run


def _sync_wandb_offline(out_dir: Path) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        return
    wandb_dir = out_dir / "wandb"
    if not wandb_dir.is_dir():
        return
    try:
        subprocess.run(
            ["wandb", "sync", str(wandb_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        logger.warning("wandb sync failed: %s", exc)


def _load_train_prompts(data_path: Path, *, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with data_path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def _prompt_text(problem: str) -> str:
    return PROMPT_TEMPLATE.format(problem=problem)


def _encode_prompt_completion(
    tokenizer: AutoTokenizer,
    problem: str,
    completion: str,
) -> tuple[torch.Tensor, int]:
    prompt = _prompt_text(problem)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
    full_ids = tokenizer(
        prompt + completion,
        return_tensors="pt",
        add_special_tokens=True,
    ).input_ids[0]
    return full_ids, int(prompt_ids.shape[0])


def _mean_completion_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    ids = input_ids.to(device).unsqueeze(0)
    logits = model(ids).logits[0]
    log_probs = F.log_softmax(logits, dim=-1)
    start = max(prompt_len - 1, 0)
    end = ids.shape[1] - 1
    if end <= start:
        return torch.zeros((), device=device, requires_grad=True)
    token_logps = [
        log_probs[pos, ids[0, pos + 1]] for pos in range(start, end)
    ]
    return torch.stack(token_logps).mean()


@torch.no_grad()
def _scalar_mean_completion_logprob(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problem: str,
    completion: str,
    *,
    device: torch.device,
) -> float:
    input_ids, prompt_len = _encode_prompt_completion(tokenizer, problem, completion)
    lp = _mean_completion_logprob(model, input_ids, prompt_len, device=device)
    return float(lp.item())


@torch.no_grad()
def _micro_batch_scalar_mean_completion_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problems: list[str],
    completions: list[str],
    *,
    device: torch.device,
) -> list[float]:
    """Mean completion logprob per (problem, completion); one forward over the batch."""
    if len(problems) != len(completions):
        raise ValueError("problems and completions length mismatch")
    if not problems:
        return []

    encoded = [
        _encode_prompt_completion(tokenizer, problem, completion)
        for problem, completion in zip(problems, completions)
    ]
    input_ids_list = [ids for ids, _ in encoded]
    prompt_lens = [prompt_len for _, prompt_len in encoded]
    seq_lens = [int(ids.shape[0]) for ids in input_ids_list]
    batch_size = len(problems)
    max_len = max(seq_lens)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    batch_ids = torch.full(
        (batch_size, max_len), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.long, device=device
    )
    for i, ids in enumerate(input_ids_list):
        length = seq_lens[i]
        batch_ids[i, :length] = ids.to(device)
        attention_mask[i, :length] = 1

    logits = model(batch_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)

    out: list[float] = []
    for i in range(batch_size):
        start = max(prompt_lens[i] - 1, 0)
        end = seq_lens[i] - 1
        if end <= start:
            out.append(0.0)
            continue
        token_logps = [
            log_probs[i, pos, batch_ids[i, pos + 1]] for pos in range(start, end)
        ]
        out.append(float(torch.stack(token_logps).mean().item()))
    return out


@torch.no_grad()
def _batched_scalar_mean_completion_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problems: list[str],
    completions: list[str],
    *,
    device: torch.device,
    micro_batch_size: int = COMPLETION_LOGPROB_MICRO_BATCH_SIZE,
) -> list[float]:
    """Batched completion logprobs; chunks of ``micro_batch_size`` per forward."""
    if len(problems) != len(completions):
        raise ValueError("problems and completions length mismatch")
    if not problems:
        return []

    results: list[float] = []
    for start in range(0, len(problems), micro_batch_size):
        chunk_p = problems[start : start + micro_batch_size]
        chunk_c = completions[start : start + micro_batch_size]
        results.extend(
            _micro_batch_scalar_mean_completion_logprobs(
                model, tokenizer, chunk_p, chunk_c, device=device
            )
        )
    return results


def _micro_batch_mean_completion_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problems: list[str],
    completions: list[str],
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    """Differentiable mean completion logprob per (problem, completion)."""
    if len(problems) != len(completions):
        raise ValueError("problems and completions length mismatch")
    if not problems:
        return []

    encoded = [
        _encode_prompt_completion(tokenizer, problem, completion)
        for problem, completion in zip(problems, completions)
    ]
    input_ids_list = [ids for ids, _ in encoded]
    prompt_lens = [prompt_len for _, prompt_len in encoded]
    seq_lens = [int(ids.shape[0]) for ids in input_ids_list]
    batch_size = len(problems)
    max_len = max(seq_lens)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    batch_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(input_ids_list):
        length = seq_lens[i]
        batch_ids[i, :length] = ids.to(device)
        attention_mask[i, :length] = 1

    logits = model(batch_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)

    out: list[torch.Tensor] = []
    for i in range(batch_size):
        start = max(prompt_lens[i] - 1, 0)
        end = seq_lens[i] - 1
        if end <= start:
            out.append(torch.zeros((), device=device, requires_grad=True))
            continue
        token_logps = [log_probs[i, pos, batch_ids[i, pos + 1]] for pos in range(start, end)]
        out.append(torch.stack(token_logps).mean())
    return out


def _per_rollout_kl_tensor(logprob: torch.Tensor, ref_logprob: float) -> torch.Tensor:
    ref_t = torch.tensor(ref_logprob, device=logprob.device, dtype=logprob.dtype)
    return logprob - ref_t


def _train_step_microbatch_backward(
    policy: torch.nn.Module,
    tokenizer: AutoTokenizer,
    groups: list[PromptRolloutGroup],
    specs_batch: list[list[_RolloutRecord]],
    *,
    device: torch.device,
    objective: ObjectiveName,
    grpo_cfg: GRPOConfig,
    objective_overrides: dict[str, Any],
    completion_logprob_micro_batch_size: int,
    optimizer: AdamW,
    budget_ctx: dict[str, Any] | None = None,
) -> TrainStepOutput:
    """Differentiable train step with per-micro-batch backward (frees graphs between chunks)."""
    flat_problems: list[str] = []
    flat_completions: list[str] = []
    flat_group_idx: list[int] = []
    flat_rollout_idx: list[int] = []
    for g_idx, specs in enumerate(specs_batch):
        for r_idx, spec in enumerate(specs):
            flat_problems.append(spec.problem)
            flat_completions.append(spec.completion)
            flat_group_idx.append(g_idx)
            flat_rollout_idx.append(r_idx)

    n_completions = len(flat_problems)
    if n_completions == 0:
        raise ValueError("empty specs_batch")

    group_advantages = [
        weighted_advantages(
            objective,
            group.rewards,
            group.cluster_ids,
            inverse_gamma=objective_overrides.get("inverse_gamma", grpo_cfg.inverse_gamma),
            w_max=objective_overrides.get("w_max", grpo_cfg.w_max),
            focal_gamma=objective_overrides.get("focal_gamma", grpo_cfg.focal_gamma),
        )
        for group in groups
    ]

    mb = max(1, completion_logprob_micro_batch_size)
    logprobs_by_group: list[list[float]] = [[] for _ in groups]

    optimizer.zero_grad(set_to_none=True)
    for start in range(0, n_completions, mb):
        if budget_ctx is not None:
            now = time.monotonic()
            last_check = float(budget_ctx.get("last_budget_check", budget_ctx["run_mono_t0"]))
            if now - last_check >= float(budget_ctx.get("poll_interval_s", 60.0)):
                elapsed = time.time() - float(budget_ctx["run_t0"])
                if _estimated_usd(elapsed, float(budget_ctx["price_per_sec"])) >= float(
                    budget_ctx["budget_cap_usd"]
                ):
                    raise BudgetCapHit()
                budget_ctx["last_budget_check"] = now

        chunk_p = flat_problems[start : start + mb]
        chunk_c = flat_completions[start : start + mb]
        chunk_g = flat_group_idx[start : start + mb]
        chunk_r = flat_rollout_idx[start : start + mb]

        chunk_logprobs = _micro_batch_mean_completion_logprobs(
            policy,
            tokenizer,
            chunk_p,
            chunk_c,
            device=device,
        )

        loss_mb = torch.zeros((), device=device)
        for lp, g_idx, r_idx in zip(chunk_logprobs, chunk_g, chunk_r):
            group = groups[g_idx]
            adv = group_advantages[g_idx][r_idx]
            loss_mb = loss_mb + _clip_surrogate_tensor(
                lp,
                group.old_logprobs[r_idx],
                adv,
                grpo_cfg.clip_ratio_low,
                grpo_cfg.clip_ratio_high,
            ) / n_completions
            if group.ref_logprobs is not None:
                loss_mb = loss_mb + grpo_cfg.kl_coef * _per_rollout_kl_tensor(
                    lp, group.ref_logprobs[r_idx]
                ) / n_completions
            logprobs_by_group[g_idx].append(float(lp.detach().item()))

        loss_mb.backward()

    grad_norm_sq = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            grad_norm_sq += float(p.grad.detach().norm(2).item()) ** 2
    grad_norm = math.sqrt(grad_norm_sq)

    optimizer.step()

    policy_losses: list[float] = []
    clip_fracs: list[float] = []
    kl_terms: list[float] = []
    all_advantages: list[float] = []
    n_rollouts = 0
    for group, logprobs, adv in zip(groups, logprobs_by_group, group_advantages):
        all_advantages.extend(adv)
        pg_loss, clip_frac = _clip_surrogate_scalar(
            logprobs,
            group.old_logprobs,
            adv,
            grpo_cfg.clip_ratio_low,
            grpo_cfg.clip_ratio_high,
        )
        policy_losses.append(pg_loss)
        clip_fracs.append(clip_frac)
        n_rollouts += len(group.rewards)
        if group.ref_logprobs is not None:
            kl_terms.append(_kl_penalty(logprobs, group.ref_logprobs))

    policy_loss = sum(policy_losses) / max(len(policy_losses), 1)
    kl_penalty = sum(kl_terms) / max(len(kl_terms), 1) if kl_terms else 0.0
    loss = policy_loss + grpo_cfg.kl_coef * kl_penalty
    clip_fraction = sum(clip_fracs) / max(len(clip_fracs), 1)
    mean_adv = sum(all_advantages) / max(len(all_advantages), 1)

    out = TrainStepOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl_penalty=kl_penalty,
        clip_fraction=clip_fraction,
        mean_advantage=mean_adv,
        n_prompts=len(groups),
        n_rollouts=n_rollouts,
    )
    out.grad_norm = grad_norm  # type: ignore[attr-defined]
    return out


def _objective_overrides(config: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = dict(config.get("objective_overrides") or {})
    inv = config.get("inverse_freq") or {}
    if "gamma" in inv:
        overrides.setdefault("inverse_gamma", float(inv["gamma"]))
    if "w_max" in inv:
        overrides.setdefault("w_max", float(inv["w_max"]))
    focal = config.get("f_grpo") or {}
    if "focal_gamma" in focal:
        overrides.setdefault("focal_gamma", float(focal["focal_gamma"]))
    return overrides


def _sample_rollouts(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problem: str,
    n: int,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> list[str]:
    seeds = [seed] if seed is not None else None
    return _sample_rollouts_batch(
        model,
        tokenizer,
        [problem],
        n,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seeds=seeds,
    )[0]


def _sample_rollouts_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problems: list[str],
    n: int,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seeds: list[int] | None,
    micro_batch_size: int = ROLLOUT_MICRO_BATCH_SIZE,
    allow_seeded_prompt_batching: bool = False,
    heartbeat_step: int | None = None,
    heartbeat_seconds: float = 60.0,
    heartbeat_completions: int = 32,
) -> list[list[str]]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        return batch_generate_rollouts(
            model,
            tokenizer,
            problems,
            n,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seeds=seeds,
            micro_batch_size=micro_batch_size,
            allow_seeded_prompt_batching=allow_seeded_prompt_batching,
            heartbeat_step=heartbeat_step,
            heartbeat_seconds=heartbeat_seconds,
            heartbeat_completions=heartbeat_completions,
        )
    finally:
        tokenizer.padding_side = prev_padding_side


def _build_step_groups(
    policy: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    batch: list[dict[str, str]],
    *,
    device: torch.device,
    n_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    step_seed: int,
    step_number: int,
    objective: ObjectiveName,
    objective_overrides: dict[str, Any],
    grpo_cfg: GRPOConfig,
    heartbeat_seconds: float,
    heartbeat_completions: int,
    rollout_micro_batch_size: int = ROLLOUT_MICRO_BATCH_SIZE,
    completion_logprob_micro_batch_size: int = COMPLETION_LOGPROB_MICRO_BATCH_SIZE,
    allow_seeded_prompt_batching: bool = False,
) -> tuple[list[PromptRolloutGroup], list[list[_RolloutRecord]], list[dict[str, Any]]]:
    groups: list[PromptRolloutGroup] = []
    specs_batch: list[list[_RolloutRecord]] = []
    rollout_diag_rows: list[dict[str, Any]] = []

    rollout_rows: list[dict[str, Any]] = []
    logprob_problems: list[str] = []
    logprob_completions: list[str] = []

    problems = [str(row["problem"]) for row in batch]
    rollout_seeds = [step_seed + i for i in range(len(batch))]
    texts_batch = _sample_rollouts_batch(
        policy,
        tokenizer,
        problems,
        n_rollouts,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seeds=rollout_seeds,
        micro_batch_size=rollout_micro_batch_size,
        allow_seeded_prompt_batching=allow_seeded_prompt_batching,
        heartbeat_step=step_number,
        heartbeat_seconds=heartbeat_seconds,
        heartbeat_completions=heartbeat_completions,
    )

    for row, texts in zip(batch, texts_batch):
        pid = str(row["prompt_id"])
        problem = str(row["problem"])
        gold = str(row["answer"])
        gold_key = canonicalize_answer(gold)
        per_prompt: list[dict[str, Any]] = []
        for text in texts:
            raw, parser_clean = extract_boxed_answer(text)
            cluster_key = canonicalize_answer(raw) if raw is not None else ""
            reward = 1.0 if (
                raw is not None
                and canonicalize_answer(raw) == gold_key
            ) else 0.0
            cid = cluster_id(cluster_key) if cluster_key else cluster_id(text)
            per_prompt.append(
                {
                    "prompt_id": pid,
                    "problem": problem,
                    "completion": text,
                    "reward": reward,
                    "cluster_id": cid,
                    "cluster_key": cluster_key,
                    "parser_clean": parser_clean,
                }
            )
            logprob_problems.append(problem)
            logprob_completions.append(text)
        rollout_rows.append(per_prompt)

    old_logprobs_all = _batched_scalar_mean_completion_logprobs(
        policy,
        tokenizer,
        logprob_problems,
        logprob_completions,
        device=device,
        micro_batch_size=completion_logprob_micro_batch_size,
    )
    ref_logprobs_all = _batched_scalar_mean_completion_logprobs(
        ref_model,
        tokenizer,
        logprob_problems,
        logprob_completions,
        device=device,
        micro_batch_size=completion_logprob_micro_batch_size,
    )

    lp_idx = 0
    for per_prompt in rollout_rows:
        rewards: list[float] = []
        cluster_ids: list[int] = []
        old_logprobs: list[float] = []
        ref_logprobs: list[float] = []
        specs: list[_RolloutRecord] = []
        pid = str(per_prompt[0]["prompt_id"]) if per_prompt else ""

        for rollout_idx, item in enumerate(per_prompt):
            old_lp = old_logprobs_all[lp_idx]
            ref_lp = ref_logprobs_all[lp_idx]
            lp_idx += 1
            rewards.append(float(item["reward"]))
            cluster_ids.append(int(item["cluster_id"]))
            old_logprobs.append(old_lp)
            ref_logprobs.append(ref_lp)
            specs.append(
                _RolloutRecord(
                    prompt_id=str(item["prompt_id"]),
                    problem=str(item["problem"]),
                    completion=str(item["completion"]),
                    reward=float(item["reward"]),
                    cluster_id=int(item["cluster_id"]),
                    old_logprob=old_lp,
                    ref_logprob=ref_lp,
                    rollout_idx=rollout_idx,
                    cluster_key=str(item["cluster_key"]),
                    parser_clean=bool(item["parser_clean"]),
                    completion_tokens=_completion_token_count(
                        tokenizer, str(item["completion"])
                    ),
                )
            )

        raw_advs = grpo_advantages(rewards)
        weighted = weighted_advantages(
            objective,
            rewards,
            cluster_ids,
            inverse_gamma=float(
                objective_overrides.get("inverse_gamma", grpo_cfg.inverse_gamma)
            ),
            w_max=float(objective_overrides.get("w_max", grpo_cfg.w_max)),
            focal_gamma=float(
                objective_overrides.get("focal_gamma", grpo_cfg.focal_gamma)
            ),
        )
        cluster_sizes = per_trajectory_cluster_counts(cluster_ids)
        correct_flags = [bool(r >= 0.5) for r in rewards]

        for spec, raw_adv, w_adv, csize in zip(specs, raw_advs, weighted, cluster_sizes):
            spec.raw_advantage = float(raw_adv)
            spec.weighted_advantage = float(w_adv)
            spec.cluster_size = int(csize)
            spec.is_minority_correct = _is_minority_correct_rollout(
                correct_flags, cluster_ids, spec.rollout_idx
            )
            rollout_diag_rows.append(
                {
                    "step": step_number,
                    "prompt_id": spec.prompt_id,
                    "rollout_idx": spec.rollout_idx,
                    "reward": spec.reward,
                    "raw_advantage": spec.raw_advantage,
                    "weighted_advantage": spec.weighted_advantage,
                    "cluster_id": spec.cluster_key,
                    "cluster_size": spec.cluster_size,
                    "is_minority_correct": spec.is_minority_correct,
                    "completion_tokens": spec.completion_tokens,
                    "parser_clean": spec.parser_clean,
                }
            )

        groups.append(
            PromptRolloutGroup(
                prompt_id=pid,
                rewards=rewards,
                cluster_ids=cluster_ids,
                logprobs=old_logprobs,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
            )
        )
        specs_batch.append(specs)

    return groups, specs_batch, rollout_diag_rows


def _commit_artifacts_volume() -> None:
    try:
        from pilot.infra.modal_app import artifacts_volume

        artifacts_volume.commit()
    except Exception as exc:
        logger.warning("artifacts_volume.commit failed: %s", exc)


def save_checkpoint(
    out_dir: Path,
    step: int,
    policy: torch.nn.Module,
    optimizer: AdamW,
    preds_offset_bytes: int,
    *,
    run_t0: float,
    price_per_sec: float,
) -> None:
    ckpt_dir = out_dir / f"checkpoint_step{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(
        {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
        ckpt_dir / "rng_state.pt",
    )
    wall_seconds = time.time() - run_t0
    state = {
        "step": step,
        "rng_state_path": str(ckpt_dir.relative_to(out_dir) / "rng_state.pt"),
        "optimizer_state_path": str(ckpt_dir.relative_to(out_dir) / "optimizer.pt"),
        "preds_offset_bytes": preds_offset_bytes,
        "wall_seconds_elapsed": wall_seconds,
        "usd_spent_estimate": _estimated_usd(wall_seconds, price_per_sec),
    }
    tmp = out_dir / "training_state.json.tmp"
    tmp.write_text(json.dumps(state))
    tmp.replace(out_dir / "training_state.json")


def load_checkpoint(
    out_dir: Path,
    state: dict[str, Any],
    policy: torch.nn.Module,
    optimizer: AdamW,
) -> None:
    _ = policy  # weights restored via from_pretrained at boot
    ckpt_dir = out_dir / f"checkpoint_step{state['step']}"
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", weights_only=False))
    rng = torch.load(ckpt_dir / "rng_state.pt", weights_only=False)
    torch.set_rng_state(rng["cpu"])
    torch.cuda.set_rng_state_all(rng["cuda"])


def _gc_checkpoints(out_dir: Path, *, retention_keep_last: int) -> None:
    ckpt_dirs = [
        d
        for d in out_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint_step")
    ]
    if not ckpt_dirs:
        return
    steps = sorted(int(d.name.replace("checkpoint_step", "")) for d in ckpt_dirs)
    keep_steps = {1}
    keep_steps.update(steps[-retention_keep_last:])
    for d in ckpt_dirs:
        step_num = int(d.name.replace("checkpoint_step", ""))
        if step_num not in keep_steps:
            shutil.rmtree(d)


def _should_commit_checkpoint(
    completed_step: int,
    max_steps: int,
    elapsed_s: float,
    ckpt_cfg: dict[str, Any],
) -> bool:
    if ckpt_cfg.get("always_save_first_step", True) and completed_step == 1:
        return True
    if ckpt_cfg.get("always_save_last_step", True) and completed_step == max_steps:
        return True
    if elapsed_s >= float(ckpt_cfg.get("target_interval_seconds", 3600)):
        return True
    if elapsed_s >= float(ckpt_cfg.get("min_interval_seconds", 1800)):
        return True
    return False


def _append_predictions(path: Path, specs_batch: list[list[_RolloutRecord]]) -> None:
    with path.open("a") as f:
        for specs in specs_batch:
            for spec in specs:
                f.write(
                    json.dumps(
                        {
                            "prompt_id": spec.prompt_id,
                            "parsed_answer": (
                                extract_boxed_answer(spec.completion)[0]
                                or extract_answer(spec.completion)
                            ),
                            "correct": bool(spec.reward),
                            "cluster_id": spec.cluster_id,
                            "completion": spec.completion,
                        }
                    )
                    + "\n"
                )


def _estimated_usd(gpu_seconds: float, price_per_sec: float) -> float:
    return gpu_seconds * price_per_sec


def run_grpo_training(
    config: dict[str, Any],
    *,
    repo_root: Path,
    artifacts_root: Path,
) -> Path:
    """GRPO / inverse_freq / f_grpo on GPU; writes artifacts under run_id."""
    run_id = str(config["run_id"])
    objective = str(config.get("objective", "grpo"))
    if objective not in ("grpo", "inverse_freq", "f_grpo"):
        raise ValueError(f"unsupported objective: {objective!r}")

    out_dir = artifact_dir(run_id, artifacts_root=artifacts_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_run_artifacts(config, artifacts_root=artifacts_root, repo_root=repo_root, out_dir=out_dir)

    log_path = out_dir / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if not any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(log_path)
        for h in root_logger.handlers
    ):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
            force=True,
        )

    shared_path = repo_root / "pilot" / "configs" / "shared_train.yaml"
    shared = yaml.safe_load(shared_path.read_text())

    seed = int(config.get("seed", shared.get("seed", 42)))
    max_steps = int(config.get("max_steps", shared.get("max_steps", 25)))
    n_rollouts = int(config.get("rollouts_per_prompt", shared.get("rollouts_per_prompt", 8)))
    batch_prompts = int(config.get("batch_prompts", shared.get("batch_prompts", 32)))
    lr = float(shared.get("learning_rate", 1e-6))
    model_id = str(shared["model_id"])
    max_new_tokens = min(
        int(config.get("max_new_tokens", shared.get("max_new_tokens", 1536))),
        1536,
    )
    temperature = float(shared.get("temperature", 1.0))
    top_p = float(shared.get("top_p", 0.95))
    rollout_micro_batch_size = int(
        config.get("rollout_micro_batch_size", ROLLOUT_MICRO_BATCH_SIZE)
    )
    completion_logprob_micro_batch_size = int(
        config.get(
            "completion_logprob_micro_batch_size",
            COMPLETION_LOGPROB_MICRO_BATCH_SIZE,
        )
    )
    allow_seeded_prompt_batching = bool(
        config.get(
            "allow_seeded_prompt_batching",
            shared.get("allow_seeded_prompt_batching", True),
        )
    )
    price_per_sec = float(shared.get("modal_price_per_sec", 0.000694))
    budget_cap_usd = float(config.get("budget_cap_usd", shared.get("budget_cap_usd", 12.0)))
    ckpt_cfg = dict(shared.get("checkpoint") or {})
    hb_cfg = dict(shared.get("heartbeat") or {})
    heartbeat_seconds = float(hb_cfg.get("seconds", 60))
    heartbeat_completions = int(hb_cfg.get("completions", 32))
    mechanism_tripwire = bool(
        config.get(
            "mechanism_tripwire_enabled",
            shared.get("mechanism_tripwire_enabled", True),
        )
    )

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_path = repo_root / str(shared.get("train_data", "pilot/data/dapo_slice_3k.jsonl"))
    train_data_sha256 = _file_sha256(data_path) if data_path.is_file() else ""
    prompts = _load_train_prompts(data_path, seed=seed)
    max_prompts = config.get("debug_max_prompts")
    if max_prompts is not None:
        prompts = prompts[: int(max_prompts)]
        logger.info("debug_max_prompts=%s", max_prompts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GRPO training requires CUDA")

    dtype = torch.bfloat16
    logger.info(
        "GRPO %s objective=%s steps=%s batch=%s N=%s model=%s seed=%s train_data_sha256=%s",
        run_id,
        objective,
        max_steps,
        batch_prompts,
        n_rollouts,
        model_id,
        seed,
        train_data_sha256[:16],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    state_path = out_dir / "training_state.json"
    resume_state: dict[str, Any] | None = None
    if state_path.exists():
        resume_state = json.loads(state_path.read_text())

    if resume_state is not None:
        ckpt_weights = out_dir / f"checkpoint_step{resume_state['step']}"
        logger.info("resuming from step %s (loading weights from %s)", resume_state["step"] + 1, ckpt_weights)
        policy = AutoModelForCausalLM.from_pretrained(
            str(ckpt_weights),
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(device)
    # B2: grad checkpointing off for perf. OOM fallback: uncomment next line and set
    # completion_logprob_micro_batch_size: 16 in shared_train.yaml.
    # policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    policy.train()
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    grpo_cfg = GRPOConfig(
        clip_ratio_low=float(shared.get("clip_ratio_low", 0.2)),
        clip_ratio_high=float(shared.get("clip_ratio_high", 0.28)),
        kl_coef=float(shared.get("kl_coef", 0.0)),
        rollouts_per_prompt=n_rollouts,
        inverse_gamma=float((config.get("inverse_freq") or {}).get("gamma", 1.0)),
        w_max=float((config.get("inverse_freq") or {}).get("w_max", 8.0)),
        focal_gamma=float((config.get("f_grpo") or {}).get("focal_gamma", 2.0)),
    )
    optimizer = AdamW(policy.parameters(), lr=lr, fused=True)
    obj_overrides = _objective_overrides(config)
    diag_path = out_dir / "step_diagnostics.jsonl"

    wandb_run = _init_wandb(
        run_name=run_id,
        run_config={**config, **{k: shared.get(k) for k in ("model_id", "seed")}},
        out_dir=out_dir,
    )

    if resume_state is None:
        _append_step_diagnostic(
            diag_path,
            {
                "step": 0,
                "phase": "run_start",
                "train_data_path": str(data_path),
                "train_data_sha256": train_data_sha256,
                "objective": objective,
                "seed": seed,
            },
        )
        _commit_artifacts_volume()

    pred_path = out_dir / "raw_predictions.jsonl"
    first_step_idx = 0
    if resume_state is not None:
        load_checkpoint(out_dir, resume_state, policy, optimizer)
        first_step_idx = int(resume_state["step"])
        offset = int(resume_state["preds_offset_bytes"])
        with pred_path.open("rb+") as f:
            f.truncate(offset)
        logger.info("resuming from step %s", first_step_idx + 1)
    else:
        pred_path.write_text("")

    t0 = time.time()
    run_mono_t0 = time.monotonic()
    last_commit_time = run_mono_t0
    budget_ctx = {
        "run_t0": t0,
        "run_mono_t0": run_mono_t0,
        "last_budget_check": run_mono_t0,
        "price_per_sec": price_per_sec,
        "budget_cap_usd": budget_cap_usd,
        "poll_interval_s": 60.0,
    }
    step_losses: list[float] = []
    step_rewards: list[float] = []
    steps_done = int(resume_state["step"]) if resume_state is not None else 0
    budget_stopped = False

    for step in range(first_step_idx, max_steps):
        step_t0 = time.time()
        elapsed = time.time() - t0
        if _estimated_usd(elapsed, price_per_sec) >= budget_cap_usd:
            logger.warning(
                "budget_cap_usd=%.2f reached at step %s (est $%.2f)",
                budget_cap_usd,
                step + 1,
                _estimated_usd(elapsed, price_per_sec),
            )
            if steps_done > 0:
                preds_offset_bytes = pred_path.stat().st_size if pred_path.exists() else 0
                save_checkpoint(
                    out_dir,
                    steps_done,
                    policy,
                    optimizer,
                    preds_offset_bytes,
                    run_t0=t0,
                    price_per_sec=price_per_sec,
                )
                _commit_artifacts_volume()
            budget_stopped = True
            break

        start = (step * batch_prompts) % len(prompts)
        batch: list[dict[str, str]] = []
        for j in range(batch_prompts):
            batch.append(prompts[(start + j) % len(prompts)])

        logger.info(
            "step %s/%s start: building groups (batch_prompts=%s, rollouts=%s, max_new_tokens=%s)",
            step + 1,
            max_steps,
            batch_prompts,
            n_rollouts,
            max_new_tokens,
        )
        phase_t0 = time.time()
        groups, specs_batch, rollout_diag_rows = _build_step_groups(
            policy,
            ref_model,
            tokenizer,
            batch,
            device=device,
            n_rollouts=n_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            step_seed=seed + step * 10007,
            step_number=step + 1,
            objective=objective,  # type: ignore[arg-type]
            objective_overrides=obj_overrides,
            grpo_cfg=grpo_cfg,
            heartbeat_seconds=heartbeat_seconds,
            heartbeat_completions=heartbeat_completions,
            rollout_micro_batch_size=rollout_micro_batch_size,
            completion_logprob_micro_batch_size=completion_logprob_micro_batch_size,
            allow_seeded_prompt_batching=allow_seeded_prompt_batching,
        )
        for row in rollout_diag_rows:
            _append_step_diagnostic(diag_path, row)
        phase_build_s = time.time() - phase_t0
        n_completions = sum(len(specs) for specs in specs_batch)
        logger.info(
            "step %s/%s groups ready: prompts=%s completions=%s build_seconds=%.1f",
            step + 1,
            max_steps,
            len(groups),
            n_completions,
            phase_build_s,
        )

        phase_t0 = time.time()
        try:
            step_out = _train_step_microbatch_backward(
                policy,
                tokenizer,
                groups,
                specs_batch,
                device=device,
                objective=objective,  # type: ignore[arg-type]
                grpo_cfg=grpo_cfg,
                objective_overrides=obj_overrides,
                completion_logprob_micro_batch_size=completion_logprob_micro_batch_size,
                optimizer=optimizer,
                budget_ctx=budget_ctx,
            )
        except BudgetCapHit:
            logger.warning(
                "budget_cap_usd=%.2f hit during train phase at step %s (est $%.2f)",
                budget_cap_usd,
                step + 1,
                _estimated_usd(time.time() - t0, price_per_sec),
            )
            preds_offset_bytes = pred_path.stat().st_size if pred_path.exists() else 0
            save_checkpoint(
                out_dir,
                step,
                policy,
                optimizer,
                preds_offset_bytes,
                run_t0=t0,
                price_per_sec=price_per_sec,
            )
            _commit_artifacts_volume()
            budget_stopped = True
            break

        phase_train_s = time.time() - phase_t0
        grad_norm = float(getattr(step_out, "grad_norm", 0.0))

        step_losses.append(float(step_out.loss))
        mean_r = sum(sum(g.rewards) for g in groups) / max(
            sum(len(g.rewards) for g in groups), 1
        )
        step_rewards.append(mean_r)
        steps_done += 1
        _append_predictions(pred_path, specs_batch)

        all_weighted: list[float] = []
        all_rewards: list[float] = []
        weighted_by_group: list[list[float]] = []
        for specs in specs_batch:
            w_adv = [s.weighted_advantage for s in specs]
            weighted_by_group.append(w_adv)
            all_weighted.extend(w_adv)
            all_rewards.extend(s.reward for s in specs)

        adv_var = 0.0
        if all_weighted:
            m_adv = sum(all_weighted) / len(all_weighted)
            adv_var = sum((a - m_adv) ** 2 for a in all_weighted) / len(all_weighted)

        parser_clean_rate = (
            sum(1 for r in rollout_diag_rows if r.get("parser_clean"))
            / max(len(rollout_diag_rows), 1)
        )
        minority_prompts = sum(
            1
            for g in groups
            if has_minority_correct_cluster(
                [bool(r >= 0.5) for r in g.rewards], g.cluster_ids
            )
        )
        clusters_per_prompt = [
            len({s.cluster_key for s in specs if s.cluster_key})
            for specs in specs_batch
        ]
        num_clusters_mean = sum(clusters_per_prompt) / max(len(clusters_per_prompt), 1)

        mechanism_signal = _mechanism_signal_per_variant(
            objective,  # type: ignore[arg-type]
            groups,
            weighted_by_group,
            inverse_gamma=float(obj_overrides.get("inverse_gamma", grpo_cfg.inverse_gamma)),
            w_max=float(obj_overrides.get("w_max", grpo_cfg.w_max)),
            focal_gamma=float(obj_overrides.get("focal_gamma", grpo_cfg.focal_gamma)),
        )

        completed_step = step + 1
        wall_step_s = time.time() - step_t0
        usd_est = _estimated_usd(time.time() - t0, price_per_sec)
        step_agg = {
            "step": completed_step,
            "phase": "step_complete",
            "wall_seconds": wall_step_s,
            "build_seconds": phase_build_s,
            "train_seconds": phase_train_s,
            "mean_reward": mean_r,
            "advantage_var": adv_var,
            "kl": float(step_out.kl_penalty),
            "clip_frac": float(step_out.clip_fraction),
            "advantage_l2": advantage_l2(all_weighted),
            "grad_norm": grad_norm,
            "parser_clean_rate": parser_clean_rate,
            "num_minority_correct_prompts": minority_prompts,
            "num_clusters_mean": num_clusters_mean,
            "usd_spent_estimate": usd_est,
            "mechanism_signal_per_variant": mechanism_signal,
        }
        _append_step_diagnostic(diag_path, step_agg)
        _commit_artifacts_volume()

        if mechanism_tripwire and completed_step == 1 and mechanism_signal < 0.9:
            logger.error(
                "mechanism tripwire: signal=%.4f < 0.9 at step 1 for %s (%s)",
                mechanism_signal,
                run_id,
                objective,
            )
            raise RuntimeError(
                f"mechanism_signal_per_variant={mechanism_signal:.4f} < 0.9 at step 1"
            )

        if wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "step": completed_step,
                    **{
                        k: v
                        for k, v in step_agg.items()
                        if k not in ("step", "phase")
                    },
                },
                step=completed_step,
            )
            wandb.log(
                {
                    "reward_hist": wandb.Histogram(all_rewards),
                    "weighted_advantage_hist": wandb.Histogram(all_weighted),
                },
                step=completed_step,
            )
            if completed_step % 5 == 0:
                sample_rows: list[list[str]] = []
                rng = random.Random(seed + completed_step)
                for specs in rng.sample(specs_batch, min(4, len(specs_batch))):
                    spec = rng.choice(specs)
                    sample_rows.append(
                        [
                            spec.prompt_id,
                            str(spec.rollout_idx),
                            spec.completion[:2000],
                            str(spec.reward),
                            str(spec.weighted_advantage),
                        ]
                    )
                wandb.log(
                    {
                        "sample_completions": wandb.Table(
                            columns=[
                                "prompt_id",
                                "rollout_idx",
                                "completion",
                                "reward",
                                "weighted_advantage",
                            ],
                            data=sample_rows,
                        )
                    },
                    step=completed_step,
                )

        logger.info(
            "step %s/%s done: loss=%.4f policy=%.4f kl=%.4f mean_reward=%.3f clip=%.3f "
            "train_seconds=%.1f total_step_seconds=%.1f mechanism=%.3f parser_clean=%.3f",
            step + 1,
            max_steps,
            step_out.loss,
            step_out.policy_loss,
            step_out.kl_penalty,
            mean_r,
            step_out.clip_fraction,
            phase_train_s,
            time.time() - step_t0,
            mechanism_signal,
            parser_clean_rate,
        )

        preds_offset_bytes = pred_path.stat().st_size
        elapsed_commit = time.monotonic() - last_commit_time
        if _should_commit_checkpoint(completed_step, max_steps, elapsed_commit, ckpt_cfg):
            save_checkpoint(
                out_dir,
                completed_step,
                policy,
                optimizer,
                preds_offset_bytes,
                run_t0=t0,
                price_per_sec=price_per_sec,
            )
            _commit_artifacts_volume()
            _gc_checkpoints(out_dir, retention_keep_last=int(ckpt_cfg.get("retention_keep_last", 2)))
            last_commit_time = time.monotonic()

    ckpt_dir: Path | None = None
    if not budget_stopped:
        ckpt_dir = out_dir / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        policy.eval()
        policy.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)

    global _trained_rollout_engine
    if not budget_stopped and ckpt_dir is not None:
        _trained_rollout_engine = HFRolloutEngine.from_checkpoint(
            ckpt_dir,
            RolloutEngineConfig(
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                torch_dtype=dtype,
                micro_batch_size=rollout_micro_batch_size,
                allow_seeded_prompt_batching=allow_seeded_prompt_batching,
            ),
        )
        logger.info("saved checkpoint to %s", ckpt_dir)

    gpu_seconds = time.time() - t0
    if not config.get("defer_cost_record"):
        record_cost(
            out_dir,
            gpu_seconds=gpu_seconds,
            price_per_sec=price_per_sec,
            run_id=run_id,
        )

    metrics = {
        "run_id": run_id,
        "objective": objective,
        "seed": seed,
        "steps_completed": steps_done,
        "max_steps": max_steps,
        "final_loss": step_losses[-1] if step_losses else None,
        "mean_train_reward": sum(step_rewards) / max(len(step_rewards), 1),
        "git_sha": git_sha(repo_root=repo_root),
        "train_data_path": str(data_path),
        "train_data_sha256": train_data_sha256,
    }
    write_metrics(out_dir / "metrics_train.json", metrics)
    (out_dir / "train_data_pin.json").write_text(
        json.dumps(
            {
                "train_data_path": str(data_path),
                "train_data_sha256": train_data_sha256,
            }
        )
    )
    pin_path = out_dir / "metrics.json"
    if pin_path.exists():
        try:
            final_metrics = json.loads(pin_path.read_text())
        except json.JSONDecodeError:
            final_metrics = {}
    else:
        final_metrics = {}
    final_metrics["train_data_sha256"] = train_data_sha256
    final_metrics["train_data_path"] = str(data_path)
    write_metrics(pin_path, final_metrics)

    if wandb_run is not None:
        import wandb

        wandb.finish()
    _sync_wandb_offline(out_dir)
    logger.info(
        "GRPO done: steps=%s mean_reward=%.3f gpu_seconds=%.1f",
        steps_done,
        metrics["mean_train_reward"],
        gpu_seconds,
    )
    return out_dir


def validate_seeded_prompt_batching_parity(
    *,
    repo_root: Path,
    n_prompts: int = 32,
    seed: int = 42,
    objective: ObjectiveName = "grpo",
    rtol_mean_reward: float = 0.01,
    rtol_advantage_l2: float = 0.05,
) -> dict[str, Any]:
    """
    B1 smoke parity: batched vs sequential rollout paths on a fixed prompt slice.

    Run from repo root (CUDA required):
      python -c "
      from pathlib import Path
      from pilot.train.hf_grpo_train import validate_seeded_prompt_batching_parity
      r = validate_seeded_prompt_batching_parity(repo_root=Path('.'))
      print(r)
      assert r['passed'], r
      "
    """
    shared_path = repo_root / "pilot" / "configs" / "shared_train.yaml"
    shared = yaml.safe_load(shared_path.read_text())
    model_id = str(shared["model_id"])
    n_rollouts = int(shared.get("rollouts_per_prompt", 8))
    max_new_tokens = int(shared.get("max_new_tokens", 2048))
    temperature = float(shared.get("temperature", 1.0))
    top_p = float(shared.get("top_p", 0.95))
    rollout_micro_batch_size = int(shared.get("rollout_micro_batch_size", ROLLOUT_MICRO_BATCH_SIZE))
    completion_logprob_micro_batch_size = int(
        shared.get("completion_logprob_micro_batch_size", COMPLETION_LOGPROB_MICRO_BATCH_SIZE)
    )
    data_path = repo_root / str(shared.get("train_data", "pilot/data/dapo_slice_3k.jsonl"))
    prompts = _load_train_prompts(data_path, seed=seed)[:n_prompts]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("parity validation requires CUDA")

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    policy.config.use_cache = False
    policy.eval()
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    step_seed = seed + 10007
    grpo_cfg = GRPOConfig()
    hb_cfg = dict(shared.get("heartbeat") or {})

    def _metrics(allow_batched: bool) -> tuple[float, float]:
        groups, _, _ = _build_step_groups(
            policy,
            ref_model,
            tokenizer,
            prompts,
            device=device,
            n_rollouts=n_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            step_seed=step_seed,
            step_number=1,
            objective=objective,
            objective_overrides={},
            grpo_cfg=grpo_cfg,
            heartbeat_seconds=float(hb_cfg.get("seconds", 60)),
            heartbeat_completions=int(hb_cfg.get("completions", 32)),
            rollout_micro_batch_size=rollout_micro_batch_size,
            completion_logprob_micro_batch_size=completion_logprob_micro_batch_size,
            allow_seeded_prompt_batching=allow_batched,
        )
        rewards = [r for g in groups for r in g.rewards]
        mean_reward = sum(rewards) / max(len(rewards), 1)
        advs = [
            a
            for g in groups
            for a in weighted_advantages(objective, g.rewards, g.cluster_ids)
        ]
        return mean_reward, advantage_l2(advs)

    mean_serial, l2_serial = _metrics(False)
    mean_batched, l2_batched = _metrics(True)
    delta_mean = abs(mean_batched - mean_serial)
    rel_l2 = abs(l2_batched - l2_serial) / max(l2_serial, 1e-12)
    passed = delta_mean < rtol_mean_reward and rel_l2 < rtol_advantage_l2
    return {
        "passed": passed,
        "mean_reward_serial": mean_serial,
        "mean_reward_batched": mean_batched,
        "delta_mean_reward": delta_mean,
        "advantage_l2_serial": l2_serial,
        "advantage_l2_batched": l2_batched,
        "relative_advantage_l2_delta": rel_l2,
        "thresholds": {
            "delta_mean_reward": rtol_mean_reward,
            "relative_advantage_l2": rtol_advantage_l2,
        },
    }
