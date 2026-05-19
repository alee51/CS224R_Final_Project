"""GPU GRPO training with HuggingFace Qwen (Run1–Run3)."""

from __future__ import annotations

import copy
import json
import logging
import random
import time
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
from pilot.train.answer_parse import extract_answer, is_correct
from pilot.train.canonicalize import cluster_id
from pilot.train.grpo_trainer import (
    GRPOConfig,
    PromptRolloutGroup,
    TrainStepOutput,
    _clip_surrogate,
    _kl_penalty,
)
from pilot.train.objectives import ObjectiveName, weighted_advantages
from pilot.train.rollout_engine import (
    HFRolloutEngine,
    PROMPT_TEMPLATE,
    ROLLOUT_MICRO_BATCH_SIZE,
    RolloutEngineConfig,
    batch_generate_rollouts,
)

logger = logging.getLogger(__name__)

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


def _batched_mean_completion_logprobs(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problems: list[str],
    completions: list[str],
    *,
    device: torch.device,
    micro_batch_size: int = COMPLETION_LOGPROB_MICRO_BATCH_SIZE,
) -> list[torch.Tensor]:
    """Differentiable batched completion logprobs in micro-batches."""
    if len(problems) != len(completions):
        raise ValueError("problems and completions length mismatch")
    if not problems:
        return []

    results: list[torch.Tensor] = []
    for start in range(0, len(problems), micro_batch_size):
        chunk_p = problems[start : start + micro_batch_size]
        chunk_c = completions[start : start + micro_batch_size]
        results.extend(
            _micro_batch_mean_completion_logprobs(
                model,
                tokenizer,
                chunk_p,
                chunk_c,
                device=device,
            )
        )
    return results


class HFPolicyModel:
    """Policy forward pass for GRPOTrainer + differentiable loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        rollout_specs: list[list[_RolloutRecord]],
        *,
        device: torch.device,
        completion_logprob_micro_batch_size: int = COMPLETION_LOGPROB_MICRO_BATCH_SIZE,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.rollout_specs = rollout_specs
        self.device = device
        self.completion_logprob_micro_batch_size = completion_logprob_micro_batch_size
        self._logprob_tensors: list[list[torch.Tensor]] = []

    def logprobs_for_rollouts(self, groups: list[PromptRolloutGroup]) -> list[list[float]]:
        flat_problems: list[str] = []
        flat_completions: list[str] = []
        row_lengths: list[int] = []
        for specs in self.rollout_specs:
            row_lengths.append(len(specs))
            for spec in specs:
                flat_problems.append(spec.problem)
                flat_completions.append(spec.completion)

        flat_logprobs = _batched_mean_completion_logprobs(
            self.model,
            self.tokenizer,
            flat_problems,
            flat_completions,
            device=self.device,
            micro_batch_size=self.completion_logprob_micro_batch_size,
        )

        self._logprob_tensors = []
        out: list[list[float]] = []
        idx = 0
        for length in row_lengths:
            row = flat_logprobs[idx : idx + length]
            idx += length
            self._logprob_tensors.append(row)
            out.append([float(lp.detach().item()) for lp in row])
        return out


def _clip_surrogate_tensor(
    logprobs: list[torch.Tensor],
    old_logprobs: list[float],
    advantages: list[float],
    clip_eps: float,
) -> torch.Tensor:
    if not logprobs:
        raise ValueError("logprobs must be non-empty")
    losses: list[torch.Tensor] = []
    for lp, old_lp, adv in zip(logprobs, old_logprobs, advantages):
        ratio = torch.exp(lp - old_lp)
        unclipped = ratio * adv
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        losses.append(-torch.minimum(unclipped, clipped_ratio))
    return torch.stack(losses).mean()


def _kl_penalty_tensor(
    logprobs: list[torch.Tensor],
    ref_logprobs: list[float],
) -> torch.Tensor:
    if not logprobs:
        return torch.zeros((), device=logprobs[0].device)
    terms = [lp - ref for lp, ref in zip(logprobs, ref_logprobs)]
    return torch.stack(terms).mean()


def _per_rollout_policy_loss_tensor(
    logprob: torch.Tensor,
    old_logprob: float,
    advantage: float,
    clip_eps: float,
) -> torch.Tensor:
    old_t = torch.tensor(old_logprob, device=logprob.device, dtype=logprob.dtype)
    ratio = torch.exp(logprob - old_t)
    adv_t = torch.tensor(advantage, device=logprob.device, dtype=logprob.dtype)
    unclipped = ratio * adv_t
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
    return -torch.minimum(unclipped, clipped_ratio)


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
            loss_mb = loss_mb + _per_rollout_policy_loss_tensor(
                lp, group.old_logprobs[r_idx], adv, grpo_cfg.clip_eps
            ) / n_completions
            if group.ref_logprobs is not None:
                loss_mb = loss_mb + grpo_cfg.kl_coef * _per_rollout_kl_tensor(
                    lp, group.ref_logprobs[r_idx]
                ) / n_completions
            logprobs_by_group[g_idx].append(float(lp.detach().item()))

        loss_mb.backward()

    optimizer.step()

    policy_losses: list[float] = []
    clip_fracs: list[float] = []
    kl_terms: list[float] = []
    all_advantages: list[float] = []
    n_rollouts = 0
    for group, logprobs, adv in zip(groups, logprobs_by_group, group_advantages):
        all_advantages.extend(adv)
        pg_loss, clip_frac = _clip_surrogate(
            logprobs, group.old_logprobs, adv, grpo_cfg.clip_eps
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

    return TrainStepOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl_penalty=kl_penalty,
        clip_fraction=clip_fraction,
        mean_advantage=mean_adv,
        n_prompts=len(groups),
        n_rollouts=n_rollouts,
    )


def _differentiable_loss(
    groups: list[PromptRolloutGroup],
    logprob_tensors: list[list[torch.Tensor]],
    objective: ObjectiveName,
    cfg: GRPOConfig,
    objective_overrides: dict[str, Any],
) -> torch.Tensor:
    policy_losses: list[torch.Tensor] = []
    kl_terms: list[torch.Tensor] = []
    device = logprob_tensors[0][0].device if logprob_tensors and logprob_tensors[0] else "cpu"

    for group, logprobs in zip(groups, logprob_tensors):
        adv = weighted_advantages(
            objective,
            group.rewards,
            group.cluster_ids,
            inverse_gamma=objective_overrides.get("inverse_gamma", cfg.inverse_gamma),
            w_max=objective_overrides.get("w_max", cfg.w_max),
            focal_gamma=objective_overrides.get("focal_gamma", cfg.focal_gamma),
        )
        policy_losses.append(
            _clip_surrogate_tensor(logprobs, group.old_logprobs, adv, cfg.clip_eps)
        )
        if group.ref_logprobs is not None:
            kl_terms.append(_kl_penalty_tensor(logprobs, group.ref_logprobs))

    policy_loss = torch.stack(policy_losses).mean() if policy_losses else torch.zeros((), device=device)
    kl_penalty = torch.stack(kl_terms).mean() if kl_terms else torch.zeros((), device=device)
    return policy_loss + cfg.kl_coef * kl_penalty


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
    rollout_micro_batch_size: int = ROLLOUT_MICRO_BATCH_SIZE,
    completion_logprob_micro_batch_size: int = COMPLETION_LOGPROB_MICRO_BATCH_SIZE,
    allow_seeded_prompt_batching: bool = False,
) -> tuple[list[PromptRolloutGroup], list[list[_RolloutRecord]]]:
    groups: list[PromptRolloutGroup] = []
    specs_batch: list[list[_RolloutRecord]] = []

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
    )

    for row, texts in zip(batch, texts_batch):
        pid = str(row["prompt_id"])
        problem = str(row["problem"])
        gold = str(row["answer"])
        per_prompt: list[dict[str, Any]] = []
        for text in texts:
            parsed = extract_answer(text)
            reward = 1.0 if is_correct(text, gold) else 0.0
            cid = cluster_id(parsed)
            per_prompt.append(
                {
                    "prompt_id": pid,
                    "problem": problem,
                    "completion": text,
                    "reward": reward,
                    "cluster_id": cid,
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

        for item in per_prompt:
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
                )
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

    return groups, specs_batch


def _append_predictions(path: Path, specs_batch: list[list[_RolloutRecord]]) -> None:
    with path.open("a") as f:
        for specs in specs_batch:
            for spec in specs:
                f.write(
                    json.dumps(
                        {
                            "prompt_id": spec.prompt_id,
                            "parsed_answer": extract_answer(spec.completion),
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
    max_steps = int(shared.get("max_steps", 100))
    n_rollouts = int(config.get("rollouts_per_prompt", shared.get("rollouts_per_prompt", 8)))
    batch_prompts = int(config.get("batch_prompts", shared.get("batch_prompts", 32)))
    lr = float(shared.get("learning_rate", 1e-6))
    model_id = str(shared["model_id"])
    max_new_tokens = int(config.get("max_new_tokens", shared.get("max_new_tokens", 2048)))
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
        config.get("allow_seeded_prompt_batching", False)
    )
    price_per_sec = float(shared.get("modal_price_per_sec", 0.000694))
    budget_cap_usd = float(config.get("budget_cap_usd", 12.0))

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_path = repo_root / str(shared.get("train_data", "pilot/data/dapo_slice_3k.jsonl"))
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
        "GRPO %s objective=%s steps=%s batch=%s N=%s model=%s seed=%s",
        run_id,
        objective,
        max_steps,
        batch_prompts,
        n_rollouts,
        model_id,
        seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    policy.gradient_checkpointing_enable()
    policy.config.use_cache = False
    policy.train()
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    grpo_cfg = GRPOConfig(
        clip_eps=float(shared.get("clip_eps", 0.2)),
        kl_coef=float(shared.get("kl_coef", 0.001)),
        rollouts_per_prompt=n_rollouts,
        inverse_gamma=float((config.get("inverse_freq") or {}).get("gamma", 1.0)),
        w_max=float((config.get("inverse_freq") or {}).get("w_max", 8.0)),
        focal_gamma=float((config.get("f_grpo") or {}).get("focal_gamma", 2.0)),
    )
    optimizer = AdamW(policy.parameters(), lr=lr)
    obj_overrides = _objective_overrides(config)

    pred_path = out_dir / "raw_predictions.jsonl"
    pred_path.write_text("")

    t0 = time.time()
    step_losses: list[float] = []
    step_rewards: list[float] = []
    steps_done = 0

    for step in range(max_steps):
        step_t0 = time.time()
        elapsed = time.time() - t0
        if _estimated_usd(elapsed, price_per_sec) >= budget_cap_usd:
            logger.warning(
                "budget_cap_usd=%.2f reached at step %s (est $%.2f)",
                budget_cap_usd,
                step,
                _estimated_usd(elapsed, price_per_sec),
            )
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
        groups, specs_batch = _build_step_groups(
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
            rollout_micro_batch_size=rollout_micro_batch_size,
            completion_logprob_micro_batch_size=completion_logprob_micro_batch_size,
            allow_seeded_prompt_batching=allow_seeded_prompt_batching,
        )
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
        )
        phase_train_s = time.time() - phase_t0

        step_losses.append(float(step_out.loss))
        mean_r = sum(sum(g.rewards) for g in groups) / max(
            sum(len(g.rewards) for g in groups), 1
        )
        step_rewards.append(mean_r)
        steps_done += 1
        _append_predictions(pred_path, specs_batch)

        logger.info(
            "step %s/%s done: loss=%.4f policy=%.4f kl=%.4f mean_reward=%.3f clip=%.3f "
            "train_seconds=%.1f total_step_seconds=%.1f",
            step + 1,
            max_steps,
            step_out.loss,
            step_out.policy_loss,
            step_out.kl_penalty,
            mean_r,
            step_out.clip_fraction,
            phase_train_s,
            time.time() - step_t0,
        )

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.eval()
    policy.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    global _trained_rollout_engine
    _trained_rollout_engine = HFRolloutEngine.from_checkpoint(
        ckpt_dir,
        RolloutEngineConfig(
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            dtype=dtype,
            micro_batch_size=rollout_micro_batch_size,
            allow_seeded_prompt_batching=allow_seeded_prompt_batching,
        ),
    )

    gpu_seconds = time.time() - t0
    if not config.get("defer_cost_record"):
        record_cost(
            out_dir,
            gpu_seconds=gpu_seconds,
            price_per_sec=price_per_sec,
            run_id=run_id,
        )

    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy.eval()
    policy.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    logger.info("saved checkpoint to %s", ckpt_dir)

    metrics = {
        "run_id": run_id,
        "objective": objective,
        "seed": seed,
        "steps_completed": steps_done,
        "max_steps": max_steps,
        "final_loss": step_losses[-1] if step_losses else None,
        "mean_train_reward": sum(step_rewards) / max(len(step_rewards), 1),
        "git_sha": git_sha(repo_root=repo_root),
    }
    write_metrics(out_dir / "metrics_train.json", metrics)
    logger.info(
        "GRPO done: steps=%s mean_reward=%.3f gpu_seconds=%.1f",
        steps_done,
        metrics["mean_train_reward"],
        gpu_seconds,
    )
    return out_dir
