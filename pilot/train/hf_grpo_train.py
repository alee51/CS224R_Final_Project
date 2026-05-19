"""GPU GRPO training with HuggingFace Qwen (Run1–Run3)."""

from __future__ import annotations

import copy
import json
import logging
import math
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
from pilot.train.grpo_trainer import GRPOConfig, GRPOTrainer, PromptRolloutGroup
from pilot.train.objectives import ObjectiveName, weighted_advantages
from pilot.train.rollout_engine import PROMPT_TEMPLATE
from pilot.train.run_proxy import _load_prompt_slice

logger = logging.getLogger(__name__)

GRPO_RUN_IDS = frozenset(
    {"run1_grpo", "run1b_grpo", "run2_inverse_freq", "run3_f_grpo"}
)


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


class HFPolicyModel:
    """Policy forward pass for GRPOTrainer + differentiable loss."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        rollout_specs: list[list[_RolloutRecord]],
        *,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.rollout_specs = rollout_specs
        self.device = device
        self._logprob_tensors: list[list[torch.Tensor]] = []

    def logprobs_for_rollouts(self, groups: list[PromptRolloutGroup]) -> list[list[float]]:
        self._logprob_tensors = []
        out: list[list[float]] = []
        for specs in self.rollout_specs:
            row: list[torch.Tensor] = []
            floats: list[float] = []
            for spec in specs:
                input_ids, prompt_len = _encode_prompt_completion(
                    self.tokenizer, spec.problem, spec.completion
                )
                lp = _mean_completion_logprob(
                    self.model, input_ids, prompt_len, device=self.device
                )
                row.append(lp)
                floats.append(float(lp.detach().item()))
            self._logprob_tensors.append(row)
            out.append(floats)
        return out


def _clip_surrogate_tensor(
    logprobs: list[torch.Tensor],
    old_logprobs: list[float],
    advantages: list[float],
    clip_eps: float,
) -> torch.Tensor:
    if not logprobs:
        return torch.zeros((), device=logprobs[0].device if logprobs else "cpu")
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
    overrides: dict[str, Any] = {}
    inv = config.get("inverse_freq") or {}
    if "gamma" in inv:
        overrides["inverse_gamma"] = float(inv["gamma"])
    if "w_max" in inv:
        overrides["w_max"] = float(inv["w_max"])
    focal = config.get("f_grpo") or {}
    if "focal_gamma" in focal:
        overrides["focal_gamma"] = float(focal["focal_gamma"])
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
    prompt = _prompt_text(problem)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kw: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "num_return_sequences": n,
        "pad_token_id": tokenizer.pad_token_id,
    }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        out = model.generate(**inputs, **gen_kw)
    if was_training:
        model.train()
    prompt_len = inputs["input_ids"].shape[1]
    texts: list[str] = []
    for seq in out:
        new_tokens = seq[prompt_len:]
        texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return texts


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
) -> tuple[list[PromptRolloutGroup], list[list[_RolloutRecord]]]:
    groups: list[PromptRolloutGroup] = []
    specs_batch: list[list[_RolloutRecord]] = []

    for i, row in enumerate(batch):
        pid = str(row["prompt_id"])
        problem = str(row["problem"])
        gold = str(row["answer"])
        texts = _sample_rollouts(
            policy,
            tokenizer,
            problem,
            n_rollouts,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=step_seed + i,
        )
        rewards: list[float] = []
        cluster_ids: list[int] = []
        old_logprobs: list[float] = []
        ref_logprobs: list[float] = []
        specs: list[_RolloutRecord] = []

        for text in texts:
            parsed = extract_answer(text)
            reward = 1.0 if is_correct(text, gold) else 0.0
            cid = cluster_id(parsed)
            old_lp = _scalar_mean_completion_logprob(
                policy, tokenizer, problem, text, device=device
            )
            ref_lp = _scalar_mean_completion_logprob(
                ref_model, tokenizer, problem, text, device=device
            )
            rewards.append(reward)
            cluster_ids.append(cid)
            old_logprobs.append(old_lp)
            ref_logprobs.append(ref_lp)
            specs.append(
                _RolloutRecord(
                    prompt_id=pid,
                    problem=problem,
                    completion=text,
                    reward=reward,
                    cluster_id=cid,
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
    if logging.getLogger().handlers:
        for h in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(h)
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
    n_rollouts = int(shared.get("rollouts_per_prompt", 8))
    batch_prompts = int(shared.get("batch_prompts", 32))
    lr = float(shared.get("learning_rate", 1e-6))
    model_id = str(shared["model_id"])
    max_new_tokens = min(int(shared.get("max_new_tokens", 2048)), 1024)
    temperature = float(shared.get("temperature", 1.0))
    top_p = float(shared.get("top_p", 0.95))
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
    trainer = GRPOTrainer(cfg=grpo_cfg)
    optimizer = AdamW(policy.parameters(), lr=lr)
    obj_overrides = _objective_overrides(config)

    pred_path = out_dir / "raw_predictions.jsonl"
    pred_path.write_text("")

    t0 = time.time()
    step_losses: list[float] = []
    step_rewards: list[float] = []
    steps_done = 0

    for step in range(max_steps):
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
        )

        policy_model = HFPolicyModel(policy, tokenizer, specs_batch, device=device)
        trainer.model = policy_model
        step_out = trainer.train_step(
            groups,
            objective,  # type: ignore[arg-type]
            objective_overrides=obj_overrides,
        )
        loss_t = _differentiable_loss(
            groups,
            policy_model._logprob_tensors,
            objective,  # type: ignore[arg-type]
            grpo_cfg,
            obj_overrides,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_t.backward()
        optimizer.step()

        step_losses.append(float(step_out.loss))
        mean_r = sum(sum(g.rewards) for g in groups) / max(
            sum(len(g.rewards) for g in groups), 1
        )
        step_rewards.append(mean_r)
        steps_done += 1
        _append_predictions(pred_path, specs_batch)

        if (step + 1) % 5 == 0 or step == 0:
            logger.info(
                "step %s/%s loss=%.4f policy=%.4f kl=%.4f mean_reward=%.3f clip=%.3f",
                step + 1,
                max_steps,
                step_out.loss,
                step_out.policy_loss,
                step_out.kl_penalty,
                mean_r,
                step_out.clip_fraction,
            )

    gpu_seconds = time.time() - t0
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
    }
    write_metrics(out_dir / "metrics.json", metrics)
    logger.info(
        "GRPO done: steps=%s mean_reward=%.3f gpu_seconds=%.1f",
        steps_done,
        metrics["mean_train_reward"],
        gpu_seconds,
    )
    return out_dir
