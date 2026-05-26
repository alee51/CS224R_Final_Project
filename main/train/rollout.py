"""vLLM rollout engine with per-token logprobs and HF weight sync."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from train.weight_sync import SyncStats, sync_hf_to_vllm

logger = logging.getLogger(__name__)


@dataclass
class RolloutCfg:
    model: str
    max_prompt_length: int = 1024
    max_response_length: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.45
    max_model_len: int = 5120
    max_num_seqs: int = 128
    enable_prefix_caching: bool = True
    logprobs: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RolloutCfg:
        return cls(
            model=str(d["model"]),
            max_prompt_length=int(d.get("max_prompt_length", 1024)),
            max_response_length=int(d.get("max_response_length", 4096)),
            temperature=float(d.get("temperature", 1.0)),
            top_p=float(d.get("top_p", 1.0)),
            gpu_memory_utilization=float(d.get("gpu_memory_utilization", 0.45)),
            max_model_len=int(d.get("max_model_len", 5120)),
            max_num_seqs=int(d.get("max_num_seqs", 128)),
            enable_prefix_caching=bool(d.get("enable_prefix_caching", True)),
            logprobs=int(d.get("logprobs", 1)),
        )


@dataclass
class RolloutResult:
    prompt_idx: int
    rollout_idx: int
    completion_ids: list[int]
    completion_text: str
    prompt_ids: list[int]
    old_logprobs: list[float]
    finish_reason: str


def _extract_old_logprobs(completion_out: Any) -> list[float]:
    """Chosen-token logprob per generated token from vLLM logprobs=1 output."""
    token_ids = list(completion_out.token_ids)
    logprobs_list = completion_out.logprobs
    if not logprobs_list:
        return [0.0] * len(token_ids)

    old: list[float] = []
    for tid, lp_dict in zip(token_ids, logprobs_list):
        if lp_dict is None:
            old.append(0.0)
            continue
        entry = lp_dict.get(tid) if isinstance(lp_dict, dict) else None
        if entry is None:
            old.append(0.0)
        else:
            old.append(float(getattr(entry, "logprob", entry)))
    return old


class RolloutEngine:
    def __init__(self, cfg: RolloutCfg) -> None:
        from vllm import LLM

        self.cfg = cfg
        logger.info("Initializing vLLM RolloutEngine for %s", cfg.model)
        self._llm = LLM(
            model=cfg.model,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_seqs=cfg.max_num_seqs,
            enable_prefix_caching=cfg.enable_prefix_caching,
        )

    @property
    def llm(self) -> Any:
        return self._llm

    def generate(
        self,
        prompts: list[str],
        n_per_prompt: int,
        seeds: list[int],
    ) -> list[RolloutResult]:
        from vllm import SamplingParams

        if len(seeds) != len(prompts) * n_per_prompt:
            raise ValueError(
                f"seeds length {len(seeds)} != prompts*n "
                f"({len(prompts)}*{n_per_prompt})"
            )

        expanded_prompts: list[str] = []
        meta: list[tuple[int, int]] = []
        seed_list: list[int] = []
        for p_idx, prompt in enumerate(prompts):
            for r_idx in range(n_per_prompt):
                expanded_prompts.append(prompt)
                meta.append((p_idx, r_idx))
                seed_list.append(
                    seeds[p_idx * n_per_prompt + r_idx]
                )

        params_list = [
            SamplingParams(
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                max_tokens=self.cfg.max_response_length,
                seed=seed,
                n=1,
                logprobs=self.cfg.logprobs,
            )
            for seed in seed_list
        ]

        outputs = self._llm.generate(expanded_prompts, params_list)
        results: list[RolloutResult] = []
        for (p_idx, r_idx), out, seed in zip(meta, outputs, seed_list):
            comp = out.outputs[0]
            results.append(
                RolloutResult(
                    prompt_idx=p_idx,
                    rollout_idx=r_idx,
                    completion_ids=list(comp.token_ids),
                    completion_text=comp.text,
                    prompt_ids=list(out.prompt_token_ids),
                    old_logprobs=_extract_old_logprobs(comp),
                    finish_reason=str(comp.finish_reason),
                )
            )
        return results

    def update_weights(self, hf_model: Any) -> SyncStats:
        return sync_hf_to_vllm(hf_model, self._llm)

    def shutdown(self) -> None:
        logger.info("RolloutEngine shutdown (vLLM engine released with process exit)")
