"""vLLM rollout engine with per-token logprobs and HF weight sync."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from train.ablation import prepare_pytorch_alloc_for_vllm_sleep, vllm_sleep_enabled
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
        llm_kwargs: dict[str, Any] = dict(
            model=cfg.model,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_num_seqs=cfg.max_num_seqs,
            enable_prefix_caching=cfg.enable_prefix_caching,
        )
        self._use_sleep = vllm_sleep_enabled()
        self._sleep_available = False
        if self._use_sleep:
            prepare_pytorch_alloc_for_vllm_sleep()
            try:
                self._llm = LLM(**llm_kwargs, enable_sleep_mode=True)
                self._sleep_available = hasattr(self._llm, "sleep")
            except TypeError:
                self._llm = LLM(**llm_kwargs)
                self._sleep_available = hasattr(self._llm, "sleep")
            if not self._sleep_available:
                logger.warning("CS224R_VLLM_SLEEP=1 but vLLM has no sleep(); disabling")
                self._use_sleep = False
        else:
            self._llm = LLM(**llm_kwargs)

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

    @staticmethod
    def _release_cuda_before_vllm_wake() -> None:
        """Drop PyTorch cached blocks before vLLM cuMem remap on wake."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @staticmethod
    def _vram_mb() -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        return 0.0

    def sleep_for_train(self) -> None:
        """Release vLLM GPU memory so HF logprob_fwd can run collocated."""
        if self._use_sleep and self._sleep_available:
            before = self._vram_mb()
            self._llm.sleep(level=1)
            after = self._vram_mb()
            logger.info(
                "vLLM sleep(level=1): %.0f MB → %.0f MB (freed %.0f MB)",
                before,
                after,
                before - after,
            )
        self._release_cuda_before_vllm_wake()

    def wake_weights_only(self) -> None:
        """Partial wake for HF→vLLM weight sync without allocating KV cache."""
        if not (self._use_sleep and self._sleep_available):
            return
        self._release_cuda_before_vllm_wake()
        before = self._vram_mb()
        self._llm.wake_up(tags=["weights"])
        after = self._vram_mb()
        logger.info(
            "vLLM wake_up(tags=['weights']): %.0f MB → %.0f MB (+%.0f MB)",
            before,
            after,
            after - before,
        )

    def wake_for_rollout(self) -> None:
        """Full wake before the next rollout step."""
        if not (self._use_sleep and self._sleep_available):
            return
        self._release_cuda_before_vllm_wake()
        before = self._vram_mb()
        self._llm.wake_up()
        after = self._vram_mb()
        logger.info(
            "vLLM wake_up(): %.0f MB → %.0f MB (+%.0f MB)",
            before,
            after,
            after - before,
        )

    def shutdown(self) -> None:
        logger.info("RolloutEngine shutdown (vLLM engine released with process exit)")
