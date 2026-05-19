"""HF rollout generation for Run0 / training (GPU)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response must be of the form Answer: <answer>.\n\n"
    "{problem}\n\n"
)

# Micro-batch size for batched `generate` (padding=True, left-padded).
ROLLOUT_MICRO_BATCH_SIZE = 8


@dataclass
class RolloutEngineConfig:
    model_id: str
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    dtype: torch.dtype = torch.bfloat16
    micro_batch_size: int = ROLLOUT_MICRO_BATCH_SIZE
    allow_seeded_prompt_batching: bool = False


def _apply_torch_seed(seed: int | None, device: str) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def batch_generate_rollouts(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    problems: list[str],
    n: int,
    *,
    device: str | torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seeds: list[int] | None = None,
    micro_batch_size: int = ROLLOUT_MICRO_BATCH_SIZE,
    allow_seeded_prompt_batching: bool = False,
) -> list[list[str]]:
    """Batched HF `generate` over problems; each problem returns ``n`` completions."""
    if not problems:
        return []
    if seeds is not None and len(seeds) != len(problems):
        raise ValueError("seeds length must match problems length")
    if n < 1:
        raise ValueError("n must be >= 1")

    device_str = str(device)
    was_training = model.training
    model.eval()
    all_texts: list[list[str]] = []

    try:
        for start in range(0, len(problems), micro_batch_size):
            chunk_probs = problems[start : start + micro_batch_size]
            chunk_seeds = (
                seeds[start : start + micro_batch_size] if seeds is not None else None
            )
            use_batched = chunk_seeds is None or len(set(chunk_seeds)) == 1

            if use_batched:
                chunk_seed = chunk_seeds[0] if chunk_seeds else None
                prompts = [PROMPT_TEMPLATE.format(problem=p) for p in chunk_probs]
                with torch.no_grad():
                    _apply_torch_seed(chunk_seed, device_str)
                    inputs = tokenizer(
                        prompts,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                    gen_kw: dict[str, Any] = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p,
                        "num_return_sequences": n,
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    out = model.generate(**inputs, **gen_kw)

                prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
                for b, plen in enumerate(prompt_lens):
                    plen = int(plen)
                    texts = [
                        tokenizer.decode(out[b * n + j][plen:], skip_special_tokens=True)
                        for j in range(n)
                    ]
                    all_texts.append(texts)
            elif allow_seeded_prompt_batching:
                prompts = [PROMPT_TEMPLATE.format(problem=p) for p in chunk_probs]
                with torch.no_grad():
                    inputs = tokenizer(
                        prompts,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
                prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

                row_outputs: list[list[str]] = [[] for _ in range(len(chunk_probs))]
                try:
                    generators = []
                    for row_seed in chunk_seeds:
                        gen = (
                            torch.Generator(device=device_str)
                            if device_str.startswith("cuda")
                            else torch.Generator()
                        )
                        gen.manual_seed(int(row_seed))
                        generators.append(gen)

                    with torch.no_grad():
                        for _ in range(n):
                            out = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                num_return_sequences=1,
                                pad_token_id=tokenizer.pad_token_id,
                                generator=generators,
                            )
                            for b, plen in enumerate(prompt_lens):
                                row_outputs[b].append(
                                    tokenizer.decode(
                                        out[b][int(plen) :],
                                        skip_special_tokens=True,
                                    )
                                )
                    all_texts.extend(row_outputs)
                except Exception as exc:
                    logger.warning(
                        "Falling back to per-prompt generate for mixed seeds: %s",
                        exc,
                    )
                    for problem, row_seed in zip(chunk_probs, chunk_seeds):
                        prompt = PROMPT_TEMPLATE.format(problem=problem)
                        with torch.no_grad():
                            _apply_torch_seed(row_seed, device_str)
                            row_inputs = tokenizer(prompt, return_tensors="pt").to(device)
                            out = model.generate(
                                **row_inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                num_return_sequences=n,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        plen = row_inputs["input_ids"].shape[1]
                        all_texts.append(
                            [
                                tokenizer.decode(seq[plen:], skip_special_tokens=True)
                                for seq in out
                            ]
                        )
            else:
                # Per-prompt seeds (run0: seed+i, GRPO: step_seed+i) keep strict legacy RNG semantics.
                for problem, row_seed in zip(chunk_probs, chunk_seeds):
                    prompt = PROMPT_TEMPLATE.format(problem=problem)
                    with torch.no_grad():
                        _apply_torch_seed(row_seed, device_str)
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        out = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            num_return_sequences=n,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    plen = inputs["input_ids"].shape[1]
                    all_texts.append(
                        [
                            tokenizer.decode(seq[plen:], skip_special_tokens=True)
                            for seq in out
                        ]
                    )
    finally:
        if was_training:
            model.train()

    return all_texts


class HFRolloutEngine:
    def __init__(self, cfg: RolloutEngineConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.micro_batch_size = cfg.micro_batch_size
        logger.info("Loading %s on %s", cfg.model_id, self.device)
        self._load_weights(cfg.model_id)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        cfg: RolloutEngineConfig | None = None,
    ) -> HFRolloutEngine:
        """Load a trained checkpoint for post-training eval."""
        ckpt = Path(checkpoint_dir)
        inst = cls.__new__(cls)
        inst.cfg = cfg or RolloutEngineConfig(model_id=str(ckpt))
        inst.device = "cuda" if torch.cuda.is_available() else "cpu"
        inst.micro_batch_size = inst.cfg.micro_batch_size
        logger.info("Loading checkpoint %s on %s", ckpt, inst.device)
        inst._load_weights(ckpt)
        return inst

    def _load_weights(self, model_source: str | Path) -> None:
        dtype = self.cfg.dtype if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def sample_rollouts(self, problem: str, n: int, *, seed: int | None = None) -> list[str]:
        seeds = [seed] if seed is not None else None
        return self.sample_rollouts_batch([problem], n, seeds=seeds)[0]

    def sample_rollouts_batch(
        self,
        problems: list[str],
        n: int,
        *,
        seeds: list[int] | None = None,
    ) -> list[list[str]]:
        return batch_generate_rollouts(
            self.model,
            self.tokenizer,
            problems,
            n,
            device=self.device,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            seeds=seeds,
            micro_batch_size=self.micro_batch_size,
            allow_seeded_prompt_batching=self.cfg.allow_seeded_prompt_batching,
        )
