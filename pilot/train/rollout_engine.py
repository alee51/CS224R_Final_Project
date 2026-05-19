"""HF rollout generation for Run0 / training (GPU)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response must be of the form Answer: <answer>.\n\n"
    "{problem}\n\n"
)


@dataclass
class RolloutEngineConfig:
    model_id: str
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    dtype: torch.dtype = torch.bfloat16


class HFRolloutEngine:
    def __init__(self, cfg: RolloutEngineConfig) -> None:
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        logger.info("Loading checkpoint %s on %s", ckpt, inst.device)
        inst._load_weights(ckpt)
        return inst

    def _load_weights(self, model_source: str | Path) -> None:
        dtype = self.cfg.dtype if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def sample_rollouts(self, problem: str, n: int, *, seed: int | None = None) -> list[str]:
        prompt = PROMPT_TEMPLATE.format(problem=problem)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kw: dict = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": True,
            "temperature": self.cfg.temperature,
            "top_p": self.cfg.top_p,
            "num_return_sequences": n,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
                if self.device == "cuda":
                    torch.cuda.manual_seed_all(seed)
            out = self.model.generate(**inputs, **gen_kw)

        prompt_len = inputs["input_ids"].shape[1]
        texts: list[str] = []
        for seq in out:
            new_tokens = seq[prompt_len:]
            texts.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
        return texts
