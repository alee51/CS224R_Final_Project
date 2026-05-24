"""Tier-1 pilot eval rollouts (AIME + HMMT Nov) after training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from pilot.eval.io import load_predictions, write_metrics
from pilot.eval.metrics import aggregate_metrics
from pilot.eval.splits import load_lock
from pilot.train.answer_parse import extract_answer, is_correct
from pilot.train.canonicalize import cluster_id

logger = logging.getLogger(__name__)


class RolloutEngine(Protocol):
    def sample_rollouts(self, problem: str, n: int, *, seed: int | None = None) -> list[str]:
        ...


def _load_eval_prompts(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _split_name(path: Path, row: dict[str, str]) -> str:
    return str(row.get("split") or path.stem)


def run_tier1_eval(
    engine: RolloutEngine,
    eval_paths: dict[str, Path],
    *,
    run_id: str,
    out_dir: Path,
    seed: int,
    n_rollouts: int = 8,
    debug_max_prompts: int | None = None,
) -> dict[str, Any]:
    """
    Roll out on tier-1 eval JSONL files and write gate-ready artifacts.

    Writes ``prompt_inputs.jsonl`` and ``raw_predictions.jsonl`` under *out_dir*.
    Returns pooled tier-1 aggregate metrics (for ``metrics.json``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_path = out_dir / "prompt_inputs.jsonl"
    pred_path = out_dir / "raw_predictions.jsonl"

    lock = load_lock()
    mconf = lock["metrics_frozen"]
    kw = dict(
        k=mconf["pass_at_k"],
        tau=mconf["cover_tau"],
        worst_q=mconf["worst_subset_quantile"],
    )

    prompt_idx = 0
    with inputs_path.open("w") as inputs_f, pred_path.open("w") as pred_f:
        for eval_split, data_path in eval_paths.items():
            if not data_path.exists():
                raise FileNotFoundError(f"Missing tier-1 eval data: {data_path}")
            prompts = _load_eval_prompts(data_path)
            if debug_max_prompts is not None:
                prompts = prompts[: int(debug_max_prompts)]
            logger.info(
                "tier-1 eval %s: %s prompts from %s",
                eval_split,
                len(prompts),
                data_path.name,
            )
            for row in prompts:
                pid = str(row["prompt_id"])
                gold = str(row["answer"])
                problem = str(row["problem"])
                split = _split_name(data_path, row) if not eval_split else eval_split

                inputs_f.write(
                    json.dumps(
                        {
                            "prompt_id": pid,
                            "eval_split": split,
                            "problem": problem,
                            "gold_answer": gold,
                        }
                    )
                    + "\n"
                )

                texts = engine.sample_rollouts(
                    problem, n_rollouts, seed=seed + prompt_idx
                )
                for text in texts:
                    parsed = extract_answer(text)
                    pred_f.write(
                        json.dumps(
                            {
                                "prompt_id": pid,
                                "eval_split": split,
                                "parsed_answer": parsed,
                                "correct": is_correct(text, gold),
                                "cluster_id": cluster_id(parsed),
                                "completion": text,
                            }
                        )
                        + "\n"
                    )
                prompt_idx += 1

    pooled = load_predictions(pred_path)
    metrics = {
        "run_id": run_id,
        "tier": "pilot_eval",
        "eval_splits": list(eval_paths.keys()),
        "n_prompts": len(pooled),
        "n_rollouts_per_prompt": n_rollouts,
        **aggregate_metrics(pooled, **kw),
    }
    write_metrics(out_dir / "metrics.json", metrics)
    logger.info(
        "tier-1 eval done: pass@1=%.3f pass@%s=%.3f n_prompts=%s",
        metrics["pass_at_1"],
        mconf["pass_at_k"],
        metrics[f"pass_at_{mconf['pass_at_k']}"],
        metrics["n_prompts"],
    )
    return metrics
