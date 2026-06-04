"""Shared I/O helpers for analysis scripts.

Each saved eval JSON produced by `main-verl/eval/run_eval.py` has the shape:

    {
      "label": "<arm>_step400",
      "ckpt_path": "...",
      "n_rollouts": 64,
      "datasets": {
        "<ds_name>": {
          "n_prompts": int,
          "pass_at_k": {"pass@1": float, ...},
          "mean_reward_at_1": float,
          "per_prompt": [
            {
              "problem_id": str,
              "ground_truth": str,
              "n_correct": int,
              "rewards": [float, ...],   # one per rollout
              "preds": [str, ...],        # parsed boxed answers
              "rollouts": [str, ...],     # raw generation text
              "logprobs": [[              # optional, when CS224R_EVAL_LOGPROBS>0
                  {tok_id: logprob, ...}, # one dict per generated token (top-N)
                  ...
              ], ...]                     # one inner list per rollout
            }
          ]
        }
      }
    }

The per-dataset incremental dump has the same shape but with only one entry in
`datasets`. Most analysis scripts treat the per-dataset dump as the unit and
glob multiple of them to build cross-arm / cross-dataset tables.

The label is parsed as "<arm>_<step>" where the trailing chunk is the step tag,
so "minority_step400" → arm="minority". The full label is also retained.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable


def expand_inputs(patterns: Iterable[str]) -> list[Path]:
    """Glob each pattern and dedupe; preserves order."""
    seen: set[str] = set()
    out: list[Path] = []
    for pat in patterns:
        # Accept either a literal path or a glob.
        matches = sorted(glob.glob(pat)) or [pat]
        for m in matches:
            p = Path(m)
            s = str(p.resolve())
            if s in seen:
                continue
            seen.add(s)
            if p.exists():
                out.append(p)
    return out


def arm_from_label(label: str) -> str:
    """grpo_step400 → grpo; base_step400 → base; minority_cot_step400 → minority_cot."""
    if "_step" in label:
        return label.rsplit("_step", 1)[0]
    return label


def iter_arm_dataset(paths: Iterable[Path], *, drop_heavy: bool = True):
    """Yield (arm, dataset_name, dataset_dict, label, top_level) tuples.

    drop_heavy: if True (default), strip per_prompt[i].logprobs (the largest field
    by far — multi-GB for n=64 evals with top-20 logprobs per token). Saves OOM
    when iterating across all 20 files. Set False for analyses that need logprobs
    (token_entropy_split, kl_from_base).
    """
    decoder = json.JSONDecoder(strict=False)  # tolerate raw control chars in rollout text
    for p in paths:
        try:
            top = decoder.decode(p.read_text())
        except Exception as exc:  # pragma: no cover
            print(f"[_io] WARN failed to load {p}: {exc}")
            continue
        label = top.get("label", p.stem)
        arm = arm_from_label(label)
        datasets = top.get("datasets", {})
        if drop_heavy:
            for ds in datasets.values():
                for pp in ds.get("per_prompt", []):
                    pp.pop("logprobs", None)
        for ds_name, ds in datasets.items():
            yield arm, ds_name, ds, label, top
        del top  # let GC reclaim before next file


def collect(patterns: Iterable[str], *, drop_heavy: bool = True) -> dict[tuple[str, str], dict]:
    """Map (arm, dataset_name) → ds dict for each file matched by the patterns."""
    out: dict[tuple[str, str], dict] = {}
    for arm, ds_name, ds, _label, _top in iter_arm_dataset(expand_inputs(patterns), drop_heavy=drop_heavy):
        # Last writer wins (later files override earlier ones for the same key).
        out[(arm, ds_name)] = ds
    return out


def collected_from_json(json_data: dict) -> dict[tuple[str, str], dict]:
    """Same shape as collect(), but built from one already-loaded JSON.

    Lets analyze() functions skip a json.load — critical when the file is
    multi-GB and would otherwise be re-parsed once per script invocation.
    """
    arm = arm_from_label(json_data.get("label", "unknown"))
    return {(arm, ds_name): ds for ds_name, ds in json_data.get("datasets", {}).items()}


def write_markdown(rel_path: str, text: str) -> Path:
    """Write a markdown file under main-verl/writeup/results/<rel_path>."""
    # analysis_io.py lives at main-verl/eval/analysis/posthoc/analysis_io.py
    # → parents[3] = main-verl/
    root = Path(__file__).resolve().parents[3] / "writeup" / "results"
    root.mkdir(parents=True, exist_ok=True)
    out = root / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out
