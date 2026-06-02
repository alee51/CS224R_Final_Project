"""|U_correct|: avg # distinct judge CoT-clusters among correct rollouts per prompt.

Matches Poly-EPO paper Fig 2 (left). For each (arm, training step):
  per_prompt = mean_{correct prompts}( |distinct judge cluster_ids among rollouts with reward>0.5| )

Degenerate cluster (-1) is excluded. GRPO has cluster_id=0 for everything (no
judge at training time) → trivially 1.0; meaningful comparison is Minority vs
Poly-EPO. For a cross-arm comparison including GRPO, run the judge over GRPO
rollouts as a separate post-hoc pass.

Also tracks `non_zero_rate` (Poly-EPO Fig 2 right): fraction of prompts in the
step with >=1 correct rollout.

Reads per-rollout JSONLs at
  main/data/probes/per_rollout_v2/{arm}/{run_id or unknown_run}/step_<N>.jsonl

Output: writeup/results/u_correct_trajectory.json + writeup/results/u_correct_summary.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("grpo", "minority", "polyepo")
DEGENERATE = -1
ROOT = Path("/Users/nancybao/Desktop/dev/cs224r_finalproject")
PER_ROLLOUT_DIR = ROOT / "main/data/probes/per_rollout_v2"
OUT_JSON = ROOT / "writeup/results/u_correct_trajectory.json"
OUT_MD = ROOT / "writeup/results/u_correct_summary.md"


def iter_step_files(arm_dir: Path):
    """Yield (step, path) using resume subdir over unknown_run when both exist."""
    by_step: dict[int, Path] = {}
    for sub in sorted(arm_dir.iterdir()):
        if not sub.is_dir():
            continue
        is_resume = sub.name != "unknown_run"
        for f in sub.glob("step_*.jsonl"):
            try:
                step = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if step not in by_step or is_resume:
                by_step[step] = f
    for step in sorted(by_step):
        yield step, by_step[step]


def load_step(path: Path) -> dict[str, list[dict]]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_prompt[row["prompt_id"]].append(row)
    return by_prompt


def step_stats(by_prompt: dict[str, list[dict]]) -> dict:
    """Compute the per-step aggregates from one step's JSONL."""
    u_correct: list[int] = []
    n_correct_per_prompt: list[int] = []
    n_prompts = len(by_prompt)
    n_prompts_with_correct = 0

    for _, rollouts in by_prompt.items():
        n_correct = sum(1 for r in rollouts if r.get("reward", 0) > 0.5)
        n_correct_per_prompt.append(n_correct)
        if n_correct == 0:
            continue
        n_prompts_with_correct += 1
        correct_rollouts = [r for r in rollouts if r.get("reward", 0) > 0.5]
        judge_set = {r.get("cluster_id") for r in correct_rollouts}
        judge_set.discard(DEGENERATE)
        u_correct.append(len(judge_set))

    return {
        "n_prompts": n_prompts,
        "n_prompts_with_correct": n_prompts_with_correct,
        "non_zero_rate": (n_prompts_with_correct / n_prompts) if n_prompts else 0.0,
        "mean_n_correct": statistics.mean(n_correct_per_prompt) if n_correct_per_prompt else 0.0,
        "u_correct_mean": statistics.mean(u_correct) if u_correct else 0.0,
    }


def analyze_arm(arm_dir: Path, sample_every: int, step_min: int, step_max: int) -> list[dict]:
    out = []
    for step, path in iter_step_files(arm_dir):
        if step < step_min or step > step_max:
            continue
        if (step - step_min) % sample_every != 0:
            continue
        s = step_stats(load_step(path))
        s["step"] = step
        out.append(s)
    return out


def fmt_md_row(arm: str, traj: list[dict]) -> list[str]:
    """Bin into [0,100], [100,200], [200,300], [300,400] and show means."""
    bins = [(0, 100), (100, 200), (200, 300), (300, 400)]
    rows = []
    for lo, hi in bins:
        b = [s for s in traj if lo <= s["step"] < hi]
        if not b:
            rows.append(f"| {arm} | {lo}-{hi-1} | no data | | |")
            continue
        jud = statistics.mean(s["u_correct_mean"] for s in b)
        nzr = statistics.mean(s["non_zero_rate"] for s in b)
        n = sum(s["n_prompts_with_correct"] for s in b)
        rows.append(f"| {arm} | {lo}-{hi-1} | {jud:.3f} | {nzr:.3f} | {n} |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--step-min", type=int, default=0)
    ap.add_argument("--step-max", type=int, default=400)
    args = ap.parse_args()

    results = {"sample_every": args.sample_every,
               "step_min": args.step_min,
               "step_max": args.step_max,
               "arms": {}}
    print(f"sampling steps {args.step_min}..{args.step_max} every {args.sample_every}\n")
    for arm in ARMS:
        d = PER_ROLLOUT_DIR / arm
        if not d.exists():
            print(f"skip {arm}: {d} not found")
            continue
        traj = analyze_arm(d, args.sample_every, args.step_min, args.step_max)
        results["arms"][arm] = traj
        if not traj:
            print(f"{arm}: no steps in range")
            continue
        # print headline summary
        first, last = traj[0], traj[-1]
        print(f"=== {arm} ({len(traj)} steps, {first['step']}..{last['step']}) ===")
        print(f"  |U_correct|   mean:  {first['u_correct_mean']:.3f}  ->  {last['u_correct_mean']:.3f}")
        print(f"  non_zero_rate mean:  {first['non_zero_rate']:.3f}  ->  {last['non_zero_rate']:.3f}")
        print()

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT_JSON}")

    # Markdown summary
    md = ["# |U_correct| training trajectory",
          "",
          "Avg # distinct judge CoT-clusters among correct rollouts per prompt, "
          "averaged over prompts in the step. Mirrors Poly-EPO paper Fig 2 (left).",
          "",
          "- Degenerate cluster (-1) excluded.",
          "- **GRPO has no judge at training time → cluster_id=0 for everything → trivially 1.0**. "
          "Cross-arm vs GRPO requires a separate post-hoc judge pass on GRPO rollouts.",
          "- **non_zero_rate**: fraction of prompts with >=1 correct rollout (Poly-EPO Fig 2 right).",
          "",
          f"Steps {args.step_min}..{args.step_max}, sampled every {args.sample_every}. "
          f"Source: per-rollout JSONLs under `main/data/probes/per_rollout_v2/`.",
          "",
          "| arm | step bin | |U_correct| | non_zero_rate | n_prompts_with_correct |",
          "|---|---|---|---|---|"]
    for arm in ARMS:
        traj = results["arms"].get(arm, [])
        if traj:
            md.extend(fmt_md_row(arm, traj))
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
