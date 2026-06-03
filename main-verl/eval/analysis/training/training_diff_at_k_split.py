"""Training-time diff@k split by solved vs unsolved prompts — hypothesis gate.

Runs on saved training-time per-rollout JSONLs (`main/data/probes/per_rollout_v2/<arm>/.../step_*.jsonl`),
not on held-out eval JSONs. Tests the "minority's diversity goes to wrong
answers" hypothesis on data already on disk, before spending Phase 1 eval
budget on the set arms.

Each row in a step JSONL:
  {global_step, prompt_id, rollout_idx, parsed_answer, reward, cluster_id, finish_reason, response_length}

Training was n=8 rollouts/prompt. For each (step, prompt) group:
  - n_correct = count of rollouts with reward > 0.5
  - solved if n_correct > 0, else unsolved
  - distinct_answers@k = #unique non-empty parsed_answer in first k rollouts,
    for k in {1, 2, 4, 8}

Per arm × partition × k: average over prompts in step, then over sampled steps.

Hypothesis-gate verdict: if minority - GRPO is LARGER in unsolved partition
than in solved partition, the "diversity to wrong answers" hypothesis holds.

Usage:
    python main-verl/eval/analysis/training_diff_at_k_split.py \
        --root main/data/probes/per_rollout_v2 \
        --step-min 200 --step-max 400 --sample-every 10
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

K_VALUES = [1, 2, 4, 8]
STEP_RE = re.compile(r"step_(\d+)\.jsonl$")


def find_step_files(arm_dir: Path, step_min: int, step_max: int, every: int):
    """Return {step: Path} keeping the most recent file if multiple match a step."""
    found: dict[int, Path] = {}
    for p in arm_dir.rglob("step_*.jsonl"):
        m = STEP_RE.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step < step_min or step > step_max:
            continue
        if step % every != 0:
            continue
        # Keep the most recently modified file if same step appears in multiple subdirs
        if step not in found or p.stat().st_mtime > found[step].stat().st_mtime:
            found[step] = p
    return dict(sorted(found.items()))


def per_step_distinct_by_partition(jsonl_path: Path):
    """Return {solved: {k: mean_distinct@k}, unsolved: {k: mean_distinct@k}, n_solved, n_unsolved}."""
    prompts: dict[str, list[dict]] = defaultdict(list)
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            prompts[row["prompt_id"]].append(row)
    solved_distinct = {k: [] for k in K_VALUES}
    unsolved_distinct = {k: [] for k in K_VALUES}
    for pid, rows in prompts.items():
        rows.sort(key=lambda r: r["rollout_idx"])
        rewards = [r["reward"] for r in rows]
        preds = [r.get("parsed_answer") for r in rows]
        n_correct = sum(1 for r in rewards if r > 0.5)
        target = solved_distinct if n_correct > 0 else unsolved_distinct
        for k in K_VALUES:
            first_k = preds[:k]
            distinct = {p for p in first_k if p}
            target[k].append(len(distinct))
    out = {"n_solved": 0, "n_unsolved": 0, "solved": {}, "unsolved": {}}
    for k in K_VALUES:
        out["solved"][k] = (sum(solved_distinct[k]) / len(solved_distinct[k])) if solved_distinct[k] else None
        out["unsolved"][k] = (sum(unsolved_distinct[k]) / len(unsolved_distinct[k])) if unsolved_distinct[k] else None
    out["n_solved"] = len(solved_distinct[K_VALUES[0]])
    out["n_unsolved"] = len(unsolved_distinct[K_VALUES[0]])
    return out


def aggregate_arm(arm_name: str, arm_dir: Path, step_min: int, step_max: int, every: int):
    files = find_step_files(arm_dir, step_min, step_max, every)
    print(f"[{arm_name}] sampling {len(files)} steps in [{step_min}, {step_max}] every {every}")
    if not files:
        return None
    per_step = []
    for step, path in files.items():
        try:
            row = per_step_distinct_by_partition(path)
            row["step"] = step
            per_step.append(row)
        except Exception as e:
            print(f"  [{arm_name}] step {step}: {e}")
    if not per_step:
        return None
    avg = {"solved": {}, "unsolved": {}}
    for k in K_VALUES:
        s_vals = [r["solved"][k] for r in per_step if r["solved"][k] is not None]
        u_vals = [r["unsolved"][k] for r in per_step if r["unsolved"][k] is not None]
        avg["solved"][k] = (sum(s_vals) / len(s_vals)) if s_vals else None
        avg["unsolved"][k] = (sum(u_vals) / len(u_vals)) if u_vals else None
    avg["n_steps"] = len(per_step)
    avg["avg_n_solved"] = sum(r["n_solved"] for r in per_step) / len(per_step)
    avg["avg_n_unsolved"] = sum(r["n_unsolved"] for r in per_step) / len(per_step)
    avg["per_step"] = per_step
    return avg


def build_markdown(results: dict, step_min: int, step_max: int, every: int) -> str:
    lines = []
    lines.append(f"# Training-time diff@k split (hypothesis gate)")
    lines.append("")
    lines.append(f"Steps sampled: [{step_min}, {step_max}], every {every}. n=8 rollouts/prompt during training.")
    lines.append("")
    lines.append(
        "Hypothesis: minority adds diversity but on **unsolved** prompts (wrong-answer diversity), "
        "not on solved prompts (correct-answer diversity)."
    )
    lines.append("")
    lines.append("If `Δ(minority − grpo)` is larger on the unsolved row than the solved row, the hypothesis holds.")
    lines.append("")
    lines.append("## distinct_answers@k by partition")
    lines.append("")
    lines.append("| arm | partition | n_prompts/step | " + " | ".join(f"k={k}" for k in K_VALUES) + " |")
    lines.append("|---|---|---|" + "---|" * len(K_VALUES))
    for arm in ("grpo", "minority", "polyepo"):
        if arm not in results:
            continue
        r = results[arm]
        s_n = r["avg_n_solved"]
        u_n = r["avg_n_unsolved"]
        s_cells = " | ".join(
            f"{r['solved'][k]:.3f}" if r["solved"][k] is not None else "—" for k in K_VALUES
        )
        u_cells = " | ".join(
            f"{r['unsolved'][k]:.3f}" if r["unsolved"][k] is not None else "—" for k in K_VALUES
        )
        lines.append(f"| {arm} | solved | {s_n:.1f} | {s_cells} |")
        lines.append(f"| {arm} | unsolved | {u_n:.1f} | {u_cells} |")
    lines.append("")
    if "minority" in results and "grpo" in results:
        lines.append("## Δ(minority − grpo) by partition")
        lines.append("")
        lines.append("| partition | " + " | ".join(f"k={k}" for k in K_VALUES) + " |")
        lines.append("|---|" + "---|" * len(K_VALUES))
        for part in ("solved", "unsolved"):
            cells = []
            for k in K_VALUES:
                m_v = results["minority"][part][k]
                g_v = results["grpo"][part][k]
                if m_v is None or g_v is None:
                    cells.append("—")
                else:
                    cells.append(f"{m_v - g_v:+.3f}")
            lines.append(f"| {part} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(
            "**Verdict gate:** if unsolved Δ > solved Δ across the k ladder, hypothesis SUPPORTED. "
            "If similar, minority's diversity is real on solved prompts too (good for the arm)."
        )
        lines.append("")
    if "polyepo" in results and "grpo" in results:
        lines.append("## Δ(polyepo − grpo) by partition (context)")
        lines.append("")
        lines.append("| partition | " + " | ".join(f"k={k}" for k in K_VALUES) + " |")
        lines.append("|---|" + "---|" * len(K_VALUES))
        for part in ("solved", "unsolved"):
            cells = []
            for k in K_VALUES:
                m_v = results["polyepo"][part][k]
                g_v = results["grpo"][part][k]
                if m_v is None or g_v is None:
                    cells.append("—")
                else:
                    cells.append(f"{m_v - g_v:+.3f}")
            lines.append(f"| {part} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="main/data/probes/per_rollout_v2")
    ap.add_argument("--step-min", type=int, default=200)
    ap.add_argument("--step-max", type=int, default=400)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument(
        "--out",
        default="main-verl/writeup/results/training_diff_at_k_split.md",
    )
    args = ap.parse_args()

    root = Path(args.root)
    arms = {"grpo": root / "grpo", "minority": root / "minority", "polyepo": root / "polyepo"}
    results = {}
    for arm, arm_dir in arms.items():
        if not arm_dir.exists():
            print(f"[{arm}] missing dir {arm_dir}")
            continue
        r = aggregate_arm(arm, arm_dir, args.step_min, args.step_max, args.sample_every)
        if r is not None:
            results[arm] = r

    md = build_markdown(results, args.step_min, args.step_max, args.sample_every)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"wrote {out_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
