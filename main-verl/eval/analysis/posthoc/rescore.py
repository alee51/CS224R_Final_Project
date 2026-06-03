"""Re-score saved eval_4b JSON outputs with verl's math.py (training grader).

Training uses `verl.utils.reward_score.math.compute_score` which does:
  last_boxed_only_string -> remove_boxed -> is_equiv(answer, ground_truth)

`is_equiv` normalizes latex (`\\frac{1}{2}` matches `0.5`, fixes `\\frac` spacing,
strips `\\text{}`, normalizes integers, etc.). My eval probe used math_dapo's
`strict_box_verify=True` which does exact-string match — stricter, hence pass@k
may be biased low.

Usage:
  python3 main/scripts/rescore_eval_with_training_grader.py path/to/eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from math import comb
from pathlib import Path

try:
    from verl.utils.reward_score import math_dapo as _math_dapo
    _HAS_MATH_DAPO = True
except Exception:
    _math_dapo = None
    _HAS_MATH_DAPO = False


# Inlined Hendrycks `is_equiv` and helpers — copied from
# verl/utils/reward_score/math.py @ 33873ec9 so we don't have to import verl
# locally.

SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft",
    "hours", "km", "units", "\\ldots", "sue", "points", "feet",
    "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds",
    "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3",
    "\\text{\n}", "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;",
    r",\!", "{,}", '"', "\\dots",
]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx:right_brace_idx + 1]


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left, s
        return s[len(left):]
    left = "\\boxed{"
    assert s[: len(left)] == left, s
    assert s[-1] == "}", s
    return s[len(left):-1]


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if substr.startswith("{"):
            new_str += substr
        else:
            try:
                assert len(substr) >= 2
            except AssertionError:
                return string
            a, b = substr[0], substr[1]
            if b != "{":
                if len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    new_str += "{" + a + "}{" + b + "}"
            else:
                if len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}" + b + post_substr
                else:
                    new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a); b = int(b)
        assert string == f"{a}/{b}"
        new_string = f"\\frac{{{a}}}{{{b}}}"
        return new_string
    except (AssertionError, ValueError):
        return string


def _remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def compute_score_math(solution_str, ground_truth):
    """verl/utils/reward_score/math.py compute_score (Hendrycks strict)."""
    try:
        boxed = last_boxed_only_string(solution_str)
        if boxed is None:
            return 0.0, None
        answer = remove_boxed(boxed)
        if is_equiv(answer, ground_truth):
            return 1.0, answer
        return 0.0, answer
    except Exception:
        return 0.0, None


def rescore(data: dict, k_values=(1, 2, 4, 8, 16, 32, 64)) -> dict:
    """Rescore each per_prompt entry using the training grader."""
    out = {"label": data["label"], "ckpt_path": data["ckpt_path"],
           "n_rollouts": data["n_rollouts"], "datasets": {}}
    n = data["n_rollouts"]
    for ds_name, ds in data["datasets"].items():
        new_per_prompt = []
        for p in ds["per_prompt"]:
            gt = p["ground_truth"]
            new_rewards = []
            new_preds = []
            for rollout in p["rollouts"]:
                r, pred = compute_score_math(rollout, gt)
                new_rewards.append(r)
                new_preds.append(pred if pred is not None else "")
            n_correct = int(sum(1 for r in new_rewards if r > 0.5))
            new_per_prompt.append({
                "problem_id": p["problem_id"],
                "ground_truth": gt,
                "n_correct": n_correct,
                "rewards": new_rewards,
                "preds": new_preds,
                "rollouts": p["rollouts"],
            })
        # pass@k
        passk = {}
        import numpy as np
        for k in k_values:
            if k > n:
                continue
            vals = []
            for pp in new_per_prompt:
                c = pp["n_correct"]
                vals.append(0.0 if c == 0 else 1.0 - comb(n - c, k) / comb(n, k))
            passk[f"pass@{k}"] = float(np.mean(vals))
        out["datasets"][ds_name] = {
            "n_prompts": len(new_per_prompt),
            "pass_at_k": passk,
            "mean_reward_at_1": float(np.mean([pp["rewards"][0] for pp in new_per_prompt])),
            "per_prompt": new_per_prompt,
        }
    return out


def math_dapo_tripwire(data: dict, n_problems: int = 20, seed: int = 42) -> dict:
    """Per eval.md §8 belt-and-suspenders: rescore n problems with
    math_dapo.compute_score(strict_box_verify=True) and check per-rollout
    agreement with the saved (math grader) reward. <90% = investigate."""
    if not _HAS_MATH_DAPO:
        return {"status": "skipped",
                "reason": "verl.utils.reward_score.math_dapo not importable; "
                          "run on Modal (or pip install verl + math_verify) to engage"}
    rng = random.Random(seed)
    out = {"status": "ran", "n_problems": n_problems, "seed": seed, "by_dataset": {}}
    for ds_name, ds in data["datasets"].items():
        prompts = ds["per_prompt"]
        idxs = rng.sample(range(len(prompts)), min(n_problems, len(prompts)))
        agree = total = both_pos = ours_only = dapo_only = 0
        for i in idxs:
            p = prompts[i]
            gt = p["ground_truth"]
            for rollout, our_r in zip(p["rollouts"], p["rewards"]):
                try:
                    dapo_r = float(_math_dapo.compute_score(rollout, gt,
                                                            strict_box_verify=True))
                except Exception:
                    dapo_r = 0.0
                ours = our_r > 0.5
                dapo = dapo_r > 0.5
                agree += int(ours == dapo)
                total += 1
                if ours and dapo: both_pos += 1
                elif ours: ours_only += 1
                elif dapo: dapo_only += 1
        rate = agree / total if total else 0.0
        out["by_dataset"][ds_name] = {
            "n_problems": len(idxs), "n_rollouts": total,
            "agree": agree, "rate": rate,
            "both_pos": both_pos, "math_only_pos": ours_only, "math_dapo_only_pos": dapo_only,
            "status": "OK" if rate >= 0.9 else "INVESTIGATE",
        }
    return out


def analyze(json_data: dict, tripwire_n: int = 20, tripwire_seed: int = 42) -> str:
    """Library API: same-grader rescore + math_dapo tripwire on one already-
    loaded eval JSON. Returns markdown summary string (does NOT write the
    rescored JSON to disk — only the diff + tripwire counts)."""
    rescored = rescore(json_data)
    lines = [f"# Rescore (same grader) + math_dapo tripwire — {json_data.get('label','')}",
             "", "## Same-grader rescore (`math.compute_score`)", ""]
    for ds_name in json_data["datasets"]:
        old = json_data["datasets"][ds_name]["pass_at_k"]
        new = rescored["datasets"][ds_name]["pass_at_k"]
        lines.append(f"### {ds_name}")
        lines.append("")
        lines.append("| k | saved | rescored | Δ |")
        lines.append("|---|---|---|---|")
        keys = sorted(set(old) & set(new), key=lambda s: int(s.split("@")[1]))
        for k in keys:
            o = old[k]; n = new[k]; d = n - o
            lines.append(f"| {k} | {o:.4f} | {n:.4f} | {'+' if d>=0 else ''}{d:.4f} |")
        only_old = sorted(set(old) - set(new))
        only_new = sorted(set(new) - set(old))
        if only_old:
            lines.append(f"")
            lines.append(f"_warn: in saved but not rescored: {only_old}_")
        if only_new:
            lines.append(f"_warn: in rescored but not saved: {only_new}_")
        lines.append("")

    lines.append("## math_dapo tripwire (eval.md §8)")
    lines.append("")
    tw = math_dapo_tripwire(json_data, n_problems=tripwire_n, seed=tripwire_seed)
    if tw["status"] == "skipped":
        lines.append(f"**SKIPPED** — {tw['reason']}")
    else:
        lines.append("| dataset | agree | rate | status | both+ | math+only | math_dapo+only |")
        lines.append("|---|---|---|---|---|---|---|")
        for ds_name, r in tw["by_dataset"].items():
            lines.append(f"| {ds_name} | {r['agree']}/{r['n_rollouts']} | "
                         f"{r['rate']:.3f} | {r['status']} | "
                         f"{r['both_pos']} | {r['math_only_pos']} | {r['math_dapo_only_pos']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("--out", default=None,
                    help="output path (default: <input>_rescored.json)")
    ap.add_argument("--tripwire-n", type=int, default=20,
                    help="n problems per dataset for math_dapo tripwire (eval.md §8)")
    ap.add_argument("--tripwire-seed", type=int, default=42)
    args = ap.parse_args()
    src = Path(args.eval_json)
    data = json.load(src.open())
    rescored = rescore(data)
    out_path = Path(args.out) if args.out else src.with_name(src.stem + "_rescored.json")
    out_path.write_text(json.dumps(rescored, indent=2))
    print(f"wrote {out_path}")
    print()
    print(analyze(data, tripwire_n=args.tripwire_n, tripwire_seed=args.tripwire_seed))


if __name__ == "__main__":
    main()
