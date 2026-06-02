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
import re
from math import comb
from pathlib import Path


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


def rescore(data: dict, k_values=(1, 4, 8, 16)) -> dict:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("--out", default=None,
                    help="output path (default: <input>_rescored.json)")
    args = ap.parse_args()
    src = Path(args.eval_json)
    data = json.load(src.open())
    rescored = rescore(data)
    out_path = Path(args.out) if args.out else src.with_name(src.stem + "_rescored.json")
    out_path.write_text(json.dumps(rescored, indent=2))
    print(f"wrote {out_path}")

    # Comparison print
    print(f"\nlabel: {data['label']}")
    for ds_name in data["datasets"]:
        old = data["datasets"][ds_name]["pass_at_k"]
        new = rescored["datasets"][ds_name]["pass_at_k"]
        print(f"  {ds_name}:")
        for k in sorted(old):
            o = old[k]; n = new[k]
            d = n - o
            print(f"    {k}: {o:.4f} -> {n:.4f} ({'+' if d>=0 else ''}{d:.4f})")


if __name__ == "__main__":
    main()
