"""DAPO Minerva reward parser — ported from verl/utils/reward_score/math_dapo.py."""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
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

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str,
    gt: str,
    gt_need_extract: bool = False,
    answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)",
) -> tuple[bool, str]:
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def extract_rank2(
    completion: str,
    gold: str,
    prompt_variant: str = "dapo_answer_v1",
) -> dict:
    clipped = completion[-300:]
    gt_norm = normalize_final_answer(gold)

    minerva_matches = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", clipped)
    if minerva_matches:
        parsed_answer_minerva = normalize_final_answer(minerva_matches[-1])
        parse_ok_minerva = bool(
            parsed_answer_minerva.strip()
            and parsed_answer_minerva != "[INVALID]"
        )
    else:
        parsed_answer_minerva = None
        parse_ok_minerva = False

    boxed_str = last_boxed_only_string(clipped)
    if boxed_str is not None:
        parsed_answer_boxed = normalize_final_answer(remove_boxed(boxed_str))
        parse_ok_boxed = bool(parsed_answer_boxed.strip())
    else:
        parsed_answer_boxed = None
        parse_ok_boxed = False

    parsed_answer: str | None = None
    extract_path = "none"
    parse_ok_rank2 = False

    if prompt_variant == "hybrid_answer_boxed":
        hybrid_matches = re.findall(r"Answer:\s*\\boxed\{([^}]+)\}", clipped)
        if hybrid_matches:
            candidate = normalize_final_answer(hybrid_matches[-1])
            if candidate.strip():
                parsed_answer = candidate
                extract_path = "hybrid"
                parse_ok_rank2 = True

    if not parse_ok_rank2 and parse_ok_boxed:
        parsed_answer = parsed_answer_boxed
        extract_path = "boxed"
        parse_ok_rank2 = True

    if not parse_ok_rank2 and parse_ok_minerva:
        parsed_answer = parsed_answer_minerva
        extract_path = "answer_line"
        parse_ok_rank2 = True

    reward = 0
    if parse_ok_rank2 and parsed_answer is not None:
        reward = 1 if parsed_answer == gt_norm else 0

    return {
        "parsed_answer_minerva": parsed_answer_minerva,
        "parse_ok_minerva": parse_ok_minerva,
        "parsed_answer_boxed": parsed_answer_boxed,
        "parse_ok_boxed": parse_ok_boxed,
        "parse_ok_rank2": parse_ok_rank2,
        "extract_path": extract_path,
        "reward": reward,
        "parsed_answer": parsed_answer if parse_ok_rank2 else None,
    }


def compute_reward(completion: str, gold: str) -> dict:
    clipped = completion[-300:]
    correct, pred = is_correct_minerva(clipped, gold)
    reward = 1 if correct else 0

    parse_ok = pred != "[INVALID]" and bool(pred.strip())

    strict_score, _ = is_correct_strict_box(completion, gold)
    strict_parse_ok = strict_score > 0

    has_boxed = last_boxed_only_string(completion) is not None
    has_answer_line = bool(
        re.findall(r"(?i)Answer\s*:\s*([^\n]+)", clipped)
    )

    parsed_is_int = False
    if parse_ok:
        normalized_digits = pred.replace(",", "").strip()
        parsed_is_int = (
            normalized_digits.lstrip("-").isdigit()
            if normalized_digits
            else False
        )

    parsed_answer: str | None = pred if parse_ok else None

    return {
        "reward": reward,
        "parse_ok": parse_ok,
        "parsed_answer": parsed_answer,
        "parsed_is_int": parsed_is_int,
        "has_boxed": has_boxed,
        "has_answer_line": has_answer_line,
        "strict_parse_ok": strict_parse_ok,
    }
