"""Minerva held-out eval (HF math-ai/minervamath) → verl parquet.

Source: math-ai/minervamath split=test (272 problems).
Schema: matches math500.parquet and aime_val.parquet.

Grader compatibility: ~81/272 problems have symbolic answers (e.g., np.arcsin(10/13),
\\frac{e^t}{3}+ce^{-2t}). The downstream grader is verl.utils.reward_score.math.compute_score
(Hendrycks is_equiv), which does latex normalization. We preprocess Python-style answers
to LaTeX form. LaTeX answers are kept as-is (already compatible). Numeric answers are unchanged.

Post-build sanity check: run compute_score(f"\\boxed{{{gold}}}", gold) for each row.
Should see score=1.0 for nearly all rows (>99%).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _MAIN_VERL_ROOT.parent

DATA_SOURCE = "minerva"
INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
REMOTE_VAL = "data/main-verl/minerva.parquet"

DEFAULT_OUT_DIR = _MAIN_VERL_ROOT / "data"


def _convert_python_to_latex(answer: str) -> str:
    """Convert Python-style math expressions to LaTeX.

    Examples:
        np.arcsin(10/13) -> \\arcsin(10/13)
        np.arccos(x) -> \\arccos(x)
        np.sqrt(2) -> \\sqrt{2}
    """
    # Replace numpy function calls with LaTeX equivalents
    replacements = {
        "np.arcsin": r"\arcsin",
        "np.arccos": r"\arccos",
        "np.arctan": r"\arctan",
        "np.sqrt": r"\sqrt",
        "np.sin": r"\sin",
        "np.cos": r"\cos",
        "np.tan": r"\tan",
        "np.exp": r"\exp",
        "np.log": r"\log",
        "np.pi": r"\pi",
        "arcsin": r"\arcsin",
        "arccos": r"\arccos",
        "arctan": r"\arctan",
        "sqrt": r"\sqrt",
    }

    result = answer
    for old, new in replacements.items():
        result = result.replace(old, new)

    return result


def _preprocess_answer(answer: str) -> str:
    """Preprocess answer for grader compatibility.

    - If Python-style (np.*), convert to LaTeX
    - If LaTeX, keep as-is
    - If numeric, keep as-is
    """
    answer = answer.strip()

    # Check if it's Python-style
    if "np." in answer:
        answer = _convert_python_to_latex(answer)

    return answer


def _load_minerva() -> list[dict]:
    """Load Minerva dataset from HuggingFace."""
    dataset = load_dataset("math-ai/minervamath", split="test")
    rows = []
    for i, record in enumerate(dataset):
        rows.append({
            "problem_id": f"minerva_{i}",
            "question": record["question"],
            "answer": record["answer"],
        })
    return rows


def _to_verl_row(record: dict) -> dict:
    """Convert Minerva row to verl schema."""
    processed_answer = _preprocess_answer(record["answer"])

    return {
        "prompt": [
            {
                "role": "user",
                "content": f"{record['question']}{INSTRUCTION_SUFFIX}",
            }
        ],
        "data_source": DATA_SOURCE,
        "reward_model": {"ground_truth": processed_answer, "style": "rule"},
        "extra_info": {
            "problem_id": record["problem_id"],
        },
    }


def build_parquet(out_dir: Path) -> tuple[Path, int]:
    """Build verl parquet from Minerva dataset."""
    print("[minerva] loading from HuggingFace...")
    rows = _load_minerva()
    print(f"[minerva] loaded {len(rows)} rows from math-ai/minervamath split=test")

    print("[minerva] preprocessing...")
    verl_rows = [_to_verl_row(r) for r in rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "minerva.parquet"

    df = pd.DataFrame(verl_rows)
    df.to_parquet(out_path, index=False)
    print(f"[minerva] wrote {out_path}")

    return out_path, len(rows)


def sanity_check(path: Path) -> None:
    """Check parquet structure and grader compatibility."""
    df = pd.read_parquet(path)

    # Schema check
    assert "prompt" in df.columns and "reward_model" in df.columns and "data_source" in df.columns
    assert (df["data_source"] == DATA_SOURCE).all()
    assert df["reward_model"].apply(lambda r: bool(r["ground_truth"])).all()

    print(f"\n[sanity_check] Schema OK: {len(df)} rows, columns={df.columns.tolist()}")

    # Grader compatibility check
    try:
        from verl.utils.reward_score.math import compute_score
    except ImportError:
        print("[sanity_check] WARNING: verl not installed, skipping grader test")
        return

    print("[sanity_check] Testing grader compatibility...")
    failures = []
    passes = 0

    for i, row in df.iterrows():
        gold = row["reward_model"]["ground_truth"]
        try:
            score = compute_score(f"\\boxed{{{gold}}}", gold)
            if score > 0.5:
                passes += 1
            else:
                failures.append((i, gold, f"score={score}"))
        except Exception as e:
            failures.append((i, gold, str(e)))

    pass_rate = passes / len(df)
    print(f"[sanity_check] Grader pass rate: {passes}/{len(df)} ({100*pass_rate:.1f}%)")

    if failures:
        print(f"\n[sanity_check] First 10 failures:")
        for idx, gold, reason in failures[:10]:
            print(f"  row {idx}: answer='{gold[:80]}...' => {reason}")

    if pass_rate < 0.80:
        print(f"\n[sanity_check] WARNING: pass rate {100*pass_rate:.1f}% < 80% threshold")

    return failures


def upload_parquet(path: Path) -> None:
    """Upload to Modal artifacts volume."""
    if str(_MAIN_VERL_ROOT) not in sys.path:
        sys.path.insert(0, str(_MAIN_VERL_ROOT))

    import modal
    from infra.modal_volume import ARTIFACTS_VOLUME_NAME

    vol = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(path), REMOTE_VAL)
    print(f"[upload] {path.name} -> /vol/{REMOTE_VAL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minerva dataset → verl parquet")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    out_path, n = build_parquet(args.out_dir)

    failures = sanity_check(out_path)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n[{ts}] minerva rows={n} -> {out_path}")

    if args.upload:
        upload_parquet(out_path)


if __name__ == "__main__":
    main()
