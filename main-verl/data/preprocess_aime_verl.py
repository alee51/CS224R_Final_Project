"""AIME-25 held-out val → verl parquet (MathReward stack — see docs/reward-decision.md).

Source: main/data/eval/aime25.jsonl (30 problems, MathArena/aime_2025).
Prompt: same maxrl polaris suffix (`\\boxed{}` contract) so MathReward parsing matches.
Router: data_source="aime25" → math.compute_score via fork's
`data_source.startswith("aime")` branch in verl/utils/reward_score/__init__.py.

Per-data_source val metrics: verl reports separately as `val/aime25/*` vs `val/polaris/*`.

Upload target on artifacts volume: data/main-verl/aime_val.parquet (mounted at /vol/...).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _MAIN_VERL_ROOT.parent

DATA_SOURCE = "aime25"
INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
REMOTE_VAL = "data/main-verl/aime_val.parquet"

DEFAULT_OUT_DIR = _MAIN_VERL_ROOT / "data"
DEFAULT_SRC = _REPO_ROOT / "main" / "data" / "eval" / "aime25.jsonl"


def _load_rows(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _to_verl_row(record: dict) -> dict:
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"{record['problem']}{INSTRUCTION_SUFFIX}",
            }
        ],
        "data_source": DATA_SOURCE,
        "reward_model": {"ground_truth": record["answer"], "style": "rule"},
        "extra_info": {
            "problem_id": str(record["prompt_id"]),
        },
    }


def build_parquet(out_dir: Path, *, src: Path = DEFAULT_SRC) -> tuple[Path, int]:
    if not src.is_file():
        raise FileNotFoundError(f"source not found: {src}")
    rows = [_to_verl_row(r) for r in _load_rows(src)]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aime_val.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path, len(rows)


def sanity_check(path: Path) -> None:
    df = pd.read_parquet(path)
    assert "prompt" in df.columns and "reward_model" in df.columns and "data_source" in df.columns
    assert (df["data_source"] == DATA_SOURCE).all()
    assert df["reward_model"].apply(lambda r: bool(r["ground_truth"])).all()


def upload_parquet(path: Path) -> None:
    if str(_MAIN_VERL_ROOT) not in sys.path:
        sys.path.insert(0, str(_MAIN_VERL_ROOT))

    import modal

    from infra.modal_volume import ARTIFACTS_VOLUME_NAME

    vol = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(path), REMOTE_VAL)
    print(f"uploaded {path.name} -> /vol/{REMOTE_VAL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AIME-25 manifest → verl parquet (+ optional Modal upload)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    out_path, n = build_parquet(args.out_dir, src=args.src)
    sanity_check(out_path)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] aime25 rows={n} -> {out_path}")

    if args.upload:
        upload_parquet(out_path)


if __name__ == "__main__":
    main()
