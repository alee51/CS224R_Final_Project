"""Polaris filtered manifest → verl parquet (MathReward stack — see docs/reward-decision.md).

Split: val = 256 rows (seed 42); train = remainder of polaris_train.jsonl (51,139 total).
<!-- TODO: confirm 256 is the right val size for trainer.test_freq=25 -->

Source: main/data/polaris_train.jsonl via paths.POLARIS_TRAIN_JSONL — no re-filter / HF re-pull.
Prompt: maxrl examples/maxrl_data_preprocess/polaris.py suffix (\\boxed{} contract).
Router: `data_source=polaris` → math.py (upstream math_reward.py) after infra patch @ image build.
Upload targets on artifacts volume: data/main-verl/polaris_{train,val}.parquet (mounted at /vol/...).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_MAIN_VERL_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _MAIN_VERL_ROOT.parent

VAL_SIZE = 256
VAL_SEED = 42
DATA_SOURCE = "polaris"
# Verbatim from maxrl examples/maxrl_data_preprocess/polaris.py (MathReward prompt contract)
INSTRUCTION_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)
REMOTE_TRAIN = "data/main-verl/polaris_train.parquet"
REMOTE_VAL = "data/main-verl/polaris_val.parquet"

DEFAULT_OUT_DIR = _MAIN_VERL_ROOT / "data"


def _polaris_train_jsonl() -> Path:
    paths_py = _REPO_ROOT / "main" / "data" / "paths.py"
    spec = importlib.util.spec_from_file_location("main_data_paths", paths_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load paths module from {paths_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Path(mod.POLARIS_TRAIN_JSONL)


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
    problem = record["problem"]
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"{problem}{INSTRUCTION_SUFFIX}",
            }
        ],
        "data_source": DATA_SOURCE,
        "reward_model": {"ground_truth": record["gold"], "style": "rule"},
        "extra_info": {
            "problem_id": record["problem_id"],
            "difficulty_band": record["difficulty_band"],
        },
    }


def _split_indices(n: int, val_size: int, seed: int) -> tuple[set[int], set[int]]:
    if val_size >= n:
        raise ValueError(f"val_size {val_size} must be < n {n}")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_idx = set(indices[:val_size])
    train_idx = set(indices[val_size:])
    return train_idx, val_idx


def build_parquets(
    out_dir: Path,
    *,
    jsonl_path: Path | None = None,
    val_size: int = VAL_SIZE,
    val_seed: int = VAL_SEED,
) -> tuple[Path, Path, dict[str, int]]:
    src = jsonl_path or _polaris_train_jsonl()
    if not src.is_file():
        raise FileNotFoundError(f"manifest not found: {src}")

    raw = _load_rows(src)
    n = len(raw)
    train_idx, val_idx = _split_indices(n, val_size, val_seed)

    train_rows = [_to_verl_row(raw[i]) for i in sorted(train_idx)]
    val_rows = [_to_verl_row(raw[i]) for i in sorted(val_idx)]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "polaris_train.parquet"
    val_path = out_dir / "polaris_val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    counts = {
        "total": n,
        "train": len(train_rows),
        "val": len(val_rows),
    }
    return train_path, val_path, counts


def sanity_check(train_path: Path) -> None:
    df = pd.read_parquet(train_path)
    assert "prompt" in df.columns and "reward_model" in df.columns and "data_source" in df.columns
    assert df["reward_model"].apply(lambda r: bool(r["ground_truth"])).all()
    assert (df["data_source"] == DATA_SOURCE).all()


def upload_parquets(train_path: Path, val_path: Path) -> None:
    if str(_MAIN_VERL_ROOT) not in sys.path:
        sys.path.insert(0, str(_MAIN_VERL_ROOT))

    import modal

    from infra.modal_volume import ARTIFACTS_VOLUME_NAME

    vol = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(train_path), REMOTE_TRAIN)
        batch.put_file(str(val_path), REMOTE_VAL)
    print(f"uploaded {train_path.name} -> /vol/{REMOTE_TRAIN}")
    print(f"uploaded {val_path.name} -> /vol/{REMOTE_VAL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Polaris manifest → verl parquet (+ optional Modal upload)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for polaris_train.parquet and polaris_val.parquet",
    )
    parser.add_argument("--upload", action="store_true", help="Upload parquets to Modal artifacts volume")
    parser.add_argument("--jsonl", type=Path, default=None, help="Override manifest path (default: paths.POLARIS_TRAIN_JSONL)")
    args = parser.parse_args()

    train_path, val_path, counts = build_parquets(args.out_dir, jsonl_path=args.jsonl)
    sanity_check(train_path)

    val_df = pd.read_parquet(val_path)
    assert (val_df["data_source"] == DATA_SOURCE).all()
    assert val_df["reward_model"].apply(lambda r: bool(r["ground_truth"])).all()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] rows total={counts['total']} train={counts['train']} val={counts['val']}")
    print(f"wrote {train_path}")
    print(f"wrote {val_path}")

    if args.upload:
        upload_parquets(train_path, val_path)


if __name__ == "__main__":
    main()
