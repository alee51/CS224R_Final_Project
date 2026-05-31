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
# Must stay in sync with grpo_smoke_1p7b.yaml `data.max_prompt_length` — drives the
# overflow fraction reported by --prompt-stats so we know what `truncation: left` discards.
DEFAULT_MAX_PROMPT_LENGTH = 1024
TOKENIZER_MODEL = "Qwen/Qwen3-1.7B-Base"
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


def _prompt_text(prompt_field) -> str:
    if isinstance(prompt_field, (list, tuple)) and prompt_field:
        return str(prompt_field[0].get("content", ""))
    return str(prompt_field)


def _percentiles(values: list[int], qs: tuple[float, ...]) -> dict[str, int]:
    if not values:
        return {f"p{int(q * 100)}": 0 for q in qs}
    s = sorted(values)
    out: dict[str, int] = {}
    for q in qs:
        idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        out[f"p{int(q * 100)}"] = s[idx]
    return out


def _try_load_qwen_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001 — network/cache miss is non-fatal here
        print(f"[prompt-stats] tokenizer load failed ({exc}); falling back to char-only stats")
        return None


def prompt_length_stats(
    train_path: Path,
    val_path: Path,
    *,
    max_prompt_length: int,
    use_tokenizer: bool,
) -> dict:
    """Length distribution + overflow fraction for prompts after MathReward suffix.

    Always reports char lengths (no deps). If `use_tokenizer=True` and transformers
    + the Qwen3 tokenizer are available, also reports token lengths against
    `max_prompt_length` — that's the threshold verl's RLHFDataset compares with
    `truncation: left` to drop left-side tokens.
    """
    tok = _try_load_qwen_tokenizer() if use_tokenizer else None

    def _stats_for(df: pd.DataFrame) -> dict:
        prompts = df["prompt"].map(_prompt_text).tolist()
        char_lens = [len(p) for p in prompts]
        out: dict = {
            "rows": len(prompts),
            "char": {
                **_percentiles(char_lens, (0.5, 0.9, 0.95, 0.99, 1.0)),
                "max": max(char_lens) if char_lens else 0,
            },
        }
        if tok is not None:
            token_lens = [len(tok.encode(p, add_special_tokens=False)) for p in prompts]
            n_over = sum(1 for n in token_lens if n > max_prompt_length)
            out["token"] = {
                **_percentiles(token_lens, (0.5, 0.9, 0.95, 0.99, 1.0)),
                "max": max(token_lens) if token_lens else 0,
                "n_over_max": n_over,
                "frac_over_max": n_over / len(token_lens) if token_lens else 0.0,
                "max_prompt_length": max_prompt_length,
                "tokenizer": TOKENIZER_MODEL,
            }
        return out

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_prompt_length": max_prompt_length,
        "truncation_mode": "left",  # mirrors grpo_smoke_1p7b.yaml data.truncation
        "train": _stats_for(pd.read_parquet(train_path)),
        "val": _stats_for(pd.read_parquet(val_path)),
    }
    out_path = train_path.with_name("polaris_prompt_lengths.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return summary


def _print_prompt_summary(summary: dict) -> None:
    for split in ("train", "val"):
        s = summary[split]
        char = s["char"]
        line = (
            f"[prompt-stats:{split}] rows={s['rows']} "
            f"char p50={char['p50']} p95={char['p95']} p99={char['p99']} max={char['max']}"
        )
        if "token" in s:
            t = s["token"]
            line += (
                f" | token p50={t['p50']} p95={t['p95']} p99={t['p99']} max={t['max']} "
                f"over_max({t['max_prompt_length']})={t['n_over_max']} "
                f"({t['frac_over_max']:.2%})"
            )
        print(line)


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
    parser.add_argument(
        "--prompt-stats",
        action="store_true",
        help=(
            "Compute prompt-length distribution and fraction of prompts that exceed "
            "--max-prompt-length (drives verl `truncation: left` overflow rate). "
            "Token stats require the Qwen3 tokenizer; falls back to char-only if "
            "transformers / network are unavailable."
        ),
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=DEFAULT_MAX_PROMPT_LENGTH,
        help="Threshold for prompt-stats overflow reporting (mirror grpo_smoke_1p7b.yaml).",
    )
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

    if args.prompt_stats:
        summary = prompt_length_stats(
            train_path,
            val_path,
            max_prompt_length=args.max_prompt_length,
            use_tokenizer=True,
        )
        _print_prompt_summary(summary)

    if args.upload:
        upload_parquets(train_path, val_path)


if __name__ == "__main__":
    main()
