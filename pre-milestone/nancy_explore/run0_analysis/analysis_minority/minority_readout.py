#!/usr/bin/env python3
"""Phase 1 — minority-correct readout (Q II).

Reads human-verified `cleaned_answers.parquet` and LLM cluster assignments from
`analysis_a/llm_clusters_summary.parquet`. Writes headline metrics + distribution plot.

No API calls.
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLEANED_PARQUET = ROOT / "data" / "cleaned_answers.parquet"
LLM_PARQUET = ROOT / "analysis_a" / "llm_clusters_summary.parquet"
OUT_MD = HERE / "minority_metrics.md"
OUT_PNG = HERE / "minority_distributions.png"
OUT_COMPOSITION = HERE / "minority_composition_correct_llm.png"
OUT_COMPOSITION_ALL500 = HERE / "minority_composition_all500_llm.png"

# Fun/exploratory stacked composition strips.
COMPOSITION_PX_PER_BAR = 22
COMPOSITION_DPI = 300
COMPOSITION_FIG_HEIGHT = 8.0
COMPOSITION_FONT_TITLE = 26
COMPOSITION_FONT_LABEL = 22
COMPOSITION_FONT_TICK = 18
COMPOSITION_RANK_DARK = [
    "#1f4e79",
    "#c45c26",
    "#2d6a4f",
    "#9b2226",
    "#6a4c93",
    "#0077b6",
    "#bc6c25",
    "#588157",
]
COMPOSITION_LIGHTEN = 0.62  # wrong-answer segments: blend base hue toward white
COMPOSITION_SEGMENT_EDGE = "#f5f5f5"

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0
N_ROLLOUTS = 8
K_PASS = 8
POLY_EPO_DEGENERATE = 100

REPO = ROOT.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from pilot.train.run_proxy import has_minority_correct_cluster  # noqa: E402


def normalize_llm_cluster_id(cid: int) -> int:
    if cid == POLY_EPO_DEGENERATE:
        return -1
    return int(cid)


def largest_correct_cluster_size(cluster_ids: list, correct: list[bool]) -> int:
    if not any(correct):
        return 0
    correct_clusters = {cid for cid, ok in zip(cluster_ids, correct) if ok}
    counts = Counter(cluster_ids)
    return max(counts[c] for c in correct_clusters)


def per_prompt_pass_at_k(correct: np.ndarray, n: int, k: int) -> float:
    c = int(correct.sum())
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def bootstrap_ci(flags: list[bool], n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    if not flags:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(flags, dtype=float)
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = sample.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_ci_pass_at_k(pass_vals: list[float], n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    if not pass_vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(pass_vals, dtype=float)
    point = float(arr.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boots[i] = sample.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_merged() -> pd.DataFrame:
    cleaned = pd.read_parquet(CLEANED_PARQUET)
    llm = pd.read_parquet(LLM_PARQUET)[["prompt_id", "rollout_idx", "llm_cluster_id"]]
    llm["llm_cluster_id"] = llm["llm_cluster_id"].map(normalize_llm_cluster_id)
    df = cleaned.merge(llm, on=["prompt_id", "rollout_idx"], how="left", validate="one_to_one")
    if df["llm_cluster_id"].isna().any():
        raise SystemExit("merge left missing llm_cluster_id rows")
    return df


def n_distinct_clusters(cluster_ids: list[int], mask: list[bool]) -> int:
    """Distinct cluster IDs among rollouts where mask is True."""
    sub = {cid for cid, ok in zip(cluster_ids, mask) if ok}
    return len(sub)


def cluster_count_summary(values: np.ndarray) -> tuple[float, float, int, int]:
    """Median, mean, min, max for a 1d array of per-prompt cluster counts."""
    if len(values) == 0:
        return float("nan"), float("nan"), 0, 0
    return float(np.median(values)), float(values.mean()), int(values.min()), int(values.max())


def build_prompt_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prompt_id, g in df.groupby("prompt_id", sort=True):
        g = g.sort_values("rollout_idx")
        correct = g["cleaned_correct"].astype(bool).tolist()
        cleaned_ids = g["cleaned_cluster_id"].astype(int).tolist()
        llm_ids = g["llm_cluster_id"].astype(int).tolist()
        n_correct = sum(correct)
        incorrect = [not c for c in correct]
        eligible = n_correct >= 1
        rows.append(
            {
                "prompt_id": prompt_id,
                "n_correct": n_correct,
                "eligible": eligible,
                "pass1": float(np.mean(correct)),
                "pass8": per_prompt_pass_at_k(np.array(correct), N_ROLLOUTS, K_PASS),
                "n_clusters_cleaned": len(set(cleaned_ids)),
                "n_clusters_llm": len(set(llm_ids)),
                "n_clusters_cleaned_correct": n_distinct_clusters(cleaned_ids, correct),
                "n_clusters_llm_correct": n_distinct_clusters(llm_ids, correct),
                "n_clusters_cleaned_incorrect": n_distinct_clusters(cleaned_ids, incorrect),
                "n_clusters_llm_incorrect": n_distinct_clusters(llm_ids, incorrect),
                "largest_correct_cleaned": largest_correct_cluster_size(cleaned_ids, correct),
                "largest_correct_llm": largest_correct_cluster_size(llm_ids, correct),
                "minority_correct_cleaned": has_minority_correct_cluster(correct, cleaned_ids) if eligible else False,
                "minority_correct_llm": has_minority_correct_cluster(correct, llm_ids) if eligible else False,
            }
        )
    return pd.DataFrame(rows)


def write_metrics_md(pp: pd.DataFrame, df: pd.DataFrame) -> None:
    n_prompts = len(pp)
    n_rollouts = len(df)
    pass1 = float(df["cleaned_correct"].mean())
    pass8_vals = pp["pass8"].tolist()
    pass8, pass8_lo, pass8_hi = bootstrap_ci_pass_at_k(pass8_vals)

    n_any_correct = int((pp["n_correct"] >= 1).sum())
    n_all_incorrect = int((pp["n_correct"] == 0).sum())

    strata = [
        (
            "All prompts (8 rollouts each)",
            pp,
            "n_clusters_cleaned",
            "n_clusters_llm",
        ),
        (
            "≥1 correct prompt (correct rollouts only)",
            pp[pp["n_correct"] >= 1],
            "n_clusters_cleaned_correct",
            "n_clusters_llm_correct",
        ),
        (
            "All-incorrect prompt (8 rollouts each)",
            pp[pp["n_correct"] == 0],
            "n_clusters_cleaned",
            "n_clusters_llm",
        ),
    ]

    lines = [
        "# Phase 1 — Run 0 cluster readout\n",
        f"**Generated:** {datetime.now(timezone.utc).date().isoformat()}  \n",
        f"**Ground truth:** `data/cleaned_answers.parquet`  \n",
        f"**LLM clusters:** `analysis_a/llm_clusters_summary.parquet` "
        f"(degenerate `{POLY_EPO_DEGENERATE}` → `-1`; all `-1` in a prompt = one cluster)  \n",
        "\n## Pass@k (human-verified correctness)\n",
        "\n| Metric | Value | 95% bootstrap CI |\n|---|---:|---:|\n",
        f"| Prompts | {n_prompts} | — |\n",
        f"| Rollouts | {n_rollouts} | — |\n",
        f"| **Pass@1** | **{100 * pass1:.2f}%** ({int(df['cleaned_correct'].sum())}/{n_rollouts}) | — |\n",
        f"| **Pass@8** | **{100 * pass8:.2f}%** | [{100 * pass8_lo:.2f}%, {100 * pass8_hi:.2f}%] |\n",
        "\nPass@8: Chen et al. unbiased Pass@k per prompt (k=8), prompt-level bootstrap "
        f"({BOOTSTRAP_N} resamples, seed={BOOTSTRAP_SEED}).\n",
        "\n## Methods — distinct clusters per prompt\n",
        "For each of 500 prompts (8 rollouts), define a per-prompt count = "
        "|{distinct cluster IDs among rollouts in that row's scope}|.\n",
        "- **Answer-hash:** `cleaned_cluster_id` (canonical human-verified answer string).\n",
        "- **LLM reasoning:** `llm_cluster_id` (`100` → `-1`; all `-1` on a prompt counts as one cluster).\n",
        "\n**Row scopes (prompt cohorts are disjoint for correct vs all-incorrect):**\n",
        "1. **All prompts** — all 8 rollouts (n=500).\n",
        f"2. **≥1 correct prompt** — only the {n_any_correct} prompts with ≥1 `cleaned_correct` rollout; "
        "count distinct clusters among **correct rollouts only** on that prompt "
        "(wrong rollouts on the same prompt are ignored).\n",
        f"3. **All-incorrect prompt** — the complementary {n_all_incorrect} prompts with "
        "**zero** correct rollouts; count distinct clusters among **all 8** rollouts "
        "(every rollout is incorrect). No prompt has all 8 correct on this run.\n",
        "\nMedian / mean / range are taken over prompts in that cohort. "
        "Mixed prompts (some correct, some wrong) appear only in row 2 for the correct-rollout "
        "count; their incorrect rollouts are **not** included in row 3.\n",
        "\n## Distinct clusters per prompt\n",
        "\n| Stratum | n prompts | Substrate | Median | Mean | Range |\n",
        "|---|--:|---|---:|---:|---:|\n",
    ]
    for label, sub, col_ans, col_llm in strata:
        n_sub = len(sub)
        for substrate, col in [("Answer-hash", col_ans), ("LLM reasoning", col_llm)]:
            med, mean, lo, hi = cluster_count_summary(sub[col].to_numpy())
            lines.append(
                f"| {label} | {n_sub} | {substrate} | **{med:.0f}** | {mean:.2f} | {lo}–{hi} |\n"
            )
    OUT_MD.write_text("".join(lines))
    print(f"Wrote {OUT_MD}")


def cluster_size_tuple(cluster_ids: list) -> tuple[int, ...]:
    """Sorted cluster sizes (desc) over all rollouts in a prompt."""
    return tuple(sorted(Counter(cluster_ids).values(), reverse=True))


def correct_cluster_size_tuple(cluster_ids: list, correct: list[bool]) -> tuple[int, ...]:
    """Sorted cluster sizes (desc) among rollouts with correct answers only."""
    freq = Counter(cid for cid, ok in zip(cluster_ids, correct) if ok)
    return tuple(sorted(freq.values(), reverse=True))


def _lighten_hex(hex_color: str, amount: float) -> str:
    """Blend hex toward white; amount in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))

    def mix(c: int) -> int:
        return int(c + (255 - c) * amount)

    return f"#{mix(r):02x}{mix(g):02x}{mix(b):02x}"


COMPOSITION_RANK_LIGHT = [_lighten_hex(c, COMPOSITION_LIGHTEN) for c in COMPOSITION_RANK_DARK]


def rollout_stack_order(cluster_ids: list[int], correct: list[bool]) -> list[int]:
    """Rollout indices 0..7: clusters largest-first; correct rollouts at bottom of each cluster."""
    by_cluster: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        by_cluster.setdefault(cid, []).append(idx)
    cluster_order = sorted(
        by_cluster.keys(),
        key=lambda c: (-len(by_cluster[c]), c),
    )
    order: list[int] = []
    for cid in cluster_order:
        members = by_cluster[cid]
        # False < True for (not correct): correct answers stack at bottom of cluster block.
        members.sort(key=lambda i: (not correct[i], i))
        order.extend(members)
    return order


def segment_fill_color(rank: int, is_correct_answer: bool) -> str:
    base = COMPOSITION_RANK_DARK if is_correct_answer else COMPOSITION_RANK_LIGHT
    return base[rank % len(base)]


def _composition_sort_key(llm_ids: list[int], correct: list[bool]) -> tuple:
    """Primary: all-rollout cluster sizes; secondary: correct-answer cluster sizes; tertiary: # correct."""
    return (
        cluster_size_tuple(llm_ids),
        correct_cluster_size_tuple(llm_ids, correct),
        sum(correct),
    )


def _build_composition_records(df: pd.DataFrame, *, eligible_only: bool) -> list[dict]:
    records = []
    for prompt_id, g in df.groupby("prompt_id", sort=True):
        g = g.sort_values("rollout_idx")
        if len(g) != N_ROLLOUTS:
            continue
        correct = g["cleaned_correct"].astype(bool).tolist()
        if eligible_only and not any(correct):
            continue
        llm_ids = g["llm_cluster_id"].astype(int).tolist()
        order = rollout_stack_order(llm_ids, correct)
        cluster_sizes = Counter(llm_ids)
        cid_to_rank = {
            cid: i
            for i, cid in enumerate(
                sorted(cluster_sizes.keys(), key=lambda c: (-cluster_sizes[c], c))
            )
        }
        records.append(
            {
                "prompt_id": prompt_id,
                "sort_key": _composition_sort_key(llm_ids, correct),
                "segment_ranks": [cid_to_rank[llm_ids[i]] for i in order],
                "segment_correct": [correct[i] for i in order],
                "minority_llm": has_minority_correct_cluster(correct, llm_ids),
                "n_correct": sum(correct),
            }
        )
    records.sort(key=lambda r: r["sort_key"], reverse=True)
    return records


def write_composition_llm(
    df: pd.DataFrame,
    *,
    out_path: Path,
    eligible_only: bool,
    title_line: str,
) -> None:
    """Eight segments per prompt (one per rollout), LLM cluster colors, sorted bars."""
    records = _build_composition_records(df, eligible_only=eligible_only)
    n = len(records)

    width_in = n * COMPOSITION_PX_PER_BAR / COMPOSITION_DPI + 2.0
    fig, ax = plt.subplots(figsize=(width_in, COMPOSITION_FIG_HEIGHT), dpi=COMPOSITION_DPI)
    x = np.arange(n)
    bottom = np.zeros(n, dtype=float)

    for seg in range(N_ROLLOUTS):
        colors = [
            segment_fill_color(r["segment_ranks"][seg], r["segment_correct"][seg])
            for r in records
        ]
        ax.bar(
            x,
            np.ones(n),
            bottom=bottom,
            width=0.92,
            color=colors,
            edgecolor=COMPOSITION_SEGMENT_EDGE,
            linewidth=0.12,
        )
        bottom += 1.0

    n_minority = sum(1 for r in records if r["minority_llm"])
    n_with_correct = sum(1 for r in records if r["n_correct"] > 0)
    ax.set_title(
        f"{title_line}\n"
        f"Dark = correct answer · light = wrong · hue = LLM cluster · "
        f"correct at bottom of each cluster · "
        f"sort: cluster sizes → correct-answer cluster sizes → # correct · "
        f"{n_with_correct} w/ correct · {n_minority} minority-LLM",
        fontsize=COMPOSITION_FONT_TITLE,
        pad=16,
    )
    ax.set_xlabel(
        "Prompts (left → right: larger LLM clusters, then richer correct-answer structure)",
        fontsize=COMPOSITION_FONT_LABEL,
        labelpad=12,
    )
    ax.set_ylabel("8 rollouts / prompt", fontsize=COMPOSITION_FONT_LABEL, labelpad=12)
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, N_ROLLOUTS + 0.5)
    ax.set_yticks(range(0, N_ROLLOUTS + 1))
    ax.tick_params(axis="y", labelsize=COMPOSITION_FONT_TICK)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=COMPOSITION_DPI, bbox_inches="tight")
    px_w = int(width_in * COMPOSITION_DPI)
    plt.close(fig)
    print(f"Wrote {out_path} (~{px_w}px wide @ {COMPOSITION_DPI} dpi, n={n})")


def write_composition_correct_llm(df: pd.DataFrame) -> None:
    write_composition_llm(
        df,
        out_path=OUT_COMPOSITION,
        eligible_only=True,
        title_line=f"Run 0 — LLM rollout composition (eligible prompts, n≤172)",
    )
    write_composition_llm(
        df,
        out_path=OUT_COMPOSITION_ALL500,
        eligible_only=False,
        title_line="Run 0 — LLM rollout composition (all 500 prompts)",
    )


def write_distributions_png(pp: pd.DataFrame) -> None:
    elig = pp[pp["eligible"]]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Run 0 — per-prompt cluster structure (Phase 1)", fontsize=12)

    axes[0, 0].hist(pp["n_clusters_cleaned"], bins=range(1, 10), align="left", color="#4c78a8", edgecolor="white")
    axes[0, 0].set_title("Answer-hash: # clusters / prompt (all prompts)")
    axes[0, 0].set_xlabel("# distinct cleaned_cluster_id")
    axes[0, 0].set_ylabel("# prompts")

    axes[0, 1].hist(pp["n_clusters_llm"], bins=range(1, 10), align="left", color="#f58518", edgecolor="white")
    axes[0, 1].set_title("LLM: # clusters / prompt (all prompts)")
    axes[0, 1].set_xlabel("# distinct llm_cluster_id")
    axes[0, 1].set_ylabel("# prompts")

    axes[1, 0].hist(
        elig["largest_correct_cleaned"],
        bins=range(0, N_ROLLOUTS + 2),
        align="left",
        color="#4c78a8",
        edgecolor="white",
    )
    axes[1, 0].set_title(f"Largest correct cluster size — answer-hash (n={len(elig)} eligible)")
    axes[1, 0].set_xlabel("rollouts in largest correct cluster")
    axes[1, 0].set_ylabel("# prompts")

    axes[1, 1].hist(
        elig["largest_correct_llm"],
        bins=range(0, N_ROLLOUTS + 2),
        align="left",
        color="#f58518",
        edgecolor="white",
    )
    axes[1, 1].set_title(f"Largest correct cluster size — LLM (n={len(elig)} eligible)")
    axes[1, 1].set_xlabel("rollouts in largest correct cluster")
    axes[1, 1].set_ylabel("# prompts")

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


def main() -> None:
    df = load_merged()
    pp = build_prompt_table(df)
    write_metrics_md(pp, df)
    write_distributions_png(pp)
    write_composition_correct_llm(df)

    pass1 = df["cleaned_correct"].mean()
    print(
        f"Sanity: Pass@1={100 * pass1:.2f}% Pass@8={100 * pp['pass8'].mean():.2f}% "
        f"clusters(all) ans_med={pp['n_clusters_cleaned'].median():.0f} "
        f"llm_med={pp['n_clusters_llm'].median():.0f}"
    )


if __name__ == "__main__":
    main()
