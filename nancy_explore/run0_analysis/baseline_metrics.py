"""Analysis D — Frozen-eval base-model baseline metrics.

Computes Pass@1, Pass@8, Cover@tau=0.15 (under three substrates),
and worst_subset_accuracy with prompt-level bootstrap 95% CIs.

Reads:
  data/predictions_reparsed.jsonl  -- 4000 rows (500 prompts x 8 rollouts)
  llm_clusters_summary.parquet     -- llm_cluster_id per (prompt_id, rollout_idx)
  pilot/preflight_lock.json        -- metric definitions (for note about k)
  substrate_results.parquet        -- optional, Analysis B embedding clusters

Writes:
  baseline_metrics.md
  baseline_metrics.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "predictions_reparsed.jsonl"
LLM_PARQUET = ROOT / "llm_clusters_summary.parquet"
PREFLIGHT = ROOT.parent.parent / "pilot" / "preflight_lock.json"
SUBSTRATE_PARQUET = ROOT / "substrate_results.parquet"  # optional, from Analysis B
SUBSTRATE_COMPARISON = ROOT / "substrate_comparison.md"
EMBED_CLUSTERS = ROOT / "embed_clusters_at_best_threshold.parquet"  # built by build_embed_clusters.py

OUT_MD = ROOT / "baseline_metrics.md"
OUT_JSON = ROOT / "baseline_metrics.json"

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 0
N_ROLLOUTS = 8
K = 8  # Pass@k
TAU = 0.15
WORST_QUANTILE = 0.25


def load_predictions() -> pd.DataFrame:
    rows = []
    with open(DATA) as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Per-prompt rollout index (assume order in file is rollout order)
    df["rollout_idx"] = df.groupby("prompt_id").cumcount()
    return df


def load_llm() -> pd.DataFrame:
    df = pd.read_parquet(LLM_PARQUET)
    return df[["prompt_id", "rollout_idx", "llm_cluster_id"]]


def load_embedding_clusters() -> tuple[pd.DataFrame | None, str | None]:
    """Load per-rollout embedding cluster assignments at the best threshold from Analysis B.

    Looks for `embed_clusters_at_best_threshold.parquet` (built by `build_embed_clusters.py`
    using the threshold reported in `substrate_comparison.md`).
    Returns (df with columns [prompt_id, rollout_idx, embed_cluster_id], source_column_name)
    or (None, None) if not available.
    """
    if not EMBED_CLUSTERS.exists():
        return None, None
    df = pd.read_parquet(EMBED_CLUSTERS)
    cols = [c for c in df.columns if c.startswith("completion_embedding@")]
    if not cols:
        return None, None
    chosen = cols[0]
    out = df[["prompt_id", "rollout_idx", chosen]].rename(columns={chosen: "embed_cluster_id"})
    return out, chosen


# ---- metric helpers ----------------------------------------------------------

def per_prompt_counts(df: pd.DataFrame, correct_col: str) -> pd.Series:
    """Number of correct rollouts per prompt."""
    return df.groupby("prompt_id")[correct_col].sum()


def per_prompt_pass1(df: pd.DataFrame, correct_col: str) -> pd.Series:
    return df.groupby("prompt_id")[correct_col].mean()


def per_prompt_pass_at_k(df: pd.DataFrame, correct_col: str, n: int, k: int) -> pd.Series:
    """Chen et al. unbiased pass@k per prompt."""
    c = per_prompt_counts(df, correct_col).astype(int)
    # pass@k_i = 1 - C(n-c_i, k) / C(n, k) when n-c_i >= k else 1
    def f(ci):
        if n - ci < k:
            return 1.0
        return 1.0 - math.comb(n - ci, k) / math.comb(n, k)
    return c.map(f)


def per_prompt_cover_largest_correct_mass(
    df: pd.DataFrame, correct_col: str, cluster_col: str, n_rollouts: int = N_ROLLOUTS
) -> pd.Series:
    """For each prompt with >=1 correct rollout, return the mass (count/n) of the
    cluster containing correct rollouts with the largest total mass.
    Prompts with no correct rollouts return NaN (ineligible)."""
    results = {}
    for prompt_id, g in df.groupby("prompt_id"):
        if g[correct_col].sum() == 0:
            results[prompt_id] = np.nan
            continue
        # Find clusters that contain at least one correct rollout.
        correct_clusters = g.loc[g[correct_col].astype(bool), cluster_col].unique()
        # Mass of each cluster = count of rollouts in that cluster / n_rollouts.
        cluster_sizes = g[cluster_col].value_counts()
        # Best (max) cluster mass among the correct-containing clusters.
        max_size = max(cluster_sizes[c] for c in correct_clusters)
        results[prompt_id] = max_size / n_rollouts
    return pd.Series(results)


def bootstrap_ci(
    prompt_ids: list[str], stat_fn: Callable[[list[str]], float],
    n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Prompt-level bootstrap. stat_fn(sampled_prompt_ids) -> scalar."""
    point = stat_fn(prompt_ids)
    rng = np.random.default_rng(seed)
    ids = np.array(prompt_ids)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True).tolist()
        boots[i] = stat_fn(sample)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def mean_over_prompts(series: pd.Series) -> Callable[[list[str]], float]:
    """Return a function that, given a list of prompt_ids, computes mean of series at those ids (with repetition)."""
    s = series  # captured
    def f(ids: list[str]) -> float:
        vals = s.loc[ids].to_numpy()
        # ignore NaN by using nanmean; for cover, ineligible prompts are NaN
        if np.all(np.isnan(vals)):
            return float("nan")
        return float(np.nanmean(vals))
    return f


def cover_at_tau(prompt_mass_series: pd.Series, tau: float) -> Callable[[list[str]], float]:
    """Cover@tau given per-prompt 'largest correct cluster mass' (NaN if ineligible).
    Returns fn mapping list of prompt ids to fraction eligible-and-mass>=tau among eligible prompts."""
    s = prompt_mass_series
    def f(ids: list[str]) -> float:
        vals = s.loc[ids].to_numpy()
        eligible = ~np.isnan(vals)
        if eligible.sum() == 0:
            return float("nan")
        meets = (vals[eligible] >= tau).sum()
        return float(meets) / float(eligible.sum())
    return f


def worst_subset_accuracy_fn(per_prompt_p1: pd.Series, df: pd.DataFrame, correct_col: str,
                              quantile: float = WORST_QUANTILE) -> Callable[[list[str]], float]:
    """Per bootstrap resample: among the resampled prompts, re-rank by per-prompt Pass@1 and
    take the worst `quantile` fraction; report mean Pass@1 over the rollouts of those prompts.
    With sampling with replacement, 'mean Pass@1 over rollouts' for a per-prompt p1 list with
    a possibly multi-counted prompt = mean of per-prompt p1 values weighted equally per
    rollout (8 per prompt), which simplifies to the unweighted mean of the per-prompt p1
    series (since each prompt has identical # rollouts).
    """
    p1 = per_prompt_p1
    def f(ids: list[str]) -> float:
        vals = p1.loc[ids].to_numpy()
        k = max(1, int(math.ceil(quantile * len(vals))))
        # worst = smallest values
        order = np.argsort(vals, kind="stable")
        worst_vals = vals[order[:k]]
        return float(worst_vals.mean())
    return f


# ---- main --------------------------------------------------------------------

def main() -> int:
    df = load_predictions()
    llm = load_llm()
    df = df.merge(llm, on=["prompt_id", "rollout_idx"], how="left")

    prompt_ids = sorted(df["prompt_id"].unique().tolist())
    assert len(prompt_ids) == 500, f"expected 500 prompts, got {len(prompt_ids)}"

    # v1 / v2 correct columns
    df["correct_v1"] = df["correct"].astype(bool)
    df["correct_v2"] = df["is_correct_v2"].astype(bool)

    # ---- Pass@1 ----
    p1_v1 = per_prompt_pass1(df, "correct_v1")
    p1_v2 = per_prompt_pass1(df, "correct_v2")
    pass1_v1 = bootstrap_ci(prompt_ids, mean_over_prompts(p1_v1))
    pass1_v2 = bootstrap_ci(prompt_ids, mean_over_prompts(p1_v2))

    # ---- Pass@k (k=8, n=8 -> collapses to "any correct") ----
    pk_v1 = per_prompt_pass_at_k(df, "correct_v1", n=N_ROLLOUTS, k=K)
    pk_v2 = per_prompt_pass_at_k(df, "correct_v2", n=N_ROLLOUTS, k=K)
    pass_at_k_v1 = bootstrap_ci(prompt_ids, mean_over_prompts(pk_v1))
    pass_at_k_v2 = bootstrap_ci(prompt_ids, mean_over_prompts(pk_v2))

    # ---- Cover@tau under three substrates (v2 only) ----
    cover_results: dict[str, dict] = {}

    # a. answer_loose (cluster_id_v2)
    mass_loose = per_prompt_cover_largest_correct_mass(df, "correct_v2", "cluster_id_v2")
    cover_loose = bootstrap_ci(prompt_ids, cover_at_tau(mass_loose, TAU))
    cover_results["answer_loose"] = {
        "point": cover_loose[0],
        "ci_low": cover_loose[1],
        "ci_high": cover_loose[2],
        "n_eligible": int((~mass_loose.isna()).sum()),
    }

    # b. completion_embedding (from Analysis B if available)
    embed_df, embed_col = load_embedding_clusters()
    if embed_df is not None:
        merged = df.merge(embed_df, on=["prompt_id", "rollout_idx"], how="left")
        if merged["embed_cluster_id"].isna().any():
            print("WARN: some rollouts missing embed_cluster_id; treating each missing as its own singleton")
            merged["embed_cluster_id"] = merged["embed_cluster_id"].astype("object")
            missing_mask = merged["embed_cluster_id"].isna()
            merged.loc[missing_mask, "embed_cluster_id"] = [
                f"_missing_{i}" for i in range(int(missing_mask.sum()))
            ]
        mass_embed = per_prompt_cover_largest_correct_mass(merged, "correct_v2", "embed_cluster_id")
        cover_embed = bootstrap_ci(prompt_ids, cover_at_tau(mass_embed, TAU))
        cover_results["completion_embedding"] = {
            "point": cover_embed[0],
            "ci_low": cover_embed[1],
            "ci_high": cover_embed[2],
            "n_eligible": int((~mass_embed.isna()).sum()),
            "source_column": embed_col,
        }
    else:
        cover_results["completion_embedding"] = {
            "point": None, "ci_low": None, "ci_high": None,
            "note": "TBD pending Analysis B (substrate_results.parquet not found)",
        }

    # c. llm_clusters (treat -1 = degenerate as its own cluster per rollout)
    df_llm = df.copy()
    # Each -1 row gets a unique cluster id so degenerate rollouts don't pool.
    mask_neg = df_llm["llm_cluster_id"] == -1
    df_llm["llm_cluster_id"] = df_llm["llm_cluster_id"].astype("object")
    df_llm.loc[mask_neg, "llm_cluster_id"] = [
        f"degen_{pid}_{i}" for pid, i in zip(
            df_llm.loc[mask_neg, "prompt_id"], df_llm.loc[mask_neg, "rollout_idx"]
        )
    ]
    mass_llm = per_prompt_cover_largest_correct_mass(df_llm, "correct_v2", "llm_cluster_id")
    cover_llm = bootstrap_ci(prompt_ids, cover_at_tau(mass_llm, TAU))
    cover_results["llm_clusters"] = {
        "point": cover_llm[0],
        "ci_low": cover_llm[1],
        "ci_high": cover_llm[2],
        "n_eligible": int((~mass_llm.isna()).sum()),
        "note": "llm_cluster_id == -1 (degenerate) treated as its own cluster per rollout",
    }

    # ---- worst_subset_accuracy ----
    ws_v1 = bootstrap_ci(prompt_ids, worst_subset_accuracy_fn(p1_v1, df, "correct_v1"))
    ws_v2 = bootstrap_ci(prompt_ids, worst_subset_accuracy_fn(p1_v2, df, "correct_v2"))

    # ---- assemble outputs ----
    out = {
        "config": {
            "n_prompts": len(prompt_ids),
            "n_rollouts_per_prompt": N_ROLLOUTS,
            "pass_at_k": K,
            "tau": TAU,
            "worst_quantile": WORST_QUANTILE,
            "bootstrap_n": BOOTSTRAP_N,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_level": 0.95,
        },
        "preflight_lock_note": (
            "preflight_lock.json declares pass_at_k=16 and bootstrap_samples=2000; "
            "the actual rollout count in Run 0 is 8, so we report Pass@8. "
            "We use bootstrap=1000 resamples per the Analysis D design doc."
        ),
        "metrics": {
            "pass_at_1": {
                "v1": {"point": pass1_v1[0], "ci_low": pass1_v1[1], "ci_high": pass1_v1[2]},
                "v2": {"point": pass1_v2[0], "ci_low": pass1_v2[1], "ci_high": pass1_v2[2]},
            },
            "pass_at_8": {
                "v1": {"point": pass_at_k_v1[0], "ci_low": pass_at_k_v1[1], "ci_high": pass_at_k_v1[2]},
                "v2": {"point": pass_at_k_v2[0], "ci_low": pass_at_k_v2[1], "ci_high": pass_at_k_v2[2]},
                "note": "n=8, k=8: Chen et al. 2021 unbiased estimator collapses to 'any rollout correct' per prompt.",
            },
            "cover_at_tau_0.15": cover_results,
            "worst_subset_accuracy": {
                "v1": {"point": ws_v1[0], "ci_low": ws_v1[1], "ci_high": ws_v1[2]},
                "v2": {"point": ws_v2[0], "ci_low": ws_v2[1], "ci_high": ws_v2[2]},
                "note": (
                    "Worst 25% of prompts by per-prompt Pass@1, re-selected within each bootstrap "
                    "resample (not pre-selected on the full set)."
                ),
            },
        },
    }

    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")

    # markdown
    def fmt_pct(d: dict) -> str:
        if d.get("point") is None:
            return d.get("note", "TBD")
        return f"{d['point']*100:.2f}% [{d['ci_low']*100:.2f}%, {d['ci_high']*100:.2f}%]"

    def fmt_pct_simple(p: float, lo: float, hi: float) -> str:
        return f"{p*100:.2f}% [{lo*100:.2f}%, {hi*100:.2f}%]"

    md = []
    md.append("# Analysis D — Frozen-eval base-model baseline")
    md.append("")
    md.append(f"_500 prompts × 8 rollouts; bootstrap 95% CIs, prompt-level resampling, "
              f"{BOOTSTRAP_N} resamples, seed={BOOTSTRAP_SEED}._")
    md.append("")
    md.append("**Lock discrepancy note.** `pilot/preflight_lock.json` records `pass_at_k=16` "
              "(and `bootstrap_samples=2000`), but Run 0 produced only 8 rollouts per prompt. "
              "Per Analysis D design (§D.2) we report **Pass@8** and flag this discrepancy. "
              "Bootstrap is set to 1000 per the Analysis D spec.")
    md.append("")
    md.append("## Headline table")
    md.append("")
    md.append("| Metric | v1 (orig parser) | v2 (fixed parser) |")
    md.append("| --- | --- | --- |")
    md.append(f"| Pass@1 | {fmt_pct_simple(*pass1_v1)} | {fmt_pct_simple(*pass1_v2)} |")
    md.append(f"| Pass@8 | {fmt_pct_simple(*pass_at_k_v1)} | {fmt_pct_simple(*pass_at_k_v2)} |")
    md.append(f"| Cover@τ=0.15 (answer_loose) | n/a | {fmt_pct(cover_results['answer_loose'])} |")
    md.append(f"| Cover@τ=0.15 (completion_embedding) | n/a | {fmt_pct(cover_results['completion_embedding'])} |")
    md.append(f"| Cover@τ=0.15 (llm_clusters) | n/a | {fmt_pct(cover_results['llm_clusters'])} |")
    md.append(f"| worst_subset_accuracy | {fmt_pct_simple(*ws_v1)} | {fmt_pct_simple(*ws_v2)} |")
    md.append("")
    md.append("## Notes on metric definitions")
    md.append("")
    md.append(f"- **Pass@8** uses the Chen et al. (2021) unbiased estimator with n={N_ROLLOUTS} "
              f"sampled completions and k={K}. With n=k=8 this collapses to "
              f"`1 if any rollout correct else 0` per prompt; the prompt-mean is then `Pass@8`.")
    md.append(f"- **Cover@τ** is computed only over prompts with ≥1 correct rollout. "
              f"Mass = `(count of rollouts in the largest cluster that contains any correct rollout) / 8`.")
    md.append(f"  - `answer_loose`: cluster substrate = `cluster_id_v2` (v2 canonical-answer hash).")
    if cover_results['completion_embedding'].get("point") is not None:
        md.append(f"  - `completion_embedding`: cluster substrate built from Analysis B's MiniLM embeddings "
                  f"at the best threshold per `substrate_comparison.md` (column = `{cover_results['completion_embedding'].get('source_column')}`, "
                  f"source: `embed_clusters_at_best_threshold.parquet` produced by `build_embed_clusters.py`).")
    else:
        md.append(f"  - `completion_embedding`: **{cover_results['completion_embedding'].get('note')}**.")
    md.append(f"  - `llm_clusters`: cluster substrate = `llm_cluster_id` from Analysis A. "
              f"Rollouts with `llm_cluster_id == -1` (degenerate) are treated as their own singleton clusters per rollout, "
              f"so they cannot anchor the 'largest correct cluster'.")
    md.append(f"- **worst_subset_accuracy**: within each bootstrap resample, prompts are ranked by per-prompt Pass@1 and the worst 25% are selected; the reported value is the mean per-prompt Pass@1 over that worst quartile. Each bootstrap draw re-selects its own worst 25%.")
    md.append("")
    md.append("## Can claim / cannot claim (per §D.4–D.5)")
    md.append("")
    md.append("**Can claim:** the proxy base model (Qwen3-1.7B-Base) achieves the values in the table above "
              "on the Run 0 prompt set (rows 0–499 of `pilot/data/dapo_slice_3k.jsonl`) under the v2 parser; "
              "any training arm needs to clear these to be a real gain on this distribution.")
    md.append("")
    md.append("**Cannot claim:** generalization to AIME-25 / HMMT / Minerva / MATH-500. "
              "Run 0 uses DaPO-3k proxy training prompts; this is a training-distribution baseline only. "
              "We also cannot claim anything about training-time behavior — these are base-model rollouts.")
    md.append("")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
