#!/usr/bin/env python3
"""E1 — Set-score simulation on Run 0 (cleaned human labels only).

Per prompt: enumerate C(8,4)=70 subsets; score minority set objectives
(ans-avg, ans-rand, cot-avg, cot-rand) and contrast f_poly; convert to
marginal set-RL advantages. Compare to GRPO and inverse_freq.

Run from repo root:
  python nancy_explore/run0_analysis/analysis_c/set_score_simulation.py
"""
from __future__ import annotations

import itertools
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLEANED_PATH = ROOT / "data" / "cleaned_answers.parquet"
LLM_PATH = ROOT / "analysis_a" / "llm_clusters_summary.parquet"

OUT_PARQUET = HERE / "objective_advantages.parquet"
OUT_PEARSON = HERE / "objective_corr_pearson.csv"
OUT_SPEARMAN = HERE / "objective_corr_spearman.csv"
OUT_MD = HERE / "set_score_simulation.md"

N = 8
SUBSET = 4
SUBSETS = list(itertools.combinations(range(N), SUBSET))
SUBSET_ARR = np.array(SUBSETS)
# rollout i appears in C(7,3)=35 subsets
INCL = [np.where((SUBSET_ARR == i).any(axis=1))[0] for i in range(N)]

RNG_SEEDS = 20
ADV_COLS = [
    "adv_grpo",
    "adv_inverse_freq_answer",
    "adv_ans_avg",
    "adv_ans_rand",
    "adv_cot_avg",
    "adv_cot_rand",
    "adv_f_poly",
]


def load_data() -> pd.DataFrame:
    cleaned = pd.read_parquet(CLEANED_PATH)[
        ["prompt_id", "rollout_idx", "cleaned_correct", "cleaned_cluster_id"]
    ]
    llm = pd.read_parquet(LLM_PATH)[["prompt_id", "rollout_idx", "llm_cluster_id"]]
    df = cleaned.merge(llm, on=["prompt_id", "rollout_idx"], how="inner", validate="one_to_one")
    assert len(df) == 4000, f"expected 4000 rows, got {len(df)}"
    assert df["prompt_id"].nunique() == 500

    # Degenerate CoT: sentinel 100 -> -1; all -1 on a prompt = one cluster for counting.
    df["llm_cluster_id"] = df["llm_cluster_id"].replace(100, -1)

    df["cleaned_correct"] = df["cleaned_correct"].astype(int)
    df = df.sort_values(["prompt_id", "rollout_idx"]).reset_index(drop=True)
    return df


def grpo_adv(df: pd.DataFrame) -> np.ndarray:
    means = df.groupby("prompt_id")["cleaned_correct"].transform("mean")
    return (df["cleaned_correct"] - means).to_numpy()


def cluster_size_within_prompt(df: pd.DataFrame, col: str) -> np.ndarray:
    return df.groupby(["prompt_id", col])[col].transform("size").to_numpy()


def inverse_freq_adv(df: pd.DataFrame) -> np.ndarray:
    """Design-doc: A_i = (r_i - mean(r_p)) / cluster_size(cleaned_cluster_id)."""
    grpo = grpo_adv(df)
    sizes = cluster_size_within_prompt(df, "cleaned_cluster_id")
    return grpo / sizes


def minority_f(
    rewards4: np.ndarray,
    clusters4: np.ndarray,
    mode: str,
    rng: random.Random | None = None,
) -> float:
    """Minority set score on a 4-rollout subset G."""
    counts = Counter(clusters4.tolist())
    min_count = min(counts.values())
    rarest = [c for c, cnt in counts.items() if cnt == min_count]
    if mode.endswith("-avg"):
        mask = np.isin(clusters4, rarest)
        return float(rewards4[mask].mean())
    if mode.endswith("-rand"):
        if rng is None:
            raise ValueError("rng required for *-rand")
        pick = rng.choice(rarest)
        return float(rewards4[clusters4 == pick].mean())
    raise ValueError(mode)


def f_poly_score(rewards4: np.ndarray, clusters4: np.ndarray) -> float:
    return float(rewards4.mean() * len(set(clusters4.tolist())) / SUBSET)


def subset_scores_for_prompt(
    rewards: np.ndarray,
    ans_clusters: np.ndarray,
    cot_clusters: np.ndarray,
    rng: random.Random | None,
) -> dict[str, np.ndarray]:
    """Return f(G) arrays length 70 for each scoring variant."""
    n_sub = len(SUBSETS)
    out = {
        "ans-avg": np.zeros(n_sub),
        "ans-rand": np.zeros(n_sub),
        "cot-avg": np.zeros(n_sub),
        "cot-rand": np.zeros(n_sub),
        "f_poly": np.zeros(n_sub),
    }
    for k, idxs in enumerate(SUBSETS):
        r4 = rewards[list(idxs)]
        a4 = ans_clusters[list(idxs)]
        c4 = cot_clusters[list(idxs)]
        out["ans-avg"][k] = minority_f(r4, a4, "ans-avg")
        out["cot-avg"][k] = minority_f(r4, c4, "cot-avg")
        out["f_poly"][k] = f_poly_score(r4, a4)
        if rng is not None:
            out["ans-rand"][k] = minority_f(r4, a4, "ans-rand", rng=rng)
            out["cot-rand"][k] = minority_f(r4, c4, "cot-rand", rng=rng)
    return out


def marginal_from_fG(fG: np.ndarray) -> np.ndarray:
    """Per-rollout marginal set advantage (length 8)."""
    baseline = fG.mean()
    set_adv = fG - baseline
    return np.array([set_adv[INCL[i]].mean() for i in range(N)])


def compute_all_set_advantages(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Fill marginal advantage columns; return per-(prompt,subset) f(G) stacks."""
    n_prompts = df["prompt_id"].nunique()
    n_sub = len(SUBSETS)
    fG_store: dict[str, np.ndarray] = {
        "ans-avg": np.zeros(n_prompts * n_sub),
        "ans-rand": np.zeros(n_prompts * n_sub),
        "cot-avg": np.zeros(n_prompts * n_sub),
        "cot-rand": np.zeros(n_prompts * n_sub),
        "f_poly": np.zeros(n_prompts * n_sub),
    }
    single_mode_mask = np.zeros(n_prompts * n_sub, dtype=bool)

    adv_ans_avg = np.zeros(len(df))
    adv_cot_avg = np.zeros(len(df))
    adv_f_poly = np.zeros(len(df))

    ans_rand_accum = np.zeros((RNG_SEEDS, len(df)))
    cot_rand_accum = np.zeros((RNG_SEEDS, len(df)))
    ans_rand_fG = np.zeros((RNG_SEEDS, n_prompts * n_sub))
    cot_rand_fG = np.zeros((RNG_SEEDS, n_prompts * n_sub))

    p_idx = 0
    for pid, g in df.groupby("prompt_id", sort=False):
        g = g.sort_values("rollout_idx")
        assert len(g) == N
        r = g["cleaned_correct"].to_numpy()
        a = g["cleaned_cluster_id"].to_numpy()
        c = g["llm_cluster_id"].to_numpy()
        idxs_g = g.index.to_numpy()

        scores_det = subset_scores_for_prompt(r, a, c, rng=None)
        for name in ("ans-avg", "cot-avg", "f_poly"):
            fG = scores_det[name]
            sl = slice(p_idx * n_sub, (p_idx + 1) * n_sub)
            fG_store[name][sl] = fG
            marg = marginal_from_fG(fG)
            if name == "ans-avg":
                adv_ans_avg[idxs_g] = marg
            elif name == "cot-avg":
                adv_cot_avg[idxs_g] = marg
            else:
                adv_f_poly[idxs_g] = marg

        for k, subset_idxs in enumerate(SUBSETS):
            a4 = a[list(subset_idxs)]
            single_mode_mask[p_idx * n_sub + k] = len(set(a4.tolist())) == 1

        sl = slice(p_idx * n_sub, (p_idx + 1) * n_sub)
        for s, seed in enumerate(range(RNG_SEEDS)):
            rng = random.Random(seed)
            scores_r = subset_scores_for_prompt(r, a, c, rng=rng)
            ans_rand_accum[s, idxs_g] = marginal_from_fG(scores_r["ans-rand"])
            cot_rand_accum[s, idxs_g] = marginal_from_fG(scores_r["cot-rand"])
            ans_rand_fG[s, sl] = scores_r["ans-rand"]
            cot_rand_fG[s, sl] = scores_r["cot-rand"]

        p_idx += 1

    fG_store["ans-rand"] = ans_rand_fG.mean(axis=0)
    fG_store["cot-rand"] = cot_rand_fG.mean(axis=0)

    df["adv_ans_avg"] = adv_ans_avg
    df["adv_cot_avg"] = adv_cot_avg
    df["adv_f_poly"] = adv_f_poly
    df["adv_ans_rand"] = ans_rand_accum.mean(axis=0)
    df["adv_cot_rand"] = cot_rand_accum.mean(axis=0)
    df["adv_ans_rand_std"] = ans_rand_accum.std(axis=0, ddof=0)
    df["adv_cot_rand_std"] = cot_rand_accum.std(axis=0, ddof=0)

    meta = {
        "fG_store": fG_store,
        "single_mode_mask": single_mode_mask,
        "ans_rand_fG_all": ans_rand_fG,
        "cot_rand_fG_all": cot_rand_fG,
    }
    return df, meta


def corr_matrix(df: pd.DataFrame, cols: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = [n for n, _ in cols]
    k = len(cols)
    mat_p = np.zeros((k, k))
    mat_s = np.zeros((k, k))
    for i, (_, a) in enumerate(cols):
        xi = df[a].to_numpy()
        for j, (_, b) in enumerate(cols):
            yj = df[b].to_numpy()
            mat_p[i, j] = pearsonr(xi, yj)[0]
            mat_s[i, j] = spearmanr(xi, yj)[0]
    return (
        pd.DataFrame(mat_p, index=names, columns=names),
        pd.DataFrame(mat_s, index=names, columns=names),
    )


def fmt_mat(mat: pd.DataFrame) -> str:
    cols = mat.columns.tolist()
    header = "| | " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [header, sep]
    for idx in mat.index:
        row = [idx]
        for c in cols:
            v = mat.loc[idx, c]
            cell = f"{v:+.3f}"
            if abs(v) > 0.9 and idx != c:
                cell = f"**{cell}**"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    df: pd.DataFrame,
    meta: dict,
    pearson_df: pd.DataFrame,
    spearman_df: pd.DataFrame,
    elapsed: float,
    grpo_sum_max: float,
) -> None:
    fG = meta["fG_store"]
    single = meta["single_mode_mask"]
    n_sub_total = len(single)

    def pair_r(a: str, b: str) -> tuple[float, float]:
        return (
            pearsonr(df[a], df[b])[0],
            spearmanr(df[a], df[b])[0],
        )

    q1 = {
        "ans": pair_r("adv_ans_avg", "adv_ans_rand"),
        "cot": pair_r("adv_cot_avg", "adv_cot_rand"),
    }
    q2 = {
        "avg": pair_r("adv_ans_avg", "adv_cot_avg"),
        "rand": pair_r("adv_ans_rand", "adv_cot_rand"),
    }

    # Q3: f(G) level correlations (35k subset scores)
    def fg_corr(a: str, b: str) -> float:
        return pearsonr(fG[a], fG[b])[0]

    q3_fg = {k: fg_corr(k, "f_poly") for k in ("ans-avg", "cot-avg")}
    n_prompts = 500
    n_sub = 70
    high_div = np.zeros(n_prompts * n_sub, dtype=bool)
    zero_minority = {k: np.zeros(n_prompts * n_sub, dtype=bool) for k in ("ans-avg", "cot-avg")}
    p_idx = 0
    for _, g in df.groupby("prompt_id", sort=False):
        g = g.sort_values("rollout_idx")
        a = g["cleaned_cluster_id"].to_numpy()
        for k, idxs in enumerate(SUBSETS):
            a4 = a[list(idxs)]
            sl = p_idx * n_sub + k
            high_div[sl] = len(set(a4.tolist())) == 4
            zero_minority["ans-avg"][sl] = fG["ans-avg"][sl] == 0.0
            zero_minority["cot-avg"][sl] = fG["cot-avg"][sl] == 0.0
        p_idx += 1

    lines = []
    lines.append("# E1 — Set-score simulation (Run 0, cleaned labels)\n")
    lines.append(f"**Runtime:** {elapsed:.1f}s  ")
    lines.append("**Inputs:** `data/cleaned_answers.parquet`, `analysis_a/llm_clusters_summary.parquet`  ")
    lines.append("**Labels:** `cleaned_correct`, `cleaned_cluster_id` only (no v2 parser fields).\n")

    lines.append("## Formula notes\n")
    lines.append("- Set-RL baseline: per-prompt mean of 70 subset scores; marginal adv = mean of `(f(G)-baseline)` over subsets containing rollout i (35 each).")
    lines.append("- `f_poly(G) = mean(r in G) * (distinct cleaned_cluster_id in G) / 4`.")
    lines.append("- `inverse_freq`: `(r_i - mean(r_p)) / cluster_size(cleaned_cluster_id)` (design doc; differs from normalized `objectives.py` weights).")
    lines.append(f"- `*-rand`: {RNG_SEEDS} seeds; reported marginal adv = mean across seeds; `*_rand_std` columns in parquet.\n")

    lines.append("## Sanity\n")
    lines.append(f"- Rows: {len(df)} (500×8). GRPO max |sum-by-prompt|: `{grpo_sum_max:.2e}`.\n")

    lines.append("## Q1 — Rand vs avg tie-break\n")
    lines.append("| Pair | Pearson r | Spearman ρ |")
    lines.append("|---|---|---|")
    lines.append(f"| ans-rand vs ans-avg | **{q1['ans'][0]:+.3f}** | {q1['ans'][1]:+.3f} |")
    lines.append(f"| cot-rand vs cot-avg | **{q1['cot'][0]:+.3f}** | {q1['cot'][1]:+.3f} |")
    lines.append("")
    lines.append("Seed variance on marginal advantages (across 4000 rollouts):")
    lines.append(f"- ans-rand std: mean `{df['adv_ans_rand_std'].mean():.4f}`, max `{df['adv_ans_rand_std'].max():.4f}`")
    lines.append(f"- cot-rand std: mean `{df['adv_cot_rand_std'].mean():.4f}`, max `{df['adv_cot_rand_std'].max():.4f}`\n")

    lines.append("## Q2 — Answer vs CoT cluster mode\n")
    lines.append("| Pair | Pearson r | Spearman ρ |")
    lines.append("|---|---|---|")
    lines.append(f"| ans-avg vs cot-avg | **{q2['avg'][0]:+.3f}** | {q2['avg'][1]:+.3f} |")
    lines.append(f"| ans-rand vs cot-rand (mean) | **{q2['rand'][0]:+.3f}** | {q2['rand'][1]:+.3f} |\n")

    lines.append("## Q3 — Minority set scores vs f_poly\n")
    lines.append("Pearson r between subset-level `f(G)` vectors (35,000 subset scores):\n")
    lines.append("| Minority f | vs f_poly r |")
    lines.append("|---|---|")
    for k, v in q3_fg.items():
        lines.append(f"| {k} | **{v:+.3f}** |")
    lines.append("")
    lines.append("Contingency: **high diversity** (4 distinct answer buckets in G) AND **minority f(G)=0**:\n")
    lines.append("| Minority def | P(zero|high div) | count high∧zero / high div |")
    lines.append("|---|---|---|")
    for k in ("ans-avg", "cot-avg"):
        hd = high_div
        both = hd & zero_minority[k]
        rate = both.sum() / hd.sum() if hd.sum() else 0.0
        lines.append(f"| {k} | **{rate*100:.1f}%** | {both.sum()} / {hd.sum()} |")
    lines.append("")
    fp_zero = fG["f_poly"] == 0.0
    lines.append(f"- f_poly=0 on {fp_zero.mean()*100:.1f}% of subsets; high-div subsets with f_poly=0: {(high_div & fp_zero).sum()} / {high_div.sum()}\n")

    lines.append("## Q4 — Distribution of f(G) on 70 subsets × 500 prompts\n")
    lines.append("(ans-rand / cot-rand: pool 20 seeds × 35k subsets = 700k scores each.)\n")
    lines.append("| Objective | frac f=0 | frac f=1 | mean f | std f |")
    lines.append("|---|---|---|---|---|")
    q4_sources = {
        "ans-avg": fG["ans-avg"],
        "cot-avg": fG["cot-avg"],
        "f_poly": fG["f_poly"],
        "ans-rand": meta["ans_rand_fG_all"].ravel(),
        "cot-rand": meta["cot_rand_fG_all"].ravel(),
    }
    for name, x in q4_sources.items():
        lines.append(
            f"| {name} | {float((x == 0).mean()):.3f} | {float((x == 1).mean()):.3f} | {x.mean():.3f} | {x.std():.3f} |"
        )
    lines.append("")

    lines.append("## Q5 — Single answer-bucket subsets (all 4 rollouts share cleaned_cluster_id)\n")
    frac_single = single.mean()
    lines.append(f"- Fraction of subsets with one answer mode: **{frac_single*100:.1f}%** ({single.sum()} / {n_sub_total}).")
    lines.append("- In that case all four rollouts tie for rarest count; minority f(G) = mean(r in G) for ans-avg/ans-rand (intended).")
    if single.any():
        sa = fG["ans-avg"][single]
        sp = fG["f_poly"][single]
        lines.append(f"- On single-mode subsets: mean ans-avg f = {sa.mean():.3f}; mean f_poly = {sp.mean():.3f}.\n")
    else:
        lines.append("")

    lines.append("## Advantage correlation matrices (4000 rollouts)\n")
    lines.append("### Pearson\n")
    lines.append(fmt_mat(pearson_df))
    lines.append("\n\n### Spearman\n")
    lines.append(fmt_mat(spearman_df))
    lines.append("\n\n*Bold = |r| > 0.9 off-diagonal.*\n")

    lines.append("## Headline vs GRPO / inverse_freq\n")
    lines.append(f"- GRPO ↔ ans-avg: Pearson **{pearson_df.loc['grpo', 'ans_avg']:+.3f}**")
    lines.append(f"- GRPO ↔ f_poly: Pearson **{pearson_df.loc['grpo', 'f_poly']:+.3f}**")
    lines.append(f"- inverse_freq ↔ ans-avg: Pearson **{pearson_df.loc['inverse_freq', 'ans_avg']:+.3f}**")
    lines.append(f"- ans-avg ↔ f_poly: Pearson **{pearson_df.loc['ans_avg', 'f_poly']:+.3f}**\n")

    OUT_MD.write_text("\n".join(lines))


def main() -> None:
    t0 = time.time()
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, {df['prompt_id'].nunique()} prompts")

    print("GRPO + inverse_freq...")
    df["adv_grpo"] = grpo_adv(df)
    df["adv_inverse_freq_answer"] = inverse_freq_adv(df)
    df["cluster_size_answer"] = cluster_size_within_prompt(df, "cleaned_cluster_id")

    print("Set-score simulation (70 subsets × 500 prompts)...")
    df, meta = compute_all_set_advantages(df)

    grpo_sum_max = df.groupby("prompt_id")["adv_grpo"].sum().abs().max()

    out_cols = [
        "prompt_id",
        "rollout_idx",
        "cleaned_correct",
        "cleaned_cluster_id",
        "llm_cluster_id",
        "cluster_size_answer",
        "adv_grpo",
        "adv_inverse_freq_answer",
        "adv_ans_avg",
        "adv_ans_rand",
        "adv_ans_rand_std",
        "adv_cot_avg",
        "adv_cot_rand",
        "adv_cot_rand_std",
        "adv_f_poly",
    ]
    df[out_cols].to_parquet(OUT_PARQUET, index=False)
    print(f"Wrote {OUT_PARQUET}")

    corr_cols = [
        ("grpo", "adv_grpo"),
        ("inverse_freq", "adv_inverse_freq_answer"),
        ("ans_avg", "adv_ans_avg"),
        ("ans_rand", "adv_ans_rand"),
        ("cot_avg", "adv_cot_avg"),
        ("cot_rand", "adv_cot_rand"),
        ("f_poly", "adv_f_poly"),
    ]
    pearson_df, spearman_df = corr_matrix(df, corr_cols)
    pearson_df.to_csv(OUT_PEARSON)
    spearman_df.to_csv(OUT_SPEARMAN)
    print(f"Wrote {OUT_PEARSON}, {OUT_SPEARMAN}")

    elapsed = time.time() - t0
    write_report(df, meta, pearson_df, spearman_df, elapsed, grpo_sum_max)
    print(f"Wrote {OUT_MD}")
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
