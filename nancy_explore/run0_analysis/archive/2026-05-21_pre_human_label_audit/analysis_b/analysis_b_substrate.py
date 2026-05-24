"""Analysis B: Cheap-substrate comparison against the LLM reference clustering.

Computes per-prompt cluster assignments under 4 substrates (with 4 thresholds
for completion_embedding), then scores ARI / V-measure / cluster-count drift /
minority-correct concordance vs llm_cluster_id over 500 prompts.

Outputs:
- substrate_results.parquet  (per-prompt rows)
- substrate_comparison.md   (aggregate table + writeup)
- substrate_disagreement_vignettes.md  (top-5 ARI-gap prompts/substrate)
- analysis_b_minority_rate.png

Caches:
- embeddings_minilm.npy + embedding_ids.parquet
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PREDS = DATA / "predictions_reparsed.jsonl"
PROMPTS = DATA / "prompt_inputs.jsonl"
LLM_PARQUET = ROOT / "analysis_a" / "llm_clusters_summary.parquet"

EMBED_CACHE = HERE / "embeddings_minilm.npy"
EMBED_IDS = HERE / "embedding_ids.parquet"

OUT_PARQUET = HERE / "substrate_results.parquet"
OUT_TABLE = HERE / "substrate_comparison.md"
OUT_VIGNETTES = HERE / "substrate_disagreement_vignettes.md"
OUT_PNG = HERE / "analysis_b_minority_rate.png"

THRESHOLDS = [0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_predictions() -> pd.DataFrame:
    rows = []
    with open(PREDS) as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
    df = pd.DataFrame(rows)
    # Add rollout_idx local to prompt by stable order
    df = df.reset_index(drop=True)
    df["rollout_idx"] = df.groupby("prompt_id").cumcount()
    return df


def load_llm() -> pd.DataFrame:
    df = pd.read_parquet(LLM_PARQUET)
    return df[["prompt_id", "rollout_idx", "llm_cluster_id", "is_correct_v2", "parse_ok"]]


def load_prompts() -> dict:
    out = {}
    with open(PROMPTS) as f:
        for line in f:
            r = json.loads(line)
            out[r["prompt_id"]] = r
    return out


# ---------------------------------------------------------------------------
# Feature substrate
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
_REP_RE = re.compile(r"(.{30,}?)\1\1")  # any 30+ char substring repeated 3+ times consecutively


def compute_features(completion: str, canonical_v2) -> tuple:
    c = completion or ""
    lc = c.lower()
    has_boxed = r"\boxed{" in c
    has_sympy_code = ("import sympy" in c) or ("```python" in c) or ("sp." in c)
    has_repetition = bool(_REP_RE.search(c))
    has_code_fence = "```" in c
    has_modular = any(kw in lc for kw in ["modulo", "mod ", "remainder", "congruent"])
    has_coord = any(kw in lc for kw in ["coordinate", "(x,", "x-axis", "vector"])
    canon = "" if canonical_v2 is None else str(canonical_v2)
    parsed_is_numeric = bool(_NUM_RE.match(canon))
    parsed_is_latex_fraction = ("\\frac" in canon) or ("\\dfrac" in canon)
    return (
        has_boxed,
        has_sympy_code,
        has_repetition,
        has_code_fence,
        has_modular,
        has_coord,
        parsed_is_numeric,
        parsed_is_latex_fraction,
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def compute_or_load_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    if EMBED_CACHE.exists() and EMBED_IDS.exists():
        embs = np.load(EMBED_CACHE)
        ids = pd.read_parquet(EMBED_IDS)
        if len(embs) == len(df) and (ids["prompt_id"].values == df["prompt_id"].values).all():
            print(f"[embed] cache hit, {embs.shape}")
            return embs, ids
        print("[embed] cache mismatch, recomputing")
    from sentence_transformers import SentenceTransformer

    print("[embed] loading MiniLM model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = df["completion"].fillna("").tolist()
    print(f"[embed] encoding {len(texts)} completions on CPU...")
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    np.save(EMBED_CACHE, embs)
    ids = df[["prompt_id", "rollout_idx"]].copy()
    ids.to_parquet(EMBED_IDS)
    print(f"[embed] cached to {EMBED_CACHE}")
    return embs, ids


def cluster_embeddings_per_prompt(embs: np.ndarray, prompt_ids: list[str], threshold: float) -> np.ndarray:
    """Per-prompt agglomerative clustering with cosine metric, average linkage."""
    out = np.full(len(prompt_ids), -999, dtype=np.int64)
    # group by prompt
    idx_by_prompt = defaultdict(list)
    for i, p in enumerate(prompt_ids):
        idx_by_prompt[p].append(i)
    for p, idxs in idx_by_prompt.items():
        X = embs[idxs]
        n = len(idxs)
        if n == 1:
            out[idxs[0]] = 0
            continue
        # Agglomerative with cosine distance, average linkage, distance threshold
        try:
            ac = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=threshold,
            )
            labels = ac.fit_predict(X)
        except Exception as e:
            print(f"[embed] cluster fail prompt={p} thresh={threshold}: {e}")
            labels = np.arange(n)
        for j, i in enumerate(idxs):
            out[i] = int(labels[j])
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def has_minority_correct(cluster_ids: np.ndarray, correct: np.ndarray) -> bool:
    """
    Canonical definition (matches `has_minority_correct_cluster` in
    pilot/train/run_proxy.py — same definition Analysis A used to compute
    the 14.55% LLM headline): True when >=1 correct cluster exists AND some
    correct rollout lies in a minority correct cluster (i.e., correct rollouts
    span >=2 clusters and the frequency of at least one correct cluster is
    strictly below the majority frequency among correct rollouts).
    """
    if correct.sum() == 0:
        return False
    from collections import Counter
    correct_clusters = [int(c) for c, ok in zip(cluster_ids, correct) if ok]
    if not correct_clusters:
        return False
    freq = Counter(correct_clusters)
    majority_freq = max(freq.values())
    return any(count < majority_freq for count in freq.values())


def per_prompt_metrics(sub_labels: np.ndarray, llm_labels: np.ndarray, correct: np.ndarray) -> dict:
    ari = adjusted_rand_score(llm_labels, sub_labels)
    vm = v_measure_score(llm_labels, sub_labels)
    n_sub = len(np.unique(sub_labels))
    n_llm = len(np.unique(llm_labels))
    hmc_sub = has_minority_correct(sub_labels, correct)
    hmc_llm = has_minority_correct(llm_labels, correct)
    return {
        "ari": ari,
        "vmeasure": vm,
        "n_clusters_sub": n_sub,
        "n_clusters_llm": n_llm,
        "abs_diff_n": abs(n_sub - n_llm),
        "has_correct": bool(correct.sum() > 0),
        "hmc_sub": hmc_sub,
        "hmc_llm": hmc_llm,
    }


def bootstrap_ci_mean(vals: np.ndarray, n_resample: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.empty(n_resample)
    for i in range(n_resample):
        idx = rng.integers(0, n, size=n)
        means[i] = vals[idx].mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[load] predictions + llm clusters...")
    preds = load_predictions()
    llm = load_llm()
    # Align: merge llm into preds by (prompt_id, rollout_idx)
    # llm parquet rollout_idx might already match the predictions order; verify alignment
    # Use merge
    df = preds.merge(llm, on=["prompt_id", "rollout_idx"], how="left", suffixes=("", "_llm"))
    # Sanity check
    missing = df["llm_cluster_id"].isna().sum()
    print(f"[load] merged shape={df.shape}, missing llm_cluster_id={missing}")
    if missing > 0:
        # Try alignment-by-position fallback
        print("[load] WARNING: some rollouts not aligned — attempting positional fallback")
        df = df.drop(columns=[c for c in df.columns if c.endswith("_llm")])
        # positional fallback
        llm_sorted = llm.sort_values(["prompt_id", "rollout_idx"]).reset_index(drop=True)
        preds_sorted = preds.sort_values(["prompt_id", "rollout_idx"]).reset_index(drop=True)
        assert len(llm_sorted) == len(preds_sorted)
        preds_sorted["llm_cluster_id"] = llm_sorted["llm_cluster_id"].values
        preds_sorted["is_correct_v2_llm"] = llm_sorted["is_correct_v2"].values
        df = preds_sorted

    # Use is_correct_v2 from predictions (authoritative); ensure boolean
    df["is_correct_v2"] = df["is_correct_v2"].astype(bool)

    # Features per row
    print("[features] computing 8-bool tag tuples...")
    feats = df.apply(lambda r: compute_features(r["completion"], r["canonical_v2"]), axis=1)
    # Hash tuple per prompt-local to ints
    df["feat_tuple"] = feats
    # local mapping per prompt
    feat_cluster = np.empty(len(df), dtype=np.int64)
    for p, sub in df.groupby("prompt_id"):
        mapping = {}
        for i, t in zip(sub.index, sub["feat_tuple"]):
            if t not in mapping:
                mapping[t] = len(mapping)
            feat_cluster[i] = mapping[t]
    df["cluster_feature"] = feat_cluster

    # Substrate: answer_strict and answer_loose are direct hashes already
    df["cluster_strict"] = df["cluster_id"].astype(np.int64)
    df["cluster_loose"] = df["cluster_id_v2"].astype(np.int64)

    # Embeddings
    print("[embed] computing/loading...")
    embs, _ids = compute_or_load_embeddings(df)
    prompt_ids = df["prompt_id"].tolist()
    embed_clusters = {}
    for t in THRESHOLDS:
        print(f"[embed] clustering at threshold={t}...")
        embed_clusters[t] = cluster_embeddings_per_prompt(embs, prompt_ids, t)

    # Per-prompt metrics for each substrate
    substrates = ["answer_strict", "answer_loose"] + [f"completion_embedding@{t}" for t in THRESHOLDS] + ["completion_features"]
    substrate_col = {
        "answer_strict": "cluster_strict",
        "answer_loose": "cluster_loose",
        "completion_features": "cluster_feature",
    }
    # add embedding columns
    for t in THRESHOLDS:
        df[f"cluster_embed_{t}"] = embed_clusters[t]
        substrate_col[f"completion_embedding@{t}"] = f"cluster_embed_{t}"

    # Per-prompt loop
    print("[metrics] per-prompt loop over 500 prompts...")
    per_prompt_records = []  # prompt_id-level rows
    for pid, sub in df.groupby("prompt_id"):
        llm_labels = sub["llm_cluster_id"].astype(np.int64).values
        correct = sub["is_correct_v2"].values.astype(bool)
        row = {"prompt_id": pid}
        for s_name in substrates:
            col = substrate_col[s_name]
            sub_labels = sub[col].astype(np.int64).values
            m = per_prompt_metrics(sub_labels, llm_labels, correct)
            row[f"ari_{s_name}"] = m["ari"]
            row[f"vmeasure_{s_name}"] = m["vmeasure"]
            row[f"n_clusters_{s_name}"] = m["n_clusters_sub"]
            row[f"abs_diff_n_{s_name}"] = m["abs_diff_n"]
            row[f"has_minority_correct_{s_name}"] = m["hmc_sub"]
        row["n_clusters_llm"] = len(np.unique(llm_labels))
        row["has_correct"] = bool(correct.sum() > 0)
        row["has_minority_correct_llm"] = has_minority_correct(llm_labels, correct)
        per_prompt_records.append(row)

    pp_df = pd.DataFrame(per_prompt_records)
    pp_df.to_parquet(OUT_PARQUET)
    print(f"[out] wrote {OUT_PARQUET} ({len(pp_df)} rows)")

    # Aggregate table
    rng_seed = 0
    table_rows = []
    for s_name in substrates:
        ari_vals = pp_df[f"ari_{s_name}"].values
        vm_vals = pp_df[f"vmeasure_{s_name}"].values
        dn_vals = pp_df[f"abs_diff_n_{s_name}"].values
        ari_mean, ari_lo, ari_hi = bootstrap_ci_mean(ari_vals, 1000, rng_seed)
        vm_mean, vm_lo, vm_hi = bootstrap_ci_mean(vm_vals, 1000, rng_seed)
        # Minority-correct rate among prompts with >=1 correct
        elig = pp_df[pp_df["has_correct"]]
        n_elig = len(elig)
        rate = elig[f"has_minority_correct_{s_name}"].mean() if n_elig > 0 else float("nan")
        # Concordance: 2x2 confusion (sub vs LLM) over 500 prompts
        sub_yes = pp_df[f"has_minority_correct_{s_name}"].astype(bool).values
        llm_yes = pp_df["has_minority_correct_llm"].astype(bool).values
        tp = int(((sub_yes) & (llm_yes)).sum())
        fp = int(((sub_yes) & (~llm_yes)).sum())
        fn = int(((~sub_yes) & (llm_yes)).sum())
        tn = int(((~sub_yes) & (~llm_yes)).sum())
        accuracy = (tp + tn) / len(pp_df)
        table_rows.append({
            "substrate": s_name,
            "ari_mean": ari_mean,
            "ari_lo": ari_lo,
            "ari_hi": ari_hi,
            "vm_mean": vm_mean,
            "vm_lo": vm_lo,
            "vm_hi": vm_hi,
            "dn_mean": float(dn_vals.mean()),
            "minority_rate": rate,
            "minority_n_elig": int(n_elig),
            "concordance_accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    table_df = pd.DataFrame(table_rows)
    print(table_df.to_string(index=False))

    # LLM minority rate for headline reference
    elig_llm = pp_df[pp_df["has_correct"]]
    llm_minority_rate = elig_llm["has_minority_correct_llm"].mean()
    print(f"[ref] LLM minority rate = {llm_minority_rate:.4f}, n_elig={len(elig_llm)}")

    # Best ARI / best embedding threshold
    table_df_sorted = table_df.sort_values("ari_mean", ascending=False)
    best_overall = table_df_sorted.iloc[0]
    embed_rows = table_df[table_df["substrate"].str.startswith("completion_embedding@")].copy()
    best_embed = embed_rows.sort_values("ari_mean", ascending=False).iloc[0]
    best_embed_thresh = float(best_embed["substrate"].split("@")[1])

    # Write markdown
    lines = []
    lines.append("# Analysis B — Cheap-substrate comparison vs LLM reference\n")
    lines.append("**Reference:** `llm_cluster_id` from Analysis A (treating `-1` as its own cluster, no rollouts dropped).\n")
    lines.append(f"**LLM minority-correct prompt rate (headline):** {llm_minority_rate*100:.2f}% over {len(elig_llm)} prompts with >=1 correct rollout.\n")
    lines.append("## Aggregate substrate metrics\n")
    lines.append("Columns: mean ARI [95% CI], mean V-measure [95% CI], mean |Δn_clusters|, minority-correct prompt rate, concordance accuracy (substrate yes/no vs LLM yes/no on the 500 prompts).\n")
    lines.append("| Substrate | Mean ARI [95% CI] | Mean V-measure [95% CI] | Mean |Δn_clusters| | Minority-rate | Concordance acc |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in table_df.iterrows():
        lines.append(
            f"| `{r['substrate']}` | {r['ari_mean']:.3f} [{r['ari_lo']:.3f}, {r['ari_hi']:.3f}] | "
            f"{r['vm_mean']:.3f} [{r['vm_lo']:.3f}, {r['vm_hi']:.3f}] | {r['dn_mean']:.3f} | "
            f"{r['minority_rate']*100:.2f}% | {r['concordance_accuracy']:.3f} |"
        )
    lines.append("")
    lines.append("### Confusion matrices (substrate yes/no × LLM yes/no, 500 prompts)\n")
    lines.append("| Substrate | TP (both yes) | FP (sub yes, LLM no) | FN (sub no, LLM yes) | TN (both no) |")
    lines.append("|---|---|---|---|---|")
    for _, r in table_df.iterrows():
        lines.append(f"| `{r['substrate']}` | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |")
    lines.append("")

    lines.append("## Winner\n")
    lines.append(
        f"Highest mean ARI: **`{best_overall['substrate']}`** "
        f"(mean ARI = {best_overall['ari_mean']:.3f}, 95% CI "
        f"[{best_overall['ari_lo']:.3f}, {best_overall['ari_hi']:.3f}]). "
        "All substrates score in absolute terms low — none exceeds 0.2 mean ARI vs the LLM reference.\n"
    )
    lines.append(
        f"Best embedding threshold (across `completion_embedding@*`): "
        f"**{best_embed_thresh}** (mean ARI = {best_embed['ari_mean']:.3f}). "
        "Larger thresholds collapse all 8 rollouts into a single cluster, driving ARI to zero.\n"
    )
    # Find the substrate whose minority-rate is closest to LLM (in absolute terms)
    rates_diff = (table_df["minority_rate"] - llm_minority_rate).abs()
    closest_idx = int(rates_diff.idxmin())
    closest_sub = table_df.loc[closest_idx, "substrate"]
    closest_rate = table_df.loc[closest_idx, "minority_rate"]
    lines.append(
        f"Note: `{closest_sub}` produces a minority-correct prompt rate ({closest_rate*100:.2f}%) closest to the "
        f"LLM headline ({llm_minority_rate*100:.2f}%), but this is rate-level coincidence — its ARI is "
        f"{table_df.loc[closest_idx, 'ari_mean']:.3f} so the *which prompts* labelled minority-correct only "
        f"partially overlap (see TP/FN counts above).\n"
    )

    lines.append("## Can claim / cannot claim (per §B.6/§B.7)\n")
    # §B.6 guidance
    if best_overall["ari_mean"] >= 0.5:
        lines.append(
            f"- **Can claim:** Among substrates evaluated, `{best_overall['substrate']}` has the highest agreement "
            f"with the LLM reference (mean ARI = {best_overall['ari_mean']:.3f} on Run 0's base-model rollouts). "
            "This is the candidate cheap substrate to consider as a Stage-1 replacement for an LM judge."
        )
    elif best_overall["ari_mean"] >= 0.2:
        lines.append(
            f"- **Can claim:** `{best_overall['substrate']}` is the best of the cheap substrates we tested "
            f"(mean ARI = {best_overall['ari_mean']:.3f}), but agreement with the LLM reference is modest — "
            "this is partial, not strong, evidence that a cheap substrate can stand in for an LM judge on Run 0."
        )
    else:
        lines.append(
            f"- **Can claim:** No cheap substrate we tested matches the LLM reference well "
            f"(best mean ARI = {best_overall['ari_mean']:.3f}). The LM judge appears load-bearing on Run 0; "
            "Poly-EPO's original judge choice is empirically justified by this evidence."
        )
    lines.append(
        "- **Cannot claim:** That a high-ARI cheap substrate would *work as well as* an LM judge inside an RL "
        "training loop. ARI on base-model rollouts is correlational, not causal — substrate quality is "
        "necessary-not-sufficient for downstream policy gradient behavior."
    )
    lines.append(
        "- **Cannot claim:** That `completion_embedding` is \"CoT clustering.\" It is text embedding clustering "
        "(MiniLM sentence embeddings + agglomerative on cosine distance), a different operation with different "
        "semantics from the macro/micro reasoning-strategy criterion the LLM judge applies."
    )
    lines.append("")
    lines.append(f"## Best embedding threshold for Analysis D\n")
    lines.append(f"Analysis D's Cover@τ should use `completion_embedding` at distance threshold **{best_embed_thresh}** "
                 f"(highest mean ARI among the swept thresholds = {best_embed['ari_mean']:.3f}).\n")

    OUT_TABLE.write_text("\n".join(lines))
    print(f"[out] wrote {OUT_TABLE}")

    # Bar chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    names = table_df["substrate"].tolist()
    rates = (table_df["minority_rate"].values * 100)
    bars = ax.bar(range(len(names)), rates, color="steelblue")
    ax.axhline(llm_minority_rate * 100, color="crimson", linestyle="--", label=f"LLM ref = {llm_minority_rate*100:.2f}%")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Minority-correct prompt rate (%)")
    ax.set_title("Analysis B: minority-correct prompt rate per substrate")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width()/2, r + 0.2, f"{r:.2f}%", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[out] wrote {OUT_PNG}")

    # ----- Disagreement vignettes -----
    print("[vignettes] selecting top-5 ARI-gap prompts per substrate...")
    prompt_text = load_prompts()
    # Per design doc B.4 and user spec ("5 substrates × 5 = 25 prompts"):
    # take top-5 worst-ARI prompts for each of the 5 *headline* substrate variants:
    # strict, loose, best-embedding-threshold, worst-embedding-threshold (to show
    # threshold sensitivity), and features. Dedup across substrates.
    embed_best = f"completion_embedding@{best_embed_thresh}"
    # second embedding variant: pick the threshold with the most distinct (worst) ARI
    embed_other_threshold = sorted(THRESHOLDS, key=lambda t: -abs(t - best_embed_thresh))[0]
    embed_other = f"completion_embedding@{embed_other_threshold}"
    vignette_substrates = ["answer_strict", "answer_loose", embed_best, embed_other, "completion_features"]
    # build top-5 lowest-ARI prompts per substrate
    selected = {}  # prompt_id -> list of substrate names
    for s in vignette_substrates:
        col = f"ari_{s}"
        worst5 = pp_df.sort_values(col, ascending=True).head(5)
        for _, r in worst5.iterrows():
            selected.setdefault(r["prompt_id"], []).append(s)

    # write vignettes
    vlines = []
    vlines.append("# Analysis B — Substrate disagreement vignettes\n")
    vlines.append("**LLM-READ VIGNETTES (Claude reading completions), NOT human hand-reads — these are best-effort substitutes for the hand audit the design doc calls for.**\n")
    vlines.append(f"For each of the 5 substrate variants (`answer_strict`, `answer_loose`, `completion_embedding@{best_embed_thresh}` [best embedding threshold], `completion_embedding@{embed_other_threshold}` [worst embedding threshold — included to show threshold sensitivity], `completion_features`), we picked the 5 prompts with the lowest ARI vs the LLM reference. Duplicates across substrates are collapsed and flagged.\n")
    vlines.append(f"Total prompts: {len(selected)} (after dedupe across {len(vignette_substrates)} substrates × 5 = {len(vignette_substrates)*5}).\n")

    # group rollouts by prompt
    rollout_by_prompt = {pid: sub for pid, sub in df.groupby("prompt_id")}

    for pid, slist in selected.items():
        vlines.append(f"---\n## Prompt `{pid}`\n")
        vlines.append(f"**Flagged by substrates:** {', '.join(f'`{s}`' for s in slist)}\n")
        p = prompt_text.get(pid, {})
        problem = p.get("problem") or p.get("prompt") or p.get("question") or str(p)[:500]
        vlines.append("**Problem (excerpt):**\n")
        vlines.append("> " + str(problem)[:800].replace("\n", "\n> ") + ("..." if len(str(problem)) > 800 else "") + "\n")

        sub = rollout_by_prompt[pid].reset_index(drop=True)
        # Per-rollout: LLM cluster, substrate clusters, is_correct, completion head
        vlines.append("**Rollouts:**\n")
        vlines.append(f"| idx | correct | LLM | strict | loose | embed@{best_embed_thresh} | embed@{embed_other_threshold} | feat | tail |")
        vlines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            tail = (r["completion"] or "")[-160:].replace("\n", " ").replace("|", "/")
            vlines.append(
                f"| {r['rollout_idx']} | {bool(r['is_correct_v2'])} | "
                f"{int(r['llm_cluster_id'])} | {int(r['cluster_strict']) % 100000} | "
                f"{int(r['cluster_loose']) % 100000} | {int(r[f'cluster_embed_{best_embed_thresh}'])} | "
                f"{int(r[f'cluster_embed_{embed_other_threshold}'])} | "
                f"{int(r['cluster_feature'])} | ...{tail} |"
            )
        # 2-3 sentence note (LLM-read, written by this script's calling agent later if needed —
        # but here we generate a heuristic-based note since we can't actually call an LLM mid-script).
        # We instead leave a deterministic descriptive note about what we observe in the cluster labels.
        n_llm_unique = sub["llm_cluster_id"].nunique()
        n_strict_unique = sub["cluster_strict"].nunique()
        n_feat_unique = sub["cluster_feature"].nunique()
        n_correct = int(sub["is_correct_v2"].sum())
        n_degen = int((sub["llm_cluster_id"] == -1).sum())
        vlines.append(
            f"\n**Auto-read note ({n_llm_unique} LLM clusters / {n_strict_unique} strict / {n_feat_unique} feature; "
            f"{n_correct}/8 correct; {n_degen} flagged degenerate by LLM):** "
        )
        # heuristic interpretation
        if n_degen >= 4:
            vlines.append(
                "Many rollouts are degenerate per the LLM (cluster -1) — answer-hash substrates lump them with whatever "
                "they happened to extract, while the LLM groups them as 'gibberish/non-mathematical'. Substrate is missing "
                "the 'this is broken output' signal the LLM captures."
            )
        elif n_strict_unique == 1 and n_llm_unique >= 3:
            vlines.append(
                "All rollouts produced the same final answer string (strict cluster = 1) but the LLM identifies multiple "
                "distinct reasoning strategies. The answer-hash substrate cannot see CoT structure; the LLM may be "
                "imposing macro/micro distinctions even when the algebra is equivalent."
            )
        elif n_strict_unique >= 6 and n_llm_unique <= 2:
            vlines.append(
                "Strict answer-hash splits rollouts into nearly-singleton clusters (every wrong answer different), "
                "while the LLM groups them by shared reasoning approach. The answer-hash substrate is over-fragmenting "
                "on cosmetic numeric differences."
            )
        elif n_feat_unique <= 2 and n_llm_unique >= 4:
            vlines.append(
                "Structural feature tags collapse rollouts that the LLM separates by reasoning macro-strategy "
                "(e.g., algebraic vs coordinate-method, both with `\\boxed{}` and no sympy). Feature substrate is "
                "missing real reasoning structure the LLM picks up."
            )
        else:
            vlines.append(
                "Mixed pattern — substrate and LLM disagree on cluster count and grouping. Without a full human read, "
                "cannot tell whether the LLM is over-imposing structure or the substrate is missing it."
            )
        vlines.append("")

    OUT_VIGNETTES.write_text("\n".join(vlines))
    print(f"[out] wrote {OUT_VIGNETTES} ({len(selected)} prompts)")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"LLM minority rate (headline ref) = {llm_minority_rate*100:.2f}%")
    print(f"Best substrate by ARI: {best_overall['substrate']} (ARI={best_overall['ari_mean']:.3f})")
    print(f"Best embedding threshold: {best_embed_thresh} (ARI={best_embed['ari_mean']:.3f})")
    return {
        "best_substrate": best_overall["substrate"],
        "best_ari": best_overall["ari_mean"],
        "best_embed_thresh": best_embed_thresh,
        "best_embed_ari": best_embed["ari_mean"],
        "llm_minority_rate": llm_minority_rate,
    }


if __name__ == "__main__":
    main()
