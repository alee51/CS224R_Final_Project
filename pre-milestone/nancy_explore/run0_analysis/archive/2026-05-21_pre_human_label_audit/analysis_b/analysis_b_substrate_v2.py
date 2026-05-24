"""Analysis B v2 — re-embed completions with a long-context, math-aware model
and re-sweep cluster thresholds against the LLM reference.

Why v2: original MiniLM (256 tokens) truncated 91.6% of Run 0 completions
(median 644 tokens, p99 1221). Qwen3-Embedding-0.6B has 32K context so the
full completion is embedded; it is also trained on Qwen3 data which is heavy
on math, so the embedding space should better reflect reasoning strategy.

Reuses the per-prompt ARI / V-measure / minority-correct concordance protocol
from analysis_b_substrate.py (so results are directly comparable).
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREDS = ROOT / "data" / "predictions_reparsed.jsonl"
LLM_PARQUET = ROOT / "analysis_a" / "llm_clusters_summary.parquet"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
MODEL_TAG = "qwen3_0p6b"

EMBED_CACHE = HERE / f"embeddings_{MODEL_TAG}.npy"
EMBED_IDS = HERE / f"embedding_ids_{MODEL_TAG}.parquet"
OUT_PARQUET = HERE / "substrate_results_v2.parquet"
OUT_MD = HERE / "substrate_comparison_v2.md"
RUN_LOG = HERE / "analysis_b_v2_run.log"

# Wider sweep than v1 because we don't know the right zone for the new embedder.
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with RUN_LOG.open("a") as f:
        f.write(line + "\n")


def load_predictions() -> pd.DataFrame:
    rows = []
    with PREDS.open() as f:
        for line in f:
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.reset_index(drop=True)
    df["rollout_idx"] = df.groupby("prompt_id").cumcount()
    return df


def load_llm() -> pd.DataFrame:
    df = pd.read_parquet(LLM_PARQUET)
    return df[["prompt_id", "rollout_idx", "llm_cluster_id", "is_correct_v2", "parse_ok"]]


def has_minority_correct(cluster_ids: np.ndarray, correct: np.ndarray) -> bool:
    if correct.sum() == 0:
        return False
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
    return {
        "ari": ari,
        "vmeasure": vm,
        "n_clusters_sub": n_sub,
        "n_clusters_llm": n_llm,
        "abs_diff_n": abs(n_sub - n_llm),
        "has_correct": bool(correct.sum() > 0),
        "hmc_sub": has_minority_correct(sub_labels, correct),
        "hmc_llm": has_minority_correct(llm_labels, correct),
    }


def bootstrap_ci_mean(vals: np.ndarray, n_resample: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(vals)
    means = np.empty(n_resample)
    for i in range(n_resample):
        idx = rng.integers(0, n, size=n)
        means[i] = vals[idx].mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_or_load_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    if EMBED_CACHE.exists() and EMBED_IDS.exists():
        embs = np.load(EMBED_CACHE)
        ids = pd.read_parquet(EMBED_IDS)
        if len(embs) == len(df) and (ids["prompt_id"].values == df["prompt_id"].values).all():
            log(f"embed cache hit: {embs.shape}")
            return embs, ids
        log("embed cache mismatch, recomputing")

    from sentence_transformers import SentenceTransformer
    log(f"loading {MODEL_NAME} ...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    log(f"  loaded in {time.time()-t0:.1f}s; max_seq_length={model.max_seq_length}")

    texts = df["completion"].fillna("").tolist()
    log(f"encoding {len(texts)} rollouts (CPU, batch_size=4) ...")
    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=4,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    log(f"encoded in {(time.time()-t0)/60:.1f} min; shape={embs.shape}")

    np.save(EMBED_CACHE, embs)
    ids = df[["prompt_id", "rollout_idx"]].copy()
    ids.to_parquet(EMBED_IDS)
    log(f"cached → {EMBED_CACHE.name}, {EMBED_IDS.name}")
    return embs, ids


def cluster_per_prompt(embs: np.ndarray, prompt_ids: list[str], threshold: float) -> np.ndarray:
    out = np.full(len(prompt_ids), -999, dtype=np.int64)
    idx_by_prompt: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(prompt_ids):
        idx_by_prompt[p].append(i)
    for p, idxs in idx_by_prompt.items():
        X = embs[idxs]
        n = len(idxs)
        if n == 1:
            out[idxs[0]] = 0
            continue
        ac = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average",
            distance_threshold=threshold,
        )
        labels = ac.fit_predict(X)
        for j, i in enumerate(idxs):
            out[i] = int(labels[j])
    return out


def main() -> None:
    RUN_LOG.write_text("")  # truncate
    log(f"model={MODEL_NAME} thresholds={THRESHOLDS}")

    preds = load_predictions()
    llm = load_llm()
    df = preds.merge(llm, on=["prompt_id", "rollout_idx"], how="left",
                     suffixes=("", "_llm"), validate="one_to_one")
    df["is_correct_v2"] = df["is_correct_v2"].astype(bool)
    log(f"merged rows={len(df)} (expect 4000)")

    embs, _ids = compute_or_load_embeddings(df)
    prompt_ids = df["prompt_id"].tolist()

    cluster_cols: dict[float, np.ndarray] = {}
    for t in THRESHOLDS:
        log(f"cluster @ threshold={t}")
        cluster_cols[t] = cluster_per_prompt(embs, prompt_ids, t)
        df[f"cluster_embed_{t}"] = cluster_cols[t]

    # Per-prompt metrics for each threshold
    log("per-prompt metric loop")
    per_prompt_rows = []
    for pid, sub in df.groupby("prompt_id"):
        llm_labels = sub["llm_cluster_id"].astype(np.int64).values
        correct = sub["is_correct_v2"].values.astype(bool)
        row = {"prompt_id": pid, "has_correct": bool(correct.sum() > 0),
               "n_clusters_llm": len(np.unique(llm_labels)),
               "has_minority_correct_llm": has_minority_correct(llm_labels, correct)}
        for t in THRESHOLDS:
            col = f"cluster_embed_{t}"
            sub_labels = sub[col].astype(np.int64).values
            m = per_prompt_metrics(sub_labels, llm_labels, correct)
            tag = f"embed@{t}_{MODEL_TAG}"
            row[f"ari_{tag}"] = m["ari"]
            row[f"vmeasure_{tag}"] = m["vmeasure"]
            row[f"n_clusters_{tag}"] = m["n_clusters_sub"]
            row[f"abs_diff_n_{tag}"] = m["abs_diff_n"]
            row[f"has_minority_correct_{tag}"] = m["hmc_sub"]
        per_prompt_rows.append(row)

    per_prompt_df = pd.DataFrame(per_prompt_rows)
    per_prompt_df.to_parquet(OUT_PARQUET, index=False)
    log(f"wrote {OUT_PARQUET.name}")

    # Aggregate
    log("aggregating substrate-level metrics")
    n_prompts = len(per_prompt_df)
    minority_llm_rate = per_prompt_df.loc[per_prompt_df["has_correct"],
                                          "has_minority_correct_llm"].mean()

    rows = []
    for t in THRESHOLDS:
        tag = f"embed@{t}_{MODEL_TAG}"
        ari_vals = per_prompt_df[f"ari_{tag}"].values
        vm_vals = per_prompt_df[f"vmeasure_{tag}"].values
        ari_mean, ari_lo, ari_hi = bootstrap_ci_mean(ari_vals)
        vm_mean, vm_lo, vm_hi = bootstrap_ci_mean(vm_vals)
        mean_diff_n = per_prompt_df[f"abs_diff_n_{tag}"].mean()
        # minority-correct rate under substrate (eligible: has_correct)
        hmc_sub = per_prompt_df.loc[per_prompt_df["has_correct"],
                                    f"has_minority_correct_{tag}"].mean()
        # concordance vs LLM (on all 500)
        sub_yes = per_prompt_df[f"has_minority_correct_{tag}"]
        llm_yes = per_prompt_df["has_minority_correct_llm"]
        # eligibility-filtered confusion (both need has_correct, since LLM minority needs >=1 correct)
        elig = per_prompt_df["has_correct"]
        tp = int(((sub_yes & llm_yes) & elig).sum())
        fp = int(((sub_yes & ~llm_yes) & elig).sum())
        fn = int(((~sub_yes & llm_yes) & elig).sum())
        tn = int(((~sub_yes & ~llm_yes) & elig).sum())
        acc = (tp + tn) / max(elig.sum(), 1)
        rows.append({
            "substrate": f"completion_embedding@{t}_{MODEL_TAG}",
            "ari_mean": ari_mean, "ari_lo": ari_lo, "ari_hi": ari_hi,
            "vm_mean": vm_mean, "vm_lo": vm_lo, "vm_hi": vm_hi,
            "mean_abs_diff_n": mean_diff_n,
            "minority_rate": hmc_sub,
            "concordance_acc_eligible": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    # Compose markdown
    log("writing markdown")
    best = max(rows, key=lambda r: r["ari_mean"])
    md = []
    md.append("# Analysis B v2 — Qwen3-Embedding-0.6B substrate sweep\n")
    md.append("Re-embed all 4000 completions with `Qwen/Qwen3-Embedding-0.6B` "
              "(32K context, math-strong) and re-cluster per-prompt.\n")
    md.append(f"**Model:** `{MODEL_NAME}`  ")
    md.append(f"**Why:** MiniLM (256-token cap) was truncating 91.6% of Run 0 completions. "
              "Qwen3-Embedding sees full completions.\n")
    md.append(f"**Reference:** `llm_cluster_id` from Analysis A (LLM minority-correct prompt rate = {minority_llm_rate*100:.2f}% on {int(per_prompt_df['has_correct'].sum())} eligible prompts).\n")
    md.append("## Aggregate substrate metrics (new embedder)\n")
    md.append("| Substrate | Mean ARI [95% CI] | Mean V-measure [95% CI] | Mean \\|Δn_clusters\\| | Minority-rate | Concordance acc (eligible) |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| `{r['substrate']}` | {r['ari_mean']:.3f} [{r['ari_lo']:.3f}, {r['ari_hi']:.3f}] | "
            f"{r['vm_mean']:.3f} [{r['vm_lo']:.3f}, {r['vm_hi']:.3f}] | {r['mean_abs_diff_n']:.3f} | "
            f"{r['minority_rate']*100:.2f}% | {r['concordance_acc_eligible']:.3f} |"
        )
    md.append("")
    md.append("## Confusion matrices vs LLM (eligible prompts: has ≥1 correct)\n")
    md.append("| Substrate | TP | FP | FN | TN |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append(f"| `{r['substrate']}` | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |")
    md.append("")
    md.append("## Headline\n")
    md.append(f"- Best mean ARI under Qwen3 embeddings: **`{best['substrate']}`** "
              f"with mean ARI = **{best['ari_mean']:.3f}** [{best['ari_lo']:.3f}, {best['ari_hi']:.3f}].")
    md.append(f"- Compared to v1 best (MiniLM@0.2, ARI 0.074) and the overall v1 best across all substrates (`answer_strict`, ARI 0.188).")
    md.append("- See `substrate_comparison.md` (v1) for the original side-by-side substrate table.")
    md.append("")
    OUT_MD.write_text("\n".join(md))
    log(f"wrote {OUT_MD.name}")
    log("done")


if __name__ == "__main__":
    main()
