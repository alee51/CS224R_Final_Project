"""Generate publication-quality figures for the diversity evaluation results."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path(__file__).parent

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "legend.framealpha": 0.9,
    "lines.linewidth":   2.0,
    "lines.markersize":  6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# Colorblind-safe palette (Wong 2011)
COLORS = {
    "base":     "#999999",  # gray
    "grpo":     "#0072B2",  # blue
    "minority": "#D55E00",  # vermillion
    "polyepo":  "#009E73",  # teal
}
MARKERS = {
    "base":     "o",
    "grpo":     "s",
    "minority": "^",
    "polyepo":  "D",
}
LABELS = {
    "base":     "Base (Qwen3-4B)",
    "grpo":     "GRPO",
    "minority": "Minority-CoT",
    "polyepo":  "Poly-EPO",
}

ARMS = ["base", "grpo", "minority", "polyepo"]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


# ── Figure 1: Pass@k on OOD benchmarks ───────────────────────────────────────

def load_passk():
    """Return {arm: {dataset: {k: val}}}"""
    coverage = json.load(open(RESULTS_DIR / "coverage_results.json"))
    out = {arm: {} for arm in ARMS}
    for fname, entry in coverage.items():
        label = entry.get("label", fname)
        arm = label.split("_step")[0]
        if arm not in ARMS:
            continue
        for ds_name, ds in entry.get("datasets", {}).items():
            pak = ds.get("pass_at_k", {})
            out[arm][ds_name] = {int(k.split("@")[1]): v for k, v in pak.items()}
    return out


def fig1_passk():
    passk = load_passk()
    datasets = [
        ("aime25",    "AIME 2025  (n=30)"),
        ("aime26",    "AIME 2026  (n=30)"),
        ("beyondaime","BeyondAIME  (n=100)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    fig.suptitle("Pass@k on OOD Benchmarks — Step 400 (4B)", y=1.01, fontsize=13)

    for ax, (ds_key, ds_title) in zip(axes, datasets):
        for arm in ARMS:
            ys = [passk[arm].get(ds_key, {}).get(k, float("nan")) for k in K_VALUES]
            ax.plot(K_VALUES, ys, color=COLORS[arm], marker=MARKERS[arm],
                    label=LABELS[arm], clip_on=False)
        ax.set_title(ds_title)
        ax.set_xlabel("k (rollouts per problem)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    axes[0].set_ylabel("Pass@k")
    axes[-1].legend(loc="upper left", frameon=True)
    fig.tight_layout()
    out = RESULTS_DIR / "fig1_passk.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"saved {out}")
    plt.close(fig)


# ── Figure 2: CoT Diversity@k ────────────────────────────────────────────────

def load_cot():
    cot = json.load(open(RESULTS_DIR / "cot_diversity_results.json"))
    out = {arm: {} for arm in ARMS}
    for key, entry in cot.items():
        if "error" in entry:
            continue
        arm = entry["arm"]
        ds  = entry["dataset"]
        agg = entry.get("mean_cot_diversity_at_k", {})
        out[arm][ds] = {int(k.lstrip("@")): v for k, v in agg.items()}
    return out


def fig2_cot():
    cot = load_cot()
    datasets = [
        ("math500",    "MATH-500  (n=500, in-distribution)"),
        ("beyondaime", "BeyondAIME  (n=100, OOD)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    fig.suptitle("CoT Diversity@k — Step 400 (4B)", y=1.01, fontsize=13)

    for ax, (ds_key, ds_title) in zip(axes, datasets):
        for arm in ARMS:
            ys = [cot[arm].get(ds_key, {}).get(k, float("nan")) for k in K_VALUES]
            ax.plot(K_VALUES, ys, color=COLORS[arm], marker=MARKERS[arm],
                    label=LABELS[arm], clip_on=False)
        ax.set_title(ds_title)
        ax.set_xlabel("k (rollouts per problem)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(K_VALUES)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    axes[0].set_ylabel("E[distinct correct CoT clusters in k rollouts]")
    axes[1].legend(loc="upper left", frameon=True)
    fig.tight_layout()
    out = RESULTS_DIR / "fig2_cot_diversity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"saved {out}")
    plt.close(fig)


# ── Figure 3: Summary bar chart at k=16 ──────────────────────────────────────

def fig3_summary():
    passk = load_passk()
    cot   = load_cot()

    k = 16
    configs = [
        # (label, source_dict, dataset_key, y_label, title)
        ("Pass@16\nAIME25",      passk, "aime25",     "Pass@16"),
        ("Pass@16\nBeyondAIME",  passk, "beyondaime",  "Pass@16"),
        ("CoT div@16\nMATH-500", cot,   "math500",     "div@16"),
        ("CoT div@16\nBeyondAIME", cot, "beyondaime",  "div@16"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8))
    fig.suptitle("Diversity & Correctness at k=16 — Step 400 (4B)", y=1.01, fontsize=13)

    for ax, (title, src, ds_key, ylabel) in zip(axes, configs):
        vals = []
        for arm in ARMS:
            d = src[arm].get(ds_key, {})
            vals.append(d.get(k, 0.0) if isinstance(d, dict) else 0.0)
        bars = ax.bar(range(4), vals, color=[COLORS[a] for a in ARMS],
                      width=0.6, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels([LABELS[a].replace(" (Qwen3-4B)", "") for a in ARMS],
                           rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(bottom=0, top=max(vals) * 1.25 if max(vals) > 0 else 0.1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        # Value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    out = RESULTS_DIR / "fig3_summary_k16.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"saved {out}")
    plt.close(fig)


# ── Figure 4: Pass@k vs CoT diversity (MATH500) — one panel ──────────────────

def fig4_correctness_vs_diversity():
    """Two scatter plots: correctness vs CoT diversity for each arm.

    MATH-500: x = pass@64 (estimated as n_correct_prompts/500), y = div@16
    BeyondAIME: x = pass@16 (from coverage_results), y = div@16
    """
    cot      = load_cot()
    coverage = json.load(open(RESULTS_DIR / "coverage_results.json"))

    # MATH500: pass@64 ≈ n_correct_prompts / n_prompts
    cot_raw = json.load(open(RESULTS_DIR / "cot_diversity_results.json"))
    math500_pass64 = {}
    for arm in ARMS:
        entry = cot_raw.get(f"{arm}_math500", {})
        if "error" not in entry and entry.get("n_prompts", 0) > 0:
            math500_pass64[arm] = entry["n_prompts_with_correct"] / entry["n_prompts"]

    # BeyondAIME: pass@16 from coverage_results
    beyondaime_pass16 = {}
    for fname, entry in coverage.items():
        label = entry.get("label", fname)
        arm = label.split("_step")[0]
        if arm not in ARMS:
            continue
        ds = entry.get("datasets", {}).get("beyondaime", {})
        if ds:
            beyondaime_pass16[arm] = ds.get("pass_at_k", {}).get("pass@16", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Correctness vs. CoT Diversity — Step 400 (4B)", y=1.01, fontsize=13)

    panels = [
        (axes[0], math500_pass64,    "math500",    "Pass@64", "MATH-500  (in-distribution)"),
        (axes[1], beyondaime_pass16, "beyondaime", "Pass@16", "BeyondAIME  (OOD)"),
    ]
    for ax, passk_dict, ds_key, xlabel, title in panels:
        for arm in ARMS:
            x = passk_dict.get(arm, float("nan"))
            y = cot[arm].get(ds_key, {}).get(16, float("nan"))
            ax.scatter(x, y, color=COLORS[arm], marker=MARKERS[arm], s=120,
                       zorder=5, label=LABELS[arm])
            ax.annotate(LABELS[arm].replace(" (Qwen3-4B)", ""),
                        (x, y), textcoords="offset points", xytext=(7, 4),
                        fontsize=9, color=COLORS[arm])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CoT Diversity@16")
        ax.set_title(title)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(linewidth=0.4, alpha=0.6)

    axes[1].legend(loc="upper left", frameon=True)
    fig.tight_layout()
    out = RESULTS_DIR / "fig4_correctness_vs_diversity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig1_passk()
    fig2_cot()
    fig3_summary()
    fig4_correctness_vs_diversity()
    print("all figures saved.")
