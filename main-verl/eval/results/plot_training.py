"""Generate training-dynamics figures for the paper.

fig5_training_accuracy.pdf  — train/pass_at_8 over 400 steps (3 arms)
fig6_training_diversity.pdf — left: train/fraction_filtered; right: |U_correct|
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

RESULTS_DIR   = Path(__file__).parent
WRITEUP_DIR   = RESULTS_DIR.parents[1] / "writeup" / "results"

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
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

COLORS  = {"grpo": "#0072B2", "minority": "#D55E00", "polyepo": "#009E73"}
LABELS  = {"grpo": "GRPO", "minority": "Minority-CoT", "polyepo": "Poly-EPO"}
MARKERS = {"grpo": "s",    "minority": "^",            "polyepo": "D"}
ARMS    = ["grpo", "minority", "polyepo"]


def ema(series, alpha=0.95):
    out = np.zeros(len(series))
    out[0] = series.iloc[0]
    for i in range(1, len(series)):
        out[i] = alpha * out[i-1] + (1 - alpha) * series.iloc[i]
    return out


def load_csv(arm_label):
    fname = {
        "grpo":     "GRPO_history.csv",
        "minority": "Minority_CoT_history.csv",
        "polyepo":  "Poly_EPO_CoT_history.csv",
    }[arm_label]
    return pd.read_csv(WRITEUP_DIR / fname)


# ── Figure 5: Training pass@8 ─────────────────────────────────────────────────

def fig5_pass():
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle("Training Accuracy (pass@8) — 400 Steps", fontsize=13)

    for arm in ARMS:
        df   = load_csv(arm)
        step = df["training/global_step"].dropna()
        raw  = df["train/pass_at_8"].dropna()
        # align by index (some rows may be NaN)
        mask = df["train/pass_at_8"].notna() & df["training/global_step"].notna()
        step = df.loc[mask, "training/global_step"].values
        raw  = df.loc[mask, "train/pass_at_8"].values
        smooth = ema(pd.Series(raw), alpha=0.9)
        ax.plot(step, raw,    color=COLORS[arm], alpha=0.15, linewidth=1.0)
        ax.plot(step, smooth, color=COLORS[arm], linewidth=2.2,
                label=LABELS[arm])

    ax.set_xlabel("Training step")
    ax.set_ylabel("Train pass@8")
    ax.set_xlim(0, 400)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.legend(loc="upper left")
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)
    fig.tight_layout()

    out = RESULTS_DIR / "fig5_training_accuracy.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"saved {out}")
    plt.close(fig)


# ── Figure 6: Fraction filtered + |U_correct| ────────────────────────────────

def fig6_diversity():
    ucorrect = json.load(open(WRITEUP_DIR / "u_correct_trajectory.json"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Training Diversity Diagnostics — 400 Steps", fontsize=13)

    # Left: fraction filtered
    ax = axes[0]
    for arm in ARMS:
        df   = load_csv(arm)
        mask = df["train/fraction_filtered"].notna() & df["training/global_step"].notna()
        step = df.loc[mask, "training/global_step"].values
        raw  = df.loc[mask, "train/fraction_filtered"].values
        smooth = ema(pd.Series(raw), alpha=0.9)
        ax.plot(step, raw,    color=COLORS[arm], alpha=0.15, linewidth=1.0)
        ax.plot(step, smooth, color=COLORS[arm], linewidth=2.2, label=LABELS[arm])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Fraction of rollouts filtered")
    ax.set_title("Rollout filtering rate")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 1)
    ax.legend(loc="center right")
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    # Right: |U_correct| (distinct correct CoT clusters per prompt, training-time)
    ax = axes[1]
    for arm in ARMS:
        entries = ucorrect["arms"].get(arm, [])
        if not entries:
            continue
        steps = [e["step"] for e in entries]
        vals  = [e["u_correct_mean"] for e in entries]
        ax.plot(steps, vals, color=COLORS[arm], marker=MARKERS[arm],
                markersize=4, linewidth=2.0, label=LABELS[arm])
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"$|U_\mathrm{correct}|$ (per prompt)")
    ax.set_title("Distinct correct CoT clusters\n(training distribution)")
    ax.set_xlim(0, 400)
    ax.set_ylim(bottom=0.9)
    note = "GRPO: trivially 1.0\n(no judge at train time)"
    ax.text(0.97, 0.10, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=COLORS["grpo"], style="italic")
    ax.legend(loc="upper left")
    ax.grid(axis="y", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    out = RESULTS_DIR / "fig6_training_diversity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig5_pass()
    fig6_diversity()
    print("done.")
