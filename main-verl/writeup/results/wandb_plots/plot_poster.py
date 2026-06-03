"""Poster-grade plot regeneration from W&B history CSVs.

Inputs (one level up): {GRPO,Minority_CoT,Poly_EPO_CoT}_history.csv.
Outputs: ./poster/*.png

Entropy back-out:
    Set arms log `actor/entropy` in sstn units (≈ per-prompt summed entropy).
    GRPO logs `actor/entropy` as seq-mean-token-mean nats/token. At step 1 all
    three arms share the same initial policy (verified by identical
    response_length/mean); ratio = 243.38 / 0.966 = 251.9. We use that as the
    calibration constant to back set arms out to per-token nats. Residual bias
    from comparing token-mean to seq-mean-token-mean is Cov(T_i,H_i)/<T>;
    plausible ±15% band shown shaded.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
OUT = HERE / "poster"
OUT.mkdir(exist_ok=True)

ARMS = {
    "GRPO":         ("GRPO_history.csv",         "#1f77b4"),
    "Minority-CoT": ("Minority_CoT_history.csv", "#ff7f0e"),
    "Poly-EPO-CoT": ("Poly_EPO_CoT_history.csv", "#2ca02c"),
}

ENTROPY_DIVISOR = 251.9
ENTROPY_BIAS_BAND = 0.15

def load(arm):
    csv, color = ARMS[arm]
    df = pd.read_csv(RESULTS / csv)
    df = df.sort_values("training/global_step").reset_index(drop=True)
    return df, color

def ema(series, alpha):
    """W&B-style EMA: y_t = alpha * y_{t-1} + (1-alpha) * x_t.
    Higher alpha = more smoothing. W&B default ~0.6; heavier ~0.85, 0.95."""
    s = series.dropna()
    if len(s) == 0:
        return s
    out = np.empty(len(s))
    out[0] = s.iloc[0]
    for i in range(1, len(s)):
        out[i] = alpha * out[i-1] + (1 - alpha) * s.iloc[i]
    return pd.Series(out, index=s.index)

def plot_metric_smoothing_variants(metric_key, title, ylabel, fname_stem,
                                    arms=("GRPO","Minority-CoT","Poly-EPO-CoT"),
                                    smoothings=(0.6, 0.85, 0.95),
                                    higher_is_better=True,
                                    figsize=(10, 5),
                                    suffix=""):
    for alpha in smoothings:
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        for arm in arms:
            df, color = load(arm)
            if metric_key not in df.columns:
                continue
            sub = df[["training/global_step", metric_key]].dropna()
            x = sub["training/global_step"].values
            y_raw = sub[metric_key].astype(float)
            y_smooth = ema(y_raw, alpha)
            ax.plot(x, y_raw, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(x, y_smooth, color=color, label=arm, linewidth=2.2)
        ax.set_xlabel("Training Step", fontsize=12)
        direction = "higher is better" if higher_is_better else "lower is better"
        ax.set_ylabel(f"{ylabel} ({direction})", fontsize=12)
        ax.set_title(f"{title}  (EMA α={alpha})", fontsize=13)
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = OUT / f"{fname_stem}_ema{int(alpha*100)}{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out.relative_to(RESULTS)}")

def plot_entropy_calibrated(alpha=0.85):
    """Entropy plot with GRPO raw + set arms backed out to nats/token via
    step-1 calibration, with ±15% bias band on set-arm lines."""
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    for arm in ("GRPO", "Minority-CoT", "Poly-EPO-CoT"):
        df, color = load(arm)
        if "actor/entropy" not in df.columns:
            continue
        sub = df[["training/global_step","actor/entropy"]].dropna()
        x = sub["training/global_step"].values
        y_raw = sub["actor/entropy"].astype(float)
        if arm == "GRPO":
            y = y_raw  # already nats/token
            band = None
            label = "GRPO (logged, nats/tok)"
        else:
            y = y_raw / ENTROPY_DIVISOR  # back out to nats/tok
            band = (y * (1 - ENTROPY_BIAS_BAND), y * (1 + ENTROPY_BIAS_BAND))
            label = f"{arm} (backed out, nats/tok)"
        y_smooth = ema(y, alpha)
        ax.plot(x, y_raw if arm == "GRPO" else y, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(x, y_smooth, color=color, label=label, linewidth=2.2)
        if band is not None:
            lo_smooth = ema(band[0], alpha)
            hi_smooth = ema(band[1], alpha)
            ax.fill_between(x, lo_smooth, hi_smooth, color=color, alpha=0.15)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Per-token entropy (nats / token)", fontsize=12)
    ax.set_title("Actor Token-Level Entropy  (set arms back-out + ±15% bias band)",
                 fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    # Footnote explaining the back-out
    note = (
        "Set arms log per-prompt summed entropy in verl; GRPO logs seq-mean-token-mean nats/token.\n"
        f"Back-out divisor = 251.9 calibrated at step 1 (all arms share initial Qwen3-4B policy → Cov ≈ 0).\n"
        "Shaded band: ±15% from non-zero Cov(T_i, H̄_i) bias when comparing token-mean to seq-mean-token-mean."
    )
    fig.text(0.01, -0.05, note, fontsize=8, style="italic", color="#444", ha="left")
    fig.tight_layout()
    out = OUT / f"entropy_calibrated_ema{int(alpha*100)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(RESULTS)}")

def plot_entropy_twin_axis(alpha=0.85):
    """Alt: GRPO on right axis (raw 0–1), set arms on left axis (raw 0–250)."""
    fig, ax_set = plt.subplots(figsize=(10, 5), dpi=150)
    ax_grpo = ax_set.twinx()
    # set arms (left)
    for arm in ("Minority-CoT","Poly-EPO-CoT"):
        df, color = load(arm)
        sub = df[["training/global_step","actor/entropy"]].dropna()
        x = sub["training/global_step"].values
        y = sub["actor/entropy"].astype(float)
        y_smooth = ema(y, alpha)
        ax_set.plot(x, y, color=color, alpha=0.18, linewidth=0.8)
        ax_set.plot(x, y_smooth, color=color, label=arm, linewidth=2.2)
    # grpo (right)
    df, color = load("GRPO")
    sub = df[["training/global_step","actor/entropy"]].dropna()
    x = sub["training/global_step"].values
    y = sub["actor/entropy"].astype(float)
    y_smooth = ema(y, alpha)
    ax_grpo.plot(x, y, color=color, alpha=0.18, linewidth=0.8)
    ax_grpo.plot(x, y_smooth, color=color, label="GRPO (right axis)", linewidth=2.2, linestyle="--")
    ax_set.set_xlabel("Training Step", fontsize=12)
    ax_set.set_ylabel("Set-arm entropy (sstn units, left)", fontsize=12)
    ax_grpo.set_ylabel("GRPO entropy (nats / token, right)", fontsize=12, color=color)
    ax_grpo.tick_params(axis='y', labelcolor=color)
    ax_set.set_title("Actor Entropy — twin axes (raw, no back-out)", fontsize=13)
    h1, l1 = ax_set.get_legend_handles_labels()
    h2, l2 = ax_grpo.get_legend_handles_labels()
    ax_set.legend(h1+h2, l1+l2, loc="upper right", fontsize=10)
    ax_set.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUT / f"entropy_twinaxis_ema{int(alpha*100)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(RESULTS)}")

if __name__ == "__main__":
    # pass@8 + fraction_filtered: ema95, square cell only
    plot_metric_smoothing_variants(
        "train/pass_at_8", "Train Pass@8", "Pass@8",
        "pass_at_8", smoothings=(0.95,), higher_is_better=True,
        figsize=(6, 5.5), suffix="_square",
    )
    plot_metric_smoothing_variants(
        "train/fraction_filtered", "Train Fraction Filtered",
        "Fraction filtered", "fraction_filtered",
        smoothings=(0.95,), higher_is_better=False,
        figsize=(6, 5.5), suffix="_square",
    )
    # entropy (deferred to paper, kept for reference): calibrated back-out + twin-axis alt
    plot_entropy_calibrated(alpha=0.85)
    plot_entropy_twin_axis(alpha=0.85)
    print("done")
