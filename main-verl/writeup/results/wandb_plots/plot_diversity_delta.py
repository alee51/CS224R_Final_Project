"""Plot Δ(distinct_answers@k=8 vs GRPO) trajectory over training steps,
split by solved vs unsolved prompts. Replaces poster Table 2.

Δ = (set arm mean distinct_answers@k=8) − (GRPO mean), computed per step
on per_rollout_v2 training-batch JSONLs (128 prompts/step). Partition is
per-arm: solved = ≥1/8 rollouts correct, unsolved = 0/8.

"distinct_answers@k=8" = #unique non-empty parsed boxed-answers among the
8 rollouts. Bounded above by parse-success rate (~95%); at k=1 the metric
degenerates to parse-success and is uninformative — not plotted here.
"""
from pathlib import Path
import json, re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/per_rollout_v2")
OUT  = Path("/Users/nancybao/Desktop/dev/cs224r_finalproject/poster-overleaf")
STEP_RE = re.compile(r"step_(\d+)\.jsonl$")
K = 8

COLORS = {"minority": "#ff7f0e", "polyepo": "#2ca02c"}
LABELS = {"minority": "Minority-CoT", "polyepo": "Poly-EPO-CoT"}

def files_for(arm, smin=1, smax=400, every=10):
    out = {}
    for p in (ROOT / arm).rglob("step_*.jsonl"):
        m = STEP_RE.search(p.name)
        if not m: continue
        s = int(m.group(1))
        if s < smin or s > smax or s % every != 0: continue
        if s not in out or p.stat().st_mtime > out[s].stat().st_mtime:
            out[s] = p
    return dict(sorted(out.items()))

def split_step(path):
    prompts = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        prompts[r["prompt_id"]].append(r)
    sv, uv = [], []
    for pid, rs in prompts.items():
        rs.sort(key=lambda r: r["rollout_idx"])
        n_correct = sum(1 for r in rs if r["reward"] > 0.5)
        preds = [r.get("parsed_answer") for r in rs[:K]]
        d = len({p for p in preds if p})
        (sv if n_correct > 0 else uv).append(d)
    return (sum(sv)/len(sv) if sv else np.nan, sum(uv)/len(uv) if uv else np.nan)

def ema(y, alpha):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y); out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * out[i-1] + (1 - alpha) * y[i]
    return out

# Build per-step series for each arm
series = {}  # arm -> {step: (solved, unsolved)}
for arm in ("grpo", "minority", "polyepo"):
    files = files_for(arm)
    series[arm] = {s: split_step(p) for s, p in files.items()}

common = sorted(set(series["grpo"]) & set(series["minority"]) & set(series["polyepo"]))
steps = np.array(common)

g_sv = np.array([series["grpo"][s][0] for s in common])
g_un = np.array([series["grpo"][s][1] for s in common])
m_sv = np.array([series["minority"][s][0] for s in common])
m_un = np.array([series["minority"][s][1] for s in common])
p_sv = np.array([series["polyepo"][s][0] for s in common])
p_un = np.array([series["polyepo"][s][1] for s in common])

m_sv_d, m_un_d = m_sv - g_sv, m_un - g_un
p_sv_d, p_un_d = p_sv - g_sv, p_un - g_un

# Compute the 200-400 window averages for annotation
mask = (steps >= 200) & (steps <= 400)
def avg(a, m): return np.nanmean(a[m])
ann = {
    "m_sv": avg(m_sv_d, mask), "m_un": avg(m_un_d, mask),
    "p_sv": avg(p_sv_d, mask), "p_un": avg(p_un_d, mask),
}
print("Δ avg over steps 200-400:")
for k, v in ann.items(): print(f"  {k}: {v:+.3f}")

ALPHA = 0.85  # EMA smoothing

fig, (ax_un, ax_sv) = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=150, sharey=True)

for ax, m_d, p_d, title in (
    (ax_un, m_un_d, p_un_d, "Unsolved prompts (0/8 correct)"),
    (ax_sv, m_sv_d, p_sv_d, "Solved prompts (≥1/8 correct)"),
):
    ax.axhline(0, color="#888", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvspan(200, 400, color="#fce5b8", alpha=0.35, zorder=0,
               label="poster avg window")
    # Minority
    ax.plot(steps, m_d, color=COLORS["minority"], alpha=0.20, linewidth=0.8)
    ax.plot(steps, ema(m_d, ALPHA), color=COLORS["minority"],
            linewidth=2.5, label=LABELS["minority"])
    # Poly
    ax.plot(steps, p_d, color=COLORS["polyepo"], alpha=0.20, linewidth=0.8)
    ax.plot(steps, ema(p_d, ALPHA), color=COLORS["polyepo"],
            linewidth=2.5, label=LABELS["polyepo"])
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Training step", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

ax_un.set_ylabel("Δ distinct answers@k=8  vs.  GRPO\n(per-prompt mean within partition)",
                 fontsize=11)
fig.suptitle("Per-step diversity gap vs. GRPO  (EMA α=0.85; partition is per-arm)",
             fontsize=13, y=1.02)
fig.tight_layout()

out_path = OUT / "diversity_delta_trajectory.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out_path}")
