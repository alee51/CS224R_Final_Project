# Findings & Research Direction Verdicts

Synthesis of the analysis we've done so far on the original proposal, the
question.md brief, and the Poly-EPO knobs ($N$, $n$, $K$). Numerical evidence
referenced here is in `simulation_results.md`.

---

## TL;DR

1. The original proposal (QC-GRPO) is **dead** under binary RLVR rewards. Drop it.
2. The `question.md` analysis is **correct and useful**, but as a standalone
   project it's one figure plus a derivation — not enough on its own.
3. The cleanest research thesis is built around the **scaling laws of $(N, n)$**.
   The paper's authors literally flag this as open in their conclusion. We have
   simulation evidence of a publishable negative result (Sweep 2 below) and a
   defensible positive proposal (annealed schedules).
4. **Recommended path**: a project framed around "$N$ is the price of
   diversity," with a principled annealing schedule of $n$ and $N$ that
   independently controls diversity pressure and compute, validated on the
   paper's synthetic domains.

---

## Mental model: how the knobs actually work

Recap of the gradient mechanics (full detail in `simulation_results.md`):
each generation $y_i$ gets a scalar weight $A_i$ that multiplies its
log-probability gradient. $A_i$ is the average of "set-score minus baseline"
over all size-$n$ subsets containing $y_i$. $f_{poly}(G) = \bar r(G) \cdot |U(G)|/n$.

Empirically confirmed by the sweeps:

| Knob | What it controls | Direction |
|---|---|---|
| $n=1$ | exact GRPO | — |
| $n \in \{2, 3\}$ | mild diversity emphasis, large gradient magnitudes | small $n$ |
| $n \in \{4, 5, 6\}$ | **active suppression** of common-correct, smaller magnitudes | medium $n$ |
| $n \to N$ | signal collapses to 0; sign flips become possible | large $n$ |
| $N$ at fixed $n$ | scales total compute and total signal magnitude | up |
| $N$ at fixed $n/N$ | **weakens the algorithm** (counter-intuitive, see Sweep 2) | — |

Two findings worth highlighting:

### Finding A: $n$ has a **regime change at $n \geq 4$** (with $N=8$)

For configurations with a dominant correct cluster:
- $n \leq 3$: common-correct gets a small positive push; rare-correct gets a
  larger positive push. Both are encouraged, just at different rates.
- $n \geq 4$: common-correct gets a *negative* push. The diversity term has
  beaten the reward term, and the algorithm actively pushes the model away
  from its dominant correct strategy.

This is qualitatively different behavior, not just a tuning knob. It explains
why the paper picked $n = N/2 = 4$: it's the smallest $n$ that triggers active
suppression of redundancy.

### Finding B: $n/N$ is **not** a scale-invariant parameter (Sweep 2)

Naively you'd expect that doubling $N$ and $n$ together preserves the
algorithm's behavior up to compute cost. **It does not.** As $(N, n)$ grows
proportionally:
- The diversity gap (rare-common) shrinks ~10× from $N=4$ to $N=20$.
- The total gradient magnitude shrinks ~6×.

Mechanism: $f_{poly}$ concentrates around its expectation by LLN, so the
variance of set scores collapses, so marginal advantages collapse. This
matters because anyone trying to scale Poly-EPO to bigger models or bigger
batches by holding the ratio constant will get **less** signal than the paper
reports, not the same. **This is a publishable scaling-law observation.**

---

## Research directions evaluated

These are all the threads that came up across our conversation. Each is rated
on a coarse "should we pursue" scale.

### D1. QC-GRPO (original proposal): quantile-conditioned baselines
> Replace the GRPO mean baseline with an $\alpha$-quantile of the within-group
> reward distribution; anneal $\alpha$ during training.

**Verdict: Drop.** Under binary RLVR rewards, the entire continuum
$\alpha \in [0, 1]$ collapses to two equivalence classes (whether the quantile
is above or below the success rate $k/N$). It's not a continuous knob, and
the discrete cases reduce to existing ideas (Pass@k variants). Could be revived
in a non-binary-reward setting (process rewards, RLHF), but that requires
reframing benchmarks and out of scope for a one-quarter project.

### D2. The "fractional diversity dampener" analysis (`question.md` RQ1–3)
> Calculate the marginal set advantage for common vs rare correct generations
> across $n$, prove that diversity is a soft (smooth) filter rather than a
> hard threshold.

**Verdict: Keep, but as an analysis chapter — not the project.** The math is
right (Section S4 of the simulation results confirms the numbers), and the
"soft filter" framing is correct. This becomes the foundation chapter that
motivates the algorithmic contribution. As a standalone thesis it's underweight
— the answer is essentially one figure plus a derivation, with no algorithmic
proposal attached.

### D3. Adaptive cluster-distribution-aware $n$
> At each step, set $n$ adaptively based on the observed cluster sizes per
> prompt. Picks the $n$ that maximizes the variance of $f_{poly}$ given the
> current distribution.

**Verdict: Borderline — too clever for a one-quarter project.** The principle
is sound (the diversity term carries information only when its variance across
sets is non-trivial), but per-prompt adaptive $n$ is operationally awkward
(different gradient batches use different $n$, statistics are harder to track,
implementation in Verl is non-trivial). Probably a follow-on paper, not this
one.

### D4. Joint $(N, n, K)$ scaling laws
> Empirically and theoretically map how the algorithm's behavior depends on
> all three knobs. The paper authors explicitly call this out as open.

**Verdict: Strong contender, especially combined with D7.** This is the most
direct lane the authors hand you in their conclusion. Sweep 2's finding alone
(the $n/N$-isn't-scale-invariant result) is enough for a Methods section.
Combined with a positive contribution (D7 below), this becomes a coherent
project.

### D5. Alternative diversity functions
> Replace $|U|/n$ with $(|U|-1)/(n-1)$, or an entropy of the within-set
> cluster distribution, or $|U|^2/n^2$, etc. Each induces different marginal
> advantages.

**Verdict: Borderline — interesting but narrow.** Easy to implement, easy to
analyze, but unlikely to produce a strong empirical result on its own. Useful
as an ablation appendix (1–2 figures) attached to a stronger thesis. Don't
center the project on it.

### D6. Term 2 (covariance) decomposition study
> The paper's Eq. (15) decomposes the marginal set advantage into
> Term 1 (mean reward × mean diversity) and Term 2 (covariance of reward and
> diversity). The whole pitch over GRPO+DIV is Term 2. Empirically measure
> when Term 2 dominates Term 1.

**Verdict: Strong contender if you want a "why does this work" project rather
than a "what should we change" project.** Genuinely original analysis, gives
you mechanistic insight, requires no algorithmic novelty — just careful
instrumentation of the existing algorithm. Lower risk than D7 because there's
no schedule to design or stabilize. Could be paired with D2 as a unified
"understanding Poly-EPO" project.

### D7. Annealed $n$ at fixed $N$ (the simplest schedule)
> Train with a schedule on $n$: start with $n$ that triggers active diversity
> pressure (e.g. $n=4$ at $N=8$), end at $n=1$ (pure GRPO). Compute is constant.

**Verdict: Strong contender — simplest defensible schedule.** Mechanically
clean (no compute confound), directly implements your stated intuition
("strong diversity early, GRPO-like late"), implementable as a one-line change
to the advantage computation. Risks: (a) it might not actually beat fixed
$n=4$ on the synthetic domains because the late-training GRPO phase might
collapse the diversity you built up; (b) "anneal $n$" sounds small as a
contribution unless paired with D4 (the scaling-law evidence) or D6 (the
mechanism).

### D8. Anneal $N$ down at fixed $n$
> Start with large $N$, end with small $N$, keep $n$ fixed. Saves compute late
> in training.

**Verdict: Drop.** As Sweep 1 shows, fixed $n$ with shrinking $N$ pushes you
toward larger $n/N$, which actually *strengthens* diversity differentiation
late in training. That's the opposite of "less diversity pressure later."
And the total signal magnitude shrinks too. Wrong direction on both axes.

### D9. Anneal $N$ up at fixed $n$
> Start small $N$, end large $N$. More compute as training progresses.

**Verdict: Drop.** Reverses the cost story (you pay more compute exactly when
training has converged and additional rollouts give diminishing returns).
Ideologically opposite to your stated preference. The diversity-pressure case
for it is also weak: at fixed $n$, growing $N$ stabilizes the rare-common gap
around 0.10 (Sweep 1) — it doesn't really "release" diversity pressure.

### D10. Fix $n/N$, vary $N$ (the "naive scaling" candidate)
> Scale $N$ and $n$ together to keep the ratio constant.

**Verdict: Don't pursue as a positive proposal, but use the negative result
in the paper.** Sweep 2 demonstrates this approach actively weakens the
algorithm by ~10×. The negative result is interesting and publishable: it
warns practitioners against the obvious scaling strategy and connects to the
authors' open question on scaling laws.

### D11. Anneal $n$ AND $N$ independently (both downward) — the recommended schedule
> Phase 1: large $N$ (e.g. 12), $n=4$ — strong total signal, diversity pressure on.
> Phase 2: paper baseline ($N=8$, $n=4$) — diversity pressure still on.
> Phase 3: small $N$ (e.g. 8 or 4), $n=1$ — pure GRPO, cheap, no diversity pressure.

**Verdict: Strongest single proposal.** Directly maps your two desiderata onto
the two knobs: $n$ controls diversity pressure (anneal down to release it),
$N$ controls compute (anneal down to save it). Honest about the fact that
they're independent levers, which is something the paper does not articulate.
The simulation evidence (Sweeps 1, 3, 4) directly motivates each knob choice.
Risks: (a) you need a clean ablation to disentangle the two annealings — at
minimum compare {anneal both, anneal only $n$, anneal only $N$, fixed
baseline}; (b) implementing variable $N$ within an existing Verl/GRPO codebase
takes some plumbing.

### D12. The "diversity preservation" empirical question (the heart of your intuition)
> Hypothesis: if you induce enough diversity early in training, the model
> retains it through the late phase even after dropping the diversity term.

**Verdict: This is the actual scientific claim you should test.** It's not
itself an algorithm — it's the question the algorithm (D11) is designed to
answer. If true, it justifies the whole "anneal $n$ down" story. If false,
it kills D11. So one of the project's experimental sections should explicitly
measure diversity (cluster count over training) under fixed-$n$ vs annealed-$n$
schedules to test this empirically.

---

## Recommended project shape

Here's what I'd actually propose, as a draft thesis:

**Title (working):** *Scaling Laws and Schedules for Set Reinforcement Learning:
Decoupling Diversity Pressure from Compute in Poly-EPO*

**Three contributions, structured around the directions above:**

1. **Mechanism (D2 + D6).** Combine the question.md soft-filter analysis with
   the Term 1 / Term 2 decomposition. Show in simulation and on small-scale
   training when Poly-EPO's covariance-based advantage genuinely beats
   reward-shaped baselines and when it doesn't.

2. **Negative scaling result (D4 + D10).** Show that the obvious "double $N$
   and $n$ together" scaling strategy weakens the algorithm by an order of
   magnitude in signal strength. This addresses the open question in the
   paper's conclusion directly.

3. **Positive schedule proposal (D11 + D12).** Propose and validate the
   independent-annealing schedule. The hypothesis is that compute can be
   reduced ~50% by tapering $N$ in late training, and that diversity built
   up under high-$n$ early training is preserved when the diversity term is
   dropped late.

**Experimental scope:** synthetic domains from the paper (multi-digit
multiplication, polynomial solving) using Qwen3-1.7B-Base. Single-axis
ablations for each schedule against the fixed-$(N, n)$ baseline. Three random
seeds. This is doable in the project timeline; the synthetic domains are
deliberately cheap.

**Out of scope (acknowledge but don't tackle):** D3 (per-prompt adaptive $n$),
D5 (alternative diversity functions), D8/D9 (one-axis annealings), QC-GRPO.

---

## Verdicts at a glance

| ID | Direction | Verdict | Why |
|---|---|---|---|
| D1 | QC-GRPO (original proposal) | **Drop** | Binary rewards make quantile baselines degenerate |
| D2 | Soft-filter / fractional dampener analysis | **Keep as foundational chapter** | Correct and useful, but not enough alone |
| D3 | Per-prompt adaptive $n$ | Borderline; defer | Operationally awkward; follow-on paper |
| D4 | $(N, n, K)$ scaling laws | **Strong** | Authors flagged this as open; we have evidence |
| D5 | Alternative diversity functions | Borderline; appendix only | Narrow contribution alone |
| D6 | Term 1 / Term 2 covariance decomposition | **Strong** | Mechanistic insight, low risk |
| D7 | Anneal $n$ at fixed $N$ | **Strong but lean** | Cleanest schedule, may need pairing |
| D8 | Anneal $N$ down at fixed $n$ | **Drop** | Wrong direction for diversity pressure |
| D9 | Anneal $N$ up at fixed $n$ | **Drop** | Reverses cost story |
| D10 | Fix $n/N$, vary $N$ | **Use as negative result** | Surprising but unfavorable scaling |
| D11 | Anneal $n$ AND $N$ independently downward | **Strongest single proposal** | Maps user's two desiderata onto two knobs |
| D12 | "Does early diversity persist?" empirical claim | **Core hypothesis** | The scientific question D11 is built around |

**Recommended bundle for the actual project:** D2 + D6 (mechanism) + D4 + D10
(scaling laws) + D11 + D12 (positive proposal + the question it answers). That's
a coherent paper: understand the mechanism, expose a scaling pitfall, propose a
schedule that exploits the mechanism while avoiding the pitfall, validate the
schedule against the empirical hypothesis.
