# Decisions

## 2026-05-18: Stop the Poly-EPO scaling/schedule direction

After working through the gradient mechanics, running the $(N, n)$ sweeps in
`simulation_results.md`, and synthesizing twelve candidate research directions
in `findings.md`, the conclusion is to **stop digging in the Poly-EPO
scaling/schedule lane** and pivot the project. Each round of analysis weakened
the central thesis rather than strengthening it: the "anneal $n$ down"
intuition is contradicted by the paper's own diversity-growth dynamics, the
Sweep 2 negative result is predictable from LLN rather than surprising, the
TA-as-paper-author dynamic makes this lane the one most likely to overlap
their existing thinking, and toy-domain validation caps the upside. Full
reasoning and the pivot options (kill-the-LM-judge being the strongest, or
switching to PKPO entirely) are in
[`why_stop_poly_epo.md`](./why_stop_poly_epo.md).
