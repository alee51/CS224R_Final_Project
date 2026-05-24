# A.7.3 — Degenerate-cluster rate sanity check

**Generated:** 2026-05-21 (overnight session)  
**Source:** `llm_clusters_summary.parquet` (4000 rollouts, all `parse_ok=True`)

## Headline

| Quantity | Value |
|---|---:|
| Rollouts assigned `llm_cluster_id == -1` (paper's `cluster_id: 100`) | 678 / 4000 |
| **LLM degenerate rate** | **16.95%** |
| Prompts with ≥1 degenerate rollout | 333 / 500 (66.6%) |
| Prompts with all 8 rollouts degenerate | 5 / 500 (1.0%) |
| Mean degenerate rollouts per prompt | 1.36 / 8 |

Distribution of degenerate count per prompt: 167 prompts have 0 degenerate, 157 have 1, 88 have 2, 47 have 3, 22 have 4, 10 have 5, 2 have 6, 2 have 7, 5 have 8.

## Cross-check against qual analysis tags

From `pilot/artifacts/run0_proxy/20260519T190202Z/analysis_v2_qual.md`:

| Qual tag | Rollouts | Rate |
|---|---:|---:|
| SymPy / Python code derailment | ~1,141 | ~28.5% |
| Repetition loops (separately reported) | ~25% | ~25% |
| Garbage last-line / long parse (>80 chars) | 367 | 9.2% |
| Nested `\boxed` regex truncation | 108 | 2.7% |

These tags **overlap** (a rollout can be sympy + truncated + long-parse) so they don't add to a clean ground-truth "degenerate" denominator.

## Interpretation

**16.95% degenerate is internally consistent**, not "wildly different" from the qual tags:

- The LLM judge's `cluster_id=100` definition (per Poly-EPO §A.1) is **gibberish / off-topic / non-mathematical / code-only**. It is not "contains sympy code" — a rollout that scaffolds a real derivation alongside sympy is still legitimate reasoning and should land in a normal cluster.
- The 28.5% "sympy/code derailment" qual tag is a presence regex (`import sympy` or triple backticks), not a judgment about whether the reasoning is degenerate. Most sympy rollouts derail but a chunk merely use sympy as a calculator inside a real argument.
- The 9.2% long-parse tag is the strictest "definitely garbage" signal in the qual analysis; 16.95% sits between that floor and the 28.5% sympy-presence ceiling, which is roughly where a "judge calls it degenerate" rate should land.

**Decision: trust the LLM clusters as the reference for Analyses B/C/D.** The degenerate rate is in the expected band; we do not need to re-judge or re-prompt.

## Cannot claim

- That 16.95% is "the true garbage rate." It is the cheap-tier LLM judge's degenerate rate; a stronger judge (Gemini 2.5-pro / Opus 4-class) might call more or fewer rollouts degenerate. The cross-tier ARI probe in §A.7.2 was explicitly skipped, so we have no direct robustness number for this.
