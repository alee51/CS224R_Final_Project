# Decisions

## 2026-05-21: Training objectives — minority set-RL ablations

**Status:** Accepted (post Run 0 offline analysis).

**Decision:** After Run 0 analysis on human labels ([`run0_exec_plan.md`](../run0_analysis/run0_exec_plan.md), E1 in [`set_score_simulation.md`](../run0_analysis/analysis_c/set_score_simulation.md)), the **headline training question** is whether **minority-voting set-RL** helps — not a repeat of the pre-redesign `inverse_freq` / cheap-substrate matrix.

**Set-RL setup (shared across minority arms):** For each prompt, score all C(8,4) subsets; `f(G)` = mean reward of rollouts in the **rarest** mode in G (ties broken at random — offline E1: ~equivalent to averaging tied modes). Convert subset scores to per-rollout **marginal advantages** (mean set advantage over subsets containing that rollout). Rollout reward remains `cleaned_correct` (0/1).

**Planned training ablations:**

| Arm | What “minority” / diversity means | Run 0 code name |
|-----|-----------------------------------|-----------------|
| **Poly-EPO (baseline)** | `mean(r in G) × (distinct clusters in G) / 4` — cluster-count diversity, not rarity | `f_poly` |
| **Minority — LLM CoT** | Rarest `llm_cluster_id` in G (offline: cached cheap-tier judge; in-loop TBD) | `cot-rand` |
| **Minority — answer-only** | Rarest `cleaned_cluster_id` (final-answer hash) in G | `ans-rand` |

**Not in the primary ablation set (unless revived later):** `inverse_freq` (old redesign matrix arm), vanilla GRPO-only as a standalone paper arm (useful contrast but not the mentor “minority voting” hypothesis), `avg` tie-break variants (offline ~same as `rand`).

**Rationale (Run 0):**

- Minority objectives are **not** the same as Poly-EPO / GRPO credit (E1 correlations ~0.4–0.6 vs minority, ~0.87 GRPO↔`f_poly`).
- Answer-hash minority sees **0%** legacy minority-correct prompts; LLM CoT sees **~14.5%** on eligible prompts — so both arms are worth training, not answer-only alone.
- Cheap text embeddings **do not** approximate LLM CoT clusters (archived Analysis B); CoT arm needs LLM labels in training or accepted proxy cost.

**Still to learn in training (not more offline E1):** Which minority arm (`ans-rand` vs `cot-rand`) improves Pass@k on the same 500-prompt slice / held-out eval.

**Supersedes:** Treating Stage 1 redesign’s three-run matrix (`run1_grpo`, `run2_inverse_freq`, `run3_f_grpo`) as the definitive research plan without adding minority arms.

---

## 2026-05-19: No shared Modal team workspace (Stage 1 pilot)

**Status:** Accepted. Supersedes team-workspace bullets in `pilot/docs/operations/PILOT_REDESIGN.md` until those doc edits landed (now aligned).

**Decision:** The team will **not** use a shared Modal team workspace for the Stage 1 pilot. Each member runs detached jobs on their **personal** Modal workspace (GitHub-username profile).

**Rationale:**

- Modal credits are allocated per personal workspace (~$400 per teammate, ~$600 for the primary operator); credits do not transfer between workspaces.
- The first pilot attempt already provisioned secrets and volumes (`pilot-artifacts`, `hf-cache`, `huggingface`, `wandb-api-key`) on the operator's personal profile; migrating to a team profile would create empty workspace-scoped resources with no credit benefit.

**Process:**

1. **Launch:** `modal profile current` on your personal profile; `modal run --detach pilot/infra/modal_app.py --run-id <run_id>`.
2. **Artifacts:** Pull from your workspace volume with `pilot/scripts/pull_run_artifacts.py` into your local clone. Cheat sheet: `pilot/docs/operations/PERSONAL_WORKSPACE_COLLAB.md`.
3. **Share checkpoints / eval outputs:** HuggingFace Hub, shared drive, or git LFS (mind size limits) — not cross-workspace Modal volumes.
4. **Metrics:** wandb project `cs224r-minority-voting` for all operators; run names include operator + `run_id`. Offline wandb on Modal + `wandb sync` after pull is acceptable.

**Supersedes:** Early `PILOT_REDESIGN.md` / `MAIN_RUNS_PLAYBOOK.md` guidance requiring `MODAL_PROFILE=team` and team-workspace migration before the matrix.

---

## 2026-05-18: Answer grading — integer match vs math equivalence

**Status:** Open implementation item for eval; not blocking train reward today.

We sampled all frozen JSONL under `pilot/data/` and compared gold `answer` fields to what our current grader does (`extract_answer` + `canonicalize_answer` + exact string match).

### Where integer / `\boxed{}` matching is enough

Use **integer extraction + normalization** (strip, optional leading-zero drop, compare as int). No SymPy / Math-Verify / LLM judge.

| Dataset | Role |
|---------|------|
| `dapo_slice_3k.jsonl` | Train (RLVR reward) |
| `aime25_eval_30.jsonl` | Pilot gate primary eval |
| `beyond_aime_eval_100.jsonl` | Paper-tier hard eval |

All gold answers in these files are plain integers (DAPO answers are already the final scalar, often `m+n`-style, not raw fractions).

### Where we need a math equivalence grader (before trusting eval metrics)

Use **Math-Verify** (preferred) or a SymPy/LaTeX normalizer — **not** verl’s demo `math_equal` with `eval` on model strings, and **not** an LLM judge.

| Dataset | Role | Why |
|---------|------|-----|
| `math500_sanity_100.jsonl` | Sanity eval | ~22% of gold answers are LaTeX, degrees, tuples, symbolic (e.g. `\frac43`, `75^\circ`, `p-q`) |
| `math500_eval_500.jsonl` | Full MATH-500 eval | ~38% non–pure-integer gold; many equivalent forms (`4/3` vs `\frac43`) |
| `hmmt_nov25_eval_30.jsonl` | Pilot gate secondary | 21/30 integers; **9/30** need symbolic grading (`1/91`, `5\pi+6\sqrt{3}`, etc.) |
| `hmmt_feb25_eval_30.jsonl` | Paper-tier eval | ~53% LaTeX / symbolic gold |

**Pilot gate today:** `aime25_eval_30` + `hmmt_nov25_eval_30` only (`preflight_lock.json` → `gate_eval_splits`). AIME is fine with integers; HMMT Nov will **under-report** accuracy on ~9 problems until we add a grader.

**Training:** stay on integer match for DAPO — gold is always an integer; difficulty is in the *problem*, not the answer format.

**Also fix:** `canonicalize_answer` currently strips all `}` and breaks LaTeX; do not rely on it for MATH-style equivalence.

---

## 2026-05-18: Stop the Poly-EPO scaling/schedule direction

After working through the gradient mechanics, running the $(N, n)$ sweeps in
[`../archive/poly_epo/../archive/poly_epo/simulation_results.md`](../archive/poly_epo/../archive/poly_epo/simulation_results.md), and synthesizing twelve candidate research directions
in [`../archive/poly_epo/../archive/poly_epo/findings.md`](../archive/poly_epo/../archive/poly_epo/findings.md), the conclusion is to **stop digging in the Poly-EPO
scaling/schedule lane** and pivot the project. Each round of analysis weakened
the central thesis rather than strengthening it: the "anneal $n$ down"
intuition is contradicted by the paper's own diversity-growth dynamics, the
Sweep 2 negative result is predictable from LLN rather than surprising, the
TA-as-paper-author dynamic makes this lane the one most likely to overlap
their existing thinking, and toy-domain validation caps the upside. Full
reasoning and the pivot options (kill-the-LM-judge being the strongest, or
switching to PKPO entirely) are in
[`../archive/poly_epo/why_stop.md`](../archive/poly_epo/why_stop.md).
