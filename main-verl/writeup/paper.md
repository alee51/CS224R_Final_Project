# Final Report — writing plan + status (2026-06-08)

**Deadline:** today, 9 PM PT (Gradescope) — 18% of course grade.
**Source file:** `paper-overleaf/final_report.tex` (Overleaf git-linked).
**Template:** `cs224r_2026.sty` (preprint mode); `~8 pages` main body + `1-page` extended abstract (separately required).

## Requirements summary (from `CS224R_Custom_Project_Guidelines.pdf` §7)

1. **One-page extended abstract** as page 1 of the report PDF — main findings + accomplishments.
2. **Main paper (~8 pages)** — motivate method, prior work, results, figures.
3. **Updated team contributions** breakdown with rationale for any change vs. proposal.
4. **AI Tools Disclosure** — what tools, what for, what was done independently.

## Section-by-section status

| § | Section | Status | Source / notes |
|---|---|---|---|
| 0 | Extended abstract (1 pg) | **TODO** | Synthesize after main body locks. Placeholder in `final_report.tex:50` (`insert extended abstract`). |
| 1 | Introduction | **TODO** | Adapt poster `Problem` + `Prior Work` blocks (`poster-overleaf/poster.tex:81–101`). Frame: GRPO mode-collapse → set-RL → minority hypothesis → empirical refutation. |
| 2 | Related Work | **TODO** | GRPO (Shao et al. 2024), Poly-EPO (Orney et al. 2026 — already in `reference.bib`), mode-collapse literature. Poster has 1-sentence Poly-EPO summary at `poster.tex:91–98`. |
| 3 | Method | **DRAFTED** | `final_report.tex` Method section: GRPO baseline + shared marginal-over-subsets kernel + Minority-CoT / Poly-EPO-CoT subset scores + judge/CoT clustering. Cites `\citep{grpo}` and `\citet{polyepo}` — bib entry for GRPO (Shao et al. 2024) still needed. |
| 4 | Experimental Setup | **DRAFTED** | `final_report.tex` §Experimental Setup: §4.1 training (verl + PE recipe + 3 deviations: LR 3e-6, 1 epoch, 4×B200), §4.2 eval (4 arms × 6 OOD datasets, n=64, pass@k + CoT-div@k), §4.3 prelim experiments (1.7B→4B, CoT-vs-answer, prompt template, mathd∨sympy audit, LR bump). Set-RL framework citation still owed in §3 Method + §2 Related Work (kernel borrowed from Set-RL, not PE-original). |
| 5 | Results | **DRAFTED** | `final_report.tex:54–220`. Source: Anastasia's `main-verl/eval/results/results_discussion.tex` (commit `ce5aea8`). Covers 3 OOD datasets pass@k + MATH-500/BeyondAIME CoT diversity. **Open: should HMMT splits be added (writeup has them; current draft drops them).** |
| 6 | Discussion | **DRAFTED** | `final_report.tex:222–340`. 5 subsections (collapse, set-RL fails, minority-as-mode-collapse, confound rebuttal, future). |
| 7 | Conclusion | **TODO (optional)** | Short — 2–3 sentences restating the three findings + framing for the diversity–correctness tradeoff. Could be dropped; §6.5 already does conclusion-y work. |
| 8 | Team Contributions | **DRAFTED (partial)** | `final_report.tex:344–347`. Covers the required who-did-what + rationale-for-shift. Light on specifics — could name algorithm files (`objective_minority.py`, `objective_poly_epo.py`), judge-prompt work, eval probe, etc. that each member touched. |
| 9 | AI Tools Disclosure | **TODO** | One paragraph: ChatGPT/Claude for code (analysis scripts, plotting), writing assistance for prose drafting; algorithm implementation (objective_minority.py, objective_poly_epo.py) done independently. |
| 10 | Limitations | **TODO** | Not strictly required, but worth a paragraph in Discussion: single-seed runs, 4B-only, Polaris-only training distribution, judge model shares base with trained arms (clustering bias risk). Currently scattered across §6.4 (confound rebuttal) + §6.5 (future work). |
| 11 | References | **TODO** | `paper-overleaf/reference.bib` is 695 B — likely only the Poly-EPO entry. Add: GRPO (Shao et al. 2024), any mode-collapse lit cited in §1/§2, MATH-500, AIME/BeyondAIME source, Qwen3 tech report. |

## What can be done immediately (no external blockers)

**Pure writing — sources are in the repo and verified:**

- Method §3 → `main-verl/writeup/algorithm_descriptions.md`
- Setup §4 → `main-verl/writeup/training.md` + `main-verl/writeup/eval.md`
- Intro §1 + Related Work §2 → `poster-overleaf/poster.tex` (already polished prose; reuse with minor extension)
- Conclusion §7 → restate §6 Discussion findings (skip if length-bound)
- AI Tools Disclosure §9 → boilerplate, ~150 words
- Limitations §10 → 1 paragraph in Discussion (single-seed / 4B-only / Polaris-only / judge-base overlap)
- References §11 → top up `reference.bib` with GRPO, mode-collapse refs, dataset citations
- Extended abstract §0 → write last

**Decisions that block § content, not eval:**
- D1: Include HMMT Feb/Nov 2025 in pass@k table? (Eval data exists in `writeup/results/comparison.md`; current draft uses only 3/5 hard-OOD.)
- D2: Include the hmmt_nov25 depth-vs-breadth crossover? (Real story nuance; documented in `writeup/results/INDEX.md` §"hmmt_nov25 explained".)
- D3: Include KL-from-base summary in results? (`writeup/results/kl_summary.md` exists; not in paper.)
- D4: Include training-time diagnostics from poster (rarity ≠ correctness, distinct-answers Δ)? Strengthens the minority-failure story but pushes length past 8 pp.
- D5: Mention polyepo × aime26 = 0/1920 repetition collapse? (Real finding, `eval_complete.md:106`; currently buried in §Results.)

## Eval status — fully resolved

Polyepo × math500 pass@k recovered from Anastasia's 2026-06-08 re-run
(151 MiB JSON on abao). Saved `pass_at_k` reproduces from `n_correct` to
<5e-5; cell is citation-safe. All 24 GEN cells in `comparison.md`.
See [`results/CAVEATS.md`](results/CAVEATS.md) "Polyepo × math500 reconciliation"
for the one-sentence provenance footnote the paper should include.

KL × polyepo × math500 was not re-run (Phase 3 ledger stays 17/18). Not
needed for the paper.

**No data-side blockers remain.**

## Open admin questions

- Anastasia is corresponding author on the GitHub remote (`alee51/CS224R_Final_Project`) — Gradescope submission protocol within group?
- Author ordering on title page already set: Anastasia → Emma → Nancy (`final_report.tex:30–45`). Confirm or change.
- Departments in author block: all three currently show department/email — Emma's affiliation is "Mathematics" in tex but "Computer Science" in poster (`poster.tex:46`). Reconcile.

## Suggested order of operations

1. ⏰ **Now** — write Method + Setup + Intro/Related Work (mostly assembly; ~2–3 hours).
2. While writing, **decide D1–D5** with collaborators.
3. After main body locks → write Conclusion → write Abstract → AI disclosure + team contribs.
4. Modal agent result → reconcile polyepo×math500 if recoverable, otherwise footnote in §4.
5. Final compile, check page count, submit.

## Pointers

- **Eval source-of-truth indices:** `writeup/results/INDEX.md` (full audit-clean tables); `eval/results/diversity_analysis.md` (paper-focused subset).
- **Method source-of-truth:** `writeup/algorithm_descriptions.md` (formulas verified against `train/objective_*.py`).
- **Training spec:** `writeup/training.md` (verified against YAML configs).
- **Eval spec:** `writeup/eval.md` (locked 2026-06-02).
- **Modal state:** `writeup/MODAL_STATUS.md` (accounts, ckpts, eval JSON inventory).
- **Poster reusable prose:** `poster-overleaf/poster.tex`.
