# Attempt-based difficulty protocol (replaces text-heuristic assessment)

## Why the blind heuristic report was wrong

Reading problem statements and tagging “olympiad-style” or prompt length does **not** establish difficulty. A short modular arithmetic problem can be trivial; a long geometry problem can be routine. Difficulty is a property of the **solver × problem** pair after genuine problem-solving effort.

The prior report also misapplied “avoid circularity”: excluding pilot rollouts avoided *reusing the same stochastic samples as proof of intrinsic hardness*, but it did **not** justify skipping attempts altogether.

## Valid methods (pick one primary per question)

| Method | What you do | Metric |
|--------|-------------|--------|
| **Analyst solve pass** | For each prompt: full reasoning, boxed/`Answer:` final, grade with `pilot.train.answer_parse.is_correct` | Solve rate, list of failures with error type |
| **Fixed-model pass@k** | Same as pilot step-1: N independent completions, same verifier | pass@k, mean reward (already in `raw_predictions.jsonl`) |
| **Human expert** | Timed or untimed solve, blinded batches | Expert solve rate, confidence ratings |

All methods require **checking the answer**, not judging from the stem.

## Procedure (analyst solve pass)

1. Load `set_a_seed43_run1b.json` / `set_b_seed42_run2.json` (or blinded X/Y via `blinding_key.json`).
2. For each record: solve from `prompt` only (no peeking at rollouts for that prompt).
3. Emit completion-shaped string ending with `\boxed{...}` or `Answer: ...`.
4. Grade: `is_correct(completion, gold)`.
5. Record per prompt: `solved`, `claimed_answer`, `notes` (algebra error, misread, timeout, etc.).
6. Aggregate per batch: solve rate, median effort (optional), failure taxonomy.
7. Unblind only in a final section.

## What we already have (model attempts)

Step-1 salvaged rollouts **are** attempt-based for **Qwen3-1.7B-Base** (8 tries/prompt):

| Batch (unblinded) | Mean pass@8 | Prompts with any correct |
|-------------------|-------------|-------------------------|
| seed 43 (run1b) | 0.062 | 8/32 |
| seed 42 (run2/3) | 0.172 | 15/32 |

That is the only quantitative attempt data in-repo today. It does **not** replace an analyst or expert pass if the question is “how hard are these for a careful reasoner?”

## Deliverable to produce

`attempt_based_difficulty_results.md` — per-prompt solve outcomes + batch comparison, no text-heuristic primary verdict.
