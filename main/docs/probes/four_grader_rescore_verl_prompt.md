# Four-grader rescore — does lenient grading lift step-0 non-zero rate?

- Rollouts: `/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/05-31/verl_prompt_4b_n800/phase1_rollouts.jsonl`
- Manifest: `/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/05-27/random_fullgold_n800/manifest.jsonl`
- Prompt variant: `verl_polaris_maxrl`
- n_rollouts used: 6400
- n_prompts: 800  (×8 rollouts)

## Headline numbers

| grader | pass@1 | pass@8 (≡ non-zero rate) | lift vs strict (non-zero) |
|---|---|---|---|
| G1 legacy strict (Rank-2 + normalize ==) | 15.66% | 44.25% | +0.00 pp |
| G2 mathd OR sympy (Rank-2 extract) | 16.56% | 47.62% | +3.38 pp |
| G3 math_verify (Rank-2 extract) | 14.03% | 38.62% | -5.63 pp |
| G4 math_verify on raw `\boxed{}` (no fallback) | 15.72% | 41.00% | -3.25 pp |

## Disagreement counts (samples capped at 50 each)

- **mathd_or_sympy_rescues_strict**: 50 samples collected.
- **math_verify_rescues_strict**: 50 samples collected.
- **math_verify_box_rescues_strict**: 50 samples collected.
- **strict_passes_math_verify_fails**: 50 samples collected.
- **strict_passes_mathd_or_sympy_fails**: 50 samples collected.

Hand-check the JSON `disagree_samples.*` arrays to decide whether
the lenient grader is genuinely correcting or sneaking false positives.
