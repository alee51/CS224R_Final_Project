# Four-grader rescore — does lenient grading lift step-0 non-zero rate?

- Rollouts: `/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/05-25/prompt_c/phase1_rollouts.jsonl`
- Manifest: `/Users/nancybao/Desktop/dev/cs224r_finalproject/main/data/probes/05-25/group_a_n800/manifest.jsonl`
- Prompt variant: `hybrid_answer_boxed`
- n_rollouts used: 6400
- n_prompts: 800  (×8 rollouts)

## Headline numbers

| grader | pass@1 | pass@8 (≡ non-zero rate) | lift vs strict (non-zero) |
|---|---|---|---|
| G1 legacy strict (Rank-2 + normalize ==) | 8.45% | 33.12% | +0.00 pp |
| G2 mathd OR sympy (Rank-2 extract) | 8.50% | 33.25% | +0.13 pp |
| G3 math_verify (Rank-2 extract) | 8.89% | 34.38% | +1.25 pp |
| G4 math_verify on raw `\boxed{}` (no fallback) | 9.52% | 35.38% | +2.25 pp |

## Disagreement counts (samples capped at 50 each)

- **mathd_or_sympy_rescues_strict**: 3 samples collected.
- **math_verify_rescues_strict**: 28 samples collected.
- **math_verify_box_rescues_strict**: 50 samples collected.

Hand-check the JSON `disagree_samples.*` arrays to decide whether
the lenient grader is genuinely correcting or sneaking false positives.
