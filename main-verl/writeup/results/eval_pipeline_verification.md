# Eval pipeline verification (dry run on base × aime25)

_2026-06-02. One-time audit triggered by the first locked-config eval JSON
landing locally (`/tmp/base_aime25.json`, 1.94 GB). Goal: confirm every
metric in `eval.md` §6 runs end-to-end on a real saved JSON BEFORE the
~20 trained-arm cells start landing._

Per-arm sanity (n_correct hist, sample tuples, rescore diff) for base ×
aime25 lives in `base_grader_sanity.md`. This doc tracks **which analysis
scripts are wired correctly**.

## Schema confirmed

Top-level: `label`, `ckpt_path`, `n_rollouts`, `datasets`. Per dataset:
`n_prompts`, `pass_at_k` (all 7 ladder values `{1,2,4,8,16,32,64}`),
`mean_reward_at_1`, `per_prompt`. Per prompt: `problem_id`, `ground_truth`,
`rendered_prompt`, `n_correct`, `rewards`, `preds`, `rollouts`, `logprobs`.

`logprobs[r][t]` is a dict `{token_id: logprob}` with **top-20** entries;
keys come back as **strings** after `json.load` (json serializer auto-
stringifies dict int-keys). This matters — see KL bug below.

## Tier 1 scripts (offline, no GPU)

| metric | script | status |
|---|---|---|
| pass@k (ladder × {1,2,4,8,16,32,64}) | saved by `run_eval.py` | ✅ saved + matches unbiased recompute exactly |
| AUC@k | `auc_at_k.py` | ✅ runs; cosmetic fix: `np.trapz` → `np.trapezoid` |
| coverage@k, distinct@k, entropy@k, majority@k | `coverage.py` | 🔧 **patched** — ladder was `{1,4,8,16}`, now `{1,2,4,8,16,32,64}` |
| diff@k split by solved/unsolved | `diff_at_k_split.py` | ✅ runs; produces solved vs unsolved partition table |
| Potential@k | `potential_at_k.py` | ✅ runs |
| Self-BLEU + distinct-n-gram | `self_bleu.py` | ✅ runs (slow; default `--max-rollouts 16 --max-problems 0`) |
| Reflective-action frequency | `reflective_actions.py` | ✅ runs; reports per-keyword + total/1k_tok |
| Per-rollout token entropy split by correct/incorrect | `token_entropy_split.py` | ✅ runs; base/aime25 gap = +0.65 bits (incorrect > correct, expected direction) |
| Rescore (same grader) | `rescore.py` | 🔧 **patched** — `k_values` was `{1,4,8,16}`, now `{1,2,4,8,16,32,64}`; also crashed when iterating saved keys not in rescored output (now intersects + warns on diff) |
| math_dapo tripwire (spec §8) | `rescore.py` (new) | 🔧 **added** — was missing entirely; now imports `verl.utils.reward_score.math_dapo`, runs strict-box-verify on 20 sampled problems/dataset, flags <90% agreement; gracefully `SKIPPED` when verl not importable locally |

## Tier 2 scripts

| metric | script | status |
|---|---|---|
| KL(π_arm ‖ π_base) per token | `kl_from_base.py` (Modal app) | 🔧 **patched (silent bug)** — saved policy logprobs have `str` token-id keys, Modal-side base teacher-force uses `int` keys. The pre-fix `set(policy.keys()) \| set(base.keys())` had **zero overlap** → every token treated as missing on one side → garbage KL. Fix: `per_token_kl` now coerces both inputs to `int`-keyed dicts at the boundary. Verified `KL(self ‖ self) = 0` after fix; pre-fix would have returned a large positive number. |
| difficulty-stratified pass@k | (not yet implemented) | ⏳ analysis-only, needs ≥1 trained-arm JSON to bucket against the base arm — defer until a trained-arm cell lands |

## Tier 3 (training-time, out of scope here)

`u_correct.py`, `cluster_correctness.py` operate on **training-time
per-rollout JSONLs**, not eval JSONs. Already exercised in
`training_diff_at_k_split.md` + `cluster_correctness.md`. Not retested here.

## Bugs caught + fixed (4)

1. **`coverage.py` k ladder** — `K_VALUES = [1,4,8,16]` would have silently
   produced a partial table for every arm × dataset.
2. **`rescore.py` k ladder** — same hardcoded `(1,4,8,16)`; would have
   silently dropped pass@2, pass@32, pass@64 from any rescore output.
3. **`rescore.py` crash on saved-vs-rescored key mismatch** — `KeyError`
   when saved JSON had pass@2 but rescore output didn't; now intersects
   and warns on diff.
4. **`kl_from_base.per_token_kl` str-vs-int token-id keys** — the most
   dangerous one because it would have produced a plausibly-shaped number
   ($KL > 0$ for all arms) that was actually pure noise from the union
   double-counting. Would have polluted every Tier 2 number.

All 4 patched in the same session. Smoke tests pass.

## Still pending (need trained-arm data)

- **Multi-arm AUC table** — `auc_at_k.py` accepts multiple paths and groups
  by `arm` automatically; just rerun with 4 JSONs as args once they land.
- **Cross-arm `diff@k split`** — same script, just feed all 4 JSONs.
- **KL end-to-end Modal run** — requires GRPO/Minority/Poly-EPO eval JSONs
  (currently blocked by FSDP rank-0 corruption — see `PHASE1_PROGRESS.md`).
- **`math_dapo` tripwire** — engages automatically the moment `rescore.py`
  runs in an environment where `verl` is importable (Modal image, or local
  with `pip install verl math_verify`).
- **difficulty-stratified pass@k** — not yet written; trivial once base is
  paired with a trained arm.

## Reproducer

```bash
# Tier 1 (all on the 1 file we have locally)
python3 main-verl/eval/analysis/posthoc/auc_at_k.py /tmp/base_aime25.json --out /tmp/eval_verify/auc_at_k.md
python3 main-verl/eval/analysis/posthoc/coverage.py /tmp/base_aime25.json
python3 main-verl/eval/analysis/posthoc/potential_at_k.py /tmp/base_aime25.json --out /tmp/eval_verify/potential_at_k.md
python3 main-verl/eval/analysis/posthoc/reflective_actions.py /tmp/base_aime25.json --out /tmp/eval_verify/reflective_actions.md
python3 main-verl/eval/analysis/posthoc/diff_at_k_split.py /tmp/base_aime25.json --out /tmp/eval_verify/diff_at_k_split.md
python3 main-verl/eval/analysis/posthoc/token_entropy_split.py /tmp/base_aime25.json --out /tmp/eval_verify/token_entropy_split.md
python3 main-verl/eval/analysis/posthoc/self_bleu.py /tmp/base_aime25.json --max-rollouts 8 --max-problems 8 --out /tmp/eval_verify/self_bleu.md
python3 main-verl/eval/analysis/posthoc/rescore.py /tmp/base_aime25.json --out /tmp/eval_verify/base_aime25_rescored.json
```
