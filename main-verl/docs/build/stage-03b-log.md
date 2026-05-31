# Stage 3b log — minority_cot with REAL judge clusters

**Stage ID:** `stage-03b`
**Image rebuild count:** 5 (Stage 3a was 4; this adds `maxrl_expose_data_to_adv_est.patch`)
**Plan:** [migration plan §4-5 + Stage 4 reuse](../verl_migration_plan.md)
**Modal app name (trainer):** `cs224r-verl-stage03b`
**Modal app name (judge):** `cs224r-verl-stage04-judge` (re-deployed on chicken602)
**Modal profile (all):** `chicken602` (test-stage policy 2026-05-30: monitor on chicken602 dashboard only)

---

## Account/profile policy (2026-05-30)

All Stage 3b artifacts use `MODAL_PROFILE=chicken602` — judge redeploy, S4.5 revalidation, verl image rebuild, 3b smoke. Rationale: user can monitor `chicken602` dashboard but not `alee72`. Accept account-contention risk (judge + trainer share one account's GPU pool) as cost-of-monitoring. The original Stage 4 deploy on `alee72` stays alive but unused — Stage 3b's `JUDGE_BASE_URL` points at the new chicken602 endpoint.

---

## Judge model swap (2026-05-30)

Switched from `Qwen/Qwen2.5-7B-Instruct` → `Qwen/Qwen3-4B-Instruct-2507` per poly_epo paper.

| Knob | Stage 4 (original) | Stage 3b (this) |
|------|---|---|
| `JUDGE_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | `Qwen/Qwen3-4B-Instruct-2507` |
| `JUDGE_MAX_MODEL_LEN` | 16384 | **40960** (YaRN factor 1.25 over 32768 native) |
| `DEFAULT_MAX_TOKENS` (output) | 2048 | **4096** (fixes S4.6b parse collapse) |
| `JudgeClientConfig.max_tokens` (output) | 2048 | **4096** |
| `rope_scaling` (vLLM) | none | `{"rope_type": "yarn", "factor": 1.25, "original_max_position_embeddings": 32768}` |

**Why 40960 + YaRN factor 1.25 (not 65536 + 2.0):** worst-case input budget is `800 (system) + 1024 (problem) + 8 × 4096 (rollouts) + 4096 (output) ≈ 38.7K`. 40960 covers worst case with ~2K spare; no need to fit beyond worst case (user 2026-05-30). YaRN 1.25× is mild extension — quality cost negligible.

**Why output 2048 → 4096:** S4.6b serial diagnostic (Stage 4 log line 117) showed 22% parse rate at 2048 due to truncated JSON mid-emission of the 8-way clustering payload. Bumping output cap removes this failure mode.

---

## Architectural fix (the actual hard part)

**Problem found 2026-05-30:** verl's `compute_advantage` dispatch at `ray_trainer.py:373` calls `adv_estimator_fn(**adv_kwargs)` where `adv_kwargs = {token_level_rewards, response_mask, config, index}`. **No rollout text and no DataProto**. So the registered `compute_minority_cot_outcome_advantage` hook physically cannot reach the rollout responses or raw prompts needed to call the judge.

**Fix:** new patch `infra/patches/maxrl_expose_data_to_adv_est.patch` adds one key (`"data": data`) to `adv_kwargs` so registered hooks can read `data.batch["responses"]` (token IDs) and `data.non_tensor_batch["raw_prompt"]` (prompt strings). Three-line patch, backward-compatible: estimators that don't take `**kwargs` are not reached by this dispatch path (it's the `else:` fallback for non-built-in estimators), and our `compute_minority_cot_outcome_advantage` already accepts `**kwargs`.

Dry-run + post-apply verification: ✅ passed against pinned `MAXRL_COMMIT=7197bbb46a2ecd866da52f6b401ff20a34fe9390`.

---

## Dispatch log

| Section | Executor | Audit | Verdict |
|---------|----------|-------|---------|
| Patches (expose_data + minority_cot rewrite) | DONE 2026-05-30 | pending | dry-run + apply OK; image rebuild 5 baked them in |
| Judge bump (server + client) | DONE 2026-05-30 | pending | Qwen3-4B-Instruct-2507 + 40960 + max_tokens 4096 |
| `train/clusters_judge.py` | DONE 2026-05-30 | pending | 6/6 unit tests PASS |
| `train/objective_minority.py` extension | DONE 2026-05-30 | pending | 8/8 existing tests still PASS |
| `infra/modal_image.py` update (rebuild 5) | DONE 2026-05-30 | pending | additive run_commands step |
| Hydra config + probe + launch script | DONE 2026-05-30 | pending | `data.return_raw_chat: true` dropped after step-1 crash (see "Two crashes" below) |
| Judge redeploy on chicken602 | DONE 2026-05-30 | pending | initial rope_scaling crash fixed by dropping kwarg (Qwen3-4B-2507 ships YaRN baked in) |
| S4.5 revalidation against Qwen3-4B | **PASS** 2026-05-30 | pending | 100% parse + 100% agreement (50 tasks × 2 passes), but 10.2s/call |
| Judge `enforce_eager=False` perf fix | DONE 2026-05-30 | pending | 10.2s → 1.63s/call steady-state (6×) — first call eats 50s CUDA-graph compile |
| Stage 3b 10-step smoke v3 | **PASS** 2026-05-30 | pending | 10/10 steps, entropy 1.04–1.33 (matches Stage 2/3a), checkpoint at `global_step_10` |

---

## Artifacts created / modified

- `main-verl/infra/patches/maxrl_expose_data_to_adv_est.patch` — NEW; adds `"data": data` to adv_kwargs.
- `main-verl/infra/patches/maxrl_minority_cot_adv_est.patch` — UPDATED; hook now calls `assign_clusters_for_minority_cot_hook` (routes mock vs judge based on Hydra `cluster_source`).
- `main-verl/judge/server.py` — Qwen3-4B + YaRN + 40960 + max_tokens 4096.
- `main-verl/judge/client.py` — model + max_tokens defaults bumped to match.
- `main-verl/train/clusters_judge.py` — NEW; real-judge cluster source with skip-overflow policy.
- `main-verl/train/objective_minority.py` — added `_group_rollouts_for_judge`, `_coerce_raw_prompt_to_str`, `assign_clusters_for_minority_cot_hook`.
- `main-verl/infra/modal_image.py` — adds patch step for expose_data (image rebuild 5).
- `main-verl/configs/minority_cot_smoke_judge_1p7b.yaml` — NEW; `cluster_source: judge`, `data.return_raw_chat: true`, `judge_max_input_tokens: 36864`.
- `main-verl/probes/minority_cot_judge_smoke.py` — NEW; pre-flights include JUDGE_BASE_URL + judge health check.
- `main-verl/scripts/launch_minority_cot_judge_smoke.sh` — NEW.
- `main-verl/tests/test_clusters_judge.py` — NEW; 6 tests covering happy/parse-fail/overflow/mixed/empty.
- `main-verl/docs/build/stage-03b-log.md` — this file.

---

## Skip-overflow policy (matches main/ Group A Phase 2)

For each prompt, `clusters_judge.py` pre-tokenizes the judge envelope (`system + user`) and checks against `judge_max_input_tokens = 36864` (40960 minus 4096 output reserve). Prompts that overflow are NOT sent to the judge; all 8 of their rollouts receive `DEGENERATE_CLUSTER_ID = -1`.

Downstream effect: `len(set(cluster_ids[p])) == 1` → `set_based_marginal_advantages` flips `keep_mask[p] = False` → zero advantage contribution for that prompt. Same fallback path the mock-collapsed case uses.

Diagnostics surface this:
- `judge_overflow_skipped` (int) — number of prompts dropped per call.
- `judge_parse_ok_rate` (float) — fraction of *in-budget* prompts the judge JSON-parsed cleanly.
- `judge_wall_s` (float) — async batch wall time.
- `judge_n_tasks` (int) — in-budget prompt count.

Expected overflow rate per Stage 2 stats: ~1-3% (response_length mean = 1007, only the 4096-saturated tail risks overflow).

---

## Two crashes encountered + fixed during bring-up

### Crash A — judge cold-start AttributeError (`rope_scaling`)

**Symptom (v1 redeploy):** Judge container crash-looped on every restart with `AttributeError: 'NoneType' object has no attribute 'update'` in `vllm/config.py:480`. S4.5 hung indefinitely waiting for a cold-start that never finished.

**Root cause:** I passed `rope_scaling={"rope_type": "yarn", ...}` as a top-level `LLM(...)` kwarg in `judge/server.py`. vLLM 0.9.0 doesn't accept that shape — it expects rope scaling via `hf_overrides`.

**Fix:** Dropped the `rope_scaling` override entirely. Qwen3-4B-Instruct-**2507** (the July 2025 variant) ships with **256K native context** with YaRN already baked into its HF config — no override needed for our 40960 budget. One commit, judge redeployed in 5s with cached image.

### Crash B — step-1 verl assertion (`raw_prompt length != batch size`)

**Symptom (smoke v2):** Pre-flights all passed, model loaded, step 0 val OK at `mean@1=0.098`, then step 1 crashed in `actor_rollout_generate_sequences` with `AssertionError: key raw_prompt length 32 is not equal to batch size 256`.

**Root cause:** I set `data.return_raw_chat: true` so the judge hook could read `data.non_tensor_batch["raw_prompt"]`. Verl's `rollout.n=8` expansion re-interleaves the tensor batch (32 prompts → 256 rollouts) but does **NOT** interleave the `raw_prompt` non-tensor field. Verl asserts on the mismatch.

**Fix:** Stop relying on `raw_prompt`. The hook now reads `data.batch["prompts"]` (tokenized, n-interleaved correctly by verl's standard path) and decodes with the trainer tokenizer (which we already load to decode rollout responses). Removed `data.return_raw_chat: true` from yaml; refactored `_group_rollouts_for_judge` to take `prompt_token_ids` instead of `raw_prompt_array`. Left-pad stripping added in `clusters_judge._strip_left_pad`. 14/14 unit tests still pass.

## Final 10-step smoke metrics

| Step | actor/entropy |
|---|---|
| 1 | 1.083 |
| 2 | 1.050 |
| 3 | 1.144 |
| 4 | 1.121 |
| 5 | 1.326 |
| 6 | 1.156 |
| 7 | 1.116 |
| 8 | 1.153 |
| 9 | 1.044 |
| 10 | 1.182 |

Range 1.04–1.33, mean ~1.14. Matches Stage 2 GRPO (1.07–1.20) and Stage 3a mock — no entropy collapse, judge cluster signal doesn't destabilize training.

Checkpoint persisted: `/vol/checkpoints/main-verl/minority_cot_smoke_judge_1p7b/global_step_10/{actor/, latest_checkpointed_iteration.txt, wandb_id.txt}`.

## TODOs / known follow-ups (deferred — not blockers for Stage 8)

- [ ] Cluster diagnostic logging — `ClusterAssignment.diagnostics` (parse rate, overflow count, distinct clusters mean) is computed but discarded by the hook. Add a `print(asg.diagnostics)` line at the end of `assign_clusters_for_minority_cot_hook` for the next smoke. Code change is live-reloadable (mounted, not baked); no image rebuild needed.
- [ ] Stage 5 (`poly_epo_cot`) — scaffolding already pre-staged in `train/objective_poly_epo.py` + 2 unmerged patches. Uses the same cluster_source pattern; should be a quick follow-up.
- [ ] Stage 6 (4B fit check) — verify Qwen3-4B trainer fits on B200×4 (rollout micro-batch tuning).
- [ ] Stage 7 — explicit `finish_reason="length"` wiring (hard Stage 8 prereq).
