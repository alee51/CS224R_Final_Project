# Training wandb metrics — final verdict (post-review)

**Written:** 2026-05-26  
**Context:** GRPO full train (`train_real.yaml`) was already in flight when this was reviewed against Poly-EPO Fig. 2 and our four-arm PLAN. **Do not change logging mid-run** for the current GRPO leg. Apply this spec when wiring **arms 2–4** (`remaining_arms.md`) and when updating monitoring docs.

**Supersedes for priority/interpretation only:** PLAN §5 *Training-time reporting*, `remaining_arms.md` §5, and `monitoring/wandb_dashboard_full.md` where this doc disagrees. Implementation IDs (C1–C4b) are unchanged.

---

## 1. Executive summary

| Category | Verdict |
| --- | --- |
| **GRPO (in flight)** | **No new wandb keys.** C1 + C1b + C2 + existing PPO/VRAM diagnostics are sufficient for routine monitoring. |
| **Set arms (answer-hash)** | Add **one** new cluster scalar (see §3.2) when `clusters_grid` exists. C4b is optional (writeup / Poly proxy), not required for ops. |
| **Minority-CoT** | Add **C4** (already in plan) **plus** in-loop judge health metrics (§3.3) — not in plan today. |
| **Poly-EPO-answer** | Same as minority-answer for cluster logging; optional subset-diversity scalar (§3.2) only if debugging `f_poly`. |
| **C3 marginal percentiles** | Keep in plan as **debug-only**; wire when convenient, not a launch blocker. |
| **Do not add** | `n_filtered_prompts`, per-prompt reward std, in-loop pass@k scalars — redundant with histogram + `fraction_filtered` + `n_kept`. |

Held-out **pass@k** across arms remains **offline eval** only (PLAN §4). No in-loop eval logging.

---

## 2. What we already log (all arms, implemented)

These are **enough** for “is the run healthy?” on every arm:

- **Coverage / learning:** `train/prompt_coverage`, `train/frac_prompts_{0..8}_correct`, `train/mean_reward`
- **GRPO-only signal proxy:** `train/mixed_reward_rate` — see §4 (do **not** use on set arms)
- **Filtering / cost:** `train/fraction_filtered`, `train/n_kept_sequences`, `train/num_chunks`, phase times, VRAM
- **Stability:** `train/ratio_*`, `train/clipped_*_frac`, `train/grad_norm_preclip`, `train/mean_neg_logprob`, finish-reason fractions
- **Grading / format:** `train/parse_ok_rate`, `train/extract_path_*`, `train/mean_reward_extract_*`
- **Shape / sanity:** `train/mean_completion_tokens`, `train/p95_completion_tokens`, `sample/completion_*` every 50 steps

Poly-EPO Fig. 2 **right** (training coverage) = `train/prompt_coverage`. The pass@k histogram is strictly more informative than a single coverage scalar.

---

## 3. Add when arms ship (prioritized)

### 3.1 Shared set-arm plumbing

When `trainer.py` builds `clusters_grid` for `minority_answer` / `poly_epo_answer` / `minority_cot`:

1. Pass `clusters` + `problem_ids` into `compute_advantages` (per `remaining_arms.md`).
2. Forward `adv_out.diagnostics` to wandb (today only `fraction_filtered` is surfaced on `StepResult`).

### 3.2 Set arms — **add** (not in PLAN today)

**Key:** `train/mean_unique_clusters_kept`

| Field | Definition |
| --- | --- |
| **Scope** | Prompts with `keep_mask == True` only (≥2 distinct cluster ids on all 8 rollouts). |
| **Per prompt** | `len(set(cluster_id for rollouts))` — **all** rollouts, not correct-only. |
| **Step aggregate** | Mean over kept prompts in the batch. |
| **Arms** | `minority_answer`, `poly_epo_answer`, `minority_cot` |

**Why:** `fraction_filtered` is binary (collapsed vs not). Set arms can train with 0% `mixed_reward_rate` (all wrong but diverse answers). This scalar tracks whether the **cluster substrate** is rich among prompts that actually get gradient — the main ops signal for set-RL that nothing else logs.

**Implementation:** Extend `aggregate_train_step_wandb_metrics(..., clusters_grid=...)` or a sibling helper; do not duplicate reward parsing.

### 3.3 Minority-CoT — **add** (partially in PLAN)

| Key | Priority | Notes |
| --- | --- | --- |
| `train/mean_unique_strategy_clusters_correct` | **Must** (C4) | Per PLAN / probe plan C4: unique **judge** cluster ids among rollouts with `reward > 0`; mean over batch. Poly-EPO Fig. 2 left. |
| `train/judge_cluster_100_frac` | **Must** | Fraction of rollouts assigned cluster id 100 (degenerate / gibberish per `poly_epo_a1.md`). |
| `train/judge_json_parse_ok_rate` | **Must** | Fraction of prompts where judge returned parseable JSON. |
| `train/judge_truncated_frac` | **Should** | Fraction of prompts skipped or truncated because judge input > `max_model_len`. |
| `train/judge_wall_clock_s` (p50 or mean) | **Should** | Step-level judge latency for cost/stall detection. |
| `train/mean_unique_clusters_kept` | **Must** | §3.2 — strategy clusters on all rollouts for kept prompts. |

Port metric names from Group A probes (`group_a_impl.md`: `cluster_100_hits`, `json_parse_ok`) for consistency, prefixed with `train/judge_*` in the train loop.

GRPO and answer-hash-only arms must **not** import judge or log C4 / judge_* keys.

### 3.4 Already in PLAN — implement with revised priority

| ID | Key(s) | Arms | Verdict |
| --- | --- | --- | --- |
| **C4b** | `train/mean_unique_answer_clusters_correct` | `minority_answer`, `poly_epo_answer` | **Optional / low.** Correct-only answer diversity is a weak proxy for minority scoring and a weak Fig. 2 left analogue. Implement if cheap (same pass as §3.2), for paper curves — **not** for routine dashboards. |
| **C3** | `train/adv_marginal_p05`, `p50`, `p95` | All set arms | **Debug-only.** Already computed in `objective.py` for set arms; forward to wandb when debugging credit assignment. **Do not block** arm launch (per `05-24_probe_plan.md`). Prefer every step only if noise is low; else every 100 steps. |
| **C4** | (see §3.3) | `minority_cot` only | **Must** for CoT arm. |

### 3.5 Poly-EPO-answer only — **optional** (not in PLAN)

**Key:** `train/mean_subset_diversity`

- Per prompt: mean over 70 size-4 subsets of `|unique cluster ids in G| / 4` (the diversity factor in `f_poly`).
- **Only** useful when debugging whether `f_poly` sees diversity in subsets; skip for default dashboards.

---

## 4. Doc / dashboard modifications (no train code)

Update when touching monitoring docs (can be done anytime; does not affect in-flight GRPO):

### 4.1 `mixed_reward_rate` — arm-specific interpretation

| Arm | Meaning |
| --- | --- |
| **GRPO** | Fraction of prompts with some but not all rollouts correct → useful GRPO signal-density proxy. |
| **Set arms** | **Not** a signal-density proxy. All-wrong prompts with multiple distinct answer/strategy clusters can have **full** set-RL gradient while `mixed_reward_rate == 0`. Use `1 - fraction_filtered` and `mean_unique_clusters_kept` instead. |

**Action:** Fix PLAN §5 row for C1 / `mixed_reward_rate` and `wandb_dashboard_full.md` §5 table (remove “low = no learning signal” for set arms).

### 4.2 `fraction_filtered` — arm-specific interpretation

| Arm | `fraction_filtered` means |
| --- | --- |
| **GRPO** | All N rollouts share the same binary reward (no group-relative signal). |
| **Set arms** | All N rollouts share one cluster id (collapsed mode → zero marginal by construction). |

Already stated in `remaining_arms.md` §5.4 — keep prominent on set-arm dashboard views.

### 4.3 Dashboard panels by arm

| Panel | GRPO | minority_answer / poly_epo_answer | minority_cot |
| --- | --- | --- | --- |
| §1 histogram + coverage | ✓ primary | ✓ primary | ✓ primary |
| `mixed_reward_rate` | ✓ | ✗ or footnote only | ✗ or footnote only |
| `mean_unique_clusters_kept` | ✗ | ✓ primary | ✓ primary |
| C4b correct answer clusters | ✗ | optional | ✗ |
| C4 strategy + judge health | ✗ | ✗ | ✓ primary |

---

## 5. Explicit do-not-add list

Do **not** implement these in `trainer.py` (redundant or misleading):

- `train/n_filtered_prompts` — use `fraction_filtered` × batch size or histogram bins
- `train/mean_reward_std_per_prompt` — determined by `frac_prompts_*_correct` for binary rewards
- `train/pass_at_{2,4,8}` — derivable from histogram in wandb
- In-loop held-out eval pass@k
- **GRPO:** `mean_unique_*` cluster metrics (no clustering substrate)
- **Routine:** C3 percentiles on default dashboard (debug panel only)

---

## 6. Poly-EPO paper mapping (reference)

| Paper Fig. 2 | Our metric | When |
| --- | --- | --- |
| Right — coverage | `train/prompt_coverage` | All arms (live) |
| Left — strategy clusters among **correct** | `train/mean_unique_strategy_clusters_correct` | `minority_cot` only (planned C4) |
| Left — cheap analogue | C4b or §3.2 | C4b = correct-only answers (weak); §3.2 = all-rollout clusters on kept prompts (better ops signal, not in paper) |

---

## 7. Implementation checklist (post–GRPO train)

Use after `remaining_arms.md` objective/clustering work lands:

- [ ] `clusters_grid` in `run_one_grpo_step` for set arms
- [ ] `train/mean_unique_clusters_kept` (§3.2)
- [ ] Surface `adv_marginal_p05/p50/p95` from `adv_out.diagnostics` (C3, debug)
- [ ] C4b optional behind same cluster pass
- [ ] `minority_cot`: judge hook + C4 + §3.3 judge health keys
- [ ] Update `wandb_dashboard_full.md` per §4
- [ ] One-line pointer in `remaining_arms.md` §5 → this doc

**No checklist items** for the GRPO leg already running unless a run is clearly broken and you need a one-off offline analysis.

---

## 8. Cross-references

- Arms 2–4 implementation: [`remaining_arms.md`](./remaining_arms.md)
- PLAN metrics table: [`../PLAN.md`](../PLAN.md) §5
- Probe IDs C1–C4b: [`../probes/05-24_probe_plan.md`](../probes/05-24_probe_plan.md)
- Live dashboard: [`../monitoring/wandb_dashboard_full.md`](../monitoring/wandb_dashboard_full.md)
- Poly-EPO extraction: [`../../../pre-milestone/pilot/docs/analysis/0519_poly_epo_methodology.md`](../../../pre-milestone/pilot/docs/analysis/0519_poly_epo_methodology.md) §6
