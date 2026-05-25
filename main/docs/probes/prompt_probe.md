# Prompt A/B/C probe — extract-format comparison

**Drafted:** 2026-05-25. **Updated:** 2026-05-25 (n800, H100-only, offline Rank-2 rescore).

**Purpose:** decide whether the DAPO `Answer:` prompt (Group A) is actually the best extraction format for Qwen3-1.7B-Base on Polaris, or whether a `\boxed{}`-native prompt aligns better with the model's pretraining bias and gives higher parse rates / mixed-reward density.

**Related:** [`group_a_results.md`](./group_a_results.md), [`prompt_extraction_research.md`](./prompt_extraction_research.md) §8 (ranked stacks).

---

## 1. Why this probe exists

Group A's DAPO `Answer:` prompt gave (200-prompt diagnostic; 800-run rescored offline):

- `has_answer_line`: ~56% (Minerva parser target)
- `has_boxed`: ~35% (pretraining bias bleeding through)
- Rank-2 (boxed-first ∪ Minerva) `parse_ok`: **~84%** (offline rescore)

Two open questions the readout couldn't answer:

1. **Is the model "fighting" the DAPO prompt?** Qwen3-1.7B-Base was pretrained on math content where `\boxed{}` is the dominant answer convention. A prompt that aligns with the model's prior might lift total compliance above 84%.
2. **Does the "prompt echo" failure mode disappear under a different prompt?** Completions that literally echo `Answer:` from the instruction with nothing after. Arm B removes that failure mode by construction.

**The clean experiment:** rerun Phase 1 on the **same 800-problem manifest** as Arm A (n800) with alternate prompts B and C; **all arms on H100**; Rank-2 metrics computed **offline** on saved `completion` text.

---

## 2. The three arms

All three arms substitute Polaris `problem` text into `{problem}`. All other knobs (model, temperature, seeds, sampling parameters, manifest) are identical — only the prompt string differs.

### Arm A — DAPO `Answer:` (control; complete)

**Source:** verbatim from `BytedTsinghua-SIA/DAPO-Math-17k` parquet, row 0, `prompt[0].content` (fetched 2026-05-24). Identical to VeRL `data_source="math_dapo"`.

**Full prompt (what the model sees):**

```text
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
```

**Python:** `dapo_answer_v1` in `main/train/prompts.py` (`DAPO_PROMPT_TEMPLATE`).

**Artifacts:** `probes/05-25/group_a_n800/phase1_rollouts.jsonl` (6400 rollouts). Wandb `mu8kj4ll`.

**Rank-2:** offline rescore only (no re-run).

---

### Arm B — VeRL MATH `\boxed{}` (new)

**Source:** verbatim from VeRL [`examples/data_preprocess/math_dataset.py`](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/math_dataset.py) — `data_source="lighteval/MATH"`. **Do not paraphrase.**

**Construction in VeRL:** `question = problem + " " + instruction_following` (suffix, single line).

**Full prompt:**

```text
{problem} Let's think step by step and output the final answer within \boxed{}.
```

**Python:** `verl_math_boxed` → `VERL_MATH_BOXED_TEMPLATE` in `main/train/prompts.py`.

**Artifacts:** `probes/05-25/prompt_b/phase1_rollouts.jsonl`

---

### Arm C — Hybrid `Answer: \boxed{}` (new)

**Source:** **none** — constructed for this probe only (not a validated RL recipe).

**Full prompt:**

```text
Solve the following math problem step by step. End your response with the final answer on its own line, formatted exactly as: Answer: \boxed{$Answer}

{problem}

Remember: the last line must be "Answer: \boxed{...}" with your final answer inside the box.
```

**Python:** `hybrid_answer_boxed` → `HYBRID_ANSWER_BOXED_TEMPLATE` in `main/train/prompts.py`.

**Artifacts:** `probes/05-25/prompt_c/phase1_rollouts.jsonl`

**Risk:** only adopt if materially beats A and B; ties lose to A (DAPO) or B (VeRL MATH).

---

## 3. What stays constant across arms

| Knob | Value |
|---|---|
| Model | `Qwen/Qwen3-1.7B-Base` (plain string, no chat template) |
| Problem manifest | **`probes/05-25/group_a_n800/manifest.jsonl`** (800 problems, `problem_id` 0–799) — **reuse, do not resample** |
| Rollouts per prompt | 8 |
| Temperature | 1.0 |
| Seeds | `global_seed + problem_id * 8 + rollout_idx` (STANDARDS) |
| `max_prompt_length` / `max_response_length` | 1024 / 4096 |
| `gpu_memory_utilization` | 0.90 |
| **GPU** | **H100 only** (`modal_price_per_sec: 0.001097`) — same SKU for A/B/C |
| Phase 2 (judge) | **Skip** (Phase-1-only via `run_phase1`) |

**Only differences:** `prompt_variant`, rollouts output path, wandb run name.

| Arm | `prompt_variant` | Rollouts path |
|-----|------------------|---------------|
| A | `dapo_answer_v1` | `probes/05-25/group_a_n800/` |
| B | `verl_math_boxed` | `probes/05-25/prompt_b/` |
| C | `hybrid_answer_boxed` | `probes/05-25/prompt_c/` |

**Wandb group (all arms):** `probe-prompt-ABC-05-25-n800`

---

## 4. Metrics (offline Rank-2 rescore)

Live Phase 1 logs minimal fields (same as Group A today: `completion`, `reward`, `parse_ok`, `has_boxed`, etc.). **All A/B/C comparison metrics** come from `main/scripts/rescore_rollouts_rank2.py` on saved jsonl.

**Per-rollout (rescored):**

| Field | Notes |
|---|---|
| `prompt_variant` | from manifest / config |
| `has_answer_line`, `has_boxed` | diagnostics |
| `parsed_answer_minerva`, `parse_ok_minerva` | Minerva on last 300 chars |
| `parsed_answer_boxed`, `parse_ok_boxed` | brace-balanced last `\boxed{}` on last 300 chars |
| `parse_ok_rank2`, `extract_path` | Rank-2 order below; `reward` = 0/1 under Rank-2 |
| `length_tokens`, `finish_reason` | from live jsonl |

**Rank-2 extraction order** (`main/train/reward.py`):

1. **Arm C only:** `Answer:\s*\\boxed\{([^}]+)\}` on clipped tail (hybrid regex)
2. Brace-balanced last `\boxed{}` inner → `normalize_final_answer`
3. Minerva `Answer:\s*([^\n]+)` → `normalize_final_answer`
4. Else `extract_path: "none"`, `parse_ok_rank2: false`

**Per-arm aggregates:** `parse_ok_rank2`, `parse_ok_minerva`, `parse_ok_boxed`, `has_boxed_rate`, `mixed_reward` (Rank-2 reward), per-band tables.

---

## 5. Decision rule

(Unchanged — compare offline rescored aggregates.)

**A wins:** Rank-2 `parse_ok` and mixed-reward within ±2pp of B and C → lock DAPO.

**B wins:** B beats A by >5pp parse_ok_rank2 **and** >2pp mixed-reward → VeRL MATH prompt + Rank-2 parser.

**C wins:** C beats A and B by >5pp on both → hybrid + note in PLAN/STANDARDS (unvalidated recipe).

**Else:** default A; document in `group_a_results.md`.

---

## 6. Implementation

| File | Change |
|------|--------|
| `main/train/prompts.py` | `VERL_MATH_BOXED_TEMPLATE`, `HYBRID_ANSWER_BOXED_TEMPLATE`, `format_problem(problem, variant=...)` |
| `main/train/reward.py` | `extract_rank2(completion, gold, prompt_variant=...)` for offline + tests |
| `main/probes/group_a_rollout_judge.py` | `prompt_variant` from yaml; `reuse_manifest: true` skips Polaris sample; log variant in wandb |
| `main/scripts/launch_probe_prompt.sh` | `modal run --detach ...::run_phase1` |
| `main/scripts/rescore_rollouts_rank2.py` | rescore any arm jsonl → summary + optional rescored jsonl |
| `main/configs/probe_b_prompt_05-25_n800.yaml` | B: `verl_math_boxed`, reuse manifest |
| `main/configs/probe_c_prompt_05-25_n800.yaml` | C: `hybrid_answer_boxed`, reuse manifest |

**Launch (repo root):**

```bash
bash main/scripts/launch_probe_prompt.sh main/configs/probe_b_prompt_05-25_n800.yaml
bash main/scripts/launch_probe_prompt.sh main/configs/probe_c_prompt_05-25_n800.yaml
```

---

## 7. Sequencing

```
Arm A n800 Phase 1 (done) → manifest on volume
  ├── Arm B Phase 1 (H100, parallel)
  └── Arm C Phase 1 (H100, parallel)
       └── offline rescore A + B + C → prompt_probe readout → lock prompt
```

**GPU $/throughput:** not mixed across arms; optional later H100 vs H200 smoke on identical slice.

---

## 8. Resolved

- Run **both** B and C (not optional).
- **Same GPU** (H100) for all arms.
- **Offline Rank-2** for A/B/C (cleaner than live duplicate logging).
- Arm C is **not** from a published recipe; A and B are verbatim cited sources.
