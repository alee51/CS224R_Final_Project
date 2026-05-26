# Main experiment — decisions log

Strategic decisions for the `main/` training stack. Pilot-era and Run 0 objective decisions remain in [`pre-milestone/nancy_explore/narrative/decisions.md`](../../pre-milestone/nancy_explore/narrative/decisions.md). Chronology: [`timeline.md`](./timeline.md).

---

## 2026-05-26: Polaris full pool — keep full gold (no integer-only filter)

**Status:** Locked for `source/polaris_train_full.jsonl` materialization (53,291 rows; not the train manifest).

**Decision:** `preprocess_polaris.py` keeps **all** non-empty HF `answer` strings (LaTeX, fractions, symbolic, etc.). Do **not** drop rows that fail `is_integer_gold`. Output → **`main/data/source/polaris_train_full.jsonl`** (see [`data/README.md`](../data/README.md)).

**Evidence:** Random full-gold n800 vs integer-stratified n800 (arm C, offline `grade_parsed_answer`) — pass@1 **~8.5%** vs **~9.4%**, pass@8 **~33%** both; see [`timeline.md`](./timeline.md) §2026-05-26 late night, [`probes/integer_vs_random_fullgold_unified_grade.md`](./probes/integer_vs_random_fullgold_unified_grade.md).

**Still integer-only:** Group A / B / C **probes** (`group_a_rollout_judge._clean_polaris_rows`) — unchanged for historical manifests.

---

## 2026-05-26: Train reward grader — mathd OR sympy (DeepScaleR / rLLM)

**Status:** Implemented in `main/train/reward.py` (`grade_parsed_answer` → `grade_answer_mathd_or_sympy`).

**Decision:** After Rank-2 extraction with **`hybrid_answer_boxed` (arm C)**, mark a rollout correct iff **`grade_answer_mathd(parsed, gold) or grade_answer_sympy(parsed, gold)`** from `main/train/math_grade_deepscaler.py`.

**Lineage (what we cite):** Vendored from [agentica-project/rllm `math_utils/utils.py`](https://github.com/agentica-project/rllm/blob/main/rllm/rewards/math_utils/utils.py) — the same rule as `grade_answer_verl` in that file (boxed extract, then mathd OR sympy). This is the grading stack used in **DeepScaleR-style** RL math training (Polaris / rLLM recipes), not DAPO’s train-time `normalize_final_answer` string equality.

**Why OR, not sympy-only:** On n800 probes, SymPy rescues all observed mathd-only gaps (0 mathd∧¬sympy on parsed rollouts), but mathd is ~100× cheaper and catches rare Hendrycks-only string cases (e.g. `\text{ ` unit strip, `k=42` LHS) if they appear in extractions. OR matches upstream and adds negligible latency vs rollouts.

**Step cost (bs=64, 8 rollouts/prompt = 512 grades/step):** extract+strict ≈ **62 ms/step**; mathd∨sympy ≈ **105 ms/step** (+**44 ms**, **~0.04%** of a ~119 s Group B H200 step) — not ~30% of step time; SymPy is expensive per call but grading is not the bottleneck.

**Empirical note (n800, Rank-2 parsed):** pass rates for `mathd ∨ sympy` equal **sympy-only** on our slice (arm A: 7.72%; arm C: 9.66% vs strict 6.0% / 8.45%). Value is vs **strict Minerva**, not vs dropping mathd.

**Not OOD eval:** Held-out pass@k still uses **Math-Verify** per STANDARDS (separate harness).

**Pairs with:** arm C prompt, Rank-2 extraction, Polaris train freeze — see timeline 2026-05-25 / 2026-05-26.

**Supersedes:** Same-day draft “sympy only”; PLAN § Reward integer-only strict match.

---

## 2026-05-26: Training `batch_size` — lock **64** (not 128) on single H200

**Status:** Locked for v1 GRPO / collocated rollout+train.

**Decision:** **`train.batch_size: 64`** on **one H200**, `rollout.gpu_memory_utilization: 0.45`, `prompt_variant: hybrid_answer_boxed`. Do **not** pursue **`batch_size: 128`** on this stack without new engineering (2-GPU, vLLM KV release before `logprob_fwd`, or microbatched logprob forwards).

**Evidence:** Group B bs=64 fits (~115 GB peak, ~25 GB headroom on 140 GB). Three bs=128 probes (util **0.45**, **0.38**, **0.40**) all completed rollout (1024 sequences) then OOM in `_completion_logprobs_hf` with **~139.4–139.7 GB** already allocated. Details: [`timeline.md`](./timeline.md) §2026-05-26 late evening.

**Note on Poly-EPO “128”:** Their **128 prompts / batch 64** is on **4× H200**, 4B, VeRL — not comparable to our **single-GPU collocated** 1.7B skeleton.

**Does not block:** `train.microbatch` up to `n_kept` (~96 observed at bs=64); §2 Polaris freeze; first real train run.

---

## 2026-05-26: Silence pylatexenc `\frac` warnings during train grading

**Status:** Implemented in `main/train/math_grade_deepscaler.py` (`logging.getLogger("pylatexenc.latex2text").setLevel(ERROR)`).

**Finding (train smoke, Modal `ap-SPj5QSem9RFgU9602NthEF`):** After each 512-sequence rollout batch (64 prompts × 8 rollouts), logs burst with `WARNING: Error in configuration: macro '\frac' failed its substitution!` — **~43 lines** over partial run (~**3%** of `compute_reward` calls per batch: 17/512, 10/512, 16/512). Warnings appear only in the **reward scoring** phase (pylatexenc `latex_to_text` inside `_parse_latex` → sympy `_normalize`), not during vLLM rollout or HF backward.

**Conclusion:** Benign. Early policy outputs **garbage LaTeX** in extracted answers (bare `\frac`, bad braces, etc.); pylatexenc can’t apply its `%s/%s` template and logs, then grading continues (mathd path or partial normalize). Gold-only scans don’t reproduce it — it’s **rollout text**, which we expect to improve under RL.

**Decision:** **Silence** these warnings at the pylatexenc logger. Reasoning: noise is from the **current policy**, not a config bug; we’re training it to produce valid math, not debugging the LaTeX parser on every step. Does not change reward semantics.

**Unrelated:** Same smoke run OOM’d on step 1 `logprob_fwd` (VRAM) — see [`decisions.md`](./decisions.md) § batch size 64.

---

## 2026-05-27: Polaris train prompt filter — proof endings + gold leak

**Status:** Materialized → **`main/data/polaris_train.jsonl`** + `polaris_train.meta.json` (frozen; canonical train manifest).

**Decision:** Drop a row iff:

```text
last_starts_prove(problem)
OR (
  gold_in_prompt(problem, gold)
  AND (
    "prove" in problem.lower()
    OR contains_show_that(problem)   # \bshow\s+that\b
  )
)
```

**Do not** drop on `gold_in_prompt` alone (keeps MCQs / in-stem choices).

**Predicate spec (frozen):** [`timeline.md`](./timeline.md) §2026-05-27 → "Predicate definitions (frozen spec)". Pins down `last_sentence` split rule, `last_starts_prove` anchoring, `gold_in_prompt` matching semantics (case-insensitive substring on whitespace-stripped raw HF answer; no length floor / no `\boxed{}` strip), and the deliberate substring-vs-`\b`-bounded asymmetry between inner `"prove"` and `contains_show_that`. Timeline spec takes precedence if `prompt_heuristics.py` is refactored.

**Implementation:** `should_drop_train_prompt_filter()` in `main/data/prompt_heuristics.py`; `main/scripts/filter_polaris_train.py` (materialize); optional labels via `label_polaris_prompts.py` → `polaris_train_labeled.jsonl`.

**Input / output:** `source/polaris_train_full.jsonl` (53,291) → `polaris_train.jsonl` (51,139). Drop **2,152 (4.0%)**; audit `polaris_train_dropped.jsonl`.

**Rationale (short):** Arm C + mathd∨sympy rewards a **boxed final answer**, not a proof writeup. Last-sentence `Prove …` items (~88% proof-style in manual spot check) are poor GRPO targets; gold-in-prompt is only toxic when paired with prove/show wording (answer leakage), not for MCQs (~9.9k rows with gold in stem, no prove/show).

**Rejected alternatives:** `gold_in_prompt` alone (−23%); `prove` anywhere on the outside (+836 find-all/multi-part); `last_contains_prove` on the outside (+236 “Given …, prove” / formatting variants). Detail: [`timeline.md`](./timeline.md) §2026-05-27.

**Manual QA:** [`probes/prove_prompt_spotcheck_80.md`](./probes/prove_prompt_spotcheck_80.md) (n=80).

**Deferred:** Loosen `last_starts` to strip leading `$`/whitespace before `^prove`; optional `len(gold) >= 3` on leak branch if short-gold false positives show up in rollouts.
