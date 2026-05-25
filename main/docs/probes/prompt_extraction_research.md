# Prompt extraction research for RL-parseable math rollouts

**Date:** 2026-05-24  
**Scope:** VeRL / DAPO / GRPO math stacks, reward–prompt pairing, and recommendations for **Qwen3-1.7B-Base** on **Polaris** (integer gold) with **vLLM** rollouts at **temp=1**.  
**Context:** Pilot soft `\boxed{}` prompt yielded ~50% completions containing `\boxed{}`; mentor pointed at VeRL `gsm8k.py`, `math_dapo.py`, and DAPO recipe configs.

---

## Executive summary

| Finding | Implication for us |
| --- | --- |
| **DAPO production data uses `Answer:` lines, not `\boxed{}` in the prompt** | VeRL’s default `math_dapo` reward (`is_correct_minerva`) is aligned with DAPO HF parquet, not with our pilot `\boxed{}` contract. |
| **`strict_box_verify` exists in `math_dapo.py` but is not wired into default DAPO/GRPO training** | Mentor’s `strict_box` path is an optional code path; live DAPO uses Minerva `Answer:` regex unless you pass `strict_box_verify=True` via a custom reward fn. |
| **VeRL `data_preprocess` scripts append short suffixes to the problem text** | Prompt contract is in the **user message string**, usually as `[{"role":"user","content": question + suffix}]`. |
| **Rollout applies `tokenizer.apply_chat_template(..., add_generation_prompt=True)`** | Chat/Instruct models get role tokens; **Qwen3-1.7B-Base has no HF chat template** — use **raw string prompts** in vLLM (as pilot did). |
| **Pilot @ temp=1:** ~50% `has_boxed`, ~21% `Answer:` line parses | Soft `\boxed{}` + strict boxed reward = sparse reward signal; stronger wording alone is insufficient without format SFT or matching parser. |

**Recommended Group A stack (rank 1):** DAPO verbatim `Answer:` prompt + `math_dapo` default parser (Minerva `Answer:` + `normalize_final_answer` + string equality vs integer gold). Log boxed/strict paths as diagnostics.

---

## 1. VeRL `examples/data_preprocess` — verbatim prompt suffixes

All RL preprocessors store prompts as **chat message lists** (typically single `user` turn). The **instruction suffix** is concatenated onto the raw problem string.

### 1.1 GSM8K

**Source:** [verl/examples/data_preprocess/gsm8k.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py)

```python
instruction_following = 'Let\'s think step by step and output the final answer after "####".'
# ...
question = question_raw + " " + instruction_following
# prompt: [{"role": "user", "content": question}]
# ground_truth: extracted via regex #### (-?[0-9\.\,]+)
```

**Reward pairing:** `data_source = "openai/gsm8k"` → [verl/utils/reward_score/gsm8k.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/gsm8k.py) — last `#### <number>` in **last 300 chars**, `method="strict"` (default).

---

### 1.2 MATH (lighteval)

**Source:** [verl/examples/data_preprocess/math_dataset.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/math_dataset.py)

```python
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
# ...
question = question + " " + instruction_following
# prompt: [{"role": "user", "content": question}]
# ground_truth: remove_boxed(last_boxed_only_string(solution)) from dataset solution field
```

**Reward pairing:** `data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", ...]` → [verl/utils/reward_score/math_reward.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_reward.py) — **last** `\boxed{...}` (brace-balanced), `is_equiv` / `strip_string` normalization (not integer-only).

---

### 1.3 Geometry3k (geo3k)

**Source:** [verl/examples/data_preprocess/geo3k.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/geo3k.py)

```python
instruction_following = (
    r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    r"The reasoning process MUST BE enclosed within   tags. "
    r"The final answer MUST BE put in \boxed{}."
)
# prompt = problem + " " + instruction_following
```

**Reward pairing:** `hiyouga/geometry3k` → [verl/utils/reward_score/geo3k.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/geo3k.py) — `extract_boxed_content` + optional **format_reward** (full response must match `.*\\boxed\{.*\}`) with `format_score=0.1`.

*Note: upstream file uses redacted thinking tag names; geo3k is the clearest VeRL example of **thinking tags + `\boxed{}` in one contract**.*

---

### 1.4 GSM8K multiturn + tool (system + user)

**Source:** [verl/examples/data_preprocess/gsm8k_multiturn_w_tool.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k_multiturn_w_tool.py)

**System message (verbatim):**

```text
You are a math expert. You are given a question and you need to solve it step by step. Reasoning step by step before any tool call. You should use the `calc_gsm8k_reward` tool after step by step solving the question, before generate final answer at least once and refine your answer if necessary. Put your final answer in the format of `#### <answer>`.
```

**User suffix:**

```python
instruction_following = "Let's think step by step and output the final answer after `####`."
```

---

### 1.5 DAPO / AIME multiturn tool scripts

**Sources:**  
- [dapo_multiturn_w_tool.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/dapo_multiturn_w_tool.py)  
- [aime2024_multiturn_w_tool.py](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/aime2024_multiturn_w_tool.py)

These scripts **do not define a new prompt template**; they load `BytedTsinghua-SIA/DAPO-Math-17k` or `AIME-2024` parquet and only inject `tools_kwargs` for code-interpreter multiturn. **The prompt text comes from the HF parquet** (see §2).

---

### 1.6 Summary table (VeRL preprocess)

| Dataset script | Instruction suffix (appended to problem) | `prompt` schema | Gold in parquet |
| --- | --- | --- | --- |
| `gsm8k.py` | `Let's think step by step and output the final answer after "####".` | `[{user}]` | `####` number string |
| `math_dataset.py` | `Let's think step by step and output the final answer within \boxed{}. ` | `[{user}]` | boxed inner from solution |
| `geo3k.py` | ``…`` + `\boxed{}` | `[{user}]` | raw `answer` field |
| `gsm8k_multiturn_w_tool.py` | system + user `####` | `[{system},{user}]` | `####` number |
| `dapo_multiturn_w_tool.py` | *(from HF parquet)* | `[{user}]` | string in `reward_model.ground_truth` |

---

## 2. DAPO-Math-17k / AIME-2024 — prompts baked in parquet (not in preprocess)

Fetched first training row from HF datasets server (`BytedTsinghua-SIA/DAPO-Math-17k`, 2026-05-24).

**`data_source`:** `math_dapo`  
**`reward_model`:** `{"style": "rule-lighteval/MATH_v2", "ground_truth": "<string>"}`  

**Verbatim user `content` template** (problem inserted in the middle):

```text
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
```

AIME-2024 eval parquet uses the **same** wrapper ([HF sample verified](https://datasets-server.huggingface.co/first-rows?dataset=BytedTsinghua-SIA/AIME-2024&config=default&split=train)).

**Critical mismatch to be aware of:** VeRL’s `math_dapo` module also implements `strict_box_verify` (last `\boxed{}` in last 100 chars), but **default DAPO training does not enable it** (§3). The **shipped DAPO prompt asks for `Answer:`**, not `\boxed{}`.

---

## 3. Reward functions and prompt pairing

### 3.1 Router: `default_compute_score`

**Source:** [verl/utils/reward_score/__init__.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/__init__.py)

| `data_source` | Module | Default extraction |
| --- | --- | --- |
| `openai/gsm8k` | `gsm8k.py` | Last `####` number in last 300 chars |
| `lighteval/MATH`, `DigitalLearningGmbH/MATH-lighteval`, `HuggingFaceH4/MATH-500` | `math_reward.py` | Last `\boxed{}`, Hendrycks-style `strip_string` |
| `math_dapo`, `math`, `math_dapo_reasoning`, `aime*` | `math_dapo.py` | **Default:** Minerva `Answer:\s*([^\n]+)` on full response (clipped to last 300 chars) |
| `hiyouga/geometry3k` | `geo3k.py` | Boxed content + format bonus |

DAPO recipe uses **`reward_model.reward_manager=dapo`** ([run_dapo_qwen2.5_32b.sh](https://github.com/verl-project/verl-recipe/blob/main/dapo/run_dapo_qwen2.5_32b.sh)), which calls the same `default_compute_score` unless overridden.

---

### 3.2 `math_dapo.py` — mentor path vs default path

**Source:** [verl/utils/reward_score/math_dapo.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_dapo.py)

**Default (`strict_box_verify=False`):**

```python
# Clip for efficiency
solution_str = solution_str[-300:]

# Minerva-style
answer_pattern = r"(?i)Answer\s*:\s*([^\n]+)"
match = re.findall(answer_pattern, solution_str)
extracted_answer = match[-1] if match else "[INVALID]"
pred = normalize_final_answer(extracted_answer)
# correct if pred == gt (string equality after normalize)
# reward: +1.0 / -1.0
```

**Optional strict boxed (`strict_box_verify=True`):**

```python
pred = pred[-100:]  # or pause_tokens slice
boxed_pred = last_boxed_only_string(pred)  # brace-balanced \boxed{
extracted_pred = remove_boxed(boxed_pred) if boxed_pred else None
# correct if extracted_pred == gt (raw string equality, no full normalize on pred)
```

Repo-wide search (2026-05-24): **`strict_box_verify` appears only in `math_dapo.py`** — not in `dapo_trainer.yaml`, `run_dapo_*.sh`, or `default_compute_score`. Enabling it requires a **custom reward function** passing `strict_box_verify=True` into `math_dapo.compute_score`.

---

### 3.3 GSM8K reward (contrast)

```python
# Last 300 chars only
solutions = re.findall("#### (\-?[0-9\.\,]+)", solution_str)
# strict: no match → reward 0 (not negative)
```

---

### 3.4 Prompt ↔ parser compatibility matrix (VeRL)

| Prompt contract | Parser | VeRL default? | Integer Polaris gold? |
| --- | --- | --- | --- |
| DAPO `Answer:` line | `math_dapo` Minerva + `normalize_final_answer` | **Yes (DAPO)** | Yes if model outputs integer after normalize |
| MATH `\boxed{}` suffix | `math_reward` / `math_dapo` strict_box | MATH GRPO examples | Yes for plain integers in box |
| GSM8K `####` | `gsm8k` strict | GSM8K GRPO | Yes |
| geo3k `` + `\boxed{}` | geo3k boxed + format partial credit | Vision/geo | Depends on grader |

---

## 4. How VeRL wraps prompts for Qwen / chat models

### 4.1 Dataset → `raw_prompt` messages

[verl/utils/dataset/rl_dataset.py](https://github.com/verl-project/verl/blob/main/verl/utils/dataset/rl_dataset.py) loads `prompt` column as `list[{"role","content"}]`, filters by length using:

```python
tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, tokenize=True, **apply_chat_template_kwargs)
```

### 4.2 Rollout → `apply_chat_template`

[verl/experimental/agent_loop/single_turn_agent_loop.py](https://github.com/verl-project/verl/blob/main/verl/experimental/agent_loop/single_turn_agent_loop.py) → [verl/utils/chat_template.py](https://github.com/verl-project/verl/blob/main/verl/utils/chat_template.py):

- `add_generation_prompt=True` for generation.
- **Qwen3.5 fallback:** dummy user message prefix if template requires ≥1 user message.
- `remove_system_prompt` option strips leading system tokens after tokenization.
- Multimodal: processor path tokenizes rendered chat string.

### 4.3 Qwen3-1.7B-Base (our model)

Pilot logs show **404** for `Qwen/Qwen3-1.7B-Base/chat_template.jinja` — no bundled chat template. VeRL chat wrapping is **ill-defined** for this checkpoint.

**Pilot / recommended inference pattern:** pass a **single plain-text prompt string** to vLLM (no `apply_chat_template`), as in [pre-milestone/pilot/train/rollout_engine.py](https://github.com/verl-project/verl/blob/main/pre-milestone/pilot/train/rollout_engine.py):

```python
PROMPT_TEMPLATE = (
    "Solve the following math problem. Reason step by step, "
    "and put your final answer within \\boxed{{}}.\n\n"
    "{problem}\n\n"
)
```

For DAPO alignment, swap the template body for the §2 parquet text (still as one string, not chat-json).

---

## 5. verl-recipe DAPO configs (prompt-adjacent)

| Item | Location | Notes |
| --- | --- | --- |
| Data download | [dapo/prepare_dapo_data.sh](https://github.com/verl-project/verl-recipe/blob/main/dapo/prepare_dapo_data.sh) | Pulls HF `dapo-math-17k.parquet` / `aime-2024.parquet` — prompts inside parquet |
| Trainer defaults | [dapo/config/dapo_trainer.yaml](https://github.com/verl-project/verl-recipe/blob/main/dapo/config/dapo_trainer.yaml) | `reward_manager: dapo`, `overlong_buffer_cfg.enable: False` by default |
| Production run | [dapo/run_dapo_qwen2.5_32b.sh](https://github.com/verl-project/verl-recipe/blob/main/dapo/run_dapo_qwen2.5_32b.sh) | `data.prompt_key=prompt`, `max_response_length=20480`, `enable_overlong_buffer=True`, `overlong_buffer_len=4096`, `filter_groups` on `acc`, **no `strict_box_verify`** |
| Overlong shaping | [dapo/README.md](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md) | Linear penalty on response length above `max_resp_len - buffer.len`; **not** the same as “overlong filtering” (dropped in paper FAQ) |
| Dynamic sampling | Same README | Resample until prompt groups have mixed `acc` — **rejection sampling at group level**, not format filtering |

**Overlong buffer (enabled in reference 32B run):**

```yaml
reward_model:
  overlong_buffer:
    enable: True
    len: 4096          # tokens
    penalty_factor: 1.0
data:
  max_response_length: 20480  # 16384 + 4096
```

Penalty applied in [DAPORewardManager](https://github.com/verl-project/verl/blob/main/verl/workers/reward_manager/dapo.py) **after** rule-based score.

---

## 6. Survey: other 2024–2026 math RL prompt/parser stacks

| Method | Prompt contract | Parser / reward | Reported format compliance | Fits Qwen3-1.7B-Base + vLLM + integer Polaris? |
| --- | --- | --- | --- | --- |
| **DAPO (BytedTsinghua-SIA)** | §2 `Answer:` template | `math_dapo` Minerva line + normalize; ±1 reward | Used at scale; parse rate not tabulated in recipe README | **Best default** — matches VeRL DAPO stack |
| **VeRL MATH GRPO** | `\boxed{}` suffix (`math_dataset.py`) | `math_reward` last boxed | Standard Hendrycks eval tradition | Good if model taught `\boxed{}`; base model ~50% boxed @ temp=1 in our pilot |
| **GSM8K / ReFT** | `####` suffix | Last `####` in tail window | High on GSM8K-style models | Wrong surface form for competition math |
| **DeepSeek-R1-Zero RL** | User/Assistant template; thinking + answer **tags** (format RM) | Rule verifier on boxed/math outcome inside tags | Emergent tags; format reward during RL | Needs Instruct-style template + tag tokens; heavy format RM engineering |
| **DeepSeek-R1 (inference guide)** | User prompt may add: “put your final answer within `\boxed{}`” | Eval uses boxed | N/A for RL | Soft recommendation only — same class as our pilot |
| **geo3k / R1-style VeRL** | `` + `\boxed{}` | Boxed extract + 10% format reward | Format reward nudges `\boxed{}` presence | Tag tokens absent on Base — would need SFT cold start |
| **Geometry3k `format_score`** | Same as geo3k | Partial credit for any `\boxed{}` in response | Implicit ~90%+ format with joint training | Optional **shaping** only; we want pure 0/1 for minority voting |
| **DAPO dynamic sampling** | Same prompt | Same parser; **discard all-0 / all-1 groups** | Improves reward variance, not parse rate | Compatible — trainer concern, not probe |
| **Dr.GRPO** | Same as GRPO | Same parser; changes **loss agg**, not format | N/A | Compatible later |
| **Poly-EPO paper** | **Not specified** for policy | Binary RLVR on answer correctness (exact match clustering) | N/A | Paper silent; our pilot `\boxed{}` was a guess |
| **OpenMathReasoning SFT (verl-recipe)** | Raw `problem` only in SFT user turn | Teacher `generated_solution` in data | SFT teaches format from teacher | **Format SFT before RL** is a separate phase — not in current PLAN |
| **Math-Verify (HF)** | Agnostic | LaTeX/sympy verify | Eval-oriented | STANDARDS: **OOD eval only**, not train reward |

---

## 7. Local pilot evidence (Qwen3-1.7B-Base, temp=1, `\boxed{}` prompt)

From `pre-milestone/pilot/artifacts/run0_proxy/20260519T190202Z/cleaned/signal_investigation.md` and Run 0 qual:

| Metric | Value |
| --- | --- |
| Completions with `\boxed{` | **2023 / 4000 (50.6%)** |
| `extract_path_clean = boxed_balanced` | 50.4% |
| `extract_path_clean = answer_line` | 21.3% |
| `runon_rejected` | 10.7% |
| Strict pilot `is_correct` (exactly one shallow `\boxed{` int) | 8.1% rollout accuracy |

**Interpretation:** With only a soft `\boxed{}` instruction, the model often follows **`Answer:`** or free-form endings anyway — matching DAPO’s parser better than our strict boxed training reward. “Stronger soft prompt + reward=0” does not fix the **train–eval parser mismatch**.

---

## 8. Ranked prompt+parser stacks for Polaris integer gold

### Rank 1 (recommended for Group A + training): **DAPO `Answer:` + `math_dapo` default**

**Prompt (verbatim from DAPO parquet, `{problem}` = Polaris `problem`):**

```text
Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:".
```

**Parser:** Port `math_dapo.compute_score` default path — last `Answer:` line in last 300 chars, `normalize_final_answer`, compare to `str(gold)` (integer). Map reward to **0/1** for our trainer (VeRL uses +1/−1).

**Rationale:**

- This is what **VeRL DAPO actually trains** against (`data_source=math_dapo`).
- Base model already emits `Answer:` ~21% of the time even under `\boxed{}` prompt; DAPO wording should **raise** parseable lines.
- No chat template required.
- Aligns with mentor’s `math_dapo.py` reference (default branch, not unused `strict_box`).

**Risk:** Non-integer expressions in `Answer:` (e.g. `\frac{a}{b}`) — rare on Polaris gold; log `parsed_is_int`.

---

### Rank 2: **DAPO prompt + hardened multi-path parser (pilot `answer_clean`)**

Same prompt as Rank 1; parser order:

1. Brace-balanced last `\boxed{}` inner  
2. Last `Answer:` line (with run-on rejection)  
3. Optional: last line fallback **disabled for reward** (parse fail = 0)

**Rationale:** Maximizes **measurable** parse rate without changing prompt; still reports `extract_path` for diagnostics. Use if Rank 1 parse rate <90% on Group A.

**Risk:** Reward–prompt contract is looser (model rewarded without `Answer:` if it boxes) — may slow format convergence.

---

### Rank 3: **MATH-style `\boxed{}` + brace-balanced strict parser**

**Prompt (VeRL `math_dataset.py` suffix):**

```text
Let's think step by step and output the final answer within \boxed{}.
```

Or STANDARDS pilot header + problem.

**Parser:** `last_boxed_only_string` + integer normalize; reward only if extract succeeds (optionally `strict_box_verify` semantics).

**Rationale:** Matches Hendrycks/MATH RL literature and mentor `strict_box` mention.

**Risk:** **~50% `has_boxed`** at temp=1 on 1.7B Base without format SFT → sparse RL signal; contradicts DAPO production pairing.

---

### Not recommended for Group A primary path

| Stack | Why not |
| --- | --- |
| GSM8K `####` | Wrong delimiter for competition-style problems |
| `strict_box_verify` alone with DAPO `Answer:` prompt | Prompt doesn’t ask for box; strict box fires rarely |
| Stronger `\boxed{}` wording only | Pilot shows compliance ~50%; 0 reward doesn’t teach format without SFT |
| Math-Verify train reward | Overkill for integer gold; STANDARDS reserves for OOD eval |

---

## 9. Recommended decision for Group A probe

**Adopt Rank 1** for Phase 1 rollouts and wandb parse-rate panels:

1. **Prompt:** DAPO verbatim template (§8 Rank 1) with Polaris `problem` substituted.  
2. **Inference:** Plain string to vLLM; **no** `apply_chat_template` on Qwen3-1.7B-Base.  
3. **Reward/parser:** `math_dapo` Minerva path, 0/1 reward, integer-normalized equality vs Polaris `answer`.  
4. **Secondary logging only:** `has_boxed`, `strict_box_verify` score, `answer_clean` paths — do **not** use alternate parsers for primary `reward` / `parse_rate` (per STANDARDS).

**If Group A `parse_rate` (primary) < ~90%:** escalate to Rank 2 same prompt; if still low, schedule **short format SFT** (geo3k-style or OpenMathReasoning-style) before RL rather than longer `\boxed{}` instructions.

**Do not** enable DAPO overlong buffer in Group A (probe uses `max_response_length=4096`; overlong shaping targets 16k+ regimes).

---

## 10. What to log in Group A to validate the choice

Per-rollout (wandb):

| Field | Purpose |
| --- | --- |
| `prompt_variant` | e.g. `dapo_answer_v1` |
| `has_boxed` | bool; `\boxed{` in completion |
| `has_answer_line` | bool; `(?i)Answer\s*:` present |
| `parsed_answer` | str or null (primary parser) |
| `parse_ok` | bool; primary parser got a non-empty normalized answer |
| `parsed_is_int` | bool; Polaris-compatible |
| `strict_box_pred` | optional; `strict_box_verify=True` extraction |
| `strict_parse_ok` | bool; strict box path got a value |
| `extract_path_clean` | if running secondary `answer_clean` (diagnostic only) |
| `reward` | 0/1 primary |
| `reward_would_be_boxed` | diagnostic 0/1 if we had used strict box only |
| `length_tokens` | tie to run-on / truncation |
| `difficulty_band` | Polaris stratification |

Per-prompt aggregates:

| Field | Purpose |
| --- | --- |
| `parse_rate` | fraction with `parse_ok` |
| `pass_rate` | fraction with `reward==1` |
| `mixed_reward` | fraction correct ∈ (0,1) across 8 rollouts |
| `has_boxed_rate` | format compliance |
| `answer_line_rate` | DAPO contract compliance |

**Decision gates (from probe plan):**

- Primary `parse_rate` > ~90% → lock parser for `main/train/reward.py`.  
- If `has_boxed_rate` high but `parse_ok` low → parser mismatch, not prompt length.  
- If `answer_line_rate` ≪ `parse_ok` → model using other formats; consider Rank 2 or SFT.

---

## 11. Implementation pointers (lift, not import)

| Component | Source to port |
| --- | --- |
| Primary parser | [verl/utils/reward_score/math_dapo.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_dapo.py) — `normalize_final_answer`, `is_correct_minerva`, clip `[-300:]` |
| Optional strict diagnostic | Same file — `is_correct_strict_box`, `[-100:]` |
| Hardened fallback | [pre-milestone/pilot/train/answer_clean.py](https://github.com/verl-project/verl/blob/main/pre-milestone/pilot/train/answer_clean.py) |
| DAPO trainer knobs (later) | [verl-recipe/dapo/run_dapo_qwen2.5_32b.sh](https://github.com/verl-project/verl-recipe/blob/main/dapo/run_dapo_qwen2.5_32b.sh) |

---

## 11b. Design space — extracting answers from non-JSON-capable models

**Why this section exists:** the DAPO `Answer:` template (§ 2, § 8 Rank 1) reads awkwardly — it repeats the format instruction twice and doesn't use `\boxed{}` like much of the MATH RL literature. This appendix documents the full design space so the choice is informed and so we don't re-derive it every time someone reads the prompt and recoils.

### The fundamental problem

Qwen3-1.7B-Base has no JSON-mode, no grammar-constrained decoding configured, no function-calling, no chat template. We need to recover a structured integer from a free-form CoT completion, automatically, every rollout, with high enough success rate that the RL reward signal isn't drowned in parse failures.

### Patterns researchers use

**1. Delimiter-based extraction (the dominant pattern).** Teach the model to emit a specific marker; regex on output.

| Marker | Lineage | Compliance on Qwen3-1.7B-Base @ temp=1 (pilot) |
| --- | --- | --- |
| `Answer:` line | Minerva (2022), DAPO (2025) | 21% (under `\boxed{}` prompt — not the contract) |
| `\boxed{...}` | Hendrycks MATH (2021), R1-lineage | ~50% (pilot Run 0) |
| `####` | GSM8K (2021), ReFT | n/a — wrong format for competition math |
| `<answer>...</answer>` tags | DeepSeek-R1-Zero | requires SFT cold start to teach tag tokens |

Failure mode of all of these: low baseline compliance → sparse reward early in training. Standard mitigation: format compliance converges to >95% within ~100-200 RL steps as the model gets rewarded for parseable outputs. Empirically reliable across DAPO, R1, Dr.GRPO.

**2. Constrained decoding** (vLLM `guided_decoding`, Outlines, lm-format-enforcer). Two sub-flavors:

- **Whole-output grammar.** Forces entire response into a schema. Kills CoT quality — model can't reason freely. Not used in published RL.
- **Free CoT + forced suffix.** Free generation until a trigger, then regex/grammar on the tail (e.g., `Answer: <integer>`). Sidesteps the format-compliance problem at the cost of vLLM complexity and slight latency. Possible in vLLM via `guided_regex` with custom stop logic. **Rarely seen in published RL papers** — most stacks just rely on delimiter convergence because it works.

**3. Two-stage extractor.** Big model generates free-form; small extractor model pulls the answer.
- Used by some eval harnesses (e.g., LLM-as-judge for MMLU subsets).
- Doubles inference cost; extractor can be wrong; introduces a second model dependency.
- The symbolic-math version of this is Math-Verify, which we already reserved for **OOD eval only** per STANDARDS.

**4. Stop-sequence completion.** Prompt ends with `Answer: ` and you stop after one line. Doesn't work with free CoT in one call. Would require two calls per rollout (CoT, then re-prompt for the answer) — doubles cost and breaks single-shot logprob reuse.

**5. JSON schema in prompt without enforcement.** Just ask for `{"answer": ...}` in the system prompt. Base models comply ~30-60% depending on training-data exposure to JSON. Worse than `Answer:` lines on Qwen3-1.7B-Base, which has more pretraining exposure to `Answer:` than to math-in-JSON.

### Why the DAPO template reads awkwardly

The repetition — first "should be of the form Answer: $Answer" before the problem, then "Remember to put your answer on its own line after 'Answer:'" after — is intentional belt-and-suspenders. Base models forget instructions mid-long-generation; bookending the format rule pulls attention back at the end. It's a known pattern in instruction-tuning data (frame → content → restate). Ugly English, validated behavior.

**Provenance:** the template was extracted verbatim from row 0 of `BytedTsinghua-SIA/DAPO-Math-17k` (HF parquet, `prompt[0].content`, fetched 2026-05-24). Not paraphrased from the paper — copy-pasted from the actual training data DAPO's production runs were trained against. The same wrapper appears in their AIME-2024 eval parquet.

### Why we picked delimiter-based (Pattern 1) with the `Answer:` marker

1. **Empirical fit to the base model.** Pilot shows ~21% `Answer:` compliance even under a `\boxed{}` prompt — meaning the model wants to emit `Answer:` naturally. The DAPO prompt should pull that rate higher.
2. **`\boxed{}` is worse at ~50% baseline** (pilot Run 0) and would need format SFT to lift to RL-usable.
3. **DAPO is what VeRL trains at scale** — `data_source="math_dapo"` → this exact prompt → known to converge on Qwen-scale base models.
4. **No model-side requirements** — no chat template, no grammar setup, no second inference call, no extra model.

### Escalation paths if Group A `parse_ok` is low

These are pre-decided so we don't re-litigate under time pressure when readouts land:

| Probe `parse_ok` | Action |
| --- | --- |
| >~90% | Lock current parser. Done. |
| 70–90% | Switch to Rank 2 (same DAPO prompt, multi-path parser — § 8 Rank 2). |
| <70% | Format SFT cold start before RL (geo3k-style or OpenMathReasoning-style, ~1 epoch on teacher solutions). |
| Persistent <70% even after SFT | Reach for **Pattern 2 free-CoT + forced suffix** as a last resort: vLLM `guided_regex` on `Answer: -?\d+$` for the last line, after free generation. Adds complexity but guarantees parseable output. **Do not pre-build this** — it's an emergency lever, not the default. |

### What's explicitly out of scope for Group A

- Adding a constrained-decoding arm. Separate decision after Group A data lands.
- Switching to `<think>...</think><answer>...</answer>` tags. Would require SFT cold start; not worth the path-divergence from DAPO baseline.
- Tool-calling / function-calling. Base model can't.
- Math-Verify for train reward. Reserved for OOD eval per STANDARDS.

---

## 12. References

- VeRL GSM8K preprocess: https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py  
- VeRL MATH preprocess: https://github.com/verl-project/verl/blob/main/examples/data_preprocess/math_dataset.py  
- VeRL geo3k preprocess: https://github.com/verl-project/verl/blob/main/examples/data_preprocess/geo3k.py  
- VeRL `math_dapo` reward: https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_dapo.py  
- VeRL reward router: https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/__init__.py  
- VeRL RL dataset / chat template: https://github.com/verl-project/verl/blob/main/verl/utils/dataset/rl_dataset.py , https://github.com/verl-project/verl/blob/main/verl/utils/chat_template.py  
- verl-recipe DAPO: https://github.com/verl-project/verl-recipe/tree/main/dapo  
- DAPO-Math-17k HF: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k  
- Polaris HF: https://huggingface.co/datasets/POLARIS-Project/Polaris-Dataset-53K  
- Project STANDARDS: [main/docs/STANDARDS.md](../STANDARDS.md)  
- Probe plan: [main/docs/probes/05-24_probe_plan.md](./05-24_probe_plan.md)
