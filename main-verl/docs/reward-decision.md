# Locked decision — VeRL MathReward stack (Stage 2+)

**Status:** LOCKED (2026-05-29)  
**Authority:** Mentor / TA OH 2026-05-28 ([`../../main/docs/verl_move_ta_meeting.md`](../../main/docs/verl_move_ta_meeting.md))  
**Scope:** All `main-verl/` training and smokes. **Ignore `main/train/reward.py`** for VeRL runs.

---

## Decision (one paragraph)

We use **upstream VeRL `math_reward.py` semantics end-to-end**: a `\boxed{}` prompt suffix in parquet, **last `\boxed{...}` extraction** from the model completion, **Hendrycks `strip_string` normalization**, and **string equality** vs `reward_model.ground_truth` → reward **0 or 1**. We do **not** use `math_dapo` (Minerva `Answer:` lines), **`math_verify`** (SymPy / HuggingFace Math-Verify), or custom graders ported from `main/`.

---

## Three layers (must stay aligned)

| Layer | Locked choice | Where it lives |
| --- | --- | --- |
| **Prompt** | Append verbatim (maxrl Polaris preprocess): `\nPlease reason step by step, and put your final answer within \boxed{}.` | `main-verl/data/preprocess_polaris_verl.py` → parquet `prompt` |
| **Extraction** | `last_boxed_only_string` → `remove_boxed` on model output | maxrl `verl/utils/reward_score/math.py` (same as upstream `math_reward.py`) |
| **Comparison** | `strip_string` both sides → string `==` (`is_equiv`) | same module |

VeRL does **not** inject a prompt for you. The instruction is **only** what we write into parquet at preprocess time.

---

## Routing (`data_source` → scorer)

| Parquet `data_source` | After our maxrl patch | Scorer module |
| --- | --- | --- |
| **`polaris`** (default) | `math.compute_score` | `verl/utils/reward_score/math.py` |
| **`math_reward`** (alias) | `math.compute_score` | same |

**Before fix (maxrl @ `7197bbb`):** `polaris` → `math_verify.py` — **wrong for this decision.**

**Fix:** maxrl fork commit **`cb8160f cs224r: route polaris/math_reward to math.py reward`** on branch `cs224r-patches`. The Modal image clones the fork at the SHA pinned by `MAXRL_BRANCH_COMMIT` in [`../infra/modal_image.py`](../infra/modal_image.py).

Hydra: `reward_model.enable: false` — built-in rule scorer via parquet routing only. **No** `custom_reward_function.path`.

---

## Explicitly rejected for VeRL

| Option | Why not |
| --- | --- |
| `math_verify` / pip `math-verify` | Symbolic equivalence — mentor asked for Hendrycks string match |
| `math_dapo` default | Parses `Answer:` lines — wrong prompt contract |
| `main/train/reward.py` (Rank-2, mathd∨sympy) | Custom stack only; TA wants VeRL built-ins |
| Hybrid / DAPO prompts from `main/` probes | Out of scope for VeRL migration |

---

## Mentor wording ↔ code map

| Mentor said | We implement |
| --- | --- |
| “MathReward to extract the answer” | `math.py` / upstream `math_reward.py` boxed extract |
| “move to mathreward + `\boxed{}`” | Parquet prompt suffix + boxed parser |
| “whatever is built in to verl” | `default_compute_score` router → `math.py` (patched fork) |

“MathReward” here is **not** the LLM-as-judge benchmark name from `STANDARDS.md`.

---

## Re-upload checklist (after prompt or patch change)

```bash
export MODAL_PROFILE=chicken602
PYTHONPATH=main-verl:main python3 -m data.preprocess_polaris_verl --out-dir main-verl/data --upload
```

Then rebuild Modal image (patch applied) and re-run GRPO smoke.

---

## Sanity test (on Modal worker after patch)

```python
from verl.utils.reward_score import default_compute_score

assert default_compute_score("polaris", r"work \boxed{42}", "42") == 1.0
assert default_compute_score("polaris", r"work \boxed{\frac{1}{2}}", "0.5") == 0.0  # math_reward strict; math_verify would pass
```

---

## References

- Upstream: [verl/utils/reward_score/math_reward.py](https://github.com/verl-project/verl/blob/main/verl/utils/reward_score/math_reward.py)
- maxrl reference preprocess: [examples/maxrl_data_preprocess/polaris.py](https://github.com/tajwarfahim/maxrl/blob/7197bbb46a2ecd866da52f6b401ff20a34fe9390/examples/maxrl_data_preprocess/polaris.py)
- Stage 2 runbook: [`build/stage-02-agent-plan.md`](build/stage-02-agent-plan.md)
- Technical survey: [`verl-reference.md`](verl-reference.md) §3.3
