# Remaining Arm Implementation Spec (Arms 2–4)

This document specifies the implementation work required to add the three set-based training arms:

- `minority_answer`
- `minority_cot`
- `poly_epo_answer`

Assumption: the existing GRPO pipeline remains unchanged.  
All new arms must reuse the same rollout, reward, optimization, checkpointing, and launch flow already used by GRPO.

---

## Scope and non-goals

### In scope

- Set-based advantage computation in `main/train/objective.py`
- Clustering helpers for answer-hash and CoT substrates
- Trainer wiring to pass cluster IDs into objective computation
- Arm-specific configs
- Unit tests for new objective and clustering logic
- Train-time metrics needed for set-arm monitoring

### Out of scope

- Evaluation stack (`eval/passk.py`, held-out eval jobs)
- Reward redesign
- Data freeze changes
- Rollout engine redesign
- Dynamic resampling / curriculum changes
- New arm definitions beyond PLAN’s four-arm set

## Shared design (all set-based arms)

### Invariants (locked by PLAN)

- `N = 8` rollouts per prompt, subset size `k = 4` -> **70** subsets per prompt, **35** subsets containing each rollout `i`.
- **Reward** unchanged: Rank-2 parse + `grade_parsed_answer` (0/1). Same `compute_reward()` for every arm.
- **Marginal advantages** (minority-answer, minority-CoT, poly-epo-answer):
  1. For each subset G, compute scalar `f(G)`.
  2. Baseline = mean of all 70 `f(G)` for that prompt.
  3. Set advantage for G: `f(G) - baseline`.
  4. Rollout i advantage: mean set-advantage over the 35 subsets that include i.
- **Loss**: same clipped surrogate as GRPO (`main/train/loss.py`); set arms use `length_norm: batch_max` (Dr.GRPO / Poly-EPO style — divide each sequence’s token sum by batch `T_max`, not per-sequence length).
- **Filtering**: `keep_mask[p] == False` drops prompt p before backward. Trainer already honors this via `loss.py`.

### Clustering policy note

`minority_answer` and `poly_epo_answer` cluster by normalized parsed-answer identity.  
The initial implementation should use canonical string normalization and hashing (as in pilot code). If stronger mathematical equivalence clustering (e.g., SymPy-equivalent forms) is added later, it should be introduced behind a clearly versioned clustering helper to preserve reproducibility.

### What differs per arm

| Arm | `clusters` source | `f(G)` |
|-----|-------------------|--------|
| `minority_answer` | Answer-hash on `parsed_answer` | Reward of **rarest** cluster in G (random tie among tied-rarest) |
| `minority_cot` | In-loop LLM judge `cluster_id` | Same minority `f(G)` |
| `poly_epo_answer` | Same as minority-answer | `mean(r in G) * (distinct clusters in G) / 4` |

### Reuse principle

Implement one shared set-RL kernel with signature `(rewards, clusters, subset_score_fn) -> AdvantageOut`.  
Each arm should be implemented as a thin wrapper over this kernel plus arm-specific cluster-source selection.  
Do **not** fork the trainer loop per arm.

**Golden reference for math:** `pre-milestone/nancy_explore/run0_analysis/analysis_c/set_score_simulation.py` (`minority_f`, `marginal_from_fG`, `f_poly_score`, `SUBSETS`, `INCL`).

---

## 1. `main/train/objective.py` - set-RL kernel and dispatch

### 1.1 Module-level constants

Precompute once (assert `n_rollouts == 8` at runtime or hardcode for v1):

```python
N_ROLLOUTS = 8
SUBSET_SIZE = 4
_SIZE4_SUBSETS: tuple[tuple[int, ...], ...]  # len 70, combinations(range(8), 4)
_INCL: list[np.ndarray]  # INCL[i] = indices into _SIZE4_SUBSETS where rollout i appears; len 8, each len 35
```

Port indexing logic from `set_score_simulation.py` (`SUBSETS`, `INCL`).

### 1.2 Core API

```python
def set_based_marginal_advantages(
    rewards: torch.Tensor,           # [n_prompts, N]
    clusters: torch.Tensor,          # [n_prompts, N], int cluster ids per rollout
    subset_score_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    rng: np.random.Generator | None,  # per-prompt seeded tiebreak for minority
) -> AdvantageOut:
```

Per prompt `p`:

1. `r = rewards[p].cpu().numpy()`, `c = clusters[p].cpu().numpy()`.
2. For each of 70 subset index tuples `idxs`, compute `r4 = r[idxs]`, `c4 = c[idxs]`, then `fG[k] = subset_score_fn(r4, c4)` (pass `rng` into minority scorer only).
3. `marg = marginal_from_fG(fG)` → length-8 vector (port `marginal_from_fG` verbatim).
4. Stack into `advantages` tensor `[n_prompts, N]`.

`keep_mask` for set arms (do not reuse GRPO's "all rewards equal" rule):

- `keep_mask[p] = False` when **all N rollouts share one cluster id** (collapsed mode -> zero marginal by construction).
- Optionally also False when `max(abs(marg)) < eps` (numerical); primary rule is single-cluster.

Diagnostics (return in `AdvantageOut.diagnostics` for wandb):

- `fraction_filtered`, `n_filtered_prompts` (same names as GRPO).
- Flattened marginal advantages for the current step (or percentiles p05/p50/p95), used for PLAN C3 logging every 100 steps.

### 1.3 Subset scorers

`_minority_subset_score(rewards4, clusters4, rng) -> float`

- `Counter(clusters4)` -> `min_count` -> list of `rarest` cluster ids.
- Tiebreak: `rng.choice(rarest)` (PLAN: random, not average-over-tied-modes).
- Return `mean(rewards4[clusters4 == pick])` (single chosen mode only).
- Used by both `minority_answer` and `minority_cot`.

`_poly_epo_subset_score(rewards4, clusters4) -> float`

- `mean(rewards4) * len(set(clusters4)) / SUBSET_SIZE` (4).
- Answer-hash clusters only (same substrate as minority-answer).

### 1.4 Arm wrappers and `compute_advantages` dispatch

```python
def _minority_advantages(rewards, clusters, *, global_seed, problem_ids) -> AdvantageOut:
    # Per-prompt RNG: np.random.default_rng(global_seed + problem_id) for tiebreak reproducibility

def _poly_epo_answer_advantages(rewards, clusters) -> AdvantageOut:
    # set_based_marginal_advantages(..., _poly_epo_subset_score, rng=None)

def compute_advantages(arm, rewards, clusters=None, *, global_seed=None, problem_ids=None):
    if arm == "grpo":
        return _grpo_advantages(rewards)
    if arm in ("minority_answer", "minority_cot"):
        assert clusters is not None
        return _minority_advantages(...)
    if arm == "poly_epo_answer":
        assert clusters is not None
        return _poly_epo_answer_advantages(...)
    raise ValueError(...)
```

`compute_advantages` currently accepts `clusters` but not `problem_ids`.  
For reproducible minority tie-breaking, include per-prompt seed context (recommended: `global_seed + problem_id`) by passing `problem_ids` from the batch into set-arm calls.

---

## 2. `main/train/clustering.py` (new file)

Provide substrate-agnostic helpers that return `list[int]` of length `N` per prompt.  
Trainer will assemble `clusters_grid: list[list[int]]` and convert to tensor.

### 2.1 Answer-hash (arms 2 and 4)

```python
def answer_hash_clusters(
    parsed_answers: list[str | None],
    parse_ok: list[bool],
) -> list[int]:
```

Normalization (port from `pre-milestone/pilot/train/canonicalize.py`):

- `canonicalize_answer(text)`: strip, remove commas, try `int` → str(int), else lowercase string.
- `cluster_id(answer) -> hash(canonicalize_answer(answer)) % (2**31)` for parse-ok rows.

Parse failures:

- Each failed rollout gets a **unique negative id** (for example, `-1 - rollout_idx`) so failed parses never share a cluster and never create artificial minority clusters.

Rationale: hybrid prompt C may produce different strings that normalize to the same answer. Clustering must use normalized parsed answers, not raw completion text.

### 2.2 CoT clusters (arm 3 only)

```python
def cot_clusters_from_judge(
    assignment: dict[int, int],  # rollout_idx -> cluster_id (already normalized: 100 -> -1)
    n_rollouts: int,
) -> list[int]:
```

- Input is the output of `_assignment_from_poly_epo_payload` in `judge/format.py` (already maps cluster 100 -> -1).
- Return `[assignment[i] for i in range(n_rollouts)]`.

Trainer obtains judge assignments first, then uses this helper to produce the same `list[int]` shape used by the answer-hash path.

---

## 3. `main/train/trainer.py` - wiring in `run_one_grpo_step`

Rename to `run_one_step` is optional but recommended, since this function is no longer GRPO-specific once set arms are added.

### 3.1 After reward loop (where `rewards_grid` / `reward_meta` exist)

**Set-based arms** (`minority_answer`, `minority_cot`, `poly_epo_answer`):

```python
SET_ARMS = frozenset({"minority_answer", "minority_cot", "poly_epo_answer"})

if cfg.arm in SET_ARMS:
    clusters_grid = []
    for p_idx in range(len(batch.prompts)):
        if cfg.arm == "minority_cot":
            # see §4 — judge 8 completions for this problem
            ids = ...  # from JudgeClient
        else:
            parsed = [reward_meta[p_idx][r]["parsed_answer"] for r in range(n_rollouts)]
            ok = [reward_meta[p_idx][r]["parse_ok"] for r in range(n_rollouts)]
            ids = answer_hash_clusters(parsed, ok)
        clusters_grid.append(ids)
    clusters_t = torch.tensor(clusters_grid, dtype=torch.long)
    adv_out = compute_advantages(
        cfg.arm, rewards_t, clusters_t,
        global_seed=cfg.global_seed,
        problem_ids=batch.problem_ids,
    )
else:
    adv_out = compute_advantages(cfg.arm, rewards_t)
```

### 3.2 `length_norm` for loss

Today `_train_step_microbatched` reads `cfg.loss["length_norm"]` only.

Two valid implementation patterns:

- **YAML-driven (minimal):** set `loss.length_norm: batch_max` in each set-arm YAML.
- **Arm-driven (safer):** derive `length_norm` from `cfg.arm`, so a YAML misconfiguration cannot accidentally train set arms with GRPO normalization.

### 3.3 Judge lifecycle (arm 3 only)

- Construct **`JudgeClient` once** at start of `train()` (or first step), not per prompt if batching allows.
- Pass `problem` text from `batch.prompts[p_idx]` and rollout dicts `{"completion": rr.completion_text}` into judge.
- **Do not** pass gold or parsed answer to judge (Group A / STANDARDS).
- On judge failure (truncation, JSON parse failure), fall back to degenerate cluster `-1` to prevent step failure.

### 3.4 Optional rename / call sites

- `run_one_grpo_step` → `run_one_step`; update `group_b_step_probe.py` imports if renamed.
- Group B probe may remain GRPO-only; no required change for arm-2 smoke runs beyond new config.

---

## 4. `main/judge/client.py` (new) - arm 3 only

### 4.1 Interface

```python
class JudgeClient:
    def cluster(self, problem: str, rollouts: list[dict]) -> list[int]:
        """Return cluster id per rollout, length == len(rollouts)."""
```

Internally:

1. `build_judge_messages(problem, rollouts)` from `judge/format.py`.
2. Run LLM (vLLM generate).
3. `_strip_json_fences` + `_assignment_from_poly_epo_payload` → assignment dict.
4. `cot_clusters_from_judge(assignment, n_rollouts)`.

### 4.2 Backends (same interface)

| Backend | When |
|---------|------|
| `LocalVllmJudge` | Second GPU or sidecar vLLM on Modal (Qwen3-4B-Instruct per Group A) |
| `ApiJudge` | Fallback if collocated train+policy+judge VRAM doesn’t fit |

Use `main/probes/group_a_rollout_judge.py` Phase 2 as the implementation reference for model setup and call flow (LLM init, `SamplingParams`, chat templating, truncation handling). Keep JSON parsing centralized through existing helpers.

### 4.3 Modal / GPU layout

- Training entrypoint remains `main/train/trainer.py` (`train_remote`), not a separate `infra/modal_app.py`.
- Arm 3 requires one explicit deployment choice:
  - **Second GPU** on same function (if Modal supports 2× GPU), or  
  - **Separate Modal function** invoked batched per train step (higher latency), or  
  - **Sequential** load: sleep policy vLLM → run judge → wake policy (slow but simplest).

Group A measurements indicate judge time is on the same order as rollout time on H100. Arm 3 therefore has materially higher inference load than rollout-only arms. Single-GPU H200 colocation for train+policy+judge remains unverified and may require a sidecar architecture.

### 4.4 Lazy import

Only import/instantiate `JudgeClient` when `cfg.arm == "minority_cot"` so GRPO and minority-answer don’t pay judge import or VRAM.

---

## 5. Training-time metrics (PLAN §5, train-time only)

**Authoritative priority / interpretation (2026-05-26 review):** [`train_wandb_metrics_verdict.md`](./train_wandb_metrics_verdict.md) — what to add when arms 2–4 ship, what to skip, and dashboard doc fixes. GRPO trains in flight without new keys.

Extend logging beyond existing C1/C1b/C2 metrics currently emitted by `aggregate_train_step_wandb_metrics`.

### 5.1 All set arms — C3

Emit every step or every 100 steps (PLAN-preferred cadence for reduced noise):

- `train/adv_marginal_p05`, `p50`, `p95` over flattened per-rollout marginal advantages from `adv_out.advantages` (only prompts with `keep_mask True`).

### 5.2 Answer-hash arms — C4b

`minority_answer` and `poly_epo_answer`:

- `train/mean_unique_answer_clusters_correct`: per prompt, among rollouts with `reward > 0`, count **distinct** answer-hash cluster ids; mean over batch.

Thread `clusters_grid` into either:
- an optional argument on `aggregate_train_step_wandb_metrics`, or
- a sibling helper that reuses the same reward metadata.

Avoid duplicated reward parsing logic.

### 5.3 CoT arm — C4

`minority_cot` only:

- `train/mean_unique_strategy_clusters_correct`: same metric but on judge cluster ids among correct rollouts only.

GRPO must **not** log C4 (no judge).

### 5.4 Existing diagnostics

Keep logging `fraction_filtered`, `n_kept`, `mean_advantage` — for set arms, `fraction_filtered` means “single-cluster (or zero-signal) prompts,” not GRPO’s “all rewards equal.”

---

## 6. Config files and launch

### 6.1 Single yaml + `arm_profiles`

All arms share `configs/train_real.yaml` (train / rollout / loss blocks). Per-arm overrides live under `arm_profiles` (checkpoint dir, wandb group, `length_norm`). Smoke-only rollout logging: `smoke_probes.rollouts_jsonl_path` (applied when `launch_mode=smoke`).

| Arm | Profile keys |
|------|----------------|
| `grpo` | `checkpoint_dir: .../train_real/`, `length_norm: per_seq`, `wandb.group: train-real` |
| `minority_answer` | `.../train_minority_answer/`, `batch_max`, `train-minority-answer` |
| `poly_epo_answer` | `.../train_poly_epo_answer/`, `batch_max`, `train-poly-epo-answer` |
| `minority_cot` | `.../train_minority_cot/`, `batch_max`, `train-minority-cot` (+ `judge:` block TBD) |

Legacy `configs/train_real_<arm>.yaml` files are **arm-only shims** (`arm: <name>`) merged into `train_real.yaml` at load time.

### 6.2 Launch script

```bash
bash main/scripts/launch_train.sh --mode smoke --arm minority_answer
```

`--arm` sets `CS224R_ARM` / Modal `--arm-override` (host env is not forwarded). `--config` still supported for shims.

---

## 7. Tests (new)

### 7.1 `tests/test_objective_minority.py`

Case A - collapsed cluster

- 8 rollouts, one cluster id, arbitrary rewards → `keep_mask False`, advantages all 0.

Case B - 7 + 1 minority

- 7 rollouts cluster A (reward 0), 1 rollout cluster B (reward 1) → B’s marginal advantage > 0, A’s < 0 (hand-check sign).

Case C - small hand-computable instance

- Use `N=4`, `k=2` only in test context (helper with configurable `n_rollouts`/`subset_size`, or test-only path). Enumerate C(4,2)=6 subsets, compute `f(G)` and marginals by hand, and assert equality. Production path remains fixed at `N=8`.

Case D - Poly-EPO subset score

- One 4-rollout group: rewards `[1,1,0,0]`, clusters `[0,0,1,1]` → `f_poly = 0.5 * 2/4 = 0.25`.

Case E - tie-break reproducibility

- Two clusters tied for rarest in a subset; same `global_seed + problem_id` → same `f(G)` across two calls.

### 7.2 `tests/test_clustering.py`

- Parse-ok identical parsed answers → same cluster id.
- Two parse failures → different negative ids.
- `canonicalize_answer("1,234")` vs `"1234"` → same id (if that’s pilot semantics).

### 7.3 Optional integration smoke test

- `arm: minority_answer`, `CS224R_TOTAL_STEPS=1`, tiny JSONL input: verify one step completes without `NotImplementedError` (Modal or local CPU with mocks).

---

## 8. Recommended implementation order

1. **`objective.py`** — kernel + `_minority_subset_score` + `_poly_epo_subset_score` + dispatch + unit tests (Cases A–E with small k where needed). — **Done (arm 2)**
2. **`clustering.py`** — `answer_hash_clusters` + tests. — **Done (arm 2)**
3. **`trainer.py`** — build `clusters_grid` for `minority_answer` / `poly_epo_answer`, pass clusters + problem_ids, arm-driven `length_norm`, C3/C4b wandb. — **Done (arm 2; judge hook pending arm 3)**
4. **`train_real_minority_answer.yaml`** — smoke 10 steps on Modal. — **Config done; Modal smoke in flight**
5. **`poly_epo_answer`** - dispatch branch + YAML (small delta once kernel exists).
6. **`judge/client.py`** + Modal GPU plan + trainer hook for `minority_cot` + `train_real_minority_cot.yaml` + C4 wandb.

Estimated size: arms 2 and 4 are ~150 LOC of core logic combined; arm 3 is ~200+ LOC and primarily infrastructure.

---

## 9. Edge cases to handle explicitly

| Situation | Behavior |
|-----------|----------|
| All 8 same answer cluster | `keep_mask False` |
| All 8 correct but same answer | Still filtered (no diversity signal) |
| Parse fail on some rollouts | Unique negative cluster ids each |
| Minority tie in subset G | `rng.choice(rarest)` per subset call, seeded per prompt |
| `n_rollouts != 8` in config | Assert fail in set arms (PLAN locks 8) |
| Judge JSON fail | Degenerate -1 for affected prompt (align Group A) |
| Zero kept prompts in batch | Existing `RuntimeError` in trainer — same as GRPO all-uniform batch |

---

## 10. Explicitly out of scope

- `eval/passk.py`, held-out AIME/HMMT rollouts, post-train checkpoint eval jobs.
- Dynamic sampling / DAPO resample-on-all-wrong flow
- `poly_epo_cot` (not in PLAN’s four arms).
- Changing reward, data freeze, or rollout engine for new arms.

---

## 11. File checklist

| File | Action |
|------|--------|
| `train/objective.py` | Add kernel, scorers, dispatch |
| `train/clustering.py` | **New** — answer + cot helpers |
| `train/trainer.py` | Build clusters, pass to objective, length_norm, judge hook, wandb |
| `judge/client.py` | **New** — arm 3 |
| `configs/train_real_minority_answer.yaml` | **New** |
| `configs/train_real_poly_epo_answer.yaml` | **New** |
| `configs/train_real_minority_cot.yaml` | **New** + `judge:` |
| `tests/test_objective_minority.py` | **New** |
| `tests/test_clustering.py` | **New** (recommended) |
| `launch_train.sh` | Optional `--config` flag |

No additional training-stack rewrites are required beyond the files listed above.