# Poly-EPO Methodology Extraction

**Paper:** Poly-EPO: Training Exploratory Reasoning Models  
**Authors:** Ifdita Hasan Orney*, Jubayer Ibn Hamid*, Shreya S Ramanujam, Shirley Wu, Hengyuan Hu, Noah Goodman, Dorsa Sadigh, Chelsea Finn (Stanford)  
**Source:** arXiv (submitted May 2026, PDF 27 pages)  
**Extracted:** 2026-05-19 for CS224R pilot alignment

---

## 1. Prompt Template

The paper does **not** provide an explicit verbatim prompt template used to wrap math problems for the policy model. Section A (Implementation Details) only describes the LM-judge clustering prompt (§A.1) and hyperparameters (§A.2). The paper states they use POLARIS-53k as training data and Qwen-3-4B-Base as the base model, but the exact system/user message wrapping math problems before feeding to the policy is not quoted anywhere in the paper.

**What IS given verbatim** is the LM-judge clustering prompt (§A.1, pp. 20-22):

**Instruction Block (static, passed to judge):**

```
Your ONLY task is to cluster the {n_responses} responses into buckets based on their
reasoning algorithm, including both the overall strategy and the methods used at key
intermediate steps.

**INPUT FORMAT:** You will receive:
1) A "Context" describing the task.
2) A numbered list of Responses from 1 to {n_responses}. Each response contains a reasoning
process and final answer.

Note: Responses may or may not explicitly state their strategy; you must infer the strategy
by analyzing the mathematical steps taken.

**CLUSTERING CRITERIA:**
(1) Macro-strategy: The overall conceptual framework (e.g., recursion vs infinite series;
prime factorization vs gcd-based formula).

(2) Micro-strategy: The specific method used to resolve key intermediate steps. Examples
include: how absolute values are removed (+- case split vs squaring), how intervals are
partitioned, or how a basis is chosen.

**CLUSTERING RULES:** - Cluster strictly based on logic and approach. NOT on wording, tone,
formatting, or final answer.

- Two responses share a cluster_id IF AND ONLY IF they use the same macro-strategy AND the
same micro-strategy at every key step.

- Arithmetic errors do NOT create new clusters if the underlying logic is identical.
- **SPECIAL CLUSTER 100:** You MUST assign `cluster_id: 100` to any response that is:
  * Gibberish (random characters, nonsense strings).
  * Irrelevant to the math problem (off-topic text).
  * Non-mathematical reasoning (e.g., writing code to solve it instead of math, or making a
    random guess at the final answer without logical steps).

**OUTPUT RULES (STRICT):** 1. Respond ONLY with a JSON object. No text outside the JSON.
2. The JSON must contain exactly {n_responses} keys: "1", "2", ..., "{n_responses}".
3. The value for each key must be:
  "chain_of_thought": "Macro: [short description]. Micro: [short description]."
  "cluster_id": integer.

4. chain_of_thought must be concise and avoid repeating the actual calculations.
```

**Instance-specific suffix (dynamic, appended per problem):**

```
**Context:**
<problem description>

**Responses:**
1. <response 1>
2. <response 2>
...
{n_responses}. <response {n_responses}>
```

**Implication for our pilot:** The paper does not specify the prompt template used for the policy model (Qwen-3-4B-Base). Our pilot currently uses `"Solve the following math problem step by step. The last line of your response must be of the form Answer: <answer>.\n\n{problem}\n\n"`. The paper cites the Tajwar et al. [TZZ+26] MLRL codebase (built on Verl), which uses Qwen3's chat format — the base model is likely prompted via its native chat template or a DeepSeek-R1-style thinking-format prompt. This is a gap in the paper's reporting.

---

## 2. Model(s) Used

**Main math experiments (§6.1):** Qwen-3-4B-Base (base, not instruction-tuned). The LM-judge for clustering is Qwen-3-4B-Instruct (a separate model, not the policy).

**Synthetic domain experiments (§6.2):** Qwen-3-1.7B-Base with Gemini-2.0-Flash as LM-judge.

**One-line summary:** Qwen-3-4B-Base (4B base model) trained on POLARIS-53k for 850 steps on 4×H200; LM-judge is Qwen-3-4B-Instruct.

**Delta from our pilot:** Our pilot uses Qwen3-1.7B-Base (from `shared_train.yaml`) and DaPO data. The paper uses Qwen3-4B-Base and POLARIS-53k. The 1.7B / synthetic-domain experiment in §6.2 is the closer analog.

---

## 3. Training Setup

From Table 1 (§A.2, p. 23):


| Parameter                     | Paper value    | Our pilot (`shared_train.yaml`)            |
| ----------------------------- | -------------- | ------------------------------------------ |
| Base model                    | Qwen-3-4B-Base | Qwen3-1.7B-Base                            |
| Generations per prompt (N)    | 8              | 8                                          |
| Set size n (for set RL)       | 4              | N/A (not set RL)                           |
| Number of sets K (for set RL) | 70             | N/A                                        |
| Max prompt length             | 1024           | not specified                              |
| Learning rate                 | 1×10⁻⁶         | 1×10⁻⁶                                     |
| KL coefficient                | 0.0            | 0.001                                      |
| Clip ratio ϵ_low              | 0.20           | 0.2                                        |
| Clip ratio ϵ_high             | 0.28           | 0.2 (symmetric)                            |
| Entropy coefficient           | 0.0            | not set                                    |
| Rollout temperature           | 1.0            | 1.0                                        |
| Prompts per batch             | 128            | 32                                         |
| Prompts per minibatch         | 64             | not explicit                               |
| Max response length           | 4096 tokens    | 2048 tokens (capped to 1024 in execute.py) |
| Training steps                | 850            | 100                                        |
| Device                        | 4×NVIDIA H200  | A100-80GB (1 GPU)                          |
| Training epochs               | 2              | not explicit                               |


**Reward function:** Binary RLVR — `r(x, y) ∈ {0, 1}` (correct/incorrect). Implicitly binary from all mathematical analysis in §5 (e.g., "r(x,y) = 0 (an incorrect response)"). No per-token scaling; per-generation binary correctness. This matches our `verifier: binary_rlvr`.

**Advantage normalization:** Paper explicitly omits standard-deviation normalization from standard GRPO: "Note that we omit the standard deviation normalization term originally used in [SWZ+24], as we found this choice led to better empirical performance — a modification consistent with recent findings in Dr.GRPO [LCL+25]" (§A, p. 23). Our pilot uses `advantage_norm: per_prompt_grpo` — unclear whether this includes std normalization; needs verification.

**Token-level loss normalization:** For GRPO baseline, Ti = |yi| (per-generation length). For Poly-EPO, Ti = Tmax (maximum response length), following Dr.GRPO Verl implementation.

**Seed/reproducibility:** No seed reported in the paper. Rollout temperature = 1.0.

---

## 4. Clustering / Canonicalization Substrate

Poly-EPO uses an **LM-judge** (Qwen-3-4B-Instruct) to cluster responses by reasoning strategy — NOT exact-match on answers.

Key design choices (§4, §A.1):

- Clusters are assigned based on **macro-strategy** (overall conceptual framework) AND **micro-strategy** (specific technique at key steps).
- Clustering is **independent of the final answer** — two responses with identical logic but different arithmetic errors share a cluster.
- Cluster 100 is reserved for degenerate responses (gibberish, off-topic, no logical steps, reward-hacking).
- All N generations for a prompt are clustered in a single judge call.
- When computing set diversity, Cluster 100 assignments are excluded from the numerator of d(x, y₁:n).

The diversity of a set is:

```
d(x, y_{1:n}) = |{C(y_1), ..., C(y_n)}| / n
```

where C(yi) is the LM-judge cluster assignment, and Cluster 100 members are excluded (§A.1, p. 23).

**Delta from our pilot:** Our pilot uses `clustering: exact_canonical` — hashing the canonicalized answer string (stripping whitespace, $, \boxed{}). This clusters by **answer identity**, not **reasoning strategy**. This is a fundamental mismatch: two responses that reach the same answer via different methods would be in the same cluster under our system, but in different clusters under Poly-EPO.

---

## 5. The Minority/Majority Voting Framing — Poly-EPO Math

**Polychromic objective** (Eq. 10, §4, p. 6):

```
f_poly(x, y_1, ..., y_n) = (1/n) * sum_{i=1}^{n} r(x, y_i) * d(x, y_{1:n})
```

= mean reward of the set × diversity of the set

**Diversity function** (Eq. 11, §4, p. 7):

```
d(x, y_{1:n}) = |{C(y_1), ..., C(y_n)}| / n
```

= number of distinct reasoning-strategy clusters / set size

**Marginal set advantage** (Eq. 8, §3, p. 5):

```
A♯_marg(x, y; f) = (1/|G(y)|) * sum_{G ∈ G(y)} A♯(x, G; f)
```

where G(y) = all constructed sets containing y, and A♯(x, G; f) = f(x, G) - mean_f.

**What this means operationally:**

- Sample N=8 generations per prompt.
- Construct K=70 sets of size n=4.
- Score each set: mean_reward × diversity.
- Each generation's advantage = average set-advantage over all sets it belongs to.
- An incorrect generation can receive **positive** advantage if it explores a rare strategy and other set members are correct (Term 1 of Eq. 15, p. 8).

**Comparison to our `inverse_freq`:**


| Property                            | Poly-EPO                                          | Our `inverse_freq`                                                                          |
| ----------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Credit scope                        | Set-level (shared among n generations)            | Per-trajectory                                                                              |
| Diversity measure                   | LM-judge strategy clusters                        | Exact-answer hash clusters                                                                  |
| What gets upweighted                | Sets with high reward AND high strategy diversity | Individual rollouts in minority answer clusters                                             |
| Incorrect but diverse gets + signal | Yes (via set credit sharing)                      | Only if cluster is rare AND reward is nonzero (reward=0 → advantage=0 after mean-centering) |
| Hyperparameter-free balance         | Yes (product structure)                           | No (w_max=8, gamma=1.0)                                                                     |


**Key structural difference:** `inverse_freq` is standard RL with reweighted per-trajectory advantages. Poly-EPO is set RL with a joint set-level objective. An incorrect rollout in `inverse_freq` has reward=0, base advantage = 0 - mean(r), so it gets a *negative* advantage regardless of cluster rarity (it only helps if the group has zero mean, which is unlikely). In Poly-EPO, incorrect rollouts in rare strategy clusters can receive positive signal if correct siblings are in the same sampled set.

---

## 6. Eval Methodology

**Benchmarks (§6.1, p. 10):**

- BeyondAIME [SC+25]
- AIME 2026
- AIME 2025
- HMMT November 2025
- HMMT February 2025
- Minerva [LAD+22]

All are held-out test sets not seen during training on POLARIS-53k.

**Metrics:**

**Pass@k coverage** (primary, Fig. 1): "the x-axis is the number of attempts k used in the evaluation while the y-axis is the coverage of the test set." This is the fraction of test-set problems solved by at least one of k attempts. k ranges appear to go up to ~64 or 128 based on the figures described. No explicit formula quoted; standard definition: `pass@k = fraction of problems where at least 1 of k rollouts is correct`.

**Majority@k** (Fig. 4): Top row is pass rate after majority voting over k samples; bottom row is majority vote share (fraction of votes for the winning answer). k appears to range from 1 to ~64.

**Training dynamics** (Fig. 2): (1) Average number of unique reasoning-strategy clusters among correct generations per prompt during training. (2) Fraction of training prompts with at least one correct rollout.

**No Cover@τ or Worst-subset metrics** are defined or used in this paper. No formal statistical significance testing reported — all results are point estimates from figures.

---

## 7. Headline Results

From §6.1 and figures (p. 10-12):

1. **Poly-EPO improves pass@k with up to 20% gains** on math reasoning test sets as k increases (abstract, §6.1).
2. **GRPO collapses diversity**: Under GRPO, the pretrained base model (Qwen-3-4B-Base) begins to outperform GRPO-trained models at pass@k as early as k=32 (§6.1, p. 13). Same degradation for GRPO+DIV.
3. **Poly-EPO grows clusters**: Under GRPO, unique reasoning clusters in correct answers decline after ~200 steps. Under GRPO+DIV, clusters stay flat. Under Poly-EPO, clusters steadily increase to substantially higher values (Fig. 2 left, p. 11).
4. **Coverage improves**: Both GRPO+DIV and Poly-EPO achieve higher fraction of prompts with at least one correct rollout during training compared to vanilla GRPO (Fig. 2 right).
5. **Majority voting**: Poly-EPO achieves equal or stronger majority-vote accuracy as k increases despite lower majority vote share (more distributed probability mass), suggesting higher-quality diversity (§6.1, p. 12-13).
6. **Synthetic domains**: On multi-digit multiplication and polynomial solving with Qwen-3-1.7B-Base, GRPO collapses to a single strategy while Poly-EPO discovers >5× as many distinct successful strategies (§6.2, p. 14).

---

## 8. Notable Methodology Choices

From the paper's implementation choices and stated rationale:

1. **No std normalization in advantage** (§A, p. 23): "we omit the standard deviation normalization term originally used in [SWZ+24], as we found this choice led to better empirical performance." This is a Dr.GRPO modification. Our pilot's `per_prompt_grpo` normalization mode needs to be checked for whether it includes std normalization.
2. **Asymmetric clip ratio** (Table 1): ϵ_low = 0.20, ϵ_high = 0.28. This is a DAPO-style choice. Our pilot uses symmetric ϵ = 0.2.
3. **Zero KL coefficient** (Table 1): KL coefficient = 0.0. No KL penalty against reference policy. Our pilot uses kl_coef = 0.001.
4. **Long response length**: Max response = 4096 tokens. Our pilot caps at 2048 (and execute.py hard-caps at 1024). This is likely too short for hard math problems.
5. **Large batch**: 128 prompts/batch (vs. our 32). Effective diversity per update is much higher.
6. **850 training steps, 2 epochs** over POLARIS-53k. Our pilot runs 100 steps. The paper shows diversity collapse in GRPO beginning at ~200 steps, suggesting 100 steps may be too few to observe the full effect.
7. **LM-judge for clustering is the instruct variant** of the same model family (Qwen-3-4B-Instruct judges Qwen-3-4B-Base outputs). Must "possess strong instruction-following capabilities."
8. **Cluster 100 exclusion**: Degenerate generations (reward hacking, gibberish) are identified by the judge and excluded from diversity computation. Our exact-match system has no such safety valve.
9. **Set size n=4, K=70 sets** from N=8 rollouts: (8 choose 4) = 70, so all possible sets are enumerated. This is the combinatorial maximum, not a random sample.

---

## 9. Ablations

The paper does **not** contain a formal ablation section. The following comparisons serve as partial ablations:

**Clustering substrate:** Not formally ablated. The paper only uses LM-judge clustering; no comparison to exact-match or embedding-based clustering. The choice of LM-judge is motivated by scalability and domain-generality arguments (§4, p. 6-7).

**Prompt template:** Not discussed or ablated.

**Step count:** Not ablated, but Fig. 2 (training dynamics) shows that GRPO diversity collapse begins around step 200 and Poly-EPO keeps growing through step ~800. This implies 100 steps is too short to see divergence.

**Model scale:** §6.2 uses 1.7B (Qwen-3-1.7B-Base) for synthetic tasks and §6.1 uses 4B for math. No head-to-head scale ablation.

**Set size n and K:** Not ablated. Authors note in §8 (Conclusion, p. 15): "The scaling laws of our general recipe for set reinforcement learning are still not well understood, especially regarding the relative roles of set size, number of rollouts, and number of constructed sets."

**GRPO+DIV as partial ablation:** GRPO+DIV uses the same LM-judge clustering as Poly-EPO but applies diversity as a per-trajectory bonus in standard RL (not set RL). It underperforms Poly-EPO on all metrics, isolating the benefit of set-level credit assignment over per-trajectory reweighting.

**Std normalization:** Mentioned as a deviation from GRPO with better performance but not formally ablated with results.

---

## 10. Constraints and Contradictions for Our Pilot Redesign

### Direct constraints

1. **100 steps is too short for signal.** The paper's Fig. 2 shows diversity dynamics only diverge meaningfully after ~200 steps. At 100 steps, we cannot distinguish Poly-EPO-style methods from GRPO because diversity collapse in GRPO hasn't fully occurred yet. Minimum recommended: ~300 steps to see the first divergence; 500+ to see the full effect. Our current `max_steps: 100` is almost certainly too short to detect the effect we are studying.
2. **Exact-match clustering is not the paper's substrate.** The paper's clustering measures *reasoning strategy diversity*, not *answer diversity*. Two rollouts with different methods reaching the same answer are separate clusters under Poly-EPO but the same cluster under our system. This makes our `inverse_freq` conceptually misaligned with what the paper measures as "minority." The paper explicitly chose LM-judge to capture strategy-level diversity. Our exact-canonical clustering gives inverse_freq a fundamentally different signal than what Poly-EPO optimizes.
3. **Max response length of 1024 tokens (execute.py cap) is too short.** The paper uses 4096. Hard math problems (AIME, HMMT) require long reasoning chains. Truncation at 1024 will systematically reward shorter (likely lower-quality) reasoning and distort reward signal.
4. **KL coefficient should be 0.0** (or very small). The paper uses kl_coef = 0.0. Our 0.001 may constrain exploration unnecessarily.
5. **Asymmetric clipping.** Paper uses ϵ_low=0.20, ϵ_high=0.28 (DAPO style). Our symmetric ϵ=0.2 may underweight positive updates.
6. **Batch size 32 vs 128.** Smaller batch means each gradient step sees less diversity. For a diversity-promoting objective this matters more than for vanilla GRPO.
7. **inverse_freq does not provide positive signal to incorrect rollouts.** This is the fundamental distinction from Poly-EPO: `r(x,y)=0` → base advantage starts at `0 - mean(r)` which is ≤ 0. Even if a cluster is rare, multiplying a negative advantage by a large weight makes it more negative. `inverse_freq` only helps correct rollouts in minority clusters. Poly-EPO upweights *exploratory* rollouts even when incorrect. Our method is closer to "correct minority voting" than "optimistic exploration."

### Guidance for redesign

- If we want to compare against what the paper actually does, we need an LM-judge clustering step (or a cheap proxy that separates strategy from answer, e.g., prefix-similarity clustering).
- If we keep exact-match clustering and `inverse_freq`, we should reframe our hypothesis: we are testing "minority-correct-answer upweighting" not "minority-strategy upweighting."
- The 1.7B / synthetic-domain experiment (§6.2) is the paper's analog to our setup. Qwen-3-1.7B-Base on synthetic tasks with Gemini-2.0-Flash judge. We use 1.7B on DaPO math problems. The gap is the judge quality and clustering substrate.
- Training data: Paper uses POLARIS-53k. We use DaPO. No comparison possible without knowing difficulty distribution overlap.
- Pass@k is the primary signal, not pass@1. At 100 steps with N=8, we should be reporting pass@1 through pass@8 at minimum; the paper goes to pass@64+ to show divergence.

