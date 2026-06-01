# Analysis A — LLM judge prompts (Poly-EPO faithful)

**Source:** Poly-EPO paper §A.1 — instruction block + both few-shot examples,
verbatim from `nancy_explore/reference/poly_epo_paper.pdf` pages 21–22.
Prior version (2026-05-29 → 2026-05-31) had only the instruction block; the
two few-shot examples were missing — see
`MEMORY/project_judge_prompt_fewshot_gap.md`. Restored 2026-05-31.

**Adaptation for Run 0 offline Analysis A:** `{n_responses}` = 8 whole rollouts per prompt (same clustering unit as Poly-EPO §A.1 — one `cluster_id` per response; judge infers macro/micro strategy from reasoning within each completion). Degenerate cluster remains `**cluster_id: 100`** per paper (downstream code maps 100 → -1).

Templates for `analysis_a_llm_clusters.py`. Placeholders: `{n_responses}`, `{problem}`, `{responses_block}`.

## System

Your ONLY task is to cluster the {n_responses} responses into buckets based on their reasoning algorithm, including both the overall strategy and the methods used at key intermediate steps.

**INPUT FORMAT:** You will receive:

1. A "Context" describing the task.
2. A numbered list of Responses from 1 to {n_responses}. Each response contains a reasoning process and final answer.

Note: Responses may or may not explicitly state their strategy; you must infer the strategy by analyzing the mathematical steps taken.

**CLUSTERING CRITERIA:**
(1) Macro-strategy: The overall conceptual framework (e.g., recursion vs infinite series; prime factorization vs gcd-based formula).

(2) Micro-strategy: The specific method used to resolve key intermediate steps. Examples include: how absolute values are removed (+- case split vs squaring), how intervals are partitioned, or how a basis is chosen.

**CLUSTERING RULES:**

- Cluster strictly based on logic and approach. NOT on wording, tone, formatting, or final answer.
- Two responses share a cluster_id IF AND ONLY IF they use the same macro-strategy AND the same micro-strategy at every key step.
- Arithmetic errors do NOT create new clusters if the underlying logic is identical.
- **SPECIAL CLUSTER 100:** You MUST assign `cluster_id: 100` to any response that is:
  - Gibberish (random characters, nonsense strings).
  - Irrelevant to the math problem (off-topic text).
  - Non-mathematical reasoning (e.g., writing code to solve it instead of math, or making a random guess at the final answer without logical steps).

**OUTPUT RULES (STRICT):**

1. Respond ONLY with a JSON object. No text outside the JSON.
2. The JSON must contain exactly {n_responses} keys: "1", "2", ..., "{n_responses}".
3. The value for each key must be an object with:
  - `"chain_of_thought"`: `"Macro: [short description]. Micro: [short description]."`
  - `"cluster_id"`: integer (use 100 for degenerate responses per rules above).
4. `chain_of_thought` must be concise and avoid repeating the actual calculations.

**Few-Shot Example 1:**

**Context:**
What is the smallest value of x such that 5x - 1 = 3x + 2?

**Responses:**
1. We can split this into two cases: 5x - 1 = 3x + 2 or 5x - 1 = -(3x + 2). Solving the first gives 2x = 3 so x = 1.5. The second gives 8x = -1 so x = -0.125.
2. The expression 5x-1 changes sign at 1/5, and 3x+2 changes at -2/3. For x < -2/3, we have -(5x-1) = -(3x+2). For -2/3 < x < 1/5, we have -(5x-1) = 3x+2. Solve -(5x - 1) = 3x + 2 for the range, yielding x = -0.125
3. Using the property that a=b implies a=b or a=-b, we get 5x-1 = 3x+2 (x=1.5) and 5x-1 = -3x-2 (x=-0.125). So the answer is x = -0.125
4. To get rid of the absolute values, square both sides: (5x - 1)^2 = (3x + 2)^2. This expands to 25x^2 - 10x + 1 = 9x^2 + 12x + 4. Solve 16x^2 - 22x - 3 = 0. So, x = -1/8, 3/2. Final answer is x = -1/8
5. Either 5x - 1 = 3x + 2 or 5x - 1 = -3x - 2. This leads to x = 3/2 and x = 0. So, final answer is x = 0.
6. I think the answer is probably 0 or maybe 1.5.
7. asdf qwer zxcv 9999 ---- ??? Let's write Python to check each x from -10 to 10: `if abs(5*x-1) == abs(3*x+2): print(x)`. The answer is -0.125.

**Expected Output:**
{
"1": {"chain_of_thought": "Macro: Algebraic casework. Micro: Direct +- case split to remove absolute values.", "cluster_id": 1},
"2": {"chain_of_thought": "Macro: Interval analysis. Micro: Testing expression sign changes across number line partitions.", "cluster_id": 2},
"3": {"chain_of_thought": "Macro: Algebraic casework. Micro: Direct +- case split to remove absolute values.", "cluster_id": 1},
"4": {"chain_of_thought": "Macro: Algebraic transformation. Micro: Squaring both sides to create and solve a quadratic equation.", "cluster_id": 3},
"5": {"chain_of_thought": "Macro: Algebraic casework. Micro: Direct +- case split to remove absolute values (contains arithmetic error).", "cluster_id": 1},
"6": {"chain_of_thought": "Macro: Non-mathematical. Micro: Random guessing without any logical derivation.", "cluster_id": 100},
"7": {"chain_of_thought": "Macro: Gibberish/Non-mathematical. Micro: Contains random non-English text, nonsense strings, and code-based iteration.", "cluster_id": 100}
}

**Few-Shot Example 2:**

**Context:**
What is the least common multiple of 72 and 96?

**Responses:**
1. 72 = 2^3 * 3^2. 96 = 2^5 * 3^1. To find the LCM, we take the highest power of each prime factor present: 2^5 * 3^2 = 32 * 9 = 288.
2. Prime factors of 72: 2, 2, 2, 3, 3. Prime factors of 96: 2, 2, 2, 2, 2, 3. The union of these sets is five 2s and two 3s. Total: 276.
3. First find the GCD using the Euclidean algorithm: 96 = 72(1) + 24; 72 = 24(3) + 0. GCD is 24. LCM is (72 * 96) / 24.
4. 72 = 8*9, 96 = 8*12. The answer is 288. The answer is 288. The answer is 288. The answer is 288.

**Expected Output:**
{
"1": {"chain_of_thought": "Macro: Prime factorization analysis. Micro: LCM via maximum exponents of prime factors.", "cluster_id": 1},
"2": {"chain_of_thought": "Macro: Prime factorization analysis. Micro: LCM via maximum exponents of prime factors (contains arithmetic error).", "cluster_id": 1},
"3": {"chain_of_thought": "Macro: Product-GCD relationship. Micro: GCD calculation via Euclidean algorithm followed by the LCM formula.", "cluster_id": 2},
"4": {"chain_of_thought": "Macro: Excessive repetition. Micro: Response loops the final answer multiple times at the end.", "cluster_id": 100}
}

## User

**Context:**
{problem}

**Responses:**
{responses_block}
