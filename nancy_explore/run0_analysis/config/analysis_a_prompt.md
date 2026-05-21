# Analysis A — LLM judge prompts (Poly-EPO faithful)

**Source:** Poly-EPO paper §A.1 — verbatim instruction block from `[pilot/docs/analysis/0519_poly_epo_methodology.md](../../../pilot/docs/analysis/0519_poly_epo_methodology.md)` (extracted from `nancy_explore/reference/poly_epo_paper.pdf`).

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

## User

**Context:**
{problem}

**Responses:**
{responses_block}