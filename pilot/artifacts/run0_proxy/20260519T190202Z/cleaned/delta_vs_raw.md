# Run 0 cleaned labels vs stored raw

**Source:** `raw_predictions.jsonl` (immutable completions)  
**Cleaner:** `pilot/train/answer_clean.py` (`extract_answer_clean`, `normalize_answer_clean`, brace-balanced `\boxed`, run-on rejection)

## Headline counts (4000 rollouts)


| Metric                                                     | Count | Rate   |
| ---------------------------------------------------------- | ----- | ------ |
| `parsed_answer` changed                                    | 514   | 12.85% |
| `correct` flipped (either direction)                       | 6     | 0.15%  |
| Prompts with different cluster/canon grouping (8 rollouts) | 500   | 100.0% |
| Correct gained (false→true) rollouts                       | 6     |        |
| Correct lost (true→false) rollouts                         | 0     |        |


## Prompt-level correctness

- Prompts with ≥1 rollout where `correct` flipped: **5**
- Prompts with any correct **gained**: **5**
- Prompts with any correct **lost**: **0**
- Mean distinct stored clusters / prompt: **7.18**
- Mean distinct clean clusters / prompt: **6.86**

### Distribution: correct rollouts per prompt (stored)


| n_correct_stored | count | %     |
| ---------------- | ----- | ----- |
| 0                | 337   | 67.4% |
| 1                | 81    | 16.2% |
| 2                | 35    | 7.0%  |
| 3                | 28    | 5.6%  |
| 4                | 10    | 2.0%  |
| 5                | 6     | 1.2%  |
| 6                | 2     | 0.4%  |
| 7                | 1     | 0.2%  |
| 8                | 0     | 0.0%  |


### Distribution: correct rollouts per prompt (clean)


| n_correct_clean | count | %     |
| --------------- | ----- | ----- |
| 0               | 335   | 67.0% |
| 1               | 83    | 16.6% |
| 2               | 33    | 6.6%  |
| 3               | 29    | 5.8%  |
| 4               | 10    | 2.0%  |
| 5               | 7     | 1.4%  |
| 6               | 2     | 0.4%  |
| 7               | 1     | 0.2%  |
| 8               | 0     | 0.0%  |


## Extract path (clean)


| Path           | Count | %     |
| -------------- | ----- | ----- |
| boxed_balanced | 2016  | 50.4% |
| answer_line    | 853   | 21.3% |
| last_line      | 705   | 17.6% |
| runon_rejected | 426   | 10.7% |


- Run-on rejected: **426** (10.7%)
- Truncated `\boxed{` (unclosed at end): **8** (0.2%)
- Nested boxed: balanced ≠ shallow-regex: **88** (2.2%)

## Flip categories (example prompt_ids)

### Correct gained (format/LaTeX/boxed fix)

`65da7224-5f07-48e3-9b01-3c9ea1dfb036`, `2e690d58-de84-4003-a33f-fbebdb71dae5`, `70aabfd8-5728-4d08-8363-94e175fc0632`, `22063de2-a7a2-4214-895f-e015e0b78f87`, `cfc7b48f-94bf-429f-b1c9-a7ac15e86b80`

### Correct lost

*None flagged at prompt level.*

### Parsed changed, correct unchanged

`1653ee27-05d2-49ea-b9fb-3cdd58a05730`, `ac8cdbc9-816e-4c77-aa3c-7ef9b3170d8b`, `a8c414e5-c522-49d6-a1af-2afcb37e3ddc`, `7ec6f22e-5008-43cf-8218-ea0c4ce775ac`, `09a07fe7-fbd8-45ee-9e87-63d37c831153`, `56a368fe-51a8-4879-9b96-053ea9485fea`, `87125f5a-b1d7-47d0-a222-a85846c2f856`, `a6bce30d-9781-402b-95ae-882c43e72b79`, `14647322-e8d8-41a1-a3e5-ac3e865663bb`, `5097fb46-8292-4218-b289-6140223ca22f`

### Run-on fallback rejected

`1653ee27-05d2-49ea-b9fb-3cdd58a05730`, `ac8cdbc9-816e-4c77-aa3c-7ef9b3170d8b`, `7ec6f22e-5008-43cf-8218-ea0c4ce775ac`, `09a07fe7-fbd8-45ee-9e87-63d37c831153`, `87125f5a-b1d7-47d0-a222-a85846c2f856`, `14647322-e8d8-41a1-a3e5-ac3e865663bb`, `5097fb46-8292-4218-b289-6140223ca22f`, `ce3091eb-d231-4776-8e7c-1765ff579257`, `57844481-891b-489d-97b3-4f73498383da`, `3624db0e-a7fc-4fc6-884b-b71efc624eaa`

### Truncated `\boxed{` / cut-off completion

`ee4283e3-709c-4a71-88b9-08c98b029a71`, `c2c7f62c-9f1f-4fe5-8582-35f8fa16a5c1`, `307a95f6-9495-4cf2-938a-f8d468847f3d`, `29e2fd56-33b5-480a-8959-456b9dc6836c`, `718b2859-ffd7-4201-9e96-c899a29b9b6f`, `822b2d99-412d-4d82-855c-1f3a313b0b1f`, `29874818-92ad-485c-b651-2893f0b6c588`, `6ea68b67-aeb7-4673-980a-ae4f8054e7eb`

### Nested boxed: regex vs balanced inner mismatch

`a8c414e5-c522-49d6-a1af-2afcb37e3ddc`, `56a368fe-51a8-4879-9b96-053ea9485fea`, `a6bce30d-9781-402b-95ae-882c43e72b79`, `ce3091eb-d231-4776-8e7c-1765ff579257`, `cfecb90b-3f7d-4493-af64-ff306ba84d0f`, `26a7856a-14a7-4ca0-827d-b050b804769a`, `370776e1-3445-4fe9-bf6d-0f99da851669`, `50a9f08d-cd07-47aa-89ed-1e185480cba8`, `f35d44e2-4321-4a47-9a56-e155c2e03a43`, `fa0527a0-1d7f-4ba1-a0dc-f6e4d7365645`

## Limitations

- **Run-on heuristic** may reject valid long math tails or accept short prose.
- **Brace-balanced boxed** prefers the *last* boxed; multiple answers in one completion are not disambiguated.
- `**normalize_answer_clean`** merges some format variants but not all mathematically equivalent forms (e.g. unsimplified radicals).
- **Truncated boxed** detection is syntactic (unclosed opener), not semantic completeness of the math.
- Stored `cluster_id` uses Python `hash()` (process-dependent); `cluster_id_clean` uses SHA-256 — compare via `canon_clean`, not raw ints.

