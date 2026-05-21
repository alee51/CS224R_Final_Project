# Fresh labeling plan

**Status:** Complete — `../labels/rollout_labels.jsonl` (4000 rows). Archived under `labeling/`.

**Agent spawn text:** from `labeling/`, run `python prepare_label_slot.py --chunk K --agent A|B` (see [`AGENT_LABEL_PROMPT.md`](AGENT_LABEL_PROMPT.md)).

## Job (one sentence)

For every rollout, two **independent** agents read **problem + last 120 chars of completion** (no gold) and each outputs **one `result`**: an **extracted final answer string** or `**runon` | `no_answer` | `needs_review`**.

Gold lives only in `chunk_KKK_keys.tsv` and dispute files — never in agent input.

---

## Anti-peeking (structural)


| Mechanism            | Purpose                                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Opaque paths**     | Each chunk gets random filenames under `blind/KKK/` (see `manifest.json`). Agents are told only their own path. |
| **Sequential A → B** | Never spawn B until A has N rows. B must not copy A if A is not finished yet.                                   |
| `**.cursorignore`**  | When preparing B, A's blind TSV is added to the isolation block so Cursor cannot Read it.                       |
| **Prompt**           | Explicit ban on `blind/`, `chunk_*_out_`*, listing dirs, and reading the other agent's file.                    |


Orchestrator reads `blind/KKK/manifest.json`; agents do not.

---

## Chunking


| Unit        | Size                                                               |
| ----------- | ------------------------------------------------------------------ |
| 1 chunk     | **50 prompts × 8 rollouts = 400 rows** (last chunk may be smaller) |
| **Phase 1** | Priority subset (~492 prompts → chunks 000–009)                    |


Chunk index `K` (0-based) = priority prompt IDs sorted, slice `[50K .. 50K+49]`.

---

## Per chunk workflow

```text
python build_chunk.py --chunk K          # in.tsv, keys.tsv, blind manifest
python prepare_label_slot.py --chunk K --agent A
spawn agent A → wait for N rows
python prepare_label_slot.py --chunk K --agent B   # blocks A in .cursorignore
spawn agent B → wait for N rows
python merge_chunk_pair.py K
```


| Role        | Path                                                |
| ----------- | --------------------------------------------------- |
| Read (both) | `chunks/chunk_KKK_in.tsv`                           |
| Write A     | `blind/KKK/<random>.tsv` (from manifest `output_a`) |
| Write B     | `blind/KKK/<other>.tsv` (`output_b`)                |
| Merge only  | `blind/KKK/manifest.json`                           |


**Input (agents):** `id`, `problem`, `tail` — no gold.

**Output:** `id`, `result` — N data rows required.

**Sidecar (agents do not read):** `chunk_KKK_keys.tsv`

---

## Merge (`merge_chunk_pair.py K`)

Loads paths from manifest. Compares blind A vs B:


| A vs B                   | Outcome                                       |
| ------------------------ | --------------------------------------------- |
| agree                    | `result` set in `labels/rollout_labels.jsonl` |
| differ or `needs_review` | `needs_human: true`, `chunk_KKK_dispute.tsv`  |


Rebuild: `python rebuild_rollout_labels.py --through 9`

---

## Scripts


| Script                      | Role                                              |
| --------------------------- | ------------------------------------------------- |
| `build_chunk.py`            | `chunk_KKK_in.tsv`, `keys.tsv`, blind manifest    |
| `prepare_label_slot.py`     | Init output TSV, cursorignore, print spawn prompt |
| `merge_chunk_pair.py`       | A/B merge → `rollout_labels.jsonl`                |
| `rebuild_rollout_labels.py` | Rebuild JSONL from all chunks                     |


