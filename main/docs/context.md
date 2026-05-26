# Context

Pointer file for agents and teammates. Read this first, then `PLAN.md`.

---

## Reading list (authoritative for what)


| Read                                                          | For                                                                                 |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `pre-milestone/main.md`                                       | Submitted milestone — current research framing, objective math, preliminary results |
| [`decisions.md`](./decisions.md)                               | Main train-stack decisions (arm C, SymPy grader, Polaris data)                        |
| [`data/README.md`](../data/README.md)                          | Polaris train vs full-pool paths (`polaris_train.jsonl` = canonical)                |
| `pre-milestone/nancy_explore/narrative/decisions.md`          | Pilot / Run 0 decisions (objectives, Modal workspace, eval grading survey)          |
| `pre-milestone/nancy_explore/narrative/timeline.md`           | Chronology through milestone                                                        |
| `pre-milestone/pilot/docs/analysis/0519_perf_consolidated.md` | Pilot performance findings, what's been applied, what's still open                  |
| `pre-milestone/pilot/docs/incidents/`                         | Pilot incident postmortems                                                          |
| `main/docs/PLAN.md`                                           | This project's working plan                                                         |
| `main/docs/LESSONS_FROM_PILOT.md`                             | Curated pilot lessons (to be written)                                               |


---

## Post-milestone facts (not yet in pre-milestone docs)

- **Budget confirmed:** $1,600 Modal credits. Per-person split: Nancy $650, Anastasia $475, Emma $475. Possible +$400 stretch.
- **Deadlines:** poster due 2026-06-03; internal target for experiments done 2026-05-31.
- **Rollout engine:** vLLM, in-process. Lightweight VeRL-flavored trainer, not the full VeRL stack.
- **Code ownership:** Nancy owns all `.py` for the main experiment.
- **Arm priority:** GRPO baseline (must), minority-answer (headline), minority-CoT (in scope, first cut if compute tight), Poly-EPO-answer (stretch only; paper `f_poly` with answer-hash diversity, no in-loop LLM judge).
- **Hardware:** **H200 locked** (single-GPU, collocated rollout+train) as of 2026-05-26 per Group B readout. H100 ruled out — OOMs at `batch_size: 64` because `_completion_logprobs_hf` is one-shot. **Production `train.batch_size: 64` locked** — bs=128 OOMs post-rollout on this stack ([`decisions.md`](./decisions.md) §2026-05-26). B200 optional further upgrade — see `main/docs/probes/B200_migration_analysis_2026-05-26T034425Z_b01999f.md`. Full reasoning in PLAN.md §5/§7.
- **Dropped from scope** (vs. earlier pilot directions): `inverse_freq`, F-GRPO, Cover@τ as training objective.
- **Polaris train data:** GRPO uses **`main/data/polaris_train.jsonl`** (51,139 rows, prompt-filtered). Unfiltered 53k pool → **`main/data/source/polaris_train_full.jsonl`** (not for training). 

