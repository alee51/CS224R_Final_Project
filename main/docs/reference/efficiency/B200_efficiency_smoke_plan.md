# B200 minority efficiency smokes (2026-05-27)

## Step 1 — mined baseline (wandb)

| Run | SKU | step≈rollout+train | t_rollout | t_train | t_logprob_fwd | t_backward | vram_peak | chunks | n_kept |
|-----|-----|-------------------|-----------|---------|---------------|------------|-----------|--------|--------|
| `wdl3fczm` | B200 | **~267s** | 58.5 | 208.5 | **11.4** | **197.0** | 148.5 | 5 | 512 |
| `q6m0tmiu` | H200 | ~391s | 93.8 | 296.9 | 19.9 | 276.2 | 131.0 | 5 | 512 |
| `1hg8fs5u` | B200 GRPO | ~118s | 58.1 | 59.6 | 4.6 | 54.4 | 144.2 | 2 | 164 |

**Readout:** On minority, **backward (gc recompute) dominates**, not logprob forwards.  
→ Prioritize **`gc_off`** and **`token_budget`** smokes before batched seq forwards.  
→ **`logprob_seq_batch`** is a small win on minority (~11s fwd); still worth one smoke.

Live prod at **~153 GB** peak ≈ smoke + headroom; budget **~180 GB** on Modal.

## Step 2 — vLLM sleep (H200 history)

`ablate-C-wakefix` (`pl1k1w3a`): sleep freed **64.8 GiB**, train step ran, **`wake_for_rollout()` cumem crash** on vLLM 0.8.5.  
**Must re-test on B200 + vLLM 0.9** (`launch_b200_efficiency_smokes.sh sleep`).

## Minimal smoke matrix (10 steps each)

| Label | What | Try? |
|-------|------|------|
| `sleep` | `vllm_sleep=1` | **Yes** — gate only; do not prod until 10 steps finish |
| `budget` | `token_budget: 130000` | **Yes** — low risk |
| `gc_off` | `gradient_checkpointing: false` | **Yes** — highest ROI if peak &lt; 175 GB |
| `seqbatch` | `logprob_seq_batch=8` | **Yes** — small fwd win |

**Skip for now:** combining sleep+gc until sleep green; `n_kept` cap (methods); 8-bit Adam (fresh branch).

Launch:

```bash
bash main/scripts/launch_b200_efficiency_smokes.sh
# or one: bash main/scripts/launch_b200_efficiency_smokes.sh gc_off
```

Summarize:

```bash
main/.venv/bin/python main/scripts/summarize_efficiency_smokes.py \
  main/docs/probes/artifacts/b200_eff_<ts>/launched.txt
```

Pick prod knobs: variant with best **Δstep / Δvram** vs `wdl3fczm`, peak &lt; 170 GB, no crash on wake.
