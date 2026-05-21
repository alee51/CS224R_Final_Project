# Decision: Skip `run0_proxy` in Stage 1 redesign matrix

- **Status:** Accepted
- **Date:** 2026-05-19
- **Decision:** Stage 1 matrix excludes `run0_proxy`; use pre-redesign Run 0 artifacts only for validity / gate context.

## Context

The first pilot matrix (pre-redesign) launched `run0_proxy` detached on 2026-05-19. Run 0 completed and was analyzed offline before the Stage 1 redesign spec was finalized.

## Reasoning

- Run 0 from the pre-redesign pilot is **complete and reviewed**: artifacts at `pilot/artifacts/run0_proxy/20260519T190202Z/`, handoff in [`RUN0_HANDOFF_FOR_REVIEW.md`](../../artifacts/run0_proxy/20260519T190202Z/RUN0_HANDOFF_FOR_REVIEW.md), plus cleaned offline analysis under `cleaned/`.
- Re-running `run0_proxy` under redesign would mostly **re-sample the same frozen model** on the same 500 DaPO rows (0–499). Parser/template fixes improve label quality but do not change the core conclusion from the completed run: **answer-cluster minority-correct prompt rate stays ~0%** on this substrate.
- Stage 1 budget and wall-clock are better spent on the **GRPO mechanism matrix** (`run1_grpo`, `run2_inverse_freq`, `run3_f_grpo`) after smoke, where training + diagnostics are the actual unknowns.
- `pilot/eval/gate.py` may still read the latest `run0_proxy` artifact directory if present; no gate re-run is required for Stage 1 launch.

## Implications

| Area | Change |
|------|--------|
| Matrix launcher | `pilot/scripts/launch_pilot_matrix.sh` — `MATRIX_RUNS` = 3 GRPO runs only |
| Matrix burst budget | **$150** nominal (3 × $50/run); `pilot/preflight_lock.json` **`pilot_total`: $200** kept as ceiling (smoke ~$10 + matrix $150 + headroom) |
| `run0_proxy` config / cap | `budget_caps_usd.run0_proxy` in preflight lock **retained** for optional manual/debug runs; not part of Stage 1 matrix |
| Ops docs | `PILOT_REDESIGN.md`, `pilot/infra/README.md`, `SMOKE_READINESS.md`, `nancy_explore/narrative/context.md` — three-run matrix language (research objectives may change) |
| Deferred §9 gate text | Re-evaluate `gate_decision.json` using **existing** Run 0 artifacts (`20260519T190202Z`), not a redesign re-run |

## Links

- Artifacts: `pilot/artifacts/run0_proxy/20260519T190202Z/`
- Handoff: `pilot/artifacts/run0_proxy/20260519T190202Z/RUN0_HANDOFF_FOR_REVIEW.md`
- Redesign spec: `pilot/docs/operations/PILOT_REDESIGN.md` (Run 0 waived bullet in §1)

## Supersedes

Any Stage 1 instruction that required a fresh `run0_proxy` in the same matrix launch as `run1`–`run3`.
