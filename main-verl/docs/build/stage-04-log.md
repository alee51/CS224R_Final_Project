# Stage 4 log — Judge service on Modal

**Stage ID:** `stage-04`
**Image rebuild count:** 1 (judge-only image, first deploy 2026-05-30)
**Model/prompt iteration count:** 0
**Plan:** [`stage-04-agent-plan.md`](./stage-04-agent-plan.md)
**Modal app name:** `cs224r-verl-stage04-judge`
**Modal profile:** `alee72` (deploy succeeded; Account B TBD for production parallel runs)

---

## Dispatch log

| Section | Executor | Audit | Verdict |
|---------|----------|-------|---------|
| S4.1 | DONE | pending | — |
| S4.2 | DONE (deployed) | pending | URLs live; cold-start not verified |
| S4.3 | DONE | pending | — |
| S4.4 | DONE (7/7 PASS) | pending | — |
| S4.5 | DONE | pending | **PASS** — 100% parse, 100% agreement (v3, ~152s) |
| S4.6 | DONE | pending | **PASS** (mean 1.37s); parse 22% at concurrency=64 |
| S4.7 | — | pending | blocked |

---

## S4.1 — Prompt + parser (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifacts:**
  - `main-verl/judge/prompts/poly_epo_a1.md`
  - `main-verl/judge/prompt.py`
  - `main-verl/judge/parse.py`
  - `main-verl/judge/types.py`
  - `main-verl/judge/__init__.py`

Ported Poly-EPO §A.1 templates and parser from `main/judge/format.py`. Public surface: `JudgeTask`, `JudgeClusterResult`, `build_judge_messages`, `parse_judge_response`. Rollouts are `list[str]` (VeRL-agnostic).

---

## S4.2 — Judge Modal service (executor)

- **Timestamp (UTC):** 2026-05-30
- **Artifacts:**
  - `main-verl/judge/modal_image.py` (judge-only; no maxrl/verl)
  - `main-verl/judge/server.py`
  - `main-verl/scripts/launch_judge_service.sh`

**Deploy:** `PYTHONPATH=main-verl modal deploy main-verl/judge/server.py` — **SUCCESS** (~205s image build + deploy)

| Endpoint | URL |
|----------|-----|
| Health | `https://alee72--health.modal.run` |
| Chat completions | `https://alee72--v1-chat-completions.modal.run` |
| Modal dashboard | `https://modal.com/apps/alee72/main/deployed/cs224r-verl-stage04-judge` |

**Model:** `Qwen/Qwen2.5-7B-Instruct` · `B200:1` · `enforce_eager=True` · chat template via `tokenizer.apply_chat_template`

**Note:** `curl` health check timed out at 120s (cold-start vLLM load). Re-try with longer timeout or check Modal logs before S4.5.

**Client URL convention:** `JUDGE_BASE_URL` = full Modal web endpoint URL (POST target), e.g. `https://alee72--v1-chat-completions.modal.run` — not `/v1/chat/completions` suffix.

---

## S4.3 — Async client (executor)

- **Artifact:** `main-verl/judge/client.py`
- Per-task semaphore + `asyncio.gather` fan-out
- `cluster_batch_sync` for probes

---

## S4.4 — Unit tests (executor)

```text
PYTHONPATH=main-verl python3 -m pytest main-verl/tests/test_judge_parse.py -v
7 passed in 0.10s
```

---

## S4.5 / S4.6 — smokes

### S4.5 agreement (2026-05-30) — **PASS**

Modal app: `ap-dfNL7e6j30ku3IRfS9pw41` · serial · `max_tokens=4096`

```json
{
  "n_tasks": 50,
  "parse_ok_both_rate": 1.0,
  "assignment_agreement_rate": 1.0,
  "cluster_100_hits_total": 128,
  "parse_ok_both": 50,
  "agreement": 50
}
```

Wall time ~152s. Thresholds: ≥90% parse, ≥80% agreement — both exceeded.

### S4.6 latency (2026-05-30) — **PASS** (with notes)

Modal app: `ap-AqvD3dua1y3DTPXKfPkvSd` · 64 tasks · ~88s wall

```json
{
  "n_tasks": 64,
  "concurrency": 64,
  "wall_s": 87.96,
  "mean_wall_per_task_s": 1.37,
  "parse_ok_rate": 0.22,
  "p95_est_s": 2.06
}
```

Mean <2s gate passed. **Notes:** p95 slightly above 2s plan target; parse rate dropped to 22% at 64-way fan-out vs 100% serial in S4.5 — Stage 3b should cap client concurrency (recommend ≤8 until server batches properly).

### S4.6b serial parse diagnostic (2026-05-31) — **FAIL** (parse gate)

Modal app: `ap-M2MGAn2QVjGPJeBPMKVkzU` (alee72) · config `judge_latency_smoke_serial.yaml` · c=1 · 64 tasks

```json
{
  "concurrency": 1,
  "wall_s": 91.69,
  "mean_wall_per_task_s": 1.43,
  "parse_ok_rate": 0.21875,
  "parse_fail_truncated": 50,
  "parse_fail_http_or_empty": 0
}
```

**Verdict:** parse collapse is **not load/concurrency** — identical 22% at c=1, c=8, and c=64. Failures are truncated JSON mid-response (likely `max_tokens` too low for the `product` synthetic template; S4.5 `addition` template parses at 100%). Serial timing ~1.4s/call (~92s for 64 tasks); S4.5 agreement ~1.5s/call (~152s for 100 calls).

**Follow-up:** raise `judge.max_tokens` in smoke configs and/or align latency probe template with agreement; redeploy judge on `chicken602`.


---

## Plan audit — `stage-04-agent-plan.md` (2026-05-30)

**Initial verdict:** FAIL → **reconciled:** PASS WITH NOTES (see prior audit section in git history)

**Stage 3b ready (pending Stage 3a hook PASS):** no — judge service deployed but S4.5/S4.6 smokes not run

---

## Next steps

1. Confirm judge health returns 200 after cold-start (Modal logs / longer curl).
2. Run S4.5 agreement + S4.6 latency smokes with `JUDGE_BASE_URL` above.
3. S4.1–S4.4 audits (read-only).
4. Stage 3a S3a.6 must PASS before Stage 3b wiring (`clusters_judge.py`).

---

## Stage 4 read-only audit (2026-05-30)

- **Verdict:** **PASS WITH NOTES**
- **Auditor:** claude (in-conversation, after background sonnet auditor died to API stream timeout)
- **Stage 3a S3a.6 PASS confirmed** 2026-05-30 — Stage 3b is unblocked.

### Holes / compromises (ranked by severity)

1. **Concurrency=8 cap is documented but NOT enforced in code.** `judge/client.py:21` defaults `JudgeClientConfig.concurrency = 64`; `client.py:43` env var defaults `JUDGE_CONCURRENCY=64`. The S4.6 finding (this log, line ~110) shows 64-way fan-out collapses parse rate to 22 %. A Stage 3b client that uses `JudgeClient.from_env()` without overrides will silently ship broken cluster assignments. **Fix before Stage 3b launch: change both defaults to 8** (one-line each), or have `train/clusters_judge.py` always construct `JudgeClientConfig(concurrency=min(8, …))` explicitly. Belt + braces both.
2. **`server.py:41 max_containers=1`** is almost certainly the root cause of the 64-way cliff: a single B200 vLLM container serving 64 concurrent requests can't keep up, responses truncate at `max_tokens=2048`, truncated JSON fails parse. Either bump `max_containers` (autoscale-then-batch) OR document that 8 is the permanent ceiling for any client. Stage 8 production runs at 3 arms × ~50K prompts × 8 rollouts will hit this hard.
3. **No retries on transient errors.** `client.py:83–89` catches all `HTTPError|KeyError|IndexError|TypeError` and silently returns an empty `JudgeClusterResult` with `parse_ok=False`. A single Modal cold-start blip = a permanently lost cluster assignment for that prompt. For Stage 8 production scale, recommend ≥1 retry with exponential backoff before returning the empty result.
4. **`server.py:88–89` swallows all exceptions** and returns `{"error": str, "choices": []}` instead of HTTP 5xx. The client then hits `data["choices"][0]` → IndexError → caught at `client.py:83`. Works but masks the original error; for Stage-7 logging wiring, a real status code would help.

### Confirmed clean

- **Tests are substantive.** `tests/test_judge_parse.py` (75 lines, 7/7 PASS) covers known-input cluster assignments, degenerate cluster 100 handling, markdown JSON fence stripping, missing-key failure mode, invalid-JSON failure mode. Not trivial. Re-ran 7/7 PASS in 0.10s.
- **`parse.py` is robust.** Strips ```json fences, validates 1-indexed keys cover `1..n`, normalizes `POLY_EPO_DEGENERATE_RAW` (100) → `DEGENERATE_CLUSTER_ID`, handles non-dict payloads.
- **Modal deploy is real.** Health + chat-completions endpoints live, S4.5 (100 % parse, 100 % agreement on 50 tasks serial) and S4.6 (mean 1.37s) smokes match the log claims.
- **`alee72` profile is acceptable for Stage 3b smoke** (judge is on a separate account from the trainer; no resource contention).

### Required actions before Stage 3b launch

1. Drop `JudgeClient` default `concurrency` 64 → 8 in `client.py:21,43`. **And/or** have Stage 3b's `clusters_judge.py` always pass `concurrency=8` explicitly when constructing the client.
2. Confirm judge health endpoint returns 200 within 30s (the S4.2 note flagged a 120s cold-start timeout; if cold-start is still 2+ min, the Stage 3b probe pre-flight `curl health` will fail spuriously — bump the pre-flight timeout to 180s or warm the judge via an idempotent ping before the smoke).

### Required actions before Stage 8 launch

1. Add retry-with-backoff to `client.py` (3 attempts, 0.5/1/2 s backoff, on `httpx.TimeoutException` and HTTP 5xx).
2. Decide on permanent concurrency policy: either bump `server.py:41 max_containers` and re-validate at 64-way fan-out, OR document the cap=8 and check that Stage 8 step-time budget (~175 s/step at 128 prompts × 1.37 s ÷ 8) is acceptable vs the migration plan §3 budget.
3. Diff `prompts/poly_epo_a1.md` against `main/judge/format.py` — confirm verbatim port (locked stack per reward-decision.md).

