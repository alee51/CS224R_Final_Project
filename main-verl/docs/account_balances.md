# Modal account balances

**Last updated:** 2026-05-31 ~22:45 PT (after killing failed Stage 8 v1 launches + emma stale judge)

| Account | Balance | Source | Owner |
|---|---|---|---|
| chicken602 (nbao0) | **$307.28** | user-visible / authoritative | Nancy |
| emma | **~$770** | calculated: $1013.42 (3:14am snapshot) − $239.23 (today post-cutoff billing) − killed stale judge `ap-Enf93pyH1UyzvvGIlVIksF` ($140 of that was burning through it) — actual likely closer to **~$770–820** as the final bill settles | Emma |
| anastasia | **~$531** | calculated: $610.90 (3:14am) − $79.77 (today) | Anastasia |
| abao | **$910.16** | user-visible / authoritative (after $506 topup ~22:30 PT) | (third-party) |
| stonedpinecones | **$1,201.00** | added 2026-05-31 ~22:50 PT, updated 22:55 PT | (friend) |
| **Total** | **~$3,719** | — | — |

## Final relaunch arm-account assignment (2026-05-31 ~23:00 PT)

| Arm | Account | Cost projection | Balance | Buffer |
|---|---|---|---|---|
| GRPO | anastasia | $444 (17.78 hr × $25) | ~$531 | +$87 |
| Minority-CoT | emma | $583 (23.33 hr × $25) | ~$770 | +$187 |
| Poly-EPO-CoT + judge | **stonedpinecones** | $639 + $320 = $959 | $1,201 | **+$242** |
| (idle) | chicken602 | — | $307 | — |

**Judge URL the CoT arms point to:** `https://stonedpinecones--v1-chat-completions.modal.run`

**Fresh-start guarantee:** all three configs use `experiment_name: *_v2` with matching `default_local_dir: /vol/checkpoints/main-verl/*_v2` — no resume from morning crashed runs.

**Pre-launch checks (mandatory):**
- Judge health 200 OK on stonedpinecones before launching CoT arms.
- PCIe / HGX bare-metal check on EVERY container within first 2 min: `nvidia-smi -q | grep '^GPU 0000'`. Kill criteria:
  - Multiple PCIe domains (e.g. `0002:`/`0003:`) → virtualized passthrough, ~40% wall slowdown.
  - OR `clocks.sm < 0.8 × clocks.max.sm` with `clocks_throttle_reasons.active = 0x0`.
- Kill+relaunch same probe to re-roll the host lottery.

## Per-account spend today (2026-05-31, post 3:14am PT cutoff)

From `modal billing report --for today --resolution h --tz local --json`:

| Account | 4am+ post | 3-4am ambig | Total post-cutoff |
|---|---|---|---|
| chicken602 | $35.85 | $4.76 | ~$40.61 |
| emma | $231.40 | $7.83 | ~$239.23 |
| anastasia | $75.10 | $4.67 | ~$79.77 |
| abao | $296.00 | $3.84 | ~$299.84 |

## Notes

- Doc 3:14am snapshot from `main-verl/docs/human notes.md`.
- emma figure assumes the killed stale judge `ap-Enf93pyH1UyzvvGIlVIksF` ($140 today) doesn't keep accruing — verified stopped at 22:45 PT.
- chicken602 user-reported $315.64 earlier dropped to $307.28 as final-app-death bills settled. Same dynamic could push emma/anastasia a few dollars lower.
- abao balance $911.80 (user-reported) → $910.16 = small final-bill settling.
