# Detached launch manifest (2026-05-19T19:01:58Z)

All runs use `modal run --detach` + default spawn. Safe to close laptop after spawn.

| Run ID | Modal app | Local artifact dir |
|--------|-----------|-------------------|
| run0_proxy | [ap-Zk6zAIs9tWpGerJHufSud1](https://modal.com/apps/chicken602/main/ap-Zk6zAIs9tWpGerJHufSud1) | `pilot/artifacts/run0_proxy/20260519T190202Z` |
| run1_grpo | [ap-CpcEIWjwiNMb8MvGCZFpAT](https://modal.com/apps/chicken602/main/ap-CpcEIWjwiNMb8MvGCZFpAT) | `pilot/artifacts/run1_grpo/20260519T190202Z` |
| run1b_grpo | [ap-EWhmIPbGpflmnM2IcrKp77](https://modal.com/apps/chicken602/main/ap-EWhmIPbGpflmnM2IcrKp77) | `pilot/artifacts/run1b_grpo/20260519T190201Z` |
| run2_inverse_freq | [ap-aAYroxfDF3TuZY5NJ1pbOP](https://modal.com/apps/chicken602/main/ap-aAYroxfDF3TuZY5NJ1pbOP) | `pilot/artifacts/run2_inverse_freq/20260519T190201Z` |
| run3_f_grpo | [ap-MO0JD72gMTybU9Sv7VCSrn](https://modal.com/apps/chicken602/main/ap-MO0JD72gMTybU9Sv7VCSrn) | `pilot/artifacts/run3_f_grpo/20260519T190201Z` |

Monitor: `modal app list` · `modal app logs <app-id>`

Pull when done:

```bash
python pilot/scripts/pull_run_artifacts.py --run-id <run_id> --local-dir <dir above>
```
