"""Create the "Stage 8 production (verl)" saved view in wandb.

Layout for the 3 production verl runs (`grpo_train_4b_1epoch`,
`minority_cot_train_4b_1epoch`, `poly_epo_cot_train_4b_1epoch`).

Filters on tags ``verl`` + ``production`` so only the Stage-8 runs land in
this workspace. The judge/CoT section is empty (NaN) for the GRPO run by
design — those keys are only emitted when ``cluster_source=judge``.

Usage:
    main/.venv/bin/python main/scripts/setup_wandb_production_view.py

Requires:
    pip install wandb-workspaces
    WANDB_API_KEY set (or ~/.netrc credentials)
"""

from dataclasses import dataclass
from typing import ClassVar, Literal

import wandb_workspaces.expr as _expr
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr

ENTITY = "224r-project"
PROJECT = "cs224r-minority-voting"
VIEW_NAME = "Stage 8 production (verl)"

# verl native timing keys for the step-time breakdown. `step` is the full
# wall; the others are non-overlapping sub-phases that should roughly sum
# to it.
TIMING_PHASES = [
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/adv",
    "timing_s/update_actor",
]


@dataclass(eq=False, frozen=True)
class RunTags(_expr.BaseMetric):
    """Filter expression for the actual run tags field.

    `ws.Tags()` in wandb-workspaces 0.3.x hardcodes `section="run"`, which the
    backend interprets as a column named `tags` (the lowercase one with the
    list icon in the filter UI) rather than the real `Tags` field (tag icon).
    Override the section to `"tags"` so it filters on the actual run tags.
    """

    name: str = "tags"
    section: ClassVar[Literal["tags"]] = "tags"


def build_workspace() -> ws.Workspace:
    return ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name=VIEW_NAME,
        runset_settings=ws.RunsetSettings(
            filters=[
                RunTags().isin(["verl"]),
                RunTags().isin(["production"]),
            ],
        ),
        sections=[
            ws.Section(
                name="1. Learning signal",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Mean reward (smoothed)",
                        y=["critic/score/mean"],
                        smoothing_type="exponential",
                        smoothing_factor=0.9,
                        smoothing_show_original=True,
                    ),
                    wr.LinePlot(
                        title="Pass@8 (any rollout correct)",
                        y=["train/pass_at_8"],
                        smoothing_type="exponential",
                        smoothing_factor=0.9,
                        smoothing_show_original=True,
                    ),
                    wr.LinePlot(
                        title="Prompts unlocked (cumulative unique)",
                        y=["train/prompts_unlocked"],
                    ),
                    wr.LinePlot(
                        title="Fraction filtered (all-same advantage)",
                        y=["train/fraction_filtered"],
                    ),
                ],
            ),
            ws.Section(
                name="2. PPO stability",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Clip frac + KL",
                        y=[
                            "actor/pg_clipfrac",
                            "actor/pg_clipfrac_lower",
                            "actor/ppo_kl",
                        ],
                    ),
                    wr.LinePlot(
                        title="Grad norm (pre-clip)",
                        y=["actor/grad_norm"],
                        log_y=True,
                    ),
                    wr.LinePlot(
                        title="Entropy + pg_loss",
                        y=["actor/entropy", "actor/pg_loss"],
                    ),
                ],
            ),
            ws.Section(
                name="3. Completion shape",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Response length (mean / max)",
                        y=["response_length/mean", "response_length/max"],
                    ),
                    wr.LinePlot(
                        title="Length-clip ratio (hits 4096)",
                        y=["response_length/clip_ratio"],
                    ),
                    wr.LinePlot(
                        title="Prompt length (mean / max)",
                        y=["prompt_length/mean", "prompt_length/max"],
                    ),
                ],
            ),
            ws.Section(
                name="4. Judge + CoT diagnostics (CoT arms only)",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Distinct clusters / step (target > 1.5)",
                        y=["train/distinct_clusters_mean"],
                    ),
                    wr.LinePlot(
                        title="Judge parse OK rate (target > 0.95)",
                        y=["train/judge_parse_ok_rate"],
                    ),
                    wr.LinePlot(
                        title="Degenerate rollouts (single-cluster collapse)",
                        y=["train/degenerate_rollouts"],
                    ),
                    wr.LinePlot(
                        title="Judge overflow skipped",
                        y=["train/judge_overflow_skipped"],
                    ),
                ],
            ),
            ws.Section(
                name="5. Perf + timing",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Step wall (s)",
                        y=["perf/time_per_step"],
                    ),
                    wr.LinePlot(
                        title="Throughput (tokens / sec / GPU)",
                        y=["perf/throughput"],
                    ),
                    wr.LinePlot(
                        title="Peak VRAM (GB, per-rank PyTorch allocator)",
                        y=["perf/max_memory_allocated_gb"],
                    ),
                    wr.LinePlot(
                        title="Timing breakdown (s, stacked)",
                        y=TIMING_PHASES,
                        plot_type="stacked-area",
                        title_y="seconds",
                    ),
                ],
            ),
            ws.Section(
                name="6. Validation (every test_freq=50 steps)",
                is_open=False,
                panels=[
                    wr.LinePlot(
                        title="val-core score (mean)",
                        y=["val-core/score/mean", "val-aux/score/mean"],
                    ),
                ],
            ),
        ],
    )


def main() -> None:
    workspace = build_workspace()
    saved = workspace.save()
    print(f"View created: {saved.url}")


if __name__ == "__main__":
    main()
