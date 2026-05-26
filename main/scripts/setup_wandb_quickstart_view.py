"""Create the "GRPO quickstart" saved view in the wandb project workspace.

Run once per project — it creates a new saved view (visible in the workspace
dropdown at the top of the run page) without touching the default workspace.

Usage:
    main/.venv/bin/python main/scripts/setup_wandb_quickstart_view.py

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
VIEW_NAME = "GRPO quickstart"


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
            filters=[RunTags().isin(["production"])],
        ),
        sections=[
            ws.Section(
                name="1. Learning signal",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Mean reward (smoothed)",
                        y=["train/mean_reward"],
                        smoothing_type="exponential",
                        smoothing_factor=0.9,
                        smoothing_show_original=True,
                    ),
                    wr.LinePlot(
                        title="Pass@k distribution",
                        y=[f"train/frac_prompts_{i}_correct" for i in range(9)],
                        plot_type="stacked-area",
                    ),
                ],
            ),
            ws.Section(
                name="2. Stability",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Importance ratio + clip frac",
                        y=[
                            "train/ratio_max",
                            "train/ratio_p95",
                            "train/clipped_high_frac",
                            "train/clipped_low_frac",
                        ],
                        log_y=True,
                    ),
                    wr.LinePlot(
                        title="Grad norm (pre-clip)",
                        y=["train/grad_norm_preclip"],
                        log_y=True,
                    ),
                ],
            ),
            ws.Section(
                name="3. Completion shape",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="Completion length",
                        y=[
                            "train/mean_completion_tokens",
                            "train/p95_completion_tokens",
                        ],
                    ),
                    wr.LinePlot(
                        title="Finish reason",
                        y=[
                            "train/frac_finish_stop",
                            "train/frac_finish_length",
                            "train/frac_finish_other",
                        ],
                        plot_type="stacked-area",
                    ),
                ],
            ),
            ws.Section(
                name="4. Safety + perf + samples",
                is_open=True,
                panels=[
                    wr.LinePlot(
                        title="VRAM peak (GB)",
                        y=["train/vram_peak_gb_step"],
                        range_y=(80, 145),
                    ),
                    wr.LinePlot(
                        title="Time per step (s, stacked)",
                        y=[
                            "train/t_rollout_s",
                            "train/t_train_fwd_bwd_s",
                            "train/t_weight_sync_s",
                        ],
                        plot_type="stacked-area",
                    ),
                    wr.MediaBrowser(
                        title="Sample completions (every 50 steps)",
                        media_keys=[
                            "sample/completion_0",
                            "sample/completion_1",
                            "sample/completion_2",
                        ],
                        num_columns=1,
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
