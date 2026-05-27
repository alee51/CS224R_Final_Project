"""Checkpoint resume helpers (no GPU)."""

from pathlib import Path

import torch
from torch import nn

from data.dataset import JsonlPromptDataset
from train.trainer import (
    apply_launch_overrides,
    find_latest_checkpoint,
    load_ckpt,
    resolve_checkpoint_dir,
    save_ckpt,
    train_cfg_from_dict,
)


def test_apply_launch_overrides_total_steps(monkeypatch):
    monkeypatch.setenv("CS224R_TOTAL_STEPS", "10")
    monkeypatch.setenv("CS224R_TRAIN_MODE", "smoke")
    out = apply_launch_overrides({"train": {"total_steps": 850, "batch_size": 64}})
    assert out["train"]["total_steps"] == 10
    assert out["train"]["batch_size"] == 64
    assert out["launch_mode"] == "smoke"


def test_find_latest_checkpoint(tmp_path: Path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "step_000002.pt").write_bytes(b"x")
    (ckpt_dir / "step_000010.pt").write_bytes(b"y")
    assert find_latest_checkpoint(ckpt_dir) == ckpt_dir / "step_000010.pt"
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_resolve_checkpoint_dir_run_scoped(monkeypatch):
    monkeypatch.setenv("CS224R_NO_RESUME", "1")
    cfg = train_cfg_from_dict(
        {
            "global_seed": 0,
            "arm": "grpo",
            "train": {"checkpoint_dir": "/vol/checkpoints/train_real/"},
            "rollout": {"model": "test"},
            "loss": {},
            "wandb": {"entity": "e", "project": "p"},
        }
    )
    path = resolve_checkpoint_dir(
        cfg, checkpoint_run_id="cs224r-train-grpo-full-nancy-05-27-1200"
    )
    assert path == Path(
        "/vol/checkpoints/train_real_cs224r-train-grpo-full-nancy-05-27-1200"
    )


def test_resolve_checkpoint_dir_legacy_yaml_dir(monkeypatch):
    monkeypatch.setenv("CS224R_NO_RESUME", "1")
    cfg = train_cfg_from_dict(
        {
            "global_seed": 0,
            "arm": "grpo",
            "train": {"checkpoint_dir": "/vol/checkpoints/train_real/"},
            "rollout": {"model": "test"},
            "loss": {},
            "wandb": {"entity": "e", "project": "p"},
        }
    )
    assert resolve_checkpoint_dir(cfg) == Path("/vol/checkpoints/train_real/")


def test_dataset_state_roundtrip(tmp_path: Path):
    import json

    path = tmp_path / "toy.jsonl"
    rows = [{"problem": f"p{i}", "gold": str(i), "problem_id": i} for i in range(8)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = JsonlPromptDataset(str(path), seed=7)
    ds.next_batch_with_ids(3)
    state = ds.state_dict()
    ds2 = JsonlPromptDataset(str(path), seed=99)
    ds2.load_state_dict(state)
    a = ds.next_batch_with_ids(2)
    b = ds2.next_batch_with_ids(2)
    assert a == b


def test_save_load_ckpt_includes_dataset(tmp_path: Path):
    import json

    path = tmp_path / "toy.jsonl"
    path.write_text(
        json.dumps({"problem": "p", "gold": "1", "problem_id": 0}) + "\n"
    )
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ds = JsonlPromptDataset(str(path), seed=0)
    ds.next_batch_with_ids(1)
    cfg = train_cfg_from_dict(
        {
            "global_seed": 0,
            "arm": "grpo",
            "train": {},
            "rollout": {"model": "test"},
            "loss": {},
            "wandb": {"entity": "e", "project": "p"},
        }
    )
    ckpt_path = tmp_path / "step_000000.pt"
    save_ckpt(ckpt_path, model, opt, 0, cfg, "wandb-test-id", ds)
    ds2 = JsonlPromptDataset(str(path), seed=99)
    model2 = nn.Linear(4, 2)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    payload = load_ckpt(ckpt_path, model2, opt2, ds2)
    assert payload["wandb_run_id"] == "wandb-test-id"
    assert payload["step"] == 0
    assert ds2.state_dict() == ds.state_dict()
