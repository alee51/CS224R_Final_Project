"""JsonlPromptDataset unit tests (no GPU)."""

import json
from pathlib import Path

from data.dataset import JsonlPromptDataset


def test_jsonl_prompt_dataset_batch(tmp_path: Path):
    path = tmp_path / "toy.jsonl"
    rows = [
        {"problem": "p0", "gold": "0", "problem_id": 10},
        {"problem": "p1", "gold": "1", "problem_id": 11},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ds = JsonlPromptDataset(str(path), seed=0)
    probs, golds, ids = ds.next_batch_with_ids(2)
    assert len(probs) == 2
    assert set(golds) == {"0", "1"}
    assert set(ids) == {10, 11}
