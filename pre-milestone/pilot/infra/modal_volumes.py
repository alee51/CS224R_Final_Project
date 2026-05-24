"""Modal Volume names and local sync helpers for pilot artifacts."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

# Long-lived volumes (storage cost is negligible vs GPU).
ARTIFACTS_VOLUME_NAME = "pilot-artifacts"
HF_CACHE_VOLUME_NAME = "hf-cache"

REMOTE_ARTIFACTS_ROOT = "/root/pilot/artifacts"
REMOTE_HF_CACHE_ROOT = "/root/.cache/huggingface"


def pull_run_artifacts_from_volume(
    run_id: str,
    local_run_dir: Path,
    *,
    volume_name: str = ARTIFACTS_VOLUME_NAME,
) -> None:
    """Download ``<run_id>/`` from the volume into *local_run_dir* (files directly).

    ``modal volume get`` always creates ``<dest>/<run_id>/…`` (rsync-like). We pull
    into a temp directory, then copy into *local_run_dir* so timestamped folders work
    without nesting or overwriting sibling runs.
    """
    local_run_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="modal-artifacts-") as tmp:
        staging = Path(tmp)
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                "--force",
                volume_name,
                f"{run_id}/",
                str(staging),
            ],
            check=True,
        )
        remote_tree = staging / run_id
        if not remote_tree.is_dir():
            raise FileNotFoundError(
                f"Expected {remote_tree} after volume get; got {list(staging.iterdir())}"
            )
        for path in remote_tree.iterdir():
            dest = local_run_dir / path.name
            if path.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(path, dest)
            else:
                shutil.copy2(path, dest)
