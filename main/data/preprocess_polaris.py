"""
One-shot Polaris freeze → polaris_train.jsonl (PLAN §2).

Not invoked by the trainer. Run manually once §2 sampling is locked.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    raise NotImplementedError(
        "preprocess_polaris.py is a stub until PLAN §2 freeze is locked. "
        "Group B uses probes/05-24/group_a/manifest.jsonl on main-artifacts."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
