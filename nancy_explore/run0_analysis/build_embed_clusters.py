"""Build per-rollout completion-embedding cluster assignments at the
best threshold from Analysis B (currently 0.2), writing
`embed_clusters_at_best_threshold.parquet` for use by Analysis D.

Per-prompt agglomerative clustering with cosine distance, distance_threshold=0.2.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parent
EMB = ROOT / "embeddings_minilm.npy"
IDS = ROOT / "embedding_ids.parquet"
OUT = ROOT / "embed_clusters_at_best_threshold.parquet"

THRESHOLD = 0.2


def main() -> None:
    ids = pd.read_parquet(IDS)
    emb = np.load(EMB)
    assert len(ids) == len(emb), (len(ids), len(emb))

    rows = []
    for prompt_id, g in ids.groupby("prompt_id", sort=False):
        idx = g.index.to_numpy()
        sub = emb[idx]
        # Agglomerative with cosine distance and distance_threshold
        model = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=THRESHOLD,
        )
        labels = model.fit_predict(sub)
        for ridx, lab in zip(g["rollout_idx"].to_numpy(), labels):
            rows.append((prompt_id, int(ridx), f"{prompt_id}:{int(lab)}"))
    out = pd.DataFrame(rows, columns=["prompt_id", "rollout_idx",
                                       f"completion_embedding@{THRESHOLD}"])
    out.to_parquet(OUT)
    print(f"wrote {OUT}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
