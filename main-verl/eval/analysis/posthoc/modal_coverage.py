"""Run coverage@k analysis on eval_4b JSONs directly on Modal (no local download).

Reads large eval JSONs from /vol/probes/eval_4b/ on the main-artifacts volume,
computes coverage@k (distinct correct answers per prompt), and writes a compact
results JSON to /vol/probes/eval_4b/coverage_results.json.

Usage (from repo root, nbao0 profile):
  MODAL_PROFILE=nbao0 PYTHONPATH=main-verl modal run \
    main-verl/eval/analysis/posthoc/modal_coverage.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import modal

ARTIFACTS_VOLUME_NAME = "main-artifacts"
ARTIFACTS_MOUNT = "/vol"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("ijson")
)

app = modal.App("cs224r-coverage-analysis")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)

ARMS = ["base", "grpo", "minority", "polyepo"]
DATASETS = ["aime25", "aime26", "beyondaime"]
K_VALUES = [1, 2, 4, 8, 16, 32, 64]

FILES = [
    f"{arm}_step400_smallood_{ds}.json"
    for arm in ARMS
    for ds in DATASETS
]


def _expected_coverage(cluster_sizes: list, n: int, k: int) -> float:
    """Exact E[distinct correct clusters in a random k-subset of n rollouts].

    For each cluster with s correct rollouts, P(cluster appears in subset) =
    1 - C(n-s, k) / C(n, k). Sum over clusters gives expected distinct count.
    """
    denom = math.comb(n, k)
    if denom == 0:
        return 0.0
    return sum(1.0 - math.comb(n - s, k) / denom for s in cluster_sizes)


def coverage_at_k(per_prompt, k):
    """Exact expected distinct correct answers in a random k-subset, averaged over prompts."""
    cov = []
    for p in per_prompt:
        cluster_counts: dict = {}
        for r, pred in zip(p["rewards"], p["preds"]):
            if r > 0.5 and pred and pred != "[INVALID]":
                cluster_counts[pred] = cluster_counts.get(pred, 0) + 1
        n = len(p["rewards"])
        cov.append(_expected_coverage(list(cluster_counts.values()), n, k))
    return sum(cov) / len(cov) if cov else 0.0


def distinct_answers_at_k(per_prompt, k):
    da = []
    for p in per_prompt:
        distinct = {pred for pred in p["preds"][:k] if pred and pred != "[INVALID]"}
        da.append(len(distinct))
    return sum(da) / len(da) if da else 0.0


def answer_entropy_at_k(per_prompt, k):
    ents = []
    for p in per_prompt:
        first_k = [pred for pred in p["preds"][:k] if pred and pred != "[INVALID]"]
        if not first_k:
            continue
        counts = Counter(first_k)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        h = -sum(prob * math.log2(prob) for prob in probs if prob > 0)
        ents.append(h)
    return (sum(ents) / len(ents)) if ents else 0.0


def analyze_file(path: Path) -> dict:
    import ijson

    results: dict = {"label": None, "n_rollouts": None, "datasets": {}}
    with path.open("rb") as f:
        for prefix, event, value in ijson.parse(f, use_float=True):
            if prefix in ("label", "n_rollouts") and event in ("string", "number"):
                results[prefix] = value
            if prefix == "datasets" and event == "start_map":
                break

    with path.open("rb") as f:
        for ds_name, ds_obj in ijson.kvitems(f, "datasets"):
            pp = ds_obj.get("per_prompt", [])
            n_roll = results["n_rollouts"] or 64
            ds_result = {
                "n_prompts": ds_obj.get("n_prompts", len(pp)),
                "pass_at_k": ds_obj.get("pass_at_k", {}),
                "coverage_at_k": {},
                "distinct_answers_at_k": {},
                "entropy_at_k": {},
            }
            for k in K_VALUES:
                if k > n_roll:
                    continue
                ds_result["coverage_at_k"][f"coverage@{k}"] = coverage_at_k(pp, k)
                ds_result["distinct_answers_at_k"][f"distinct@{k}"] = distinct_answers_at_k(pp, k)
                ds_result["entropy_at_k"][f"entropy@{k}"] = answer_entropy_at_k(pp, k)
            results["datasets"][ds_name] = ds_result

    return results


@app.function(
    image=image,
    volumes={ARTIFACTS_MOUNT: artifacts_volume},
    timeout=3600,
    memory=8192,
)
def run_coverage() -> dict:
    eval_dir = Path(ARTIFACTS_MOUNT) / "probes" / "eval_4b"
    all_results = {}

    for fname in FILES:
        path = eval_dir / fname
        if not path.exists():
            print(f"SKIP (not found): {fname}")
            continue
        print(f"analyzing {fname} ({path.stat().st_size // 1024 // 1024} MB)...")
        try:
            result = analyze_file(path)
            all_results[fname] = result
            for ds_name, ds in result["datasets"].items():
                cov16 = ds["coverage_at_k"].get("coverage@16", "?")
                p16 = ds["pass_at_k"].get("pass@16", "?")
                print(f"  {ds_name}: coverage@16={cov16:.3f}  pass@16={p16:.3f}" if isinstance(cov16, float) else f"  {ds_name}: {cov16}")
        except Exception as e:
            print(f"ERROR {fname}: {e}")
            all_results[fname] = {"error": str(e)}

    out_path = eval_dir / "coverage_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=float))
    artifacts_volume.commit()
    print(f"\nwrote {out_path}")
    return all_results


@app.local_entrypoint()
def main():
    results = run_coverage.remote()
    print("\n=== COVERAGE SUMMARY ===")
    for fname, r in results.items():
        if "error" in r:
            print(f"{fname}: ERROR — {r['error']}")
            continue
        arm = r.get("label", fname)
        for ds_name, ds in r.get("datasets", {}).items():
            cov = ds["coverage_at_k"]
            pas = ds["pass_at_k"]
            print(f"\n{arm} | {ds_name} (n={ds['n_prompts']})")
            print(f"  pass@16={pas.get('pass@16', '?'):.3f}  coverage@16={cov.get('coverage@16', '?'):.3f}")
            print(f"  coverage: " + "  ".join(f"@{k}={v:.2f}" for k, v in cov.items()))
