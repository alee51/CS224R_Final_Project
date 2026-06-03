"""Modal-side runner for Tier 1 analysis + grader sanity (eval.md §6.1 + §8).

v3: each script exposes `analyze(json_data) -> str`. The worker loads the JSON
ONCE per file and calls each analyzer in-process — vs v2 which re-loaded the
full multi-GB JSON inside each subprocess (9× the I/O on math500).

Mounts main-artifacts, picks up the saved eval JSONs in /vol/probes/eval_4b/,
runs every Tier 1 metric + grader sanity (with math_dapo tripwire, since verl
is in the image) inside the container, and writes only small markdown
summaries back to /vol/probes/eval_4b/_summaries/<label>/.

Usage:
    MODAL_PROFILE=abao modal run main-verl/eval/analysis/posthoc/posthoc_app.py::analyze \\
        --input-glob "/vol/probes/eval_4b/<arm>_step400_*.json"

Default skips the combined mega-files (`aime25-aime26-...`) and schemaprobe
files; pass --skip-patterns "" to override.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Local `modal run` needs main-verl/ on sys.path so `from infra.*` works.
# Inside the Modal container PYTHONPATH=/root/main-verl is set by the image.
for _p in Path(__file__).resolve().parents:
    if (_p / "infra" / "modal_image.py").exists():
        sys.path.insert(0, str(_p))
        break

import modal

from infra.modal_image import app_name, image as _base_image
from infra.modal_volume import ARTIFACTS_MOUNT, ARTIFACTS_VOLUME_NAME

app = modal.App(app_name())
image = _base_image
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)


def _sanity_md(label: str, data: dict) -> str:
    """Spec §8.1-2: n_correct histogram + sample tuples."""
    import collections
    out = [f"# Grader sanity — {label}", ""]
    for ds_name, ds in data.get("datasets", {}).items():
        out.append(f"## {ds_name} (n={ds['n_prompts']} prompts)")
        out.append("")
        out.append(f"- saved `pass_at_k`: `{ds['pass_at_k']}`")
        out.append(f"- mean_reward_at_1: `{ds.get('mean_reward_at_1', float('nan')):.4f}`")
        prompts = ds["per_prompt"]
        ncs = [p["n_correct"] for p in prompts]
        hist = sorted(collections.Counter(ncs).items())
        out.append(f"- n_correct distribution: `{dict(hist)}`")
        empty = sum(1 for p in prompts for q in p["preds"] if q == "")
        total = sum(len(p["preds"]) for p in prompts)
        out.append(f"- empty `preds`: {empty}/{total} ({100*empty/max(total,1):.1f}%)")
        out.append("")
        out.append("Sample (problem_id, gt, preds[:3], rewards[:3]):")
        out.append("```")
        for i in range(min(3, len(prompts))):
            p = prompts[i]
            out.append(
                f"[{i}] id={p['problem_id']!r} gt={p['ground_truth']!r} "
                f"n_correct={p['n_correct']} preds[:3]={p['preds'][:3]} "
                f"rewards[:3]={p['rewards'][:3]}"
            )
        out.append("```")
        out.append("")
    return "\n".join(out)


def _analyze_one(input_path: str, summaries_root: str,
                 self_bleu_max_rollouts: int = 8,
                 self_bleu_max_problems: int = 0) -> dict:
    """Load JSON once, call each analyzer in-process."""
    import json
    import time
    import traceback

    inp = Path(input_path)
    label = inp.stem
    out_dir = Path(summaries_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"input": str(inp), "label": label, "out_dir": str(out_dir)}

    # ONE JSON load per file (vs 9× in v2). This is the headline speedup.
    t0 = time.time()
    try:
        with inp.open() as f:
            data = json.load(f)
        results["json_load_s"] = round(time.time() - t0, 1)
    except Exception as e:
        results["json_load"] = f"FAIL: {e}"
        return results

    # Import the analyzers — done inside the worker so the orchestrator
    # container doesn't need to load any of them.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import auc_at_k, coverage, potential_at_k, reflective_actions
    import diff_at_k_split, token_entropy_split, self_bleu, rescore

    # (script_name, callable returning markdown string)
    analyzers = [
        ("sanity",            lambda d: _sanity_md(label, d)),
        ("auc_at_k",          auc_at_k.analyze),
        ("coverage",          coverage.analyze),
        ("potential_at_k",    potential_at_k.analyze),
        ("reflective_actions",reflective_actions.analyze),
        ("diff_at_k_split",   diff_at_k_split.analyze),
        ("token_entropy_split", token_entropy_split.analyze),
        ("self_bleu",         lambda d: self_bleu.analyze(d,
                                  max_rollouts=self_bleu_max_rollouts,
                                  max_problems=self_bleu_max_problems)),
        ("rescore",           rescore.analyze),
    ]

    for name, fn in analyzers:
        t0 = time.time()
        try:
            md = fn(data)
            (out_dir / f"{name}.md").write_text(md)
            results[name] = f"ok ({time.time()-t0:.1f}s)"
        except Exception as e:
            (out_dir / f"{name}.err").write_text(
                f"{e}\n\n{traceback.format_exc()}"
            )
            results[name] = f"FAIL: {type(e).__name__}: {e}"

    # Per-file summary
    lines = [f"# Posthoc summary — {label}", "",
             f"- json_load_s: {results.get('json_load_s', '?')}"]
    for k, v in results.items():
        if k in ("input", "label", "out_dir", "json_load_s"):
            continue
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("Files:")
    for fn_ in sorted(p.name for p in out_dir.iterdir() if p.is_file()):
        lines.append(f"- `{fn_}`")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))
    return results


@app.function(
    image=image,
    timeout=2 * 3600,
    memory=65 * 1024,  # 64 GB; math500 is 23 GB on disk → ~50-60 GB after json.load
    cpu=4.0,
    volumes={ARTIFACTS_MOUNT: artifacts_volume},
)
def analyze_one(input_path: str, summaries_root: str,
                self_bleu_max_rollouts: int = 8,
                self_bleu_max_problems: int = 0) -> dict:
    Path(summaries_root).mkdir(parents=True, exist_ok=True)
    r = _analyze_one(input_path, summaries_root,
                     self_bleu_max_rollouts=self_bleu_max_rollouts,
                     self_bleu_max_problems=self_bleu_max_problems)
    artifacts_volume.commit()
    return r


@app.function(
    image=image,
    timeout=4 * 3600,
    volumes={ARTIFACTS_MOUNT: artifacts_volume},
)
def analyze(
    input_glob: str = "/vol/probes/eval_4b/*_step400_*.json",
    summaries_root: str = "/vol/probes/eval_4b/_summaries",
    skip_patterns: str = "schemaprobe,aime25-aime26",
    self_bleu_max_rollouts: int = 8,
    self_bleu_max_problems: int = 0,
) -> None:
    import glob as _glob
    import time

    Path(summaries_root).mkdir(parents=True, exist_ok=True)
    skips = [s for s in skip_patterns.split(",") if s]
    files = sorted(_glob.glob(input_glob))
    files = [f for f in files if not any(s in Path(f).name for s in skips)]
    print(f"[posthoc v3] glob={input_glob}  matched={len(files)}  (after skip={skips})")
    for f in files:
        print(f"  - {f}  ({Path(f).stat().st_size/1e6:.1f} MB)")

    print(f"\n[posthoc v3] spawning {len(files)} analyze_one() containers in parallel...")
    handles = [analyze_one.spawn(f, summaries_root,
                                 self_bleu_max_rollouts, self_bleu_max_problems)
               for f in files]
    all_results = []
    for f, h in zip(files, handles):
        try:
            r = h.get()
        except Exception as e:
            r = {"input": f, "label": Path(f).stem, "error": str(e)}
        all_results.append(r)
        print(f"\n[posthoc v3] === {Path(f).name} done ===")
        for k, v in r.items():
            if k in ("input", "label", "out_dir"):
                continue
            print(f"  {k}: {v}")

    master = [
        "# Posthoc analysis — master summary (v3)",
        "",
        f"_Run at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}_",
        "",
        f"Processed {len(all_results)} JSONs from `{input_glob}`.",
        "",
    ]
    for r in all_results:
        master.append(f"## {r['label']}")
        master.append(f"- json_load_s: {r.get('json_load_s', '?')}")
        for k, v in r.items():
            if k in ("input", "label", "out_dir", "json_load_s"):
                continue
            master.append(f"- **{k}**: {v}")
        master.append("")
    (Path(summaries_root) / "MASTER_SUMMARY.md").write_text("\n".join(master))
    artifacts_volume.commit()
    print(f"\n[posthoc v3] done. Summaries at: {summaries_root}/")
