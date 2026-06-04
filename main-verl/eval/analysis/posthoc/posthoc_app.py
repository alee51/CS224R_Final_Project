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
    # container doesn't need to load any of them. The analyzers live at
    # /root/main-verl/eval/analysis/posthoc/ on Modal (add_local_dir) and at
    # __file__'s parent locally.
    for _cand in (
        Path("/root/main-verl/eval/analysis/posthoc"),
        Path(__file__).resolve().parent,
    ):
        if (_cand / "auc_at_k.py").exists():
            sys.path.insert(0, str(_cand))
            break
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


def _sample_rollouts_md(label: str, data: dict, n_per_prompt: int = 3,
                        n_prompts: int = 5, max_chars: int = 2500) -> str:
    """For diagnosing weight-load / degeneration issues — dump a handful of
    actual rollout texts per prompt, mixed across (correct, empty-pred, wrong)."""
    out = [f"# Rollout samples — {label}", "",
           f"Up to {n_per_prompt} rollouts/prompt × {n_prompts} prompts.",
           "Truncated to first/last {max_chars//2}c each so loops are visible.",
           ""]
    for ds_name, ds in data.get("datasets", {}).items():
        out.append(f"## {ds_name}")
        out.append("")
        prompts = ds["per_prompt"][:n_prompts]
        for pi, p in enumerate(prompts):
            out.append(f"### prompt {pi}: id=`{p['problem_id']}`  gt=`{p['ground_truth']}`  n_correct={p['n_correct']}")
            out.append("")
            # bucket rollouts: correct / empty-pred / wrong
            buckets = {"correct": [], "empty_pred": [], "wrong": []}
            for ri, (text, pred, rwd) in enumerate(zip(p["rollouts"], p["preds"], p["rewards"])):
                if not text: continue
                bucket = "correct" if rwd > 0.5 else ("empty_pred" if not pred else "wrong")
                buckets[bucket].append((ri, text, pred, rwd))
            for bucket, items in buckets.items():
                take = items[:n_per_prompt]
                for ri, text, pred, rwd in take:
                    out.append(f"#### [{bucket}] rollout {ri}  pred=`{pred!r}`  reward={rwd}  len={len(text)}c")
                    out.append("```")
                    if len(text) > max_chars:
                        out.append(text[:max_chars//2])
                        out.append(f"...[{len(text) - max_chars} chars elided]...")
                        out.append(text[-max_chars//2:])
                    else:
                        out.append(text)
                    out.append("```")
                    out.append("")
        out.append("")
    return "\n".join(out)


def _repetition_metrics(text: str, ngram: int = 10) -> dict:
    """Quantify how much a rollout loops. Returns:
        max_ngram_repeat: how many times the most-repeated n-gram appears
        repeat_ratio: fraction of n-gram occurrences that are NON-unique
                       (1 - distinct_n) — higher = more repetition
        max_run_repeat: longest run of consecutive identical n-grams
        tail_repeats_head: does the last 500 chars contain text from the
                            first 2000 chars? signature of late-rollout loop
    """
    import re
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    if len(toks) < ngram:
        return {"max_ngram_repeat": 0, "repeat_ratio": 0.0,
                "max_run_repeat": 0, "tail_repeats_head": False, "n_tokens": len(toks)}
    from collections import Counter
    ngrams = [tuple(toks[i:i+ngram]) for i in range(len(toks)-ngram+1)]
    c = Counter(ngrams)
    most_common_count = c.most_common(1)[0][1]
    distinct = len(c)
    total = len(ngrams)
    repeat_ratio = 1 - distinct / total if total else 0.0
    # longest run of consecutive identical ngrams
    max_run = 1
    cur_run = 1
    for i in range(1, len(ngrams)):
        if ngrams[i] == ngrams[i-1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    # Does the last 80-char window appear earlier in the text? (loop tail)
    tail = text[-300:-100] if len(text) > 500 else ""
    head = text[:max(0, len(text)-500)]
    tail_repeats = bool(tail) and tail in head
    return {
        "max_ngram_repeat": most_common_count,
        "repeat_ratio": round(repeat_ratio, 3),
        "max_run_repeat": max_run,
        "tail_repeats_head": tail_repeats,
        "n_tokens": len(toks),
    }


@app.function(
    image=image,
    timeout=2 * 3600,
    memory=65 * 1024,
    cpu=4.0,
    volumes={ARTIFACTS_MOUNT: artifacts_volume},
)
def repetition_diagnostic(
    input_glob: str = "/vol/probes/eval_4b/*_step400_smallood_aime25.json",
    summaries_root: str = "/vol/probes/eval_4b/_summaries",
    skip_patterns: str = "schemaprobe,aime25-aime26",
    ngram: int = 10,
) -> None:
    """Test H1 (reflection-loop degeneration). For each (arm, dataset), bucket
    rollouts by reward (correct / empty_pred / wrong), then compute aggregate
    repetition stats per bucket. If trained-arm empty_pred has dramatically
    higher repetition than base's empty_pred → H1 supported."""
    import glob as _glob, json
    Path(summaries_root).mkdir(parents=True, exist_ok=True)
    skips = [s for s in skip_patterns.split(",") if s]
    files = sorted(_glob.glob(input_glob))
    files = [f for f in files if not any(s in Path(f).name for s in skips)]
    print(f"[rep_diag] {len(files)} files")
    out_rows = []
    for f in files:
        label = Path(f).stem
        arm = label.split("_step400")[0]
        print(f"  loading {f}...")
        with open(f) as fh:
            data = json.load(fh)
        for ds_name, ds in data["datasets"].items():
            bucket_stats = {"correct": [], "empty_pred": [], "wrong": []}
            for p in ds["per_prompt"]:
                for text, pred, rwd in zip(p["rollouts"], p["preds"], p["rewards"]):
                    if not text:
                        continue
                    bucket = "correct" if rwd > 0.5 else ("empty_pred" if not pred else "wrong")
                    bucket_stats[bucket].append(_repetition_metrics(text, ngram=ngram))
            for bucket, items in bucket_stats.items():
                if not items:
                    continue
                n = len(items)
                avg_repeat_ratio = sum(x["repeat_ratio"] for x in items) / n
                avg_max_ngram = sum(x["max_ngram_repeat"] for x in items) / n
                avg_max_run = sum(x["max_run_repeat"] for x in items) / n
                pct_tail_loop = sum(1 for x in items if x["tail_repeats_head"]) / n
                avg_tokens = sum(x["n_tokens"] for x in items) / n
                # "looping rollout" heuristic: max_ngram_repeat >= 5 OR tail_repeats_head
                pct_looping = sum(1 for x in items
                                   if x["max_ngram_repeat"] >= 5 or x["tail_repeats_head"]) / n
                out_rows.append({
                    "arm": arm, "ds": ds_name, "bucket": bucket, "n": n,
                    "avg_tokens": round(avg_tokens, 1),
                    "avg_repeat_ratio": round(avg_repeat_ratio, 3),
                    "avg_max_ngram_repeat": round(avg_max_ngram, 1),
                    "avg_max_run": round(avg_max_run, 1),
                    "pct_tail_repeats_head": round(pct_tail_loop, 3),
                    "pct_looping_heuristic": round(pct_looping, 3),
                })
    # Write a comparison markdown
    md = ["# Repetition diagnostic — H1 (reflection-loop degeneration) check", "",
          f"_n-gram size = {ngram}; metrics computed per rollout, averaged per bucket._",
          "",
          "**Reading**: if trained-arm `empty_pred` row has ≫ repetition vs base `empty_pred`,",
          "the looping hypothesis is supported. If similar, H1 is wrong.",
          "",
          "| arm | dataset | bucket | n | avg_tokens | avg_repeat_ratio | avg_max_ngram_repeat | avg_max_run | %tail_loop | %looping |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out_rows:
        md.append(f"| {r['arm']} | {r['ds']} | {r['bucket']} | {r['n']} | "
                  f"{r['avg_tokens']} | {r['avg_repeat_ratio']} | "
                  f"{r['avg_max_ngram_repeat']} | {r['avg_max_run']} | "
                  f"{r['pct_tail_repeats_head']} | {r['pct_looping_heuristic']} |")
    out_path = Path(summaries_root) / "repetition_diagnostic.md"
    out_path.write_text("\n".join(md) + "\n")
    artifacts_volume.commit()
    print(f"[rep_diag] wrote {out_path}")
    for r in out_rows:
        print(f"  {r}")


@app.function(
    image=image,
    timeout=2 * 3600,
    memory=65 * 1024,
    cpu=4.0,
    volumes={ARTIFACTS_MOUNT: artifacts_volume},
)
def sample_rollouts(input_glob: str = "/vol/probes/eval_4b/*_step400_*.json",
                    summaries_root: str = "/vol/probes/eval_4b/_summaries",
                    skip_patterns: str = "schemaprobe,aime25-aime26",
                    n_per_prompt: int = 3,
                    n_prompts: int = 5,
                    max_chars: int = 3000) -> None:
    """Dump human-readable rollout samples for each matched JSON. Tiny output
    files (per-cell markdown); never pulls the full JSON."""
    import glob as _glob, json
    Path(summaries_root).mkdir(parents=True, exist_ok=True)
    skips = [s for s in skip_patterns.split(",") if s]
    files = sorted(_glob.glob(input_glob))
    files = [f for f in files if not any(s in Path(f).name for s in skips)]
    print(f"[samples] {len(files)} files")
    for f in files:
        label = Path(f).stem
        out_dir = Path(summaries_root) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  loading {f} ...")
        with open(f) as fh:
            data = json.load(fh)
        md = _sample_rollouts_md(label, data, n_per_prompt=n_per_prompt,
                                 n_prompts=n_prompts, max_chars=max_chars)
        (out_dir / "rollout_samples.md").write_text(md)
        print(f"  wrote {out_dir / 'rollout_samples.md'} ({len(md)/1000:.1f} KB)")
    artifacts_volume.commit()


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
