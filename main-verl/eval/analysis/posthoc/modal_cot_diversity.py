"""CoT diversity@k: expected distinct correct reasoning clusters in a random k-subset.

For each prompt, sends all 64 rollouts to the LLM judge (same judge used during
training) to get CoT cluster assignments. Filters to correct rollouts, then
computes E[distinct correct CoT clusters in k-subset] exactly via the formula:

    sum_i  1 - C(n - s_i, k) / C(n, k)

where s_i = number of correct rollouts in CoT cluster i, n = total rollouts.

This is the "all subsets of size k" version — no sampling, exact expected value.

Datasets: math500 + beyondaime. Arms: base, grpo, minority, polyepo.

Usage (from repo root):
    MODAL_PROFILE=nbao0 PYTHONPATH=main-verl modal run \
        main-verl/eval/analysis/posthoc/modal_cot_diversity.py

Output: /vol/probes/eval_4b/cot_diversity_results.json
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import modal

ARTIFACTS_VOLUME_NAME = "main-artifacts"
ARTIFACTS_MOUNT = "/vol"
JUDGE_BASE_URL = "https://lee-anastasia-y--v1-chat-completions.modal.run"
JUDGE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEGENERATE_CLUSTER_RAW = 100
DEGENERATE_CLUSTER_ID = -1

# Local path evaluated only during image build (never inside the container).
_LOCAL_PROMPT = "/Users/alee/stanford-cs/224r/CS224R_Final_Project/main-verl/judge/prompts/poly_epo_a1.md"
CONTAINER_PROMPT = "/root/judge/prompts/poly_epo_a1.md"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("ijson", "httpx")
    .add_local_file(_LOCAL_PROMPT, CONTAINER_PROMPT)
)

app = modal.App("cs224r-cot-diversity")
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=False)

ARMS = ["base", "grpo", "minority", "polyepo"]
DATASETS = {
    "math500":    "math500",    # filename fragment
    "beyondaime": "beyondaime",
}
FILES = {
    (arm, ds): f"{arm}_step400_{shard}_{shard}.json" if ds == "math500"
               else f"{arm}_step400_smallood_{ds}.json"
    for arm in ARMS
    for ds, shard in [("math500", "math500"), ("beyondaime", "beyondaime")]
}
K_VALUES = [1, 2, 4, 8, 16, 32, 64]


# ── prompt helpers ────────────────────────────────────────────────────────────

def _extract_problem(rendered_prompt: str) -> str:
    """Pull user message content out of a Qwen3 chat-formatted string."""
    m = re.search(r"<\|im_start\|>user\n(.*?)\n<\|im_end\|>", rendered_prompt, re.DOTALL)
    if m:
        text = m.group(1).strip()
    else:
        text = rendered_prompt.strip()
    # Strip the trailing instruction appended by the eval formatter.
    for suffix in (
        "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        "\nPlease reason step by step, and put your final answer within \\boxed{}",
    ):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def _build_judge_payload(problem: str, rollouts: list[str]) -> dict:
    n = len(rollouts)
    prompt_path = Path(CONTAINER_PROMPT)
    text = prompt_path.read_text()
    system_m = re.search(r"## System\s*\n+(.*?)(?=\n## User|\Z)", text, re.DOTALL)
    user_m   = re.search(r"## User\s*\n+(.*?)\Z",                  text, re.DOTALL)
    system   = system_m.group(1).strip().replace("{n_responses}", str(n))
    responses_block = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rollouts))
    user = (user_m.group(1).strip()
            .replace("{problem}", problem)
            .replace("{responses_block}", responses_block)
            .replace("{n_responses}", str(n)))
    return {
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }


def _parse_judge_response(raw: str, n_rollouts: int) -> dict[int, int]:
    """Return rollout_idx → cluster_id (0-indexed, -1 = degenerate)."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {i: DEGENERATE_CLUSTER_ID for i in range(n_rollouts)}
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return {i: DEGENERATE_CLUSTER_ID for i in range(n_rollouts)}
    assignment = {}
    for i in range(n_rollouts):
        entry = obj.get(str(i + 1), {})
        raw_id = entry.get("cluster_id", DEGENERATE_CLUSTER_RAW) if isinstance(entry, dict) else DEGENERATE_CLUSTER_RAW
        try:
            cid = int(raw_id)
        except (TypeError, ValueError):
            cid = DEGENERATE_CLUSTER_RAW
        assignment[i] = DEGENERATE_CLUSTER_ID if cid == DEGENERATE_CLUSTER_RAW else cid
    return assignment


# ── exact all-subsets formula ─────────────────────────────────────────────────

def expected_distinct_correct_clusters(cluster_sizes: list[int], n: int, k: int) -> float:
    """E[distinct correct CoT clusters in a random k-subset of n rollouts].

    cluster_sizes[i] = number of correct rollouts in CoT cluster i.
    Uses 1 - C(n-s, k) / C(n, k) per cluster, summed.
    """
    denom = math.comb(n, k)
    if denom == 0:
        return 0.0
    return sum(1.0 - math.comb(n - s, k) / denom for s in cluster_sizes)


# ── judge caller ──────────────────────────────────────────────────────────────

MAX_ROLLOUT_CHARS = 1500  # truncate each rollout before sending to judge
MIN_CORRECT_TO_JUDGE = 2  # skip prompts with fewer correct rollouts


async def call_judge(client, problem: str, rollouts: list[str], sem: asyncio.Semaphore) -> dict[int, int]:
    """Judge a list of rollouts; returns local_idx → cluster_id."""
    payload = _build_judge_payload(problem, rollouts)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.post(JUDGE_BASE_URL, json=payload, timeout=180.0)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("choices"):
                    raise ValueError(f"empty choices in response: {data}")
                content = data["choices"][0]["message"]["content"]
                return _parse_judge_response(content, len(rollouts))
            except Exception as e:
                if attempt == 2:
                    print(f"  judge call failed after 3 attempts: {e}")
                    return {i: DEGENERATE_CLUSTER_ID for i in range(len(rollouts))}
                await asyncio.sleep(2 ** attempt)


async def judge_correct_rollouts(prompts: list[dict]) -> list[dict]:
    """For each prompt, judge only correct rollouts (truncated). Returns per-prompt
    dicts with 'correct_indices', 'cluster_by_correct_idx' (local→cluster_id)."""
    import httpx

    sem = asyncio.Semaphore(8)

    async def process_one(pp):
        rewards = pp["rewards"]
        rollouts = pp.get("rollouts", [])
        correct_indices = [i for i, r in enumerate(rewards) if r > 0.5 and i < len(rollouts)]
        if len(correct_indices) < MIN_CORRECT_TO_JUDGE:
            return {"correct_indices": correct_indices, "cluster_by_correct_idx": {}}
        correct_rollouts = [rollouts[i][:MAX_ROLLOUT_CHARS] for i in correct_indices]
        local_assignment = await call_judge(client, pp["problem"], correct_rollouts, sem)
        return {"correct_indices": correct_indices, "cluster_by_correct_idx": local_assignment}

    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[process_one(pp) for pp in prompts])


# ── streaming reader ──────────────────────────────────────────────────────────

def stream_prompts(path: Path) -> list[dict]:
    """Stream per_prompt entries from a large eval JSON, keeping only needed fields."""
    import ijson
    prompts = []
    with path.open("rb") as f:
        for ds_name, ds_obj in ijson.kvitems(f, "datasets"):
            for pp in ds_obj.get("per_prompt", []):
                prompts.append({
                    "problem_id":      pp["problem_id"],
                    "ground_truth":    pp["ground_truth"],
                    "rewards":         pp["rewards"],
                    "rollouts":        pp.get("rollouts", []),
                    "rendered_prompt": pp.get("rendered_prompt", ""),
                    "problem":         _extract_problem(pp.get("rendered_prompt", "")),
                })
    return prompts


# ── main analysis ─────────────────────────────────────────────────────────────

def analyze_arm_dataset(path: Path, arm: str, ds: str) -> dict:
    print(f"  streaming {path.name} ({path.stat().st_size // 1024 // 1024} MB)...")
    prompts = stream_prompts(path)
    print(f"  {len(prompts)} prompts loaded, calling judge on correct rollouts only...")

    judge_results = asyncio.run(judge_correct_rollouts(prompts))

    per_prompt_results = []
    for pp, jr in zip(prompts, judge_results):
        n = len(pp["rewards"])
        correct_indices = jr["correct_indices"]
        local_assignment = jr["cluster_by_correct_idx"]  # local_idx → cluster_id

        # Map local cluster ids to counts of correct rollouts per cluster.
        cluster_counts: dict[int, int] = defaultdict(int)
        for local_idx, cid in local_assignment.items():
            if cid != DEGENERATE_CLUSTER_ID:
                cluster_counts[cid] += 1

        sizes = list(cluster_counts.values())
        n_correct = len(correct_indices)
        n_clusters = len(sizes)

        result = {
            "problem_id": pp["problem_id"],
            "n_correct": n_correct,
            "n_cot_clusters_correct": n_clusters,
            "cot_diversity_at_k": {},
        }
        for k in K_VALUES:
            if k <= n:
                result["cot_diversity_at_k"][f"@{k}"] = expected_distinct_correct_clusters(sizes, n, k)
        per_prompt_results.append(result)

    # Aggregate: mean over all prompts (prompts with 0 correct contribute 0).
    n_prompts = len(per_prompt_results)
    n_with_correct = sum(1 for p in per_prompt_results if p["n_correct"] > 0)
    agg = {}
    for k in K_VALUES:
        key = f"@{k}"
        vals = [p["cot_diversity_at_k"].get(key, 0.0) for p in per_prompt_results]
        agg[key] = sum(vals) / n_prompts if n_prompts else 0.0

    print(f"  done: {n_with_correct}/{n_prompts} prompts have ≥1 correct rollout")
    print(f"  cot_diversity@16={agg.get('@16', 0):.4f}")

    return {
        "arm": arm,
        "dataset": ds,
        "n_prompts": n_prompts,
        "n_prompts_with_correct": n_with_correct,
        "mean_cot_diversity_at_k": agg,
        "per_prompt": per_prompt_results,
    }


@app.function(
    image=image,
    volumes={ARTIFACTS_MOUNT: artifacts_volume},
    timeout=21600,  # 6 hours — 8 arms × ~40min each
    memory=8192,
)
def run_cot_diversity() -> dict:
    sys.path.insert(0, "/root/main-verl")
    eval_dir = Path(ARTIFACTS_MOUNT) / "probes" / "eval_4b"
    out_dir = eval_dir
    all_results = {}

    # Resume: load any per-arm results already saved from a prior run.
    for arm in ARMS:
        for ds in DATASETS:
            checkpoint = out_dir / f"cot_diversity_{arm}_{ds}.json"
            if checkpoint.exists():
                print(f"RESUME: loading {checkpoint.name}")
                all_results[f"{arm}_{ds}"] = json.loads(checkpoint.read_text())

    for arm in ARMS:
        for ds in DATASETS:
            key = f"{arm}_{ds}"
            if key in all_results:
                print(f"SKIP (already done): {arm} | {ds}")
                continue
            fname = FILES[(arm, ds)]
            path = eval_dir / fname
            if not path.exists():
                print(f"SKIP (not found): {fname}")
                continue
            print(f"\n=== {arm} | {ds} ===")
            try:
                result = analyze_arm_dataset(path, arm, ds)
                all_results[key] = result
                # Save per-arm checkpoint immediately.
                checkpoint = out_dir / f"cot_diversity_{arm}_{ds}.json"
                checkpoint.write_text(json.dumps(
                    {kk: vv for kk, vv in result.items() if kk != "per_prompt"},
                    indent=2, default=float,
                ))
                artifacts_volume.commit()
                print(f"  saved checkpoint: {checkpoint.name}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback; traceback.print_exc()
                all_results[key] = {"error": str(e)}

    out = out_dir / "cot_diversity_results.json"
    out.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "per_prompt"} for k, v in all_results.items()},
        indent=2, default=float,
    ))
    artifacts_volume.commit()
    print(f"\nwrote summary to {out}")
    return all_results


@app.local_entrypoint()
def main():
    # Use spawn so the job survives local client disconnect / laptop sleep.
    call = run_cot_diversity.spawn()
    print(f"Spawned run_cot_diversity, call_id={call.object_id}")
    print("Results will be written to /vol/probes/eval_4b/cot_diversity_results.json")
    print("Check progress: modal app list --profile nbao0")
