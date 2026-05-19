#!/usr/bin/env python3
"""
Apply pre-registered gate rules on tier-1 pilot_eval splits only.

Usage:
  python pilot/eval/gate.py --artifacts-dir pilot/artifacts --lock pilot/preflight_lock.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pilot.eval.bootstrap import bootstrap_all
from pilot.eval.io import load_predictions, write_metrics
from pilot.eval.metrics import aggregate_metrics
from pilot.eval.splits import load_lock, pilot_gate_splits

DECISIONS = (
    "ESCALATE",
    "PIVOT_WORST_SUBSET",
    "PIVOT_SUBSTRATE_OR_ARCH",
    "STOP_NO_SIGNAL",
    "PENDING",
)

TAIL_KEYS = ("cover_at_tau", "worst_subset_accuracy")


def _eval_run_on_splits(
    artifacts_dir: Path,
    run_id: str,
    lock: dict,
    gate_splits: list[str],
) -> dict[str, Any]:
    pred = artifacts_dir / run_id / "raw_predictions.jsonl"
    if not pred.exists():
        return {"status": "missing", "run_id": run_id}

    mconf = lock["metrics_frozen"]
    kw = dict(
        k=mconf["pass_at_k"],
        tau=mconf["cover_tau"],
        worst_q=mconf["worst_subset_quantile"],
    )
    metric_keys = [
        "pass_at_1",
        f"pass_at_{mconf['pass_at_k']}",
        "cover_at_tau",
        "worst_subset_accuracy",
    ]

    per_split: dict[str, Any] = {}
    pooled_prompts = []
    for split in gate_splits:
        prompts = load_predictions(pred, eval_splits=[split])
        per_split[split] = {
            "aggregate": aggregate_metrics(prompts, **kw),
            "bootstrap_ci": bootstrap_all(
                prompts,
                metric_keys,
                n_samples=mconf["bootstrap_samples"],
                ci_level=mconf["ci_level"],
                seed=mconf["bootstrap_seed"],
                **kw,
            ),
            "n_prompts": len(prompts),
        }
        pooled_prompts.extend(prompts)

    pooled_agg = aggregate_metrics(pooled_prompts, **kw)
    pooled_ci = bootstrap_all(
        pooled_prompts,
        metric_keys,
        n_samples=mconf["bootstrap_samples"],
        ci_level=mconf["ci_level"],
        seed=mconf["bootstrap_seed"],
        **kw,
    )
    return {
        "status": "ok",
        "run_id": run_id,
        "gate_splits": gate_splits,
        "per_split": per_split,
        "aggregate": pooled_agg,
        "bootstrap_ci": pooled_ci,
    }


def _delta_pp(a: float, b: float) -> float:
    return (a - b) * 100.0


def _ci_excludes_zero(ci: dict[str, float]) -> bool:
    return ci["ci_low"] > 0 or ci["ci_high"] < 0


def decide(lock: dict, runs: dict[str, dict]) -> dict[str, Any]:
    gates = lock["gates_frozen"]
    run1 = runs.get("run1_grpo", {})
    run1b = runs.get("run1b_grpo", {})
    run2 = runs.get("run2_inverse_freq", {})
    run3 = runs.get("run3_f_grpo", {})

    rationale: list[str] = []
    checks: dict[str, bool] = {}
    noise_cap = float(gates.get("noise_floor_pp_max", 6.0))

    from pilot.infra.artifacts import resolve_latest_run_dir

    run0_metrics_path = resolve_latest_run_dir("run0_proxy") / "metrics.json"
    if run0_metrics_path.exists():
        r0m = json.loads(run0_metrics_path.read_text())
        minority_rate = r0m.get("minority_correct_prompt_rate", 0.0)
        checks["run0_minority_ok"] = minority_rate >= gates["run0_minority_correct_rate_min"]
        if not checks["run0_minority_ok"]:
            return _pack(
                "PIVOT_WORST_SUBSET",
                rationale
                + [f"Run0 minority-correct rate {minority_rate:.3f} < {gates['run0_minority_correct_rate_min']}"],
                checks,
                runs,
            )
    else:
        return _pack("PENDING", ["Run0 artifacts incomplete"], checks, runs)

    for rid in ("run1_grpo", "run1b_grpo"):
        if runs.get(rid, {}).get("status") != "ok":
            return _pack("PENDING", [f"{rid} artifacts incomplete"], checks, runs)

    m1, m1b = run1["aggregate"], run1b["aggregate"]
    noise = max(abs(_delta_pp(m1[k], m1b[k])) for k in TAIL_KEYS)
    checks["noise_floor_ok"] = noise <= noise_cap
    rationale.append(
        f"Run1 vs Run1b pooled pilot_eval tail max |Δ| = {noise:.2f} pp (cap {noise_cap} pp)"
    )
    if not checks["noise_floor_ok"]:
        return _pack(
            "STOP_NO_SIGNAL",
            rationale + [f"Baseline noise exceeds {noise_cap} pp on tier-1 eval"],
            checks,
            runs,
        )

    if run2.get("status") != "ok":
        return _pack("PENDING", ["Run2 artifacts incomplete"], checks, runs)

    m2 = run2["aggregate"]
    c2 = run2["bootstrap_ci"]
    tail_gain = max(_delta_pp(m2[k], m1[k]) for k in TAIL_KEYS)
    pass1_drop = _delta_pp(m1["pass_at_1"], m2["pass_at_1"])
    tail_ci_ok = any(
        _ci_excludes_zero(c2[k]) and c2[k]["point"] > run1["bootstrap_ci"][k]["point"]
        for k in TAIL_KEYS
    )
    checks["tail_gain_ok"] = tail_gain >= gates["tail_gain_pp_min"] and tail_ci_ok
    checks["pass1_ok"] = pass1_drop <= gates["pass1_regression_pp_max"]
    rationale.append(
        f"Run2 vs Run1 pooled tier-1: tail gain max {tail_gain:.2f} pp; Pass@1 drop {pass1_drop:.2f} pp"
    )

    if not checks["tail_gain_ok"]:
        return _pack(
            "PIVOT_WORST_SUBSET",
            rationale + ["inverse_freq did not move tier-1 tail metrics beyond pre-registered threshold"],
            checks,
            runs,
        )

    if not checks["pass1_ok"]:
        return _pack(
            "PIVOT_SUBSTRATE_OR_ARCH",
            rationale + [f"Pass@1 regression {pass1_drop:.2f}pp exceeds cap"],
            checks,
            runs,
        )

    if run3.get("status") != "ok":
        return _pack(
            "ESCALATE",
            rationale + ["Run2 promising vs GRPO; Run3 not run — run F-GRPO before final escalate claim"],
            checks,
            runs,
        )

    m3 = run3["aggregate"]
    f_gap = max(abs(_delta_pp(m2[k], m3[k])) for k in TAIL_KEYS)
    checks["f_grpo_distinct"] = f_gap > gates["f_grpo_equivalence_pp"]
    rationale.append(f"Run2 vs Run3 tier-1 tail |Δ| = {f_gap:.2f} pp")
    if not checks["f_grpo_distinct"]:
        return _pack(
            "PIVOT_SUBSTRATE_OR_ARCH",
            rationale + ["Run2 ≈ F-GRPO on tier-1 eval"],
            checks,
            runs,
        )

    return _pack(
        "ESCALATE",
        rationale + ["inverse_freq beats GRPO on tier-1, distinct from F-GRPO, Pass@1 within cap"],
        checks,
        runs,
    )


def _pack(decision: str, rationale: list[str], checks: dict, runs: dict) -> dict:
    assert decision in DECISIONS
    return {
        "decision": decision,
        "rationale": rationale,
        "checks": checks,
        "tier": "pilot_eval",
        "runs_evaluated": {k: v.get("status", "missing") for k, v in runs.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", type=Path, default=Path("pilot/artifacts"))
    ap.add_argument("--lock", type=Path, default=Path("pilot/preflight_lock.json"))
    ap.add_argument("--out", type=Path, default=Path("pilot/gate_decision.json"))
    ap.add_argument("--summary-out", type=Path, default=Path("pilot/pilot_metrics_summary.json"))
    args = ap.parse_args()

    lock = load_lock(args.lock)
    gate_splits = pilot_gate_splits(lock)
    run_ids = ["run0_proxy", "run1_grpo", "run1b_grpo", "run2_inverse_freq", "run3_f_grpo"]
    runs = {
        rid: _eval_run_on_splits(args.artifacts_dir, rid, lock, gate_splits)
        for rid in run_ids
        if rid != "run0_proxy"
    }
    runs["run0_proxy"] = {"status": "proxy", "run_id": "run0_proxy"}

    for rid, data in runs.items():
        if data.get("status") == "ok":
            out_dir = args.artifacts_dir / rid
            write_metrics(out_dir / "metrics.json", data)
            write_metrics(out_dir / "metrics_ci.json", data["bootstrap_ci"])

    gate = decide(lock, runs)
    args.out.write_text(json.dumps(gate, indent=2) + "\n")
    summary = {
        "lock_version": lock["version"],
        "model_id": lock["model_id"],
        "gate_splits": gate_splits,
        "runs": runs,
        "gate": gate,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
