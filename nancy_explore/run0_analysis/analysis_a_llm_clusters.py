#!/usr/bin/env python3
"""Analysis A: LLM reasoning-cluster ground truth for Run 0.

Clusters 8 rollouts per prompt via a cheap-tier LLM judge (default: Google
``gemini-3.1-flash``). Caches per-prompt JSON under the Run 0 artifact tree;
writes ``llm_clusters_summary.parquet`` and ``analysis_a_summary.md``.

Usage (from repo root):
    python nancy_explore/run0_analysis/analysis_a_llm_clusters.py --dry-run
    python nancy_explore/run0_analysis/analysis_a_llm_clusters.py --pilot
    python nancy_explore/run0_analysis/analysis_a_llm_clusters.py --tier cheap
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pilot.train.run_proxy import has_minority_correct_cluster  # noqa: E402

RUN0_DIR = Path(__file__).resolve().parent
DATA_DIR = RUN0_DIR / "data"
REParsed_PATH = DATA_DIR / "predictions_reparsed.jsonl"
PROMPTS_PATH = DATA_DIR / "prompt_inputs.jsonl"
ENV_PATH = RUN0_DIR / ".env"

CONFIG_YAML = RUN0_DIR / "config" / "llm_judge_models.yaml"
PROMPT_MD = RUN0_DIR / "config" / "analysis_a_prompt.md"

ARTIFACT_RUN = REPO / "pilot/artifacts/run0_proxy/20260519T190202Z"
CACHE_DIR = ARTIFACT_RUN / "llm_clusters"

SUMMARY_PARQUET = RUN0_DIR / "llm_clusters_summary.parquet"
SUMMARY_MD = RUN0_DIR / "analysis_a_summary.md"

N_BOOT = 1000
BOOT_SEED = 42
PILOT_N = 5
# Free tier gemini-3.1-flash-lite: 15 RPM — global throttle below
MAX_WORKERS = 2
MAX_RETRIES = 8
RETRY_BASE_SEC = 2.0
RETRY_429_SEC = 65.0
# 15 req/min → 4.0s minimum; 4.1s keeps a small margin under the cap
MIN_API_INTERVAL_SEC = 4.1

N_RESPONSES = 8
PROMPT_FORMAT = "poly_epo_paper_a1"
POLY_EPO_DEGENERATE_CLUSTER = 100

_api_rate_lock = threading.Lock()
_last_api_call = 0.0


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _load_yaml_config() -> dict:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("pyyaml required: pip install pyyaml") from e
    with CONFIG_YAML.open() as f:
        return yaml.safe_load(f)


def _resolve_model(provider: str, tier: str, cfg: dict) -> str:
    providers = cfg.get("providers") or {}
    if provider not in providers:
        raise SystemExit(f"unknown provider {provider!r}; choose from {list(providers)}")
    tier_map = providers[provider]
    if tier not in tier_map:
        raise SystemExit(f"unknown tier {tier!r} for {provider}; choose from {list(tier_map)}")
    return tier_map[tier]


def _google_api_keys() -> list[tuple[str, str]]:
    """(env_var_name, key) in priority order; dedupe by key value."""
    order = (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY_2",
        "GEMINI_API_KEY_2",
    )
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name in order:
        val = (os.environ.get(name) or "").strip()
        if val and val not in seen:
            seen.add(val)
            out.append((name, val))
    return out


@dataclass
class GoogleKeyPool:
    """Thread-safe pool; rotates to GOOGLE_API_KEY_2 on daily quota errors."""

    keys: list[str]
    labels: list[str]
    _idx: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_env(cls, *, start_key: int = 1) -> GoogleKeyPool:
        pairs = _google_api_keys()
        pool = cls(
            keys=[k for _, k in pairs],
            labels=[n for n, _ in pairs],
        )
        if start_key > 1:
            want = {2: ("GOOGLE_API_KEY_2", "GEMINI_API_KEY_2")}
            labels = want.get(start_key, ())
            for i, label in enumerate(pool.labels):
                if label in labels:
                    pool._idx = i
                    return pool
            raise SystemExit(
                f"--api-key {start_key} requested but none of {labels} found in .env"
            )
        return pool

    def current(self) -> str | None:
        if not self.keys:
            return None
        with self._lock:
            return self.keys[self._idx]

    def current_label(self) -> str | None:
        if not self.labels:
            return None
        with self._lock:
            return self.labels[self._idx]

    def rotate_on_daily_limit(self) -> bool:
        with self._lock:
            if self._idx + 1 >= len(self.keys):
                return False
            self._idx += 1
            print(
                f"Daily quota hit on {self.labels[self._idx - 1]}; "
                f"switching to {self.labels[self._idx]}",
                file=sys.stderr,
            )
            return True


def _is_daily_quota_error(err: str) -> bool:
    """Distinguish daily cap from per-minute 429 (retry-after ~1s)."""
    e = err.lower()
    if "perday" in e or "per_day" in e or "requestsperday" in e:
        return True
    if "perminute" in e or "per_minute" in e:
        return False
    if "please retry in" in e:
        return False
    if "quota exceeded" in e and ("day" in e or "daily" in e):
        return True
    return False


def _bootstrap_ci(
    prompt_flags: list[bool],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    if not prompt_flags:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(prompt_flags)
    rates: list[float] = []
    for _ in range(n_boot):
        sample = [prompt_flags[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(sample) / n)
    rates.sort()
    lo = rates[int(0.025 * n_boot)]
    hi = rates[int(0.975 * n_boot)]
    return (sum(prompt_flags) / n, lo, hi)


def _load_prompt_templates() -> tuple[str, str]:
    """Return (system_template, user_template) from Poly-EPO prompt markdown."""
    if not PROMPT_MD.is_file():
        raise SystemExit(f"missing prompt template: {PROMPT_MD}")

    text = PROMPT_MD.read_text()
    system_m = re.search(
        r"## System\s*\n+(.*?)(?=\n## User|\Z)",
        text,
        re.DOTALL,
    )
    user_m = re.search(r"## User\s*\n+(.*?)\Z", text, re.DOTALL)
    system = (system_m.group(1).strip() if system_m else "").strip()
    user = (user_m.group(1).strip() if user_m else "").strip()
    if not system or not user:
        raise SystemExit(f"invalid prompt template sections in {PROMPT_MD}")
    return system, user


def _build_responses_block(rollouts: list[dict]) -> str:
    """Poly-EPO instance format: numbered responses 1..N (1-indexed)."""
    blocks: list[str] = []
    for idx, r in enumerate(rollouts):
        n = idx + 1
        completion = r.get("completion", "")
        blocks.append(f"{n}. {completion}")
    return "\n".join(blocks)


def build_messages(problem: str, rollouts: list[dict]) -> tuple[str, str]:
    system_tpl, user_tpl = _load_prompt_templates()
    system = system_tpl.replace("{n_responses}", str(N_RESPONSES))
    user = (
        user_tpl.replace("{problem}", problem)
        .replace("{responses_block}", _build_responses_block(rollouts))
    )
    return system, user


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _normalize_cluster_id(cid: int) -> int:
    """Paper uses 100 for degenerate; downstream metrics use -1."""
    return -1 if cid == POLY_EPO_DEGENERATE_CLUSTER else cid


def _normalize_cluster_assignment(raw: Any) -> dict[int, int]:
    if not isinstance(raw, dict):
        raise ValueError("cluster_assignment must be an object")
    out: dict[int, int] = {}
    for k, v in raw.items():
        idx = int(k)
        if idx < 0 or idx > 7:
            raise ValueError(f"rollout index out of range: {idx}")
        out[idx] = _normalize_cluster_id(int(v))
    if set(out.keys()) != set(range(N_RESPONSES)):
        raise ValueError(f"cluster_assignment must cover 0-7, got {sorted(out)}")
    return out


def _normalize_clusters(raw: Any, assignment: dict[int, int]) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError("clusters must be an array")
    clusters: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each cluster must be an object")
        cid = int(item["cluster_id"])
        members_raw = item.get("member_rollouts") or item.get("members") or item.get(
            "rollouts"
        )
        if members_raw is not None:
            members = [int(x) for x in members_raw]
        else:
            members = [i for i, a in assignment.items() if a == cid]
        sig = str(
            item.get("reasoning_signature")
            or item.get("signature")
            or item.get("description")
            or ""
        )
        clusters.append(
            {
                "cluster_id": cid,
                "member_rollouts": members,
                "reasoning_signature": sig,
            }
        )
    return clusters


def _assignment_from_poly_epo_payload(payload: dict) -> tuple[dict[int, int], list[dict]]:
    """Paper §A.1 format: keys \"1\"..\"N\" with chain_of_thought + cluster_id."""
    assignment: dict[int, int] = {}
    cot_by_idx: dict[int, str] = {}
    for key, val in payload.items():
        if not str(key).isdigit():
            continue
        rollout_1idx = int(key)
        if rollout_1idx < 1 or rollout_1idx > N_RESPONSES:
            raise ValueError(f"rollout key out of range: {key}")
        idx = rollout_1idx - 1
        if not isinstance(val, dict):
            raise ValueError(f"response {key} must be an object")
        cid = _normalize_cluster_id(int(val["cluster_id"]))
        assignment[idx] = cid
        cot_by_idx[idx] = str(val.get("chain_of_thought", ""))
    if set(assignment.keys()) != set(range(N_RESPONSES)):
        raise ValueError(
            f"Poly-EPO JSON must have keys 1-{N_RESPONSES}, got {sorted(assignment)}"
        )
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, cid in assignment.items():
        by_cluster[cid].append(idx)
    clusters: list[dict] = []
    for cid, members in sorted(by_cluster.items()):
        macro_micro = "; ".join(
            cot_by_idx[m][:120] for m in sorted(members)[:2]
        )
        clusters.append(
            {
                "cluster_id": cid,
                "member_rollouts": sorted(members),
                "reasoning_signature": macro_micro or f"cluster_{cid}",
            }
        )
    return assignment, clusters


def _assignment_from_payload(payload: dict) -> tuple[dict[int, int], list[dict] | None]:
    digit_keys = [k for k in payload if str(k).isdigit()]
    if len(digit_keys) >= N_RESPONSES:
        return _assignment_from_poly_epo_payload(payload)
    if "cluster_assignment" in payload and payload["cluster_assignment"] is not None:
        assignment = _normalize_cluster_assignment(payload["cluster_assignment"])
        return assignment, None
    raw_assign = (
        payload.get("assignments")
        or payload.get("rollout_assignments")
        or payload.get("rollout_assignment")
    )
    if isinstance(raw_assign, list):
        out: dict[int, int] = {}
        for item in raw_assign:
            if not isinstance(item, dict):
                raise ValueError("each assignment entry must be an object")
            idx_raw = (
                item.get("rollout_idx")
                if item.get("rollout_idx") is not None
                else item.get("rollout_index")
                if item.get("rollout_index") is not None
                else item.get("rollout_id")
                if item.get("rollout_id") is not None
                else item.get("idx")
            )
            if idx_raw is None:
                raise ValueError(f"assignment entry missing rollout index: {item}")
            out[int(idx_raw)] = _normalize_cluster_id(int(item["cluster_id"]))
        return _normalize_cluster_assignment(out), None
    raise ValueError("missing cluster_assignment or assignments")


def parse_llm_json(text: str) -> tuple[list[dict], dict[int, int]]:
    payload = json.loads(_strip_json_fences(text))
    assignment, clusters_poly = _assignment_from_payload(payload)
    if clusters_poly is not None:
        return clusters_poly, assignment
    clusters = _normalize_clusters(payload.get("clusters"), assignment)
    if set(assignment.keys()) != set(range(N_RESPONSES)):
        raise ValueError(f"assignment must cover 0-7, got {sorted(assignment)}")
    for idx, cid in assignment.items():
        if cid == -1:
            continue
        member_cids = [c["cluster_id"] for c in clusters if idx in c["member_rollouts"]]
        if member_cids and member_cids[0] != cid:
            raise ValueError(f"assignment[{idx}]={cid} inconsistent with clusters")
    return clusters, assignment


def call_gemini(
    model: str,
    system: str,
    user: str,
    api_key: str,
) -> str:
    """Call Google Gemini via ``google-genai`` (preferred) or ``google.generativeai``."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        text = response.text
        if not text:
            raise RuntimeError("empty response from google-genai")
        return text
    except ImportError:
        pass

    try:
        import google.generativeai as genai_legacy

        genai_legacy.configure(api_key=api_key)
        gmodel = genai_legacy.GenerativeModel(
            model_name=model,
            system_instruction=system,
            generation_config=genai_legacy.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        response = gmodel.generate_content(user)
        text = response.text
        if not text:
            raise RuntimeError("empty response from google.generativeai")
        return text
    except ImportError as e:
        raise SystemExit(
            "Install google-genai (preferred): pip install google-genai\n"
            "  or: pip install google-generativeai"
        ) from e


@dataclass
class PromptJob:
    prompt_id: str
    problem: str
    gold_answer: str
    rollouts: list[dict]


def _cache_path(prompt_id: str) -> Path:
    return CACHE_DIR / f"{prompt_id}.json"


def _cache_usable(cache: dict | None) -> bool:
    """Only accept successful caches from the current Poly-EPO §A.1 protocol."""
    if not cache:
        return False
    return bool(
        cache.get("parse_ok")
        and cache.get("cluster_assignment")
        and cache.get("prompt_format") == PROMPT_FORMAT
    )


def _throttle_api() -> None:
    global _last_api_call
    with _api_rate_lock:
        now = time.monotonic()
        wait = MIN_API_INTERVAL_SEC - (now - _last_api_call)
        if wait > 0:
            time.sleep(wait)
        _last_api_call = time.monotonic()


def _retry_sleep(attempt: int, err: str) -> float:
    if _is_daily_quota_error(err):
        return RETRY_BASE_SEC
    if "429" in err or "RESOURCE_EXHAUSTED" in err:
        return RETRY_429_SEC
    return RETRY_BASE_SEC * (2**attempt)


def purge_stale_caches(*, dry_run: bool = False) -> tuple[int, int]:
    """Remove cache files from old prompts or failed runs."""
    if not CACHE_DIR.is_dir():
        return 0, 0
    removed = kept = 0
    for path in CACHE_DIR.glob("*.json"):
        try:
            cache = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            if not dry_run:
                path.unlink(missing_ok=True)
            removed += 1
            continue
        if _cache_usable(cache):
            kept += 1
            continue
        if not dry_run:
            path.unlink(missing_ok=True)
        removed += 1
    return removed, kept


def _read_cache(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def process_prompt(
    job: PromptJob,
    *,
    provider: str,
    tier: str,
    model: str,
    dry_run: bool,
    force: bool,
    key_pool: GoogleKeyPool,
) -> dict:
    cache_path = _cache_path(job.prompt_id)
    if not force:
        existing = _read_cache(cache_path)
        if existing:
            if _cache_usable(existing):
                return existing
            if dry_run and existing.get("dry_run"):
                return existing

    system, user = build_messages(job.problem, job.rollouts)
    base: dict[str, Any] = {
        "prompt_id": job.prompt_id,
        "provider": provider,
        "tier": tier,
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completion_truncation": False,
        "prompt_format": PROMPT_FORMAT,
        "system_prompt": system,
        "user_prompt": user,
        "parse_ok": False,
        "error": None,
        "raw_response": None,
        "clusters": None,
        "cluster_assignment": None,
    }

    if dry_run:
        base["dry_run"] = True
        _write_cache(cache_path, base)
        return base

    api_key = key_pool.current()
    if not api_key:
        base["error"] = "missing GOOGLE_API_KEY / GOOGLE_API_KEY_2 (or GEMINI_* aliases)"
        _write_cache(cache_path, base)
        return base

    last_err: str | None = None
    raw: str | None = None
    for attempt in range(MAX_RETRIES):
        api_key = key_pool.current()
        if not api_key:
            base["error"] = "no API keys available"
            _write_cache(cache_path, base)
            return base
        try:
            _throttle_api()
            raw = call_gemini(model, system, user, api_key)
            clusters, assignment = parse_llm_json(raw)
            base.update(
                {
                    "raw_response": raw,
                    "clusters": clusters,
                    "cluster_assignment": {str(k): v for k, v in assignment.items()},
                    "parse_ok": True,
                    "error": None,
                    "api_key_label": key_pool.current_label(),
                }
            )
            _write_cache(cache_path, base)
            return base
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            base["raw_response"] = raw
            base["error"] = last_err
            if _is_daily_quota_error(last_err) and key_pool.rotate_on_daily_limit():
                time.sleep(2.0)
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(_retry_sleep(attempt, last_err))
            else:
                _write_cache(cache_path, base)
    return base


def _write_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise SystemExit(
            "pyarrow required for parquet output: pip install pyarrow"
        ) from e
    table = pa.Table.from_pylist(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _build_summary_rows(
    jobs: list[PromptJob],
    caches: dict[str, dict],
) -> list[dict]:
    rows: list[dict] = []
    for job in jobs:
        cache = caches.get(job.prompt_id, {})
        parse_ok = bool(cache.get("parse_ok"))
        assignment_raw = cache.get("cluster_assignment") or {}
        assignment: dict[int, int] = {}
        if parse_ok and assignment_raw:
            assignment = {int(k): int(v) for k, v in assignment_raw.items()}
        for idx, r in enumerate(job.rollouts):
            rows.append(
                {
                    "prompt_id": job.prompt_id,
                    "rollout_idx": idx,
                    "llm_cluster_id": assignment.get(idx) if parse_ok else None,
                    "is_correct_v2": bool(r.get("is_correct_v2")),
                    "parsed_answer_v2": r.get("parsed_answer_v2", ""),
                    "canonical_v2": r.get("canonical_v2", ""),
                    "parse_ok": parse_ok,
                    "provider": cache.get("provider"),
                    "tier": cache.get("tier"),
                    "model": cache.get("model"),
                }
            )
    return rows


def _minority_metrics(jobs: list[PromptJob], caches: dict[str, dict]) -> tuple[list[bool], int, int]:
    """Prompt-level flags for minority-correct under LLM clusters (parsed prompts only)."""
    flags: list[bool] = []
    n_eligible = 0
    n_parsed = 0
    for job in jobs:
        cache = caches.get(job.prompt_id, {})
        if not cache.get("parse_ok"):
            continue
        n_parsed += 1
        assign = cache.get("cluster_assignment") or {}
        cluster_ids = [int(assign[str(i)]) for i in range(8)]
        correct = [bool(r.get("is_correct_v2")) for r in job.rollouts]
        if not any(correct):
            continue
        n_eligible += 1
        flags.append(has_minority_correct_cluster(correct, cluster_ids))
    return flags, n_eligible, n_parsed


def _build_summary_md(
    *,
    provider: str,
    tier: str,
    model: str,
    n_prompts: int,
    n_parsed: int,
    n_eligible: int,
    rate: float,
    lo: float,
    hi: float,
    degenerate_frac: float,
    dry_run: bool,
) -> str:
    lines = [
        "# Analysis A — LLM reasoning clusters (summary)\n",
        f"**Generated:** {datetime.now(timezone.utc).date().isoformat()}  \n",
        f"**Provider / tier / model:** `{provider}` / `{tier}` / `{model}`  \n",
        f"**Cache dir:** `{CACHE_DIR.relative_to(REPO)}`  \n",
    ]
    if dry_run:
        lines.append("\n> **Dry-run:** prompts built only; no API calls.\n")
    lines.extend(
        [
            "\n## Headline: minority_correct_prompt_rate_llm\n",
            "Among prompts with ≥1 correct rollout (v2 parser), fraction where correct "
            "rollouts span ≥2 LLM clusters and at least one correct cluster is not the "
            "largest (same definition as `has_minority_correct_cluster` in "
            "`pilot/train/run_proxy.py`).\n",
            "\n| Metric | Value |\n|---|---:|\n",
            f"| Prompts attempted | {n_prompts} |\n",
            f"| Prompts with successful parse | {n_parsed} |\n",
            f"| Prompts with ≥1 correct (eligible) | {n_eligible} |\n",
        ]
    )
    if n_eligible:
        lines.append(
            f"| **minority_correct_prompt_rate_llm** | **{100*rate:.2f}%** "
            f"(95% CI [{100*lo:.2f}%, {100*hi:.2f}%]) |\n"
        )
    else:
        lines.append("| **minority_correct_prompt_rate_llm** | N/A (no eligible prompts) |\n")
    lines.append(
        f"\n| Rollouts in degenerate cluster (`cluster_id == -1`) | "
        f"{100*degenerate_frac:.2f}% of parsed assignments |\n"
    )
    lines.append(
        "\n## Next steps (manual)\n"
        "- Hand-check 10 prompts → `llm_clusters_handcheck.md` (see design doc §A.7)\n"
        "- Optional cheap↔moderate ARI on 50 prompts → `llm_judge_cross_tier.md`\n"
    )
    return "".join(lines)


def _degenerate_rate(caches: dict[str, dict]) -> float:
    n = 0
    deg = 0
    for cache in caches.values():
        if not cache.get("parse_ok"):
            continue
        assign = cache.get("cluster_assignment") or {}
        for i in range(8):
            n += 1
            if int(assign.get(str(i), assign.get(i, 0))) == -1:
                deg += 1
    return deg / n if n else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 0 Analysis A: LLM reasoning clusters")
    parser.add_argument("--pilot", action="store_true", help=f"First {PILOT_N} prompts only")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of prompts")
    parser.add_argument("--tier", choices=["cheap", "moderate", "expensive"], default="cheap")
    parser.add_argument("--provider", default=None, help="Override yaml default_provider")
    parser.add_argument("--dry-run", action="store_true", help="Build/cache prompts only")
    parser.add_argument("--force", action="store_true", help="Ignore successful cache hits")
    parser.add_argument(
        "--purge-stale",
        action="store_true",
        help="Delete cache files that are not poly_epo_paper_a1 parse_ok",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Concurrent API calls (global throttle still enforces RPM)",
    )
    parser.add_argument(
        "--api-key",
        type=int,
        choices=[1, 2],
        default=1,
        help="Start with GOOGLE_API_KEY (1) or GOOGLE_API_KEY_2 (2)",
    )
    args = parser.parse_args()

    _load_env(ENV_PATH)
    cfg = _load_yaml_config()
    provider = args.provider or cfg.get("default_provider", "google")
    model = os.environ.get("GEMINI_MODEL") or _resolve_model(provider, args.tier, cfg)

    if not REParsed_PATH.is_file():
        raise SystemExit(f"missing {REParsed_PATH}; run reparse_rescore.py first")
    if not PROMPTS_PATH.is_file():
        raise SystemExit(f"missing {PROMPTS_PATH}")

    prompts = {r["prompt_id"]: r for r in _load_jsonl(PROMPTS_PATH)}
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for row in _load_jsonl(REParsed_PATH):
        by_prompt[row["prompt_id"]].append(row)

    prompt_ids = sorted(by_prompt.keys())
    if args.pilot:
        prompt_ids = prompt_ids[:PILOT_N]
    if args.limit is not None:
        prompt_ids = prompt_ids[: args.limit]

    jobs: list[PromptJob] = []
    for pid in prompt_ids:
        rollouts = by_prompt[pid]
        if len(rollouts) != 8:
            raise SystemExit(f"prompt {pid}: expected 8 rollouts, got {len(rollouts)}")
        pr = prompts[pid]
        jobs.append(
            PromptJob(
                prompt_id=pid,
                problem=pr["problem"],
                gold_answer=pr["gold_answer"],
                rollouts=rollouts,
            )
        )

    key_pool = (
        GoogleKeyPool.from_env(start_key=args.api_key)
        if provider == "google"
        else GoogleKeyPool([], [])
    )
    if provider != "google":
        raise SystemExit("only provider=google is implemented in this script")
    if not args.dry_run and not key_pool.keys:
        print(
            "Warning: no GOOGLE_API_KEY / GOOGLE_API_KEY_2; will write error stubs to cache",
            file=sys.stderr,
        )
    elif key_pool.keys:
        print(
            f"API keys loaded: {', '.join(key_pool.labels)}",
            file=sys.stderr,
        )

    if args.purge_stale:
        removed, kept = purge_stale_caches()
        print(f"Purged {removed} stale cache file(s); kept {kept} valid poly_epo caches")

    caches: dict[str, dict] = {}
    to_run: list[PromptJob] = []
    for job in jobs:
        path = _cache_path(job.prompt_id)
        if not args.force:
            cached = _read_cache(path)
            if cached:
                if _cache_usable(cached):
                    caches[job.prompt_id] = cached
                    continue
                if args.dry_run and cached.get("dry_run"):
                    caches[job.prompt_id] = cached
                    continue
        to_run.append(job)

    print(
        f"Analysis A: {len(jobs)} prompts, {len(to_run)} to process, "
        f"model={model}, dry_run={args.dry_run}, workers={args.workers}"
    )

    def _run(job: PromptJob) -> tuple[str, dict]:
        result = process_prompt(
            job,
            provider=provider,
            tier=args.tier,
            model=model,
            dry_run=args.dry_run,
            force=args.force,
            key_pool=key_pool,
        )
        return job.prompt_id, result

    if to_run:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run, job): job for job in to_run}
            for fut in as_completed(futures):
                pid, result = fut.result()
                caches[pid] = result
                if result.get("dry_run"):
                    status = "dry-run (prompt cached)"
                elif result.get("parse_ok"):
                    status = "ok"
                else:
                    status = f"err: {result.get('error')}"
                print(f"  {pid}: {status}")

    for job in jobs:
        if job.prompt_id not in caches:
            caches[job.prompt_id] = _read_cache(_cache_path(job.prompt_id)) or {}

    summary_rows = _build_summary_rows(jobs, caches)
    if not args.dry_run:
        _write_parquet(summary_rows, SUMMARY_PARQUET)
        minority_flags, n_eligible, n_parsed = _minority_metrics(jobs, caches)
        rate, lo, hi = _bootstrap_ci(minority_flags)
        md = _build_summary_md(
            provider=provider,
            tier=args.tier,
            model=model,
            n_prompts=len(jobs),
            n_parsed=n_parsed,
            n_eligible=n_eligible,
            rate=rate,
            lo=lo,
            hi=hi,
            degenerate_frac=_degenerate_rate(caches),
            dry_run=False,
        )
        SUMMARY_MD.write_text(md)
        print(f"Wrote {SUMMARY_PARQUET}")
        print(f"Wrote {SUMMARY_MD}")
        print(
            f"minority_correct_prompt_rate_llm={100*rate:.2f}% "
            f"[{100*lo:.2f}%, {100*hi:.2f}%] (n_eligible={n_eligible})"
        )
    else:
        print(f"Dry-run: cached prompts under {CACHE_DIR}")


if __name__ == "__main__":
    main()
