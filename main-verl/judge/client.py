"""Async HTTP client for the Stage 4 judge service."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from dataclasses import dataclass

import httpx

from judge.parse import parse_judge_response
from judge.prompt import build_judge_messages
from judge.types import JudgeClusterResult, JudgeTask


# Stage 4 S4.6 finding (2026-05-30): parse rate collapses 100% → 22% at concurrency=64
# against the single-container vLLM server. Default capped at 8 until server.py
# `max_containers=1` is bumped or batching is reworked. See stage-04-log.md audit
# "Required actions before Stage 3b launch" item 1.
JUDGE_CONCURRENCY_CAP = 8


@dataclass
class JudgeClientConfig:
    base_url: str
    auth_token: str | None
    model: str
    concurrency: int = JUDGE_CONCURRENCY_CAP
    timeout_s: float = 120.0
    temperature: float = 0.0
    # Stage 3b (2026-05-30): bumped 2048 → 4096 to fix S4.6b parse collapse.
    # 8-way clustering JSON with chain_of_thought fields needs >2048 tokens.
    max_tokens: int = 4096


class JudgeClient:
    # Retry budget for transient HTTP failures (network, timeout, bad status).
    # Does NOT apply to KeyError/IndexError/TypeError (malformed payload).
    MAX_RETRIES = 2  # 1 initial attempt + 2 retries = 3 total
    _RETRY_BASE_S = 0.5  # backoff: 0.5s, 1.5s (base * 2**attempt, capped at 2s)
    _RETRY_CAP_S = 2.0

    def __init__(self, config: JudgeClientConfig) -> None:
        self.config = config
        # JUDGE_BASE_URL is the full Modal web-endpoint URL (POST target).
        self._completions_url = config.base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> JudgeClient:
        base_url = os.environ.get("JUDGE_BASE_URL", "")
        if not base_url:
            raise ValueError("JUDGE_BASE_URL is required")
        return cls(
            JudgeClientConfig(
                base_url=base_url,
                auth_token=os.environ.get("JUDGE_AUTH_TOKEN"),
                model=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
                concurrency=min(
                    int(os.environ.get("JUDGE_CONCURRENCY", str(JUDGE_CONCURRENCY_CAP))),
                    JUDGE_CONCURRENCY_CAP,
                ),
                timeout_s=float(os.environ.get("JUDGE_TIMEOUT_S", "120")),
                temperature=float(os.environ.get("JUDGE_TEMPERATURE", "0")),
                max_tokens=int(os.environ.get("JUDGE_MAX_TOKENS", "4096")),
            )
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return headers

    async def _one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        task: JudgeTask,
    ) -> JudgeClusterResult:
        async with sem:
            system, user = build_judge_messages(task.problem, task.rollouts)
            body = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            last_http_exc: httpx.HTTPError | None = None
            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        self._completions_url,
                        json=body,
                        headers=self._headers(),
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return parse_judge_response(content, n_rollouts=len(task.rollouts))
                except httpx.HTTPError as exc:
                    last_http_exc = exc
                    if attempt < self.MAX_RETRIES:
                        delay = min(self._RETRY_BASE_S * (2 ** attempt), self._RETRY_CAP_S)
                        print(f"[judge-retry] attempt {attempt + 1}/{self.MAX_RETRIES + 1} failed ({exc!r}); retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
                except (KeyError, IndexError, TypeError):
                    # Malformed payload — retrying won't help.
                    return JudgeClusterResult(
                        assignment={},
                        clusters=[],
                        parse_ok=False,
                        raw_response=None,
                    )
            # All attempts exhausted.
            return JudgeClusterResult(
                assignment={},
                clusters=[],
                parse_ok=False,
                raw_response=None,
            )

    async def cluster_batch(self, tasks: list[JudgeTask]) -> list[JudgeClusterResult]:
        if not tasks:
            return []
        sem = asyncio.Semaphore(self.config.concurrency)
        async with httpx.AsyncClient(timeout=self.config.timeout_s) as client:
            return await asyncio.gather(
                *[self._one(client, sem, task) for task in tasks]
            )

    def cluster_batch_sync(self, tasks: list[JudgeTask]) -> list[JudgeClusterResult]:
        """Run ``cluster_batch`` from sync code (e.g. Ray adv_estimator hook).

        If the caller is already inside a running event loop (common in Ray /
        verl workers), ``asyncio.run`` raises. In that case we delegate to a
        one-off thread that owns its own loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.cluster_batch(tasks))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.cluster_batch(tasks)).result()
