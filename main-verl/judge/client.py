"""Async HTTP client for the Stage 4 judge service."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
from dataclasses import dataclass

import httpx

from judge.parse import parse_judge_response
from judge.prompt import build_judge_messages, build_poly_epo_schema
from judge.types import JudgeClusterResult, JudgeTask


# Max concurrent HTTP requests (each may carry a multi-prompt vLLM batch).
JUDGE_CONCURRENCY_CAP = 8
# Production default: matches train_batch_size=128 split into 2 chunks (one per
# judge container under max_containers=2). Probes/scripts that pass arm_config=None
# previously fell through to 16, generating mis-sized POSTs in the service logs.
DEFAULT_HTTP_BATCH_SIZE = 64


@dataclass
class JudgeClientConfig:
    base_url: str
    auth_token: str | None
    model: str
    concurrency: int = JUDGE_CONCURRENCY_CAP
    timeout_s: float = 120.0
    temperature: float = 0.0
    # Stage 3b (2026-05-30): bumped 2048 → 4096 to fix S4.6b parse collapse.
    max_tokens: int = 4096
    # Prompts per POST when server supports ``requests[]`` batching (see judge/server.py).
    http_batch_size: int = DEFAULT_HTTP_BATCH_SIZE
    # Wall-clock budget for one batched POST (long prefill × N prompts).
    batch_timeout_s: float = 600.0


class JudgeClient:
    MAX_RETRIES = 2
    _RETRY_BASE_S = 0.5
    _RETRY_CAP_S = 2.0

    def __init__(self, config: JudgeClientConfig) -> None:
        self.config = config
        self._completions_url = config.base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> JudgeClient:
        base_url = os.environ.get("JUDGE_BASE_URL", "")
        if not base_url:
            raise ValueError("JUDGE_BASE_URL is required")
        http_batch_size = int(
            os.environ.get("JUDGE_HTTP_BATCH_SIZE", str(DEFAULT_HTTP_BATCH_SIZE))
        )
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
                http_batch_size=max(1, http_batch_size),
                batch_timeout_s=float(os.environ.get("JUDGE_BATCH_TIMEOUT_S", "600")),
            )
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        return headers

    @staticmethod
    def _chunk_tasks(tasks: list[JudgeTask], chunk_size: int) -> list[list[JudgeTask]]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        return [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]

    @staticmethod
    def _messages_for_task(task: JudgeTask) -> list[dict[str, str]]:
        system, user = build_judge_messages(task.problem, task.rollouts)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _parse_single_payload(data: dict, task: JudgeTask) -> JudgeClusterResult:
        content = data["choices"][0]["message"]["content"]
        return parse_judge_response(content, n_rollouts=len(task.rollouts))

    @staticmethod
    def _parse_batch_payload(
        data: dict, tasks: list[JudgeTask]
    ) -> list[JudgeClusterResult]:
        results_raw = data.get("results")
        if not isinstance(results_raw, list):
            raise TypeError("batch response missing results[]")
        if len(results_raw) != len(tasks):
            raise ValueError(
                f"batch results length {len(results_raw)} != tasks {len(tasks)}"
            )
        out: list[JudgeClusterResult] = []
        for item, task in zip(results_raw, tasks):
            try:
                if isinstance(item, dict) and item.get("error"):
                    out.append(
                        JudgeClusterResult(
                            assignment={},
                            clusters=[],
                            parse_ok=False,
                            raw_response=f"<batch item error: {item['error']}>",
                        )
                    )
                    continue
                out.append(JudgeClient._parse_single_payload(item, task))
            except (KeyError, IndexError, TypeError) as exc:
                out.append(
                    JudgeClusterResult(
                        assignment={},
                        clusters=[],
                        parse_ok=False,
                        raw_response=f"<malformed batch item: {exc!r}>",
                    )
                )
        return out

    @staticmethod
    def _malformed_result(exc: Exception) -> JudgeClusterResult:
        return JudgeClusterResult(
            assignment={},
            clusters=[],
            parse_ok=False,
            raw_response=f"<malformed API payload: {exc!r}>",
        )

    @staticmethod
    def _http_failed_result(exc: httpx.HTTPError | None) -> JudgeClusterResult:
        return JudgeClusterResult(
            assignment={},
            clusters=[],
            parse_ok=False,
            raw_response=f"<HTTP failed: {exc!r}>",
        )

    def _timeout_for_batch(self, batch_len: int) -> float:
        if batch_len <= 1:
            return self.config.timeout_s
        return max(self.config.batch_timeout_s, self.config.timeout_s)

    async def _post_single(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        task: JudgeTask,
    ) -> JudgeClusterResult:
        async with sem:
            body = {
                "model": self.config.model,
                "messages": self._messages_for_task(task),
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "guided_json": build_poly_epo_schema(len(task.rollouts)),
            }
            last_http_exc: httpx.HTTPError | None = None
            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        self._completions_url,
                        json=body,
                        headers=self._headers(),
                        timeout=self.config.timeout_s,
                    )
                    resp.raise_for_status()
                    return self._parse_single_payload(resp.json(), task)
                except httpx.HTTPError as exc:
                    last_http_exc = exc
                    if attempt < self.MAX_RETRIES:
                        delay = min(
                            self._RETRY_BASE_S * (2 ** attempt), self._RETRY_CAP_S
                        )
                        print(
                            f"[judge-retry] single attempt {attempt + 1}/"
                            f"{self.MAX_RETRIES + 1} failed ({exc!r}); "
                            f"retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                except (KeyError, IndexError, TypeError) as exc:
                    return self._malformed_result(exc)
            return self._http_failed_result(last_http_exc)

    async def _post_http_batch(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        tasks: list[JudgeTask],
    ) -> list[JudgeClusterResult]:
        if not tasks:
            return []
        async with sem:
            # All tasks in a batch share one SamplingParams on the server side,
            # so all rollout counts in the batch must match (true for N_ROLLOUTS=8).
            body = {
                "model": self.config.model,
                "requests": [
                    {"messages": self._messages_for_task(task)} for task in tasks
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "guided_json": build_poly_epo_schema(len(tasks[0].rollouts)),
            }
            timeout = self._timeout_for_batch(len(tasks))
            last_http_exc: httpx.HTTPError | None = None
            for attempt in range(self.MAX_RETRIES + 1):
                try:
                    resp = await client.post(
                        self._completions_url,
                        json=body,
                        headers=self._headers(),
                        timeout=timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if "error" in data and "results" not in data:
                        err = str(data["error"])
                        return [
                            JudgeClusterResult(
                                assignment={},
                                clusters=[],
                                parse_ok=False,
                                raw_response=f"<batch error: {err}>",
                            )
                            for _ in tasks
                        ]
                    return self._parse_batch_payload(data, tasks)
                except httpx.HTTPError as exc:
                    last_http_exc = exc
                    if attempt < self.MAX_RETRIES:
                        delay = min(
                            self._RETRY_BASE_S * (2 ** attempt), self._RETRY_CAP_S
                        )
                        print(
                            f"[judge-retry] batch(n={len(tasks)}) attempt "
                            f"{attempt + 1}/{self.MAX_RETRIES + 1} failed ({exc!r}); "
                            f"retrying in {delay:.1f}s"
                        )
                        await asyncio.sleep(delay)
                except (KeyError, IndexError, TypeError, ValueError) as exc:
                    bad = self._malformed_result(exc)
                    return [bad for _ in tasks]
            failed = self._http_failed_result(last_http_exc)
            return [failed for _ in tasks]

    async def cluster_batch(self, tasks: list[JudgeTask]) -> list[JudgeClusterResult]:
        if not tasks:
            return []
        sem = asyncio.Semaphore(self.config.concurrency)
        batch_size = self.config.http_batch_size
        if batch_size > 1:
            n_chunks = len(self._chunk_tasks(tasks, batch_size))
            print(
                f"[judge-client] http_batch_size={batch_size} "
                f"n_tasks={len(tasks)} n_http_posts={n_chunks} "
                f"concurrency={self.config.concurrency}",
                flush=True,
            )
        async with httpx.AsyncClient(timeout=self.config.batch_timeout_s) as client:
            if batch_size <= 1:
                return await asyncio.gather(
                    *[self._post_single(client, sem, task) for task in tasks]
                )
            chunks = self._chunk_tasks(tasks, batch_size)
            chunk_results = await asyncio.gather(
                *[self._post_http_batch(client, sem, chunk) for chunk in chunks]
            )
            return [result for chunk in chunk_results for result in chunk]

    def cluster_batch_sync(self, tasks: list[JudgeTask]) -> list[JudgeClusterResult]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.cluster_batch(tasks))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.cluster_batch(tasks)).result()
