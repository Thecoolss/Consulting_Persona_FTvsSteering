"""Thin Bedrock client wrapper shared by data generation and evaluation.

Handles: the Anthropic-on-Bedrock request format, retries with backoff,
throttling delay, and running token/cost accounting.
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionError as BotoConnectionError, \
    ReadTimeoutError, ConnectTimeoutError

import config


class ModelRefusal(Exception):
    """The model returned stop_reason='refusal' with no content."""


@dataclass
class CostTracker:
    """Accumulates token usage and estimated USD cost across calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    _by_model: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, model_id: str, in_tok: int, out_tok: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            m = self._by_model.setdefault(model_id, {"in": 0, "out": 0})
            m["in"] += in_tok
            m["out"] += out_tok

    def estimated_usd(self) -> float:
        total = 0.0
        for model_id, usage in self._by_model.items():
            price = config.BEDROCK_PRICING.get(model_id)
            if not price:
                continue
            total += usage["in"] / 1_000_000 * price["input"]
            total += usage["out"] / 1_000_000 * price["output"]
        return total

    def summary(self) -> str:
        return (
            f"{self.calls} calls | in={self.input_tokens:,} out={self.output_tokens:,} "
            f"tok | est. ${self.estimated_usd():.3f}"
        )


class BedrockClient:
    """Minimal Anthropic-message client over Bedrock runtime."""

    def __init__(self, region: str | None = None):
        # Explicit timeouts so a stalled connection raises (and is retried)
        # instead of hanging the whole sequential loop indefinitely.
        boto_cfg = Config(
            connect_timeout=15,
            read_timeout=config.BEDROCK_READ_TIMEOUT,
            retries={"max_attempts": 0, "mode": "standard"},  # we handle retries
        )
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region or config.BEDROCK_REGION,
            config=boto_cfg,
        )
        self.cost = CostTracker()

    def invoke(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> str:
        """Send a single-user-turn message, return the assistant text.

        Retries on throttling/transient errors with exponential backoff.
        """
        body: dict = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        last_err: Exception | None = None
        throttle_retries = 0
        attempt = 0
        while attempt < config.BEDROCK_MAX_RETRIES:
            try:
                resp = self.client.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                payload = json.loads(resp["body"].read())
                usage = payload.get("usage", {})
                self.cost.add(
                    model_id,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )
                time.sleep(config.BEDROCK_CALL_DELAY_SECONDS)
                content = payload.get("content") or []
                if payload.get("stop_reason") == "refusal" or not content:
                    # The model declined and returned no text. Retrying the same
                    # model won't change that, so surface it to the caller.
                    raise ModelRefusal(f"{model_id} refused (stop_reason="
                                       f"{payload.get('stop_reason')})")
                return content[0]["text"]
            except (ReadTimeoutError, ConnectTimeoutError, BotoConnectionError) as exc:
                # Stalled/slow connection — back off and retry.
                last_err = exc
                time.sleep(2 ** attempt)
                attempt += 1
                continue
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                last_err = exc
                if code in ("ThrottlingException", "TooManyRequestsException"):
                    # Rate limit: wait it out with jittered backoff (capped at 30s).
                    # These retries have their own, larger budget so a busy
                    # minute doesn't turn into a failed call.
                    throttle_retries += 1
                    if throttle_retries > config.BEDROCK_THROTTLE_RETRIES:
                        break
                    time.sleep(min(30, 2 ** throttle_retries) * (0.5 + random.random()))
                    continue
                if code in ("ServiceUnavailableException", "ModelTimeoutException"):
                    time.sleep(2 ** attempt)
                    attempt += 1
                    continue
                raise
        raise RuntimeError(f"Bedrock call failed after retries: {last_err}")

    def invoke_many(self, tasks, model_id, max_tokens=1024, temperature=0.0,
                    max_workers=None, on_result=None):
        """Run many prompts concurrently. Preserves input order in the result list.

        tasks: list of prompt strings (or (key, prompt) tuples — key is ignored
               for ordering but handy for callers).
        on_result: optional callback(index, text_or_exc) called as each finishes,
                   for live progress.
        Returns: list aligned to `tasks`; entries are the text, or None on failure.
        """
        workers = max_workers or config.BEDROCK_MAX_CONCURRENCY
        prompts = [t[1] if isinstance(t, tuple) else t for t in tasks]
        results: list = [None] * len(prompts)

        def one(i: int):
            return i, self.invoke(prompts[i], model_id, max_tokens, temperature)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, i) for i in range(len(prompts))]
            for fut in as_completed(futures):
                try:
                    i, text = fut.result()
                    results[i] = text
                    if on_result:
                        on_result(i, text)
                except Exception as exc:  # noqa: BLE001
                    if on_result:
                        on_result(-1, exc)
        return results
