"""Thin Bedrock client wrapper shared by data generation and evaluation.

Handles: the Anthropic-on-Bedrock request format, retries with backoff,
throttling delay, and running token/cost accounting.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import boto3
from botocore.exceptions import ClientError

import config


@dataclass
class CostTracker:
    """Accumulates token usage and estimated USD cost across calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    _by_model: dict = field(default_factory=dict)

    def add(self, model_id: str, in_tok: int, out_tok: int) -> None:
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
        self.client = boto3.client(
            "bedrock-runtime", region_name=region or config.BEDROCK_REGION
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
        for attempt in range(config.BEDROCK_MAX_RETRIES):
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
                return payload["content"][0]["text"]
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                last_err = exc
                if code in ("ThrottlingException", "TooManyRequestsException",
                            "ServiceUnavailableException", "ModelTimeoutException"):
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Bedrock call failed after retries: {last_err}")
