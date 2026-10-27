"""Shared helpers for loading test prompts and writing response JSON files."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def load_test_prompts() -> list[dict]:
    with open(config.TEST_PROMPTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def save_responses(filename: str, records: list[dict]) -> str:
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(records)} responses to {path}")
    return path


def make_record(method: str, variant: str, prompt_rec: dict, response: str,
                gen_seconds: float) -> dict:
    return {
        "method": method,
        "variant": variant,
        "prompt_id": prompt_rec["id"],
        "category": prompt_rec.get("category", ""),
        "prompt": prompt_rec["prompt"],
        "response": response,
        "generation_time_seconds": round(gen_seconds, 3),
    }
