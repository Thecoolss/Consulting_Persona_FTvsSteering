"""Phase 5 — LLM-as-Judge evaluation via AWS Bedrock (Claude Sonnet 4.5).

Loads every response file in outputs/, scores each response on all 5 dimensions,
and writes evaluation/results/evaluation_results.json. Tracks running cost.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --dry-run      # a handful of responses
    python evaluation/run_evaluation.py --files outputs/baseline_responses.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common.bedrock import BedrockClient
from evaluation.judge_prompts import RUBRICS


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_score(raw: str) -> dict:
    cleaned = strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1:
            return {"score": None, "justification": "parse_error: " + cleaned[:80]}
        try:
            data = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return {"score": None, "justification": "parse_error"}
    score = data.get("score")
    try:
        score = int(score)
        if score < 1 or score > 5:
            score = max(1, min(5, score))
    except (TypeError, ValueError):
        score = None
    return {"score": score, "justification": str(data.get("justification", ""))[:300]}


def load_response_files(file_args: list[str] | None) -> list[dict]:
    paths = file_args or sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*_responses.json")))
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            records.extend(json.load(fh))
    if not records:
        sys.exit("No response records found in outputs/. Run the methods first.")
    return records


def score_one(client: BedrockClient, prompt: str, response: str, dimension: str) -> dict:
    rubric = RUBRICS[dimension].format(prompt=prompt, response=response)
    raw = client.invoke(rubric, config.BEDROCK_JUDGE_MODEL, max_tokens=256, temperature=0.0)
    parsed = parse_score(raw)
    if parsed["score"] is None:  # one retry with a nudge
        raw = client.invoke(rubric + "\n\nReturn ONLY valid JSON.",
                             config.BEDROCK_JUDGE_MODEL, max_tokens=256, temperature=0.0)
        parsed = parse_score(raw)
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--files", nargs="*")
    args = ap.parse_args()

    records = load_response_files(args.files)
    if args.dry_run:
        records = records[:3]

    client = BedrockClient()
    results = []
    total = len(records)
    print(f"Judging {total} responses x {len(config.EVAL_DIMENSIONS)} dimensions "
          f"via {config.BEDROCK_JUDGE_MODEL}")

    for idx, rec in enumerate(records):
        scores = {}
        for dim in config.EVAL_DIMENSIONS:
            scores[dim] = score_one(client, rec["prompt"], rec["response"], dim)
        results.append({
            "method": rec["method"],
            "variant": rec["variant"],
            "prompt_id": rec["prompt_id"],
            "category": rec.get("category", ""),
            "prompt": rec["prompt"],
            "response": rec["response"],
            "scores": scores,
        })
        if (idx + 1) % 10 == 0 or args.dry_run:
            print(f"  [{idx+1}/{total}] {client.cost.summary()}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(results)} scored records to {out_path}")
    print(f"Judge cost: {client.cost.summary()}")
    print("Next: python evaluation/analyze_results.py")


if __name__ == "__main__":
    main()
