"""Phase 5 — LLM-as-Judge evaluation via AWS Bedrock (Claude Sonnet 4.5).

Loads every response file in outputs/, scores each response on all 5 dimensions
(one judge call per dimension, so scores don't bleed into each other), and writes
evaluation/results/evaluation_results.json.

Calls run concurrently and every score is appended to a cache file as soon as it
arrives (evaluation/results/judge_cache.jsonl). Re-running skips anything already
scored, so an interrupted run never pays twice for the same call.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --dry-run      # 3 responses x 5 dimensions
    python evaluation/run_evaluation.py --files outputs/steering_v2_responses.json
    python evaluation/run_evaluation.py --workers 12
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common.bedrock import BedrockClient, ModelRefusal
from evaluation.judge_prompts import RUBRICS

CACHE_PATH = os.path.join(config.RESULTS_DIR, "judge_cache.jsonl")
RESULTS_PATH = os.path.join(config.RESULTS_DIR, "evaluation_results.json")


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
    try:
        score = max(1, min(5, int(data.get("score"))))
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


def cache_key(rec: dict, dim: str) -> str:
    return f"{rec['method']}|{rec['variant']}|{rec['prompt_id']}|{dim}"


def load_cache() -> dict:
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    if row.get("score") is not None:   # retry failures on rerun
                        cache[row["key"]] = row
    return cache


def _judge(client: BedrockClient, rubric: str, model_id: str) -> dict:
    raw = client.invoke(rubric, model_id, max_tokens=256, temperature=0.0)
    parsed = parse_score(raw)
    if parsed["score"] is None:  # one retry with a nudge
        raw = client.invoke(rubric + "\n\nReturn ONLY valid JSON.",
                            model_id, max_tokens=256, temperature=0.0)
        parsed = parse_score(raw)
    return parsed


def score_one(client: BedrockClient, prompt: str, response: str, dimension: str) -> dict:
    """Score with the main judge; on a refusal, fall back once and tag the result."""
    rubric = RUBRICS[dimension].format(prompt=prompt, response=response)
    try:
        parsed = _judge(client, rubric, config.BEDROCK_JUDGE_MODEL)
        parsed["judge"] = "main"
    except ModelRefusal:
        parsed = _judge(client, rubric, config.BEDROCK_JUDGE_FALLBACK_MODEL)
        parsed["judge"] = "fallback"
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    records = load_response_files(args.files)
    if args.dry_run:
        records = records[:3]

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    cache = load_cache()
    tasks = [(rec, dim) for rec in records for dim in config.EVAL_DIMENSIONS
             if cache_key(rec, dim) not in cache]
    total = len(records) * len(config.EVAL_DIMENSIONS)
    print(f"Judge: {config.BEDROCK_JUDGE_MODEL}", flush=True)
    print(f"{len(records)} responses x {len(config.EVAL_DIMENSIONS)} dimensions = {total} scores | "
          f"{total - len(tasks)} cached | {len(tasks)} to call | workers={args.workers}", flush=True)

    client = BedrockClient()
    lock = threading.Lock()
    done = {"n": 0, "fail": 0}
    t0 = time.time()

    def work(rec, dim):
        return rec, dim, score_one(client, rec["prompt"], rec["response"], dim)

    with open(CACHE_PATH, "a", encoding="utf-8") as cache_fh, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, rec, dim) for rec, dim in tasks]
        for fut in as_completed(futures):
            try:
                rec, dim, parsed = fut.result()
            except Exception as exc:  # noqa: BLE001
                with lock:
                    done["fail"] += 1
                print(f"  call failed: {exc}", flush=True)
                continue
            row = {"key": cache_key(rec, dim), **parsed}
            with lock:
                cache[row["key"]] = row
                cache_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                cache_fh.flush()
                done["n"] += 1
                n = done["n"]
            if n % 100 == 0 or n == len(tasks):
                rate = n / max(time.time() - t0, 1e-6)
                eta = (len(tasks) - n) / max(rate, 1e-6)
                print(f"  [{n}/{len(tasks)}] {client.cost.summary()} | eta {eta/60:.1f} min",
                      flush=True)

    # ---- Assemble per-response results from the cache ----
    results, missing = [], 0
    for rec in records:
        scores = {}
        for dim in config.EVAL_DIMENSIONS:
            row = cache.get(cache_key(rec, dim))
            if row is None or row.get("score") is None:
                missing += 1
                scores[dim] = {"score": None, "justification": "missing"}
            else:
                scores[dim] = {"score": row["score"], "justification": row["justification"],
                               "judge": row.get("judge", "main")}
        results.append({
            "method": rec["method"],
            "variant": rec["variant"],
            "prompt_id": rec["prompt_id"],
            "category": rec.get("category", ""),
            "prompt": rec["prompt"],
            "response": rec["response"],
            "scores": scores,
        })

    out_path = RESULTS_PATH if not args.dry_run else os.path.join(
        config.RESULTS_DIR, "evaluation_results_dryrun.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(results)} records to {out_path}")
    print(f"Missing scores: {missing} | failed calls: {done['fail']} "
          f"| time {time.time() - t0:.0f}s")
    print(f"This run's judge cost: {client.cost.summary()}")
    if missing:
        print("Re-run the same command to retry the missing scores (cached ones are skipped).")
    else:
        print("Next: python evaluation/analyze_results.py")


if __name__ == "__main__":
    main()
