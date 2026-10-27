"""Generate contrastive pairs for activation steering.

For each of ~300 prompts we need two responses to the SAME question:
  - positive: full consulting persona  (reused from the fine-tuning dataset)
  - negative: plain, direct, no-jargon  (generated fresh via Bedrock Haiku)

Outputs two JSONL files with MATCHING indices (line i of positive <-> line i of
negative share the same instruction).

Usage:
    python data/generate_steering_pairs.py            # ~300 pairs
    python data/generate_steering_pairs.py --dry-run  # 3 pairs
    python data/generate_steering_pairs.py --target 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from common.bedrock import BedrockClient

NEG_PROMPT_TEMPLATE = """Answer the following question in PLAIN, DIRECT language.

Rules:
- Short, simple sentences. No corporate jargon. No frameworks. No buzzwords.
- No "executive summary", no "next steps", no 2x2 matrices.
- Just answer the question clearly and concisely, the way a plain-spoken person would.
- 60-180 words.

Question: {question}

Answer:"""


def load_finetuning_pairs() -> list[dict]:
    if not os.path.exists(config.TRAIN_DATA_PATH):
        sys.exit(
            f"{config.TRAIN_DATA_PATH} not found. Run generate_finetuning_data.py first."
        )
    pairs = []
    with open(config.TRAIN_DATA_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", type=int, default=config.NUM_STEERING_PAIRS)
    args = ap.parse_args()

    target = 3 if args.dry_run else args.target
    source = load_finetuning_pairs()
    if len(source) < target:
        print(f"WARNING: only {len(source)} fine-tuning pairs available; "
              f"generating {len(source)} steering pairs instead of {target}.")
        target = len(source)

    selected = source[:target]
    client = BedrockClient()
    os.makedirs(config.DATA_DIR, exist_ok=True)

    positives, negatives = [], []
    t0 = time.time()
    for i, pair in enumerate(selected):
        question = pair["instruction"]
        neg_prompt = NEG_PROMPT_TEMPLATE.format(question=question)
        try:
            neg = client.invoke(neg_prompt, config.BEDROCK_GENERATOR_MODEL,
                                 max_tokens=512, temperature=0.3)
        except Exception as exc:
            print(f"  [{i+1}/{target}] ERROR {exc} — skipping")
            continue
        positives.append({"instruction": question, "response": pair["response"]})
        negatives.append({"instruction": question, "response": neg.strip()})
        if (i + 1) % 20 == 0 or args.dry_run:
            print(f"  [{i+1}/{target}] {client.cost.summary()}")

    with open(config.STEERING_POS_PATH, "w", encoding="utf-8") as fh:
        for p in positives:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(config.STEERING_NEG_PATH, "w", encoding="utf-8") as fh:
        for n in negatives:
            fh.write(json.dumps(n, ensure_ascii=False) + "\n")

    print("\n--- SUMMARY ---")
    print(f"Pairs written: {len(positives)} "
          f"(positive -> {config.STEERING_POS_PATH}, negative -> {config.STEERING_NEG_PATH})")
    print(f"Time: {time.time() - t0:.1f}s | Bedrock: {client.cost.summary()}")


if __name__ == "__main__":
    main()
