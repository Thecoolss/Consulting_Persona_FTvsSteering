"""Steering v2 data — matched contrastive pairs.

The v1 negatives were written from scratch: 60-180 words of prose against
150-400-word persona answers with lists. A mean-difference vector built from
that encodes "longer, more lists" as well as "consultant", which eats the
usable steering range.

Here each persona answer is REWRITTEN in plain language, keeping the same facts,
roughly the same length, and the same paragraph/list layout. Style is then the
only systematic difference between the two sides.

Pairs are kept only if the plain rewrite is 0.8-1.25x the persona length and
contains at most 1 consulting-jargon term.

Usage:
    python data/generate_matched_pairs.py
    python data/generate_matched_pairs.py --dry-run
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
from data.generate_finetuning_data import jargon_count, word_count

POS_OUT = os.path.join(config.DATA_DIR, "steering_positive_v2.jsonl")
NEG_OUT = os.path.join(config.DATA_DIR, "steering_negative_v2.jsonl")

REWRITE_PROMPT = """Rewrite the answer below in plain, direct, everyday language.

Keep exactly the same:
- facts, advice and level of detail
- length (about {words} words)
- layout: the same number of paragraphs, and the same numbered or bulleted lists if there are any

Remove completely:
- business and consulting jargon (synergies, leverage, stakeholder, deliverables, workstream, north star, net-net, ROI, value-add, operationalize, etc.)
- framework talk (2x2 matrices, SWOT, "lens", "three horizons")
- "executive summary", "next steps", "action items" headers or phrases

Write it the way a clear, friendly expert would explain it to a friend.
Return ONLY the rewritten answer, no preamble.

Question: {question}

Answer to rewrite:
{answer}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    with open(config.STEERING_POS_PATH, encoding="utf-8") as fh:
        positives = [json.loads(l) for l in fh if l.strip()]
    if args.dry_run:
        positives = positives[:3]

    prompts = [REWRITE_PROMPT.format(words=word_count(p["response"]),
                                     question=p["instruction"], answer=p["response"])
               for p in positives]
    client = BedrockClient()
    done = {"n": 0}

    def progress(_i, _r):
        done["n"] += 1
        if done["n"] % 25 == 0 or args.dry_run:
            print(f"  [{done['n']}/{len(prompts)}] {client.cost.summary()}", flush=True)

    t0 = time.time()
    print(f"Rewriting {len(prompts)} persona answers via {config.BEDROCK_GENERATOR_MODEL} "
          f"(workers={args.workers})", flush=True)
    rewrites = client.invoke_many(prompts, config.BEDROCK_GENERATOR_MODEL,
                                  max_tokens=900, temperature=0.3,
                                  max_workers=args.workers, on_result=progress)

    kept, drop = [], {"failed": 0, "length": 0, "jargon": 0}
    for pos, neg in zip(positives, rewrites):
        if not neg:
            drop["failed"] += 1
            continue
        neg = neg.strip()
        ratio = word_count(neg) / max(word_count(pos["response"]), 1)
        if not 0.8 <= ratio <= 1.25:
            drop["length"] += 1
            continue
        if jargon_count(neg) > 1:
            drop["jargon"] += 1
            continue
        kept.append((pos, neg, ratio))

    out_pos = POS_OUT if not args.dry_run else POS_OUT + ".dryrun"
    out_neg = NEG_OUT if not args.dry_run else NEG_OUT + ".dryrun"
    with open(out_pos, "w", encoding="utf-8") as fp, open(out_neg, "w", encoding="utf-8") as fn:
        for pos, neg, _ in kept:
            fp.write(json.dumps({"instruction": pos["instruction"], "response": pos["response"]},
                                ensure_ascii=False) + "\n")
            fn.write(json.dumps({"instruction": pos["instruction"], "response": neg},
                                ensure_ascii=False) + "\n")

    print("\n--- SUMMARY ---")
    print(f"Kept {len(kept)}/{len(positives)} pairs | dropped {drop}")
    if kept:
        ratios = sorted(r for _, _, r in kept)
        pj = sum(jargon_count(p["response"]) for p, _, _ in kept) / len(kept)
        nj = sum(jargon_count(n) for _, n, _ in kept) / len(kept)
        print(f"Length ratio neg/pos: median {ratios[len(ratios)//2]:.2f} "
              f"(min {ratios[0]:.2f}, max {ratios[-1]:.2f})")
        print(f"Jargon per answer: persona {pj:.1f} vs plain {nj:.2f}")
    print(f"Time {time.time() - t0:.0f}s | {client.cost.summary()} -> {out_pos}, {out_neg}")
    if args.dry_run and kept:
        p, n, _ = kept[0]
        print(f"\nQ: {p['instruction']}\n--- persona ---\n{p['response'][:400]}\n--- plain ---\n{n[:400]}")


if __name__ == "__main__":
    main()
