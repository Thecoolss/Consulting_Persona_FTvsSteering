"""Method 3, part 1 — extract activation steering vectors from contrastive pairs.

For each layer in STEERING_LAYER_RANGE, compute a normalized
(mean_positive - mean_negative) direction over assistant-response tokens and
save it to steering_vectors/, along with per-layer activation norms.

Layer/scale selection is done separately on held-out prompts:
    python methods/steering/sweep_steering.py

Usage:
    python methods/steering/extract_vectors.py
    python methods/steering/extract_vectors.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import model_utils
from methods.steering import steering_utils


def load_pairs(path: str) -> list[tuple[str, str]]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out.append((rec["instruction"], rec["response"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_utils.preflight_hf()
    model_utils.set_seeds()

    positives = load_pairs(config.STEERING_POS_PATH)
    negatives = load_pairs(config.STEERING_NEG_PATH)
    n = min(len(positives), len(negatives))
    positives, negatives = positives[:n], negatives[:n]
    if args.dry_run:
        positives, negatives = positives[:6], negatives[:6]
    print(f"Contrastive pairs: {len(positives)} | capture={config.STEERING_CAPTURE} "
          f"(response tokens only)", flush=True)

    layers = list(range(config.STEERING_LAYER_RANGE[0], config.STEERING_LAYER_RANGE[1] + 1))

    print("Loading base model (4-bit)...", flush=True)
    tokenizer = model_utils.load_tokenizer()
    model = model_utils.load_base_model()

    t0 = time.time()
    vectors, norms = steering_utils.build_steering_vectors(
        model, tokenizer, positives, negatives, layers)
    steering_utils.save_vectors(vectors, norms=norms)
    elapsed = time.time() - t0

    print(f"\nExtracted {len(vectors)} layer vectors in {elapsed:.1f}s "
          f"(est. ${elapsed / 3600 * config.EC2_HOURLY_RATE:.3f}) -> {config.STEERING_VECTORS_DIR}")
    print(f"{'layer':>5s} {'resid_norm':>10s} {'raw_diff':>9s}")
    for l in layers:
        print(f"{l:5d} {norms[l]['mean_resid_norm']:10.1f} {norms[l]['raw_diff_norm']:9.2f}")
    print("\nNext: python methods/steering/sweep_steering.py")


if __name__ == "__main__":
    main()
