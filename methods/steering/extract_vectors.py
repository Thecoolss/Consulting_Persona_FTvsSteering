"""Method 3, part 1 — extract activation steering vectors from contrastive pairs.

For each layer in STEERING_LAYER_RANGE, compute a normalized (mean_positive -
mean_negative) direction and save it to steering_vectors/. Then run a quick
validation: apply each layer's vector to a few prompts at the default scale and
print the outputs so you can pick the best layer for the full run.

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


def load_jsonl(path: str) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # format each example as a chat turn so activations reflect real usage
                out.append(rec["instruction"] + "\n\n" + rec["response"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_utils.preflight_hf()
    model_utils.set_seeds()

    positives = load_jsonl(config.STEERING_POS_PATH)
    negatives = load_jsonl(config.STEERING_NEG_PATH)
    n = min(len(positives), len(negatives))
    positives, negatives = positives[:n], negatives[:n]
    if args.dry_run:
        positives, negatives = positives[:6], negatives[:6]
    print(f"Contrastive pairs: {len(positives)} | capture={config.STEERING_CAPTURE}")

    layers = list(range(config.STEERING_LAYER_RANGE[0], config.STEERING_LAYER_RANGE[1] + 1))

    print("Loading base model (4-bit)...")
    tokenizer = model_utils.load_tokenizer()
    model = model_utils.load_base_model()

    t0 = time.time()
    vectors = steering_utils.build_steering_vectors(
        model, tokenizer, positives, negatives, layers)
    steering_utils.save_vectors(vectors)
    elapsed = time.time() - t0
    print(f"Extracted {len(vectors)} layer vectors in {elapsed:.1f}s "
          f"(est. ${elapsed/3600*config.EC2_HOURLY_RATE:.3f}) -> {config.STEERING_VECTORS_DIR}")

    # ---- Validation: eyeball each layer on a couple of prompts ----
    val_prompts = [
        "How should I prepare for a job interview?",
        "Explain how photosynthesis works.",
    ]
    print("\n=== Layer validation (scale="
          f"{config.STEERING_SCALE_DEFAULT}) ===")
    for l in layers:
        print(f"\n--- layer {l} ---")
        for vp in val_prompts:
            with steering_utils.steering(model, vectors[l], l, config.STEERING_SCALE_DEFAULT):
                text, _ = model_utils.generate(model, tokenizer, vp, system_msg=None)
            print(f"  Q: {vp}\n  A: {text[:220]}...")

    print("\nPick the best layer(s) from the above, then run run_steering.py "
          "(set --layer).")
    print("Next: python s3_sync.py upload --dir steering_vectors")


if __name__ == "__main__":
    main()
