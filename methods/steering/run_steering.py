"""Method 3, part 2 — generate responses with activation steering applied.

Loads the base model + the chosen layer's steering vector and runs all 50 test
prompts at a single --scale (or every scale in STEERING_SCALES_TO_TRY if
omitted). The variant field records the layer and scale.

Take --layer and --scale from steering_vectors/sweep_results.json ("pick"),
which sweep_steering.py chooses on held-out prompts. Picking them from test-set
outputs would leak the evaluation data into the method.

Usage:
    python methods/steering/run_steering.py --layer 12 --scale 6
    python methods/steering/run_steering.py --layer 12 --scale 6 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import io_utils, model_utils
from methods.steering import steering_utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True, help="layer from sweep_results.json")
    ap.add_argument("--scale", type=float, default=None, help="single scale; else sweep")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_utils.preflight_hf()
    model_utils.set_seeds()

    prompts = io_utils.load_test_prompts()
    if args.dry_run:
        prompts = prompts[:3]

    scales = [args.scale] if args.scale is not None else config.STEERING_SCALES_TO_TRY

    print("Loading base model (4-bit)...")
    tokenizer = model_utils.load_tokenizer()
    model = model_utils.load_base_model()
    vector = steering_utils.load_vector(args.layer)

    records = []
    for scale in scales:
        variant = f"layer{args.layer}_scale{scale}"
        print(f"\n[steering:{variant}] — {len(prompts)} prompts")
        for i, p in enumerate(prompts):
            with steering_utils.steering(model, vector, args.layer, scale):
                text, secs = model_utils.generate(model, tokenizer, p["prompt"],
                                                   system_msg=None)
            records.append(io_utils.make_record("steering", variant, p, text, secs))
            print(f"  [{i+1}/{len(prompts)}] {secs:.1f}s")

    io_utils.save_responses("steering_responses.json", records)
    print("\nNext: python s3_sync.py upload --dir outputs")


if __name__ == "__main__":
    main()
