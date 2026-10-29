"""Method 3, part 2 — generate responses with activation steering applied.

v1 (single layer, absolute scale):
    python methods/steering/run_steering.py --layer 12 --scale 6
    Writes outputs/steering_responses.json.

v2 (config picked by sweep_steering_v2.py):
    python methods/steering/run_steering.py --v2
    Reads steering_vectors_v2/sweep_results.json ("pick") and writes
    outputs/steering_v2_responses.json, so v1 results are never overwritten.

Take the layer/scale or v2 config from the sweep on held-out prompts. Picking
them from test-set outputs would leak the evaluation data into the method.

Add --dry-run to run only 3 prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import io_utils, model_utils
from methods.steering import steering_utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, help="v1: layer from sweep_results.json")
    ap.add_argument("--scale", type=float, default=None, help="v1: single scale; else sweep")
    ap.add_argument("--v2", action="store_true", help="use the steering v2 sweep pick")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.v2 and args.layer is None:
        ap.error("pass --layer (v1) or --v2")

    model_utils.preflight_hf()
    model_utils.set_seeds()

    prompts = io_utils.load_test_prompts()
    if args.dry_run:
        prompts = prompts[:3]

    print("Loading base model (4-bit)...", flush=True)
    tokenizer = model_utils.load_tokenizer()
    model = model_utils.load_base_model()

    records = []
    if args.v2:
        with open(os.path.join(config.STEERING_V2_DIR, "sweep_results.json"),
                  encoding="utf-8") as fh:
            pick = json.load(fh)["pick"]
        layers, alpha, pn = pick["layers"], pick["alpha"], pick["preserve_norm"]
        norms = steering_utils.load_norms(config.STEERING_V2_DIR)
        vectors = steering_utils.load_vector_set(layers, config.STEERING_V2_DIR)
        strengths = {l: alpha * norms[l] for l in layers}
        variant = (f"v2_L{layers[0]}" + (f"-{layers[-1]}" if len(layers) > 1 else "")
                   + f"_a{alpha}" + ("_pn" if pn else ""))
        print(f"\n[steering:{variant}] — {len(prompts)} prompts", flush=True)
        for i, p in enumerate(prompts):
            with steering_utils.multi_steering(model, vectors, strengths, preserve_norm=pn):
                text, secs = model_utils.generate(model, tokenizer, p["prompt"], system_msg=None)
            records.append(io_utils.make_record("steering", variant, p, text, secs))
            print(f"  [{i+1}/{len(prompts)}] {secs:.1f}s", flush=True)
        out_name = "steering_v2_responses.json"
    else:
        scales = [args.scale] if args.scale is not None else config.STEERING_SCALES_TO_TRY
        vector = steering_utils.load_vector(args.layer)
        for scale in scales:
            variant = f"layer{args.layer}_scale{scale}"
            print(f"\n[steering:{variant}] — {len(prompts)} prompts", flush=True)
            for i, p in enumerate(prompts):
                with steering_utils.steering(model, vector, args.layer, scale):
                    text, secs = model_utils.generate(model, tokenizer, p["prompt"],
                                                      system_msg=None)
                records.append(io_utils.make_record("steering", variant, p, text, secs))
                print(f"  [{i+1}/{len(prompts)}] {secs:.1f}s", flush=True)
        out_name = "steering_responses.json"

    if args.dry_run:
        out_name = out_name.replace(".json", "_dryrun.json")
    io_utils.save_responses(out_name, records)
    print("\nNext: python s3_sync.py upload --dir outputs")


if __name__ == "__main__":
    main()
