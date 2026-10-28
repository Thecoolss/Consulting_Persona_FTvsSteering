"""Pick the steering layer and scale on HELD-OUT prompts.

Held-out prompts are fine-tuning instructions that were not used to build the
steering vectors and are not in the 50-prompt test set, so the choice never
peeks at the evaluation data.

Selection rule: among (layer, scale) configs with zero degenerate generations,
take the one with the highest mean jargon count (ties go to the smaller scale).
It must beat the unsteered baseline, otherwise the script exits with code 2.

Writes steering_vectors/sweep_results.json with the full grid and the pick.

Usage:
    python methods/steering/sweep_steering.py
    python methods/steering/sweep_steering.py --layers 14 16 18 --scales 4 6 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import io_utils, model_utils
from data.generate_finetuning_data import jargon_count
from methods.steering import steering_utils


def heldout_prompts(n: int) -> list[str]:
    """One instruction per category, excluding steering-pair and test prompts."""
    with open(config.STEERING_POS_PATH, encoding="utf-8") as fh:
        used = {json.loads(l)["instruction"] for l in fh if l.strip()}
    test = {p["prompt"] for p in io_utils.load_test_prompts()}
    with open(config.TRAIN_DATA_PATH, encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    out, seen = [], set()
    for r in recs:
        if r["instruction"] in used or r["instruction"] in test or r["category"] in seen:
            continue
        seen.add(r["category"])
        out.append(r["instruction"])
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=[12, 14, 16, 18, 20])
    ap.add_argument("--scales", type=float, nargs="*", default=[4.0, 6.0, 8.0, 10.0])
    ap.add_argument("--n-prompts", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    model_utils.set_seeds()
    prompts = heldout_prompts(args.n_prompts)
    print(f"Held-out prompts ({len(prompts)}):", flush=True)
    for p in prompts:
        print("  -", p)

    tokenizer = model_utils.load_tokenizer()
    model = model_utils.load_base_model()

    def run(layer=None, scale=None) -> dict:
        texts = []
        for p in prompts:
            if layer is None:
                text, _ = model_utils.generate(model, tokenizer, p, None,
                                               max_new_tokens=args.max_new_tokens)
            else:
                vec = steering_utils.load_vector(layer)
                with steering_utils.steering(model, vec, layer, scale):
                    text, _ = model_utils.generate(model, tokenizer, p, None,
                                                   max_new_tokens=args.max_new_tokens)
            texts.append(text)
        return {
            "jargon": mean(jargon_count(t) for t in texts),
            "degenerate": sum(steering_utils.looks_degenerate(t) for t in texts),
            "sample": texts[0][:240],
        }

    t0 = time.time()
    base = run()
    print(f"\nUnsteered baseline: jargon={base['jargon']:.2f} "
          f"degenerate={base['degenerate']}/{len(prompts)}", flush=True)

    grid = []
    print(f"\n{'layer':>5s} {'scale':>6s} {'jargon':>7s} {'degen':>6s}", flush=True)
    for layer in args.layers:
        for scale in args.scales:
            r = run(layer, scale)
            r.update(layer=layer, scale=scale)
            grid.append(r)
            print(f"{layer:5d} {scale:6.1f} {r['jargon']:7.2f} "
                  f"{r['degenerate']:3d}/{len(prompts)}", flush=True)

    safe = [g for g in grid if g["degenerate"] == 0 and g["jargon"] > base["jargon"]]
    pick = max(safe, key=lambda g: (g["jargon"], -g["scale"])) if safe else None

    os.makedirs(config.STEERING_VECTORS_DIR, exist_ok=True)
    out_path = os.path.join(config.STEERING_VECTORS_DIR, "sweep_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"prompts": prompts, "baseline": base, "grid": grid, "pick": pick},
                  fh, indent=2, ensure_ascii=False)

    print(f"\nSweep time: {time.time() - t0:.0f}s -> {out_path}")
    if pick is None:
        print("NO SAFE CONFIG: nothing beat the baseline without degenerating.")
        sys.exit(2)
    print(f"PICK: layer={pick['layer']} scale={pick['scale']} "
          f"jargon={pick['jargon']:.2f} (baseline {base['jargon']:.2f})")
    print(f"Sample: {pick['sample']}")


if __name__ == "__main__":
    main()
