"""Method 1 — system-prompt engineering (plus the plain base-model control).

Loads Llama 3.1 8B in 4-bit and runs all 50 test prompts through:
  - baseline  : NO system prompt (the control — how the raw model answers)
  - minimal / medium / detailed : the three persona system-prompt variants

Writes outputs/prompting_responses.json (variants) and outputs/baseline_responses.json.

Usage:
    python methods/prompting/run_prompting.py
    python methods/prompting/run_prompting.py --dry-run          # 3 prompts
    python methods/prompting/run_prompting.py --variants detailed
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import io_utils, model_utils
from methods.prompting import system_prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["minimal", "medium", "detailed"])
    args = ap.parse_args()

    model_utils.preflight_hf()
    model_utils.set_seeds()

    prompts = io_utils.load_test_prompts()
    if args.dry_run:
        prompts = prompts[:3]

    print("Loading base model (4-bit)...")
    tokenizer = model_utils.load_tokenizer()
    model = model_utils.load_base_model()

    # ---- Baseline control: no system prompt ----
    baseline_records = []
    print(f"\n[baseline] no system prompt — {len(prompts)} prompts")
    for i, p in enumerate(prompts):
        text, secs = model_utils.generate(model, tokenizer, p["prompt"], system_msg=None)
        baseline_records.append(
            io_utils.make_record("baseline", "none", p, text, secs))
        print(f"  [{i+1}/{len(prompts)}] {secs:.1f}s")
    io_utils.save_responses("baseline_responses.json", baseline_records)

    # ---- Prompting variants ----
    variant_records = []
    for variant in args.variants:
        system_msg = system_prompts.get_variant(variant)
        print(f"\n[prompting:{variant}] — {len(prompts)} prompts")
        for i, p in enumerate(prompts):
            text, secs = model_utils.generate(model, tokenizer, p["prompt"], system_msg)
            variant_records.append(
                io_utils.make_record("prompting", variant, p, text, secs))
            print(f"  [{i+1}/{len(prompts)}] {secs:.1f}s")
    io_utils.save_responses("prompting_responses.json", variant_records)

    print("\nDone. Remember: python s3_sync.py upload --dir outputs")


if __name__ == "__main__":
    main()
