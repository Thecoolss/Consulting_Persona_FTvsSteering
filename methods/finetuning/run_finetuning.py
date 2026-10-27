"""Method 2 inference — generate responses with the QLoRA fine-tuned model.

Loads the 4-bit base model + the trained LoRA adapter and runs all 50 test
prompts. Uses only the MINIMAL system prompt, the persona should come from the
trained weights, not the prompt.

Usage:
    python methods/finetuning/run_finetuning.py
    python methods/finetuning/run_finetuning.py --dry-run
    python methods/finetuning/run_finetuning.py --merged   # use merged_model/ instead
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import io_utils, model_utils
from methods.prompting.system_prompts import MINIMAL


def load_finetuned(use_merged: bool):
    from transformers import AutoModelForCausalLM
    if use_merged:
        if not os.path.isdir(config.MERGED_MODEL_DIR):
            sys.exit(f"{config.MERGED_MODEL_DIR} not found. Run merge_adapter.py first.")
        model = AutoModelForCausalLM.from_pretrained(
            config.MERGED_MODEL_DIR,
            quantization_config=model_utils.bnb_config(),
            device_map="auto",
            token=model_utils.load_env_token(),
        )
        return model
    # base + adapter (default, recommended)
    from peft import PeftModel
    if not os.path.isdir(config.ADAPTER_DIR):
        sys.exit(f"{config.ADAPTER_DIR} not found. Train first (train_qlora.py).")
    base = model_utils.load_base_model()
    return PeftModel.from_pretrained(base, config.ADAPTER_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merged", action="store_true", help="use merged_model/")
    args = ap.parse_args()

    model_utils.preflight_hf()
    model_utils.set_seeds()

    prompts = io_utils.load_test_prompts()
    if args.dry_run:
        prompts = prompts[:3]

    print("Loading fine-tuned model...")
    tokenizer = model_utils.load_tokenizer()
    model = load_finetuned(args.merged)

    records = []
    print(f"[finetuning] — {len(prompts)} prompts")
    for i, p in enumerate(prompts):
        text, secs = model_utils.generate(model, tokenizer, p["prompt"], MINIMAL)
        records.append(io_utils.make_record("finetuning", "qlora", p, text, secs))
        print(f"  [{i+1}/{len(prompts)}] {secs:.1f}s")

    io_utils.save_responses("finetuning_responses.json", records)
    print("\nNext: python s3_sync.py upload --dir outputs")


if __name__ == "__main__":
    main()
