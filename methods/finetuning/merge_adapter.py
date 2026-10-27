"""Optional — merge the LoRA adapter into the base model weights.

Produces a standalone merged model in merged_model/. This is OPTIONAL; the
default inference path (run_finetuning.py) loads base + adapter directly, which
keeps the artifact tiny. Use this only if you want a single portable model.

Note: merging requires loading the base model in a mergeable dtype (fp16/bf16),
not 4-bit, so this needs more VRAM/RAM than the quantized inference path.

Usage:
    python methods/finetuning/merge_adapter.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from common import model_utils


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    model_utils.preflight_hf()

    if not os.path.isdir(config.ADAPTER_DIR):
        sys.exit(f"{config.ADAPTER_DIR} not found. Train first.")

    print("Loading base model in bf16 for merging (not 4-bit)...")
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=model_utils.load_env_token(),
    )
    model = PeftModel.from_pretrained(base, config.ADAPTER_DIR)
    print("Merging adapter...")
    model = model.merge_and_unload()

    os.makedirs(config.MERGED_MODEL_DIR, exist_ok=True)
    model.save_pretrained(config.MERGED_MODEL_DIR)
    model_utils.load_tokenizer().save_pretrained(config.MERGED_MODEL_DIR)
    print(f"Merged model saved to {config.MERGED_MODEL_DIR}")


if __name__ == "__main__":
    main()
