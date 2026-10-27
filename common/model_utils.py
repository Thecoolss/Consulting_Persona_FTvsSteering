"""Shared model/tokenizer loading and generation helpers (GPU/EC2 side).

Imported by the prompting, finetuning, and steering methods so quantization and
generation config stay identical across all of them (fair comparison).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def set_seeds() -> None:
    import torch
    import transformers
    torch.manual_seed(config.SEED)
    transformers.set_seed(config.SEED)


def load_env_token() -> str | None:
    """Load HF_TOKEN from .env (or environment) so gated Llama can be pulled."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(config.PROJECT_ROOT, ".env"))
    return os.environ.get("HF_TOKEN")


def preflight_hf() -> None:
    """Fail fast with a clear message if the HF token / Llama access is missing.

    Avoids discovering a gating problem minutes into a GPU run.
    """
    token = load_env_token()
    if not token:
        sys.exit("HF_TOKEN not found in .env — cannot download the gated Llama model.")
    try:
        from huggingface_hub import model_info
        info = model_info(config.BASE_MODEL, token=token)
        print(f"[preflight] HF access OK: {info.id}")
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"[preflight] Cannot access {config.BASE_MODEL} with this token: {exc}\n"
            "Accept the Llama 3.1 license on HF and check the token."
        )


def bnb_config():
    """4-bit NF4 quantization config shared by all methods."""
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.QUANT_TYPE,
        bnb_4bit_use_double_quant=config.DOUBLE_QUANT,
        bnb_4bit_compute_dtype=torch.bfloat16 if config.BF16 else torch.float16,
    )


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.BASE_MODEL, token=load_env_token())
    # Llama tokenizers have no pad token by default.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(for_training: bool = False):
    """Load Llama 3.1 8B in 4-bit. Used by every method for parity."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        quantization_config=bnb_config(),
        device_map="auto",
        token=load_env_token(),
    )
    if for_training:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    return model


def build_chat_inputs(tokenizer, user_msg: str, system_msg: str | None):
    """Apply Llama 3.1 chat template to a (system?, user) turn."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )


def generate(model, tokenizer, user_msg: str, system_msg: str | None = None) -> tuple[str, float]:
    """Generate one response with the SHARED greedy config. Returns (text, seconds)."""
    import torch
    input_ids = build_chat_inputs(tokenizer, user_msg, system_msg).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
            do_sample=config.GEN_DO_SAMPLE,
            temperature=config.GEN_TEMPERATURE if config.GEN_DO_SAMPLE else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - t0
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip(), elapsed
