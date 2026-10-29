"""Activation-steering utilities: capture activations, build steering vectors,
and apply them at inference via forward hooks.

CAPTURE (config.STEERING_CAPTURE):
  Each contrastive example is rendered with the Llama 3.1 chat template
  (user = instruction, assistant = response), and activations are read ONLY at
  the assistant-response token positions.
  "mean" (default): average over the response tokens. Spreads the persona
      signal across the whole answer, which suits a diffuse stylistic trait.
  "last": the final response token only. Simpler but noisier.

Why response-only matters: the first token (<|begin_of_text|>) carries a huge
activation (an "attention sink", norm ~500 vs ~20 elsewhere). Averaging over all
tokens lets that outlier leak into the vector, weighted by sequence length, so
the vector picks up "answer length" instead of "consultant style".

STEERING: the hook adds scale * unit_vector to the residual stream at one layer,
at every position EXCEPT position 0 during prefill, so the attention-sink token
is never perturbed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config


def _decoder_layers(model):
    """Find the list of decoder blocks, handling PEFT/quantization wrappers."""
    node = model
    for _ in range(6):
        if hasattr(node, "layers"):
            return node.layers
        if hasattr(node, "model"):
            node = node.model
        elif hasattr(node, "base_model"):
            node = node.base_model
        else:
            break
    raise AttributeError("Could not locate decoder layers on this model.")


def _layer_module(model, layer_idx: int):
    """Return the decoder block whose OUTPUT is the residual stream at this layer."""
    return _decoder_layers(model)[layer_idx]


def _format_pair(tokenizer, instruction: str, response: str):
    """Chat-format one example. Returns (input_ids [1, seq], n_prompt_tokens)."""
    user = [{"role": "user", "content": instruction}]
    prompt_text = tokenizer.apply_chat_template(
        user, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": response}], tokenize=False)
    n_prompt = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt",
                    truncation=True, max_length=config.MAX_SEQ_LENGTH)["input_ids"]
    return ids, n_prompt


def capture_activations(model, tokenizer, pairs, layers, capture=None):
    """Mean response-token activation per layer over all (instruction, response) pairs.

    Returns (means, norms): means[l] is a [hidden] tensor; norms[l] is the average
    per-token residual-stream norm over response tokens (for scale reference).
    """
    import torch

    capture = capture or config.STEERING_CAPTURE
    sums = {l: None for l in layers}
    norm_sums = {l: 0.0 for l in layers}
    count = 0
    captured = {}

    def make_hook(layer_idx):
        def hook(_module, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()
        return hook

    handles = [_layer_module(model, l).register_forward_hook(make_hook(l)) for l in layers]
    try:
        for instruction, response in pairs:
            ids, n_prompt = _format_pair(tokenizer, instruction, response)
            ids = ids.to(model.device)
            with torch.no_grad():
                model(input_ids=ids)
            start = min(n_prompt, ids.shape[1] - 1)
            for l in layers:
                resp = captured[l][0, start:].float()  # [resp_tokens, hidden]
                vec = resp[-1] if capture == "last" else resp.mean(dim=0)
                sums[l] = vec if sums[l] is None else sums[l] + vec
                norm_sums[l] += resp.norm(dim=-1).mean().item()
            count += 1
    finally:
        for h in handles:
            h.remove()

    n = max(count, 1)
    return ({l: sums[l] / n for l in layers}, {l: norm_sums[l] / n for l in layers})


def build_steering_vectors(model, tokenizer, positives, negatives, layers):
    """vector[l] = normalize(mean_pos[l] - mean_neg[l]). Also returns mean norms."""
    pos, pos_norms = capture_activations(model, tokenizer, positives, layers)
    neg, neg_norms = capture_activations(model, tokenizer, negatives, layers)
    vectors, norms = {}, {}
    for l in layers:
        diff = pos[l] - neg[l]
        length = diff.norm()
        vectors[l] = (diff / length) if length > 0 else diff
        norms[l] = {
            "mean_resid_norm": (pos_norms[l] + neg_norms[l]) / 2,
            "raw_diff_norm": float(length),
        }
    return vectors, norms


@contextmanager
def steering(model, vector, layer_idx: int, scale: float):
    """Add (scale * vector) to the residual stream at one layer, sparing position 0."""
    v = vector.to(model.device)

    def hook(_module, _inp, output):
        hs = output[0] if isinstance(output, tuple) else output
        add = (scale * v).to(hs.dtype)
        if hs.shape[1] > 1:      # prefill: leave the attention-sink token alone
            hs = hs.clone()
            hs[:, 1:, :] += add
        else:                    # decode step: one new token
            hs = hs + add
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    handle = _layer_module(model, layer_idx).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def multi_steering(model, vectors: dict, strengths: dict, preserve_norm: bool = False):
    """Steering v2: add strengths[l] * vectors[l] at several layers at once.

    strengths are absolute (already multiplied by the layer's typical activation
    norm by the caller), so a smaller push at each of several layers can add up
    without any single layer being pushed off-distribution.

    preserve_norm: rescale each steered token back to its pre-steering norm.
    This keeps activation magnitudes in the range the model was trained on,
    which is the main cause of the repetitive collapse seen with v1.
    Position 0 (attention sink) is never steered.
    """
    import torch

    handles = []

    def make_hook(v, s):
        def hook(_module, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            add = (s * v).to(hs.dtype)
            start = 1 if hs.shape[1] > 1 else 0
            seg = hs[:, start:, :]
            new = seg + add
            if preserve_norm:
                old_n = seg.float().norm(dim=-1, keepdim=True)
                new_n = new.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                new = (new.float() * (old_n / new_n)).to(hs.dtype)
            if start:
                hs = torch.cat([hs[:, :1, :], new], dim=1)
            else:
                hs = new
            if isinstance(output, tuple):
                return (hs,) + tuple(output[1:])
            return hs
        return hook

    try:
        for layer, v in vectors.items():
            handles.append(_layer_module(model, layer).register_forward_hook(
                make_hook(v.to(model.device), strengths[layer])))
        yield
    finally:
        for h in handles:
            h.remove()


def load_vector_set(layers, directory) -> dict:
    return {l: load_vector(l, directory) for l in layers}


def load_norms(directory) -> dict:
    with open(os.path.join(directory, "norms.json"), encoding="utf-8") as fh:
        return {int(k): v["mean_resid_norm"] for k, v in json.load(fh).items()}


def looks_degenerate(text: str) -> bool:
    """Crude check for broken generations: too short or highly repetitive."""
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 25:
        return True
    return len({w.lower() for w in words}) / len(words) < 0.3


def save_vectors(vectors, directory=None, norms=None):
    import torch
    directory = directory or config.STEERING_VECTORS_DIR
    os.makedirs(directory, exist_ok=True)
    for l, v in vectors.items():
        torch.save(v.cpu(), os.path.join(directory, f"layer_{l}.pt"))
    if norms is not None:
        with open(os.path.join(directory, "norms.json"), "w") as fh:
            json.dump({str(k): v for k, v in norms.items()}, fh, indent=2)


def load_vector(layer_idx: int, directory=None):
    import torch
    directory = directory or config.STEERING_VECTORS_DIR
    return torch.load(os.path.join(directory, f"layer_{layer_idx}.pt"))
