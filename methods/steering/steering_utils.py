"""Activation-steering utilities: capture activations, build steering vectors,
and apply them at inference via forward hooks.

Design choice — CAPTURE STRATEGY (config.STEERING_CAPTURE):
  "mean" (default): average the residual-stream activation over all RESPONSE
      tokens of each example. This spreads the persona signal across the whole
      answer and gives a smoother, more stable steering direction — better for a
      diffuse stylistic trait like "consultant voice".
  "last": use only the final token's activation. Simpler but noisier, since the
      direction is estimated from a single point per example.

We hook the residual stream at model.model.layers[i] (the block output).
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config


def _decoder_layers(model):
    """Find the list of decoder blocks, handling PEFT/quantization wrappers.

    Walks `.model`/`.base_model` attributes until it finds `.layers`.
    """
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


def capture_activations(model, tokenizer, texts, layers, capture=None):
    """Run each text through the model and collect mean activations per layer.

    Returns: dict[layer_idx] -> tensor[hidden] (mean over examples of the per-example
    aggregated activation).
    """
    import torch

    capture = capture or config.STEERING_CAPTURE
    sums = {l: None for l in layers}
    count = 0

    captured = {}

    def make_hook(layer_idx):
        def hook(_module, _inp, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()
        return hook

    handles = [_layer_module(model, l).register_forward_hook(make_hook(l)) for l in layers]
    try:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=config.MAX_SEQ_LENGTH).to(model.device)
            with torch.no_grad():
                model(**ids)
            for l in layers:
                hs = captured[l][0]  # [seq, hidden]
                if capture == "last":
                    vec = hs[-1]
                else:  # mean over tokens
                    vec = hs.mean(dim=0)
                vec = vec.float()
                sums[l] = vec if sums[l] is None else sums[l] + vec
            count += 1
    finally:
        for h in handles:
            h.remove()

    return {l: (sums[l] / max(count, 1)) for l in layers}


def build_steering_vectors(model, tokenizer, positives, negatives, layers):
    """steering_vector[l] = normalize(mean_pos[l] - mean_neg[l])."""
    import torch
    pos = capture_activations(model, tokenizer, positives, layers)
    neg = capture_activations(model, tokenizer, negatives, layers)
    vectors = {}
    for l in layers:
        diff = pos[l] - neg[l]
        norm = diff.norm()
        vectors[l] = (diff / norm) if norm > 0 else diff
    return vectors


@contextmanager
def steering(model, vector, layer_idx: int, scale: float):
    """Context manager: add (scale * vector) to the residual stream at a layer."""
    import torch

    v = vector.to(model.device)

    def hook(_module, _inp, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + scale * v.to(hs.dtype)
            return (hs,) + tuple(output[1:])
        return output + scale * v.to(output.dtype)

    handle = _layer_module(model, layer_idx).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def save_vectors(vectors, directory=None):
    import torch
    directory = directory or config.STEERING_VECTORS_DIR
    os.makedirs(directory, exist_ok=True)
    for l, v in vectors.items():
        torch.save(v.cpu(), os.path.join(directory, f"layer_{l}.pt"))


def load_vector(layer_idx: int, directory=None):
    import torch
    directory = directory or config.STEERING_VECTORS_DIR
    return torch.load(os.path.join(directory, f"layer_{layer_idx}.pt"))
