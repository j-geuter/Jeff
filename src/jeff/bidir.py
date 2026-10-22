"""Bidirectional attention (Jeff 1.1 models whose jeff_config.json has "bidirectional": "full").

The decision is read at the last prompt token and nothing is generated, so the prompt does not have to be read
causally. These models were trained with their full-attention layers reading the prompt in both directions: every
token, including the state, can attend to the question and the option list after it. Gemma 4: the full-attention
layers attend bidirectionally (keys masked only for padding); the sliding-window layers keep their causal sliding mask
(mode "full"); mode "all" also opens the sliding-window layers within their window. Qwen3.5 / 3.6: mode "full" opens
the full-attention layers (one in four); the Gated DeltaNet linear-attention layers are a recurrence and stay causal.
The reasoning (jeff.reasoning) is written by the base model with causal attention (causal()).

    enable(transformer, "full")      # once, after loading: marks the layers, records the mode
    out = transformer(input_ids=ids, attention_mask=attention_arg(transformer, ids, mask), use_cache=False)
"""
from __future__ import annotations

import contextlib

import torch

MODES = (None, "full", "all")


def _check(mode) -> None:
    if mode not in MODES:
        raise ValueError(f"bidirectional must be one of {MODES}, got {mode!r}")


def enable(transformer, mode) -> int:
    """Mark the full-attention layers non-causal; returns how many layers were changed (0 without a mode)."""
    _check(mode)
    if not mode:
        return 0
    types = getattr(transformer.config, "layer_types", None)
    if not types:
        raise ValueError("bidirectional attention needs a model with config.layer_types (Gemma 4)")
    if mode == "all" and "linear_attention" in types:
        raise ValueError("bidirectional='all' is not defined for linear-attention layers (Qwen3.5): use 'full'")
    changed = 0
    for layer, kind in zip(transformer.layers, types):
        if kind == "full_attention" or (mode == "all" and kind == "sliding_attention"):
            layer.self_attn.is_causal = False
            changed += 1
    if not changed:
        raise ValueError("no full-attention layer found")
    transformer._jev_bidirectional = mode
    return changed


@contextlib.contextmanager
def causal(transformer):
    """Temporarily causal again (the base model writes the reasoning); the bidirectional layers and mode are
    restored afterwards."""
    mode = mode_of(transformer)
    if not mode:
        yield
        return
    changed = [layer for layer in transformer.layers                  # Qwen3.5's linear-attention layers have no self_attn
               if getattr(getattr(layer, "self_attn", None), "is_causal", True) is False]
    for layer in changed:
        layer.self_attn.is_causal = True
    transformer._jev_bidirectional = None
    try:
        yield
    finally:
        for layer in changed:
            layer.self_attn.is_causal = False
        transformer._jev_bidirectional = mode


def mode_of(transformer):
    return getattr(transformer, "_jev_bidirectional", None)


def attention_arg(transformer, input_ids: torch.Tensor, attention: torch.Tensor, mode="model"):
    """The attention_mask argument for the decoder stack: the 2D padding mask (no mode), or a dict per layer type.
    mode "model" (default) uses the mode recorded by enable()."""
    mode = mode_of(transformer) if mode == "model" else mode
    _check(mode)
    if not mode:
        return attention
    from transformers.masking_utils import create_bidirectional_mask, create_sliding_window_causal_mask
    config = transformer.config
    batch, width = input_ids.shape
    stub = torch.empty((batch, width, 1), dtype=transformer.dtype, device=input_ids.device)   # shape / dtype / device only
    position_ids = torch.arange(width, device=input_ids.device).unsqueeze(0)
    out = {}
    for kind in set(config.layer_types):
        if kind == "full_attention":
            out[kind] = create_bidirectional_mask(config=config, inputs_embeds=stub, attention_mask=attention)
        elif kind == "linear_attention":     # Qwen3.5 Gated DeltaNet layers: a causal recurrence, padding mask only
            from transformers.masking_utils import create_recurrent_attention_mask
            out[kind] = create_recurrent_attention_mask(config=config, inputs_embeds=stub, attention_mask=attention)
        elif kind == "sliding_attention" and mode == "all":
            window = int(config.sliding_window)
            out[kind] = create_bidirectional_mask(config=config, inputs_embeds=stub, attention_mask=attention,
                                                  and_mask_function=lambda b, h, q, kv: (q - kv).abs() < window,
                                                  allow_is_bidirectional_skip=False)
        elif kind == "sliding_attention":
            out[kind] = create_sliding_window_causal_mask(config=config, inputs_embeds=stub, attention_mask=attention,
                                                          past_key_values=None, position_ids=position_ids)
        else:
            raise ValueError(f"unsupported layer type {kind}")
    return out
