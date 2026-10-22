"""The Jeff decision head: one logit per answer option from the language model's last hidden state.

    z = c * tanh(W0 h / c) + A std(h) + b,        std(h) = (h - mu) / sigma

h is the post-final-norm hidden state at the last prompt position (what the LM head reads). W0 are the LM-head rows of
the answer letters A-P, so the first term is the model's own letter readout. c is the LM's final-logit soft-capping
(Gemma 4: 30); Qwen models have none, and the first term is then W0 h. mu and sigma were computed once on training
data. The correction A, b was learned; it started at zero, so training started exactly at the letter readout. Slot j
belongs to the option labeled LETTERS[j]; a question with k options uses the first k slots.

Release format: head_config.json and head.safetensors, with fp32 tensors base_rows (W0), delta.weight (A), delta.bias
(b), mu and sigma. The released files also hold an abstain unit (abstain_net.*), used only in training; it never
enters the option probabilities and is not loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from .prompting import MAX_OPTIONS


class DecisionHead(nn.Module):
    def __init__(self, d_model: int, k_slots: int = MAX_OPTIONS, softcap: float | None = None):
        super().__init__()
        self.d_model, self.k_slots = d_model, k_slots
        self.softcap = float(softcap) if softcap else None
        self.register_buffer("base_rows", torch.zeros(k_slots, d_model))
        self.delta = nn.Linear(d_model, k_slots, bias=True)
        self.register_buffer("mu", torch.zeros(d_model))
        self.register_buffer("sigma", torch.ones(d_model))

    def cap(self, z: torch.Tensor) -> torch.Tensor:
        """The LM's final-logit soft-capping (identity without one)."""
        return self.softcap * torch.tanh(z / self.softcap) if self.softcap is not None else z

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [B, d_model] -> logits [B, k_slots]; a question with k options reads the first k."""
        return self.cap(h @ self.base_rows.T) + self.delta((h - self.mu) / self.sigma)


def load_head(folder) -> DecisionHead:
    """The decision head of a released model folder, in fp32."""
    from safetensors.torch import load_file
    folder = Path(folder)
    config = json.loads((folder / "head_config.json").read_text())
    if (config.get("kind") != "residual" or config.get("extra_dim") or config.get("extra_layers")
            or config.get("k_slots") != MAX_OPTIONS):
        raise ValueError(f"{folder}: unsupported decision head {config}")
    head = DecisionHead(config["d_model"], config["k_slots"], config.get("softcap"))
    tensors = load_file(str(folder / "head.safetensors"))
    head.load_state_dict({k: v for k, v in tensors.items() if not k.startswith("abstain_net.")})  # strict
    return head.float()
