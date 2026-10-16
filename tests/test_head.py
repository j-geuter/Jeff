"""The decision head: its formula and the release format."""
import json

import pytest
import torch
from safetensors.torch import save_file

from jeff.head import DecisionHead, load_head

D = 8


def random_head(softcap=None) -> DecisionHead:
    torch.manual_seed(0)
    head = DecisionHead(D, softcap=softcap)
    with torch.no_grad():
        head.base_rows.normal_()
        head.delta.weight.normal_()
        head.delta.bias.normal_()
        head.mu.normal_()
        head.sigma.uniform_(0.5, 2.0)
    return head


@pytest.mark.parametrize("softcap", [None, 30.0, 2.0])
def test_formula(softcap):
    head, h = random_head(softcap), torch.randn(3, D)
    readout = h @ head.base_rows.T
    if softcap:
        readout = softcap * torch.tanh(readout / softcap)
    correction = ((h - head.mu) / head.sigma) @ head.delta.weight.T + head.delta.bias
    torch.testing.assert_close(head(h), readout + correction)


def test_zero_correction_is_the_letter_readout():
    head, h = DecisionHead(D), torch.randn(2, D)
    with torch.no_grad():
        head.base_rows.normal_()
        head.delta.weight.zero_()
        head.delta.bias.zero_()
    torch.testing.assert_close(head(h), h @ head.base_rows.T)


def save_release(folder, head: DecisionHead, config: dict | None = None, drop: set = frozenset()) -> None:
    """head_config.json + head.safetensors as in the released models, including the unused abstain unit."""
    config = {"d_model": D, "k_slots": 16, "abstain": True, "kind": "residual", "extra_dim": 0, "extra_layers": [],
              "logit_stats": False, "t_min": 0.05, **({"softcap": head.softcap} if head.softcap else {}), **(config or {})}
    (folder / "head_config.json").write_text(json.dumps(config))
    tensors = {k: v.contiguous() for k, v in head.state_dict().items() if k not in drop}
    tensors.update({"abstain_net.weight": torch.zeros(1, D), "abstain_net.bias": torch.full((1,), -3.0)})
    save_file(tensors, str(folder / "head.safetensors"))


def test_load_release_format(tmp_path):
    head = random_head(30.0)
    save_release(tmp_path, head)
    loaded, h = load_head(tmp_path), torch.randn(2, D)
    assert loaded.softcap == 30.0
    torch.testing.assert_close(loaded(h), head(h), rtol=0, atol=0)


def test_missing_tensor_is_an_error(tmp_path):
    save_release(tmp_path, random_head(), drop={"mu"})
    with pytest.raises(RuntimeError, match="mu"):
        load_head(tmp_path)


@pytest.mark.parametrize("change", [{"kind": "linear"}, {"kind": "temperature"}, {"extra_dim": 4}, {"k_slots": 8}])
def test_other_heads_are_refused(tmp_path, change):
    save_release(tmp_path, random_head(), change)
    with pytest.raises(ValueError, match="unsupported"):
        load_head(tmp_path)
