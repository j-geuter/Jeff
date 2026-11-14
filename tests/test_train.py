"""Training (jeff.train) and export (jeff.export) on tiny random models (CPU, no downloads)."""
import json
import random
import shutil

import pytest
import torch
from safetensors.torch import load_file

pytest.importorskip("peft")                       # pip install ".[train]"

from jeff import export as E
from jeff import train as T
from jeff.head import DecisionHead
from jeff.prompting import encode_prompt

CONFIG = {"name": "tiny", "bidirectional": None,
          "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["q_proj", "v_proj", "gate_proj", "down_proj"]},
          "lr_lora": 1e-3, "lr_head": 1e-3, "batch_tokens": 1500, "grad_accum": 2, "max_tokens": 8192,
          "max_steps": 4, "warmup": 1, "decay_start": 2, "checkpoint_every": 2}


def bank_rows(n: int = 24) -> list[dict]:
    """Rows in DecisionBank's format: choice / yes-no / score questions, JSON and text states, some without a target."""
    rng, rows = random.Random(0), []
    for i in range(n):
        qtype = ("choice", "noul", "score")[i % 3]
        k = {"choice": 2 + i % 5, "noul": 2, "score": 5}[qtype]
        state = {"ticket": i, "text": "x" * rng.randint(5, 120)} if i % 2 else f"note {i}: " + "y" * rng.randint(5, 120)
        target = None if i % 4 == 0 else [rng.random() + 0.01 for _ in range(k)]
        rows.append({"id": f"r{i}", "state": json.dumps(state) if i % 2 else state, "state_is_json": bool(i % 2),
                     "question": f"Question {i}?", "qtype": qtype, "label": i % k, "target_distribution": target,
                     "options": [{"id": f"o{j}", "description": f"option {j}"} for j in range(k)]})
    return rows


@pytest.fixture
def bank(tmp_path):
    folder = tmp_path / "bank"
    folder.mkdir()
    rows = bank_rows()
    for shard in range(2):                       # two shards, read in name order
        with open(folder / f"train-{shard:05d}-of-00002.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows[shard * 12:(shard + 1) * 12])
    return folder


def test_rows_keep_order_and_parse_json_states(bank):
    rows = T.load_rows(str(bank))
    assert [r["id"] for r in rows] == [f"r{i}" for i in range(24)]
    assert rows[1]["state"] == {"ticket": 1, "text": rows[1]["state"]["text"]} and isinstance(rows[0]["state"], str)


def test_shuffled_options_carry_label_and_target():
    rng = random.Random(1)
    row = bank_rows()[1]                          # a choice row with a target
    for _ in range(20):
        out = T.shuffled(row, rng)
        assert out["options"][out["label"]] == row["options"][row["label"]]
        for option, p in zip(out["options"], out["target_distribution"]):
            assert p == row["target_distribution"][row["options"].index(option)]
    score = bank_rows()[2]
    assert T.shuffled(score, rng) is score        # ordered levels stay in order


def test_targets(tiny):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tiny("gemma", reasoning=False)[0])
    rows = bank_rows()
    soft = T.encode(tokenizer, rows[1], 8192)
    dist = torch.tensor(rows[1]["target_distribution"])
    torch.testing.assert_close(soft["t"][: soft["k"]], dist / dist.sum())
    assert soft["t"][soft["k"]:].sum() == 0 and soft["ids"] == encode_prompt(tokenizer, rows[1], 8192)
    hard = T.encode(tokenizer, rows[0], 8192)     # no target: the label
    assert hard["t"].tolist() == [float(j == rows[0]["label"]) for j in range(16)]
    assert T.encode(tokenizer, rows[1], 10) is None   # too long: left out


def test_length_batches():
    rng = random.Random(0)
    items = [{"ids": [0] * rng.randint(1, 300)} for _ in range(500)]
    batches = T.length_batches(items, 1000, random.Random(3), chunk=4)
    assert sorted(i for b in batches for i in b) == list(range(500))
    assert all(len(b) == 1 or max(len(items[i]["ids"]) for i in b) * len(b) <= 1000 for b in batches)
    assert batches == T.length_batches(items, 1000, random.Random(3), chunk=4)


def test_lr_schedule():
    config = {"warmup": 250, "decay_start": 1400, "max_steps": 1500}
    assert T.lr_scale(1, config) == pytest.approx(1 / 250)
    assert T.lr_scale(250, config) == T.lr_scale(1400, config) == 1.0
    assert T.lr_scale(1450, config) == pytest.approx(0.5)
    assert T.lr_scale(1500, config) == 1e-3       # floor
    with pytest.raises(ValueError):
        T.lr_scale(1, {"warmup": 10, "decay_start": 5, "max_steps": 20})


def test_soft_ce_ignores_masked_slots():
    logits = torch.tensor([[2.0, 0.0, float("-inf")]])
    target = torch.tensor([[0.25, 0.75, 0.0]])
    expected = -(0.25 * torch.log_softmax(logits[:, :2], -1)[0, 0] + 0.75 * torch.log_softmax(logits[:, :2], -1)[0, 1])
    torch.testing.assert_close(T.soft_ce(logits, target), expected)


@pytest.mark.parametrize("kind", ["gemma", "qwen"])
def test_head_starts_at_the_letter_readout(tiny, kind):
    model, tokenizer = T.load_base(str(tiny(kind, reasoning=False)[0]), None, "cpu")
    head = DecisionHead(model.config.hidden_size, 16, T.softcap_of(model))
    with torch.no_grad():
        head.base_rows.copy_(T.letter_rows(model, tokenizer))
        head.delta.weight.zero_()
        head.delta.bias.zero_()
        item = T.encode(tokenizer, bank_rows()[1], 8192)
        h = T.last_hidden(T.decoder_of(model), [item], tokenizer.pad_token_id, "cpu")
        logits = T.option_logits(head, h, torch.tensor([item["k"]]))
        lm = model(input_ids=torch.tensor([item["ids"]])).logits[0, -1].float()
    letters = [tokenizer.convert_tokens_to_ids(c) for c in "ABCDEFGHIJKLMNOP"]
    torch.testing.assert_close(logits[0, : item["k"]], lm[letters][: item["k"]], atol=0.1, rtol=0.02)  # bf16 LM head
    assert torch.isinf(logits[0, item["k"]:]).all()


def run(tiny, bank, out, seed=1, kind="gemma", **changes):
    folder = tiny(kind, reasoning=False)[0]
    config = {**CONFIG, "model": {"source": str(folder)}, **changes}
    return T.train(config, seed, out, str(bank))


def weights(out):
    return {**load_file(str(out / "adapter" / "adapter_model.safetensors")), **load_file(str(out / "head.safetensors"))}


def test_resume_continues_exactly(tiny, bank, tmp_path):
    whole = weights(run(tiny, bank, tmp_path / "whole"))
    out = run(tiny, bank, tmp_path / "split")
    assert sorted(p.name for p in (out / "state").iterdir()) == ["step-000002", "step-000004"]   # the last two
    shutil.rmtree(out / "state" / "step-000004")          # as if stopped after the state of step 2
    shutil.rmtree(out / "adapter")
    (out / "head.safetensors").unlink()
    resumed = weights(run(tiny, bank, out))
    assert whole.keys() == resumed.keys()
    assert all(torch.equal(whole[k], resumed[k]) for k in whole)
    assert whole["delta.weight"].abs().sum() > 0                  # training moved the head


def test_resume_refuses_another_config(tiny, bank, tmp_path):
    run(tiny, bank, tmp_path / "r")
    with pytest.raises(ValueError, match="another configuration"):
        run(tiny, bank, tmp_path / "r", lr_head=2e-3)


@pytest.mark.parametrize("kind", ["gemma", "qwen"])
def test_export_averages_runs_and_serves(tiny, bank, tmp_path, kind):
    from jeff.model import JeffModel
    runs = [run(tiny, bank, tmp_path / f"s{seed}", seed=seed, kind=kind, max_steps=3,
                bidirectional="full" if kind == "qwen" else None) for seed in (1, 2, 3)]
    out = E.export("Jeff-1.1-Large", runs, tmp_path / "model", device="cpu")
    # the averaged adapter adds the mean of the runs' weight changes
    adapter = load_file(str(out / "adapter" / "adapter_model.safetensors"))
    members = [load_file(str(r / "adapter" / "adapter_model.safetensors")) for r in runs]
    name = next(k for k in adapter if k.endswith("lora_A.weight"))
    b = name.replace("lora_A", "lora_B")
    mean = sum(m[b].float() @ m[name].float() for m in members) / 3
    torch.testing.assert_close(adapter[b] @ adapter[name], mean, atol=1e-6, rtol=1e-5)
    assert json.loads((out / "adapter" / "adapter_config.json").read_text())["r"] == 12
    # the head is the mean of the runs' heads
    head = load_file(str(out / "head.safetensors"))
    heads = [load_file(str(r / "head.safetensors")) for r in runs]
    assert all(torch.equal(head[k], sum(h[k] for h in heads) / 3) for k in head)
    assert json.loads((out / "serving_rule.json").read_text()) == E.SERVING_RULES["Jeff-1.1-Large"]
    # the folder serves, and its answer is the averaged head on the merged model
    model = JeffModel(out, device="cpu", commit=False)
    assert model.decider.bidirectional == ("full" if kind == "qwen" else None)
    row = {**bank_rows()[3], "state": {"ticket": 3}}
    probs = model.decider.decide(row)[0]
    merged, tokenizer = T.load_base(str(out), None, "cpu")
    decoder = T.decoder_of(merged)
    if kind == "qwen":
        from jeff import bidir
        bidir.enable(decoder, "full")
    with torch.no_grad():
        h = T.last_hidden(decoder, [T.encode(tokenizer, row, 8192)], tokenizer.pad_token_id, "cpu")
        reference = torch.softmax(model.decider.head(h)[0, : len(row["options"])], -1)
    torch.testing.assert_close(torch.tensor(probs, dtype=torch.float32), reference, atol=1e-4, rtol=1e-3)
    # reasoning: the released settings, and a question that reasons (share 1: every question)
    assert (out / "reasoning.json").read_bytes() == (E.RELEASED / "Jeff-1.1-Large.reasoning.json").read_bytes()
    thinking = JeffModel(out, device="cpu", reasoning=True, think_share=1.0, think_budget=4,
                         reasoning_engine="transformers")
    thinking.decider.decide(row)
    assert thinking.decider.last["thought"] and thinking.decider.last["output_tokens"] > 0


def test_released_reasoning_settings_for_every_model():
    from jeff.reasoning import TYPES
    for name in E.SERVING_RULES:
        config = json.loads((E.RELEASED / f"{name}.reasoning.json").read_text())
        assert all(len(config["tau_table"][q]) == 101 for q in TYPES) and config["budget"] in (256, 512)


def test_export_needs_a_released_name(tmp_path):
    with pytest.raises(ValueError, match="released temperatures"):
        E.export("Jeff-9", [tmp_path], tmp_path / "out", device="cpu")
