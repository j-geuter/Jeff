"""Jeff 1.1 serving on tiny random models (CPU, see conftest.py): the base model behind the adapter, bidirectional
attention, reasoning when unsure and its limits, questions with many options and long inputs; Jeff 1.0 folders keep
their one-pass answers."""
import json
import threading
import urllib.request
from collections import deque
from http.server import HTTPServer

import pytest
import torch

from jeff import reasoning as R
from jeff.model import JeffModel, split_groups
from jeff.prompting import encode_prompt
from jeff.serve import make_handler

STATE = {"ticket": "Refund not received after 14 days", "customer": "J. Doe"}
REQUEST = {"route": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Payments and refunds", "tech": "Technical problems", "sales": "Orders"}},
           "urgent": {"type": "noul", "instructions": "Is this urgent?"},
           "severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["low", "mid", "high"]}}


def load(folder, **options) -> JeffModel:
    return JeffModel(folder, device="cpu", **options)


def one_pass(model: JeffModel, row: dict) -> list[float]:
    """The one-pass answer computed by hand: hidden state at the last prompt token -> head -> softmax."""
    ids = encode_prompt(model.decider.tokenizer, row, model.decider.max_tokens)
    with torch.no_grad():
        h = model.decider.hidden(ids).to(torch.bfloat16).float()
        return torch.softmax(model.decider.head(h)[0, : len(row["options"])], dim=-1).tolist()


def test_one_pass_is_unchanged_and_reasoning_is_off_by_default(tiny):
    folder, _ = tiny("gemma")
    model = load(folder)
    assert model.decider.reasoner is None and model.reasoning is None and model.decider.bidirectional is None
    for key, question in REQUEST.items():
        row = model.row(STATE, question, key)
        probs, _, n, _ = model.decider.decide(row)
        assert probs == pytest.approx(one_pass(model, row), abs=1e-6) and n > 0
        assert model.decider.last == {"output_tokens": 0, "thought": False, "shortened": False}


def test_settings_shares_budget_and_limits(tiny):
    folder, _ = tiny("gemma")
    table = json.loads((folder / "reasoning.json").read_text())["tau_table"]["noul"]
    s = R.settings(folder)
    assert s["tau"] == {"choice": -1.0, "noul": table[20], "score": table[60]} and s["budget"] == 4
    assert all(s[key] is None for key in R.LIMITS)
    s = R.settings(folder, jevbench_limits=True)
    assert (s["max_output_per_decision"], s["max_share"], s["request_seconds"]) == (2.0, 0.45, 30.0)
    s = R.settings(folder, jevbench_limits=True, max_output_per_decision=5, budget=8)
    assert s["max_output_per_decision"] == 5 and s["budget"] == 8 and s["max_share"] == 0.45
    assert R.settings(folder, request_seconds=3)["request_seconds"] == 3
    assert R.settings(folder, share=1.0)["tau"] == {"choice": 2.0, "noul": 2.0, "score": 2.0}
    s = R.settings(folder, share={"noul": 0.0, "choice": 0.5})
    assert s["tau"] == {"choice": table[50], "noul": -1.0, "score": table[60]}
    for bad in ({"share": 1.5}, {"share": {"rank": 0.1}}, {"budget": 0}):
        with pytest.raises(ValueError):
            R.settings(folder, **bad)


def test_models_without_reasoning_settings_refuse_reasoning(tiny):
    folder, _ = tiny("gemma", reasoning=False)
    with pytest.raises(ValueError, match="no reasoning settings"):
        load(folder, reasoning=True)
    assert load(folder).decide(STATE, REQUEST)["route"]["choice"] in REQUEST["route"]["criteria"]


@pytest.mark.parametrize("kind", ["gemma", "qwen"])
def test_adapter_hooks_give_back_the_base_model(tiny, kind):
    """Released weights bf16(W + s B A) minus the adapter's term: each adapted layer is the base layer up to bfloat16
    rounding, and the whole model is much closer to the base model than the released one is."""
    folder, info = tiny(kind)
    model = load(folder, reasoning=True)
    delta = model.decider.reasoner.delta
    assert delta.layers == len(info["lora"]) and delta.scale == info["scale"]
    released, base = dict(model.decider.model.named_modules()), dict(info["base"].named_modules())
    torch.manual_seed(0)
    for name in list(info["lora"])[:6]:
        x = torch.randn(3, base[name].in_features, dtype=torch.bfloat16)
        with torch.no_grad(), delta.base():
            off = released[name](x).float()
        on = released[name](x).float()
        want = base[name](x).float()
        assert torch.allclose(off, want, atol=0.05, rtol=0.02)
        assert (on - want).abs().max() > 10 * (off - want).abs().max()
    ids = torch.randint(10, 90, (1, 24))
    with torch.no_grad():
        on = model.decider.model(ids).logits.float()
        with delta.base():
            off = model.decider.model(ids).logits.float()
        want = info["base"](ids).logits.float()
    assert (off - want).norm() < 0.2 * (on - want).norm()
    assert not delta.active


def test_bidirectional_models_read_both_ways_and_reason_causally(tiny):
    causal, _ = tiny("gemma")
    folder, _ = tiny("gemma", "full")                     # the same weights (same seed)
    a, b = load(causal), load(folder, reasoning=True, think_share=1.0)
    assert b.decider.bidirectional == "full"
    full = [layer for layer in b.decider.base.layers if layer.self_attn.is_causal is False]
    assert len(full) == 2                                 # the full-attention layers; sliding layers stay causal
    row = a.row(STATE, REQUEST["route"], "route")
    assert one_pass(a, row) != pytest.approx(one_pass(b, row), abs=1e-4)
    b.decider.decide(row)
    assert b.decider.last["thought"]
    assert all(layer.self_attn.is_causal is False for layer in full)      # restored after generating causally


@pytest.mark.parametrize("kind,bidirectional", [("gemma", None), ("gemma", "full"), ("qwen", "full")])
def test_reasoning_writes_reads_and_reports_tokens(tiny, kind, bidirectional):
    folder, _ = tiny(kind, bidirectional)
    model = load(folder, reasoning=True, think_share=1.0)
    for key, question in REQUEST.items():
        answer = model.decide(STATE, {key: question})[key]
        assert model.decider.last["thought"] and 0 <= model.decider.last["output_tokens"] <= 4
        if question["type"] == "noul":
            assert not 0.2 < answer["noul"] < 0.8        # the yes/no commitment still applies
        else:
            assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert model.decider.reasoner.served[0] == 3


def test_reasoned_answer_is_the_pooled_read_at_its_temperature(tiny):
    """The served distribution is softmax(z / think_t) with z the mean of the one-pass and read log-probabilities."""
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0)
    reasoner = model.decider.reasoner
    row = model.row(STATE, REQUEST["route"], "route")
    ids = encode_prompt(model.decider.tokenizer, row, 8192)
    first = one_pass(model, row)
    with torch.no_grad():
        probs, logits, _ = reasoner.think(ids, 3, "choice", [float(v) for v in torch.tensor(first).log()])
    reasoner.config["mix"] = False
    with torch.no_grad():
        read, _, _ = reasoner.think(ids, 3, "choice", [0.0, 0.0, 0.0])        # read alone (pooling off)
    think_t, rule_t = reasoner.config["temperatures"]["choice"], model.rule["choice_t"]
    z = (torch.tensor(read).log() * think_t / rule_t + torch.tensor(first).log()) / 2   # undo the scaling of `read`
    want = torch.softmax(z / think_t, dim=-1)
    served = model.answer("choice", ["billing", "tech", "sales"], probs)["probabilities"]
    assert list(served.values()) == pytest.approx(want.tolist(), abs=1e-5)


def test_share_zero_never_reasons(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=0.0)
    model.decide(STATE, REQUEST)
    assert model.decider.reasoner.served == [3, 0]


def test_cost_limit_bounds_mean_output_tokens(tiny):
    """The mean output per question over everything served never exceeds the limit, even when every question is
    unsure and every reasoning used its whole budget."""
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0, max_output_per_decision=1.0)
    assert model.reasoning["max_share"] is None
    thought = []
    for _ in range(12):
        model.decide(STATE, {"route": REQUEST["route"]})
        thought.append(model.decider.last["thought"])
        n, spent = model.decider.reasoner.served
        assert spent / n <= 1.0
    assert thought[:3] == [False] * 3 and thought[3]      # (0 + 4) / 4 = 1 is the first allowed


def test_jevbench_limits(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, jevbench_limits=True, think_share=1.0)
    assert model.reasoning["max_output_per_decision"] == 2.0 and model.reasoning["request_seconds"] == 30.0
    for _ in range(8):
        model.decide(STATE, {"route": REQUEST["route"]})
        n, spent = model.decider.reasoner.served
        assert spent / n <= 2.0
    assert model.decider.reasoner.deadline is not None


def test_share_limit_bounds_the_recent_share(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0)
    reasoner = model.decider.reasoner
    reasoner.config.update(max_share=0.5, warmup=0)
    reasoner._recent = deque(maxlen=4)
    thought = []
    for _ in range(12):
        model.decide(STATE, {"route": REQUEST["route"]})
        thought.append(model.decider.last["thought"])
    assert thought[0] and 4 <= sum(thought) <= 7


def test_request_time_limit_stops_reasoning(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0, request_seconds=0.0)
    model.decide(STATE, REQUEST)
    assert model.decider.reasoner.served == [3, 0]


def test_reasoning_needs_the_question_type(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True)
    row = model.row(STATE, REQUEST["route"], "route")
    assert row["qtype"] == "choice"
    with pytest.raises(ValueError, match="question type"):
        model.decider.decide({k: v for k, v in row.items() if k != "qtype"})


def test_many_options_are_answered_in_groups(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0)
    options = {f"o{i}": f"Option number {i}" for i in range(40)}
    answer = model.decide(STATE, {"wide": {"type": "choice", "instructions": "Which fits?", "criteria": options}})["wide"]
    assert len(answer["probabilities"]) == 40 and answer["choice"] in options
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert model.decider.last["groups"] == 3 and not model.decider.last["thought"]   # wide questions do not reason
    for n in range(17, 257):                               # consecutive, at most 16, sizes differ by at most 1
        groups = split_groups(n, 16)
        sizes = [len(g) for g in groups]
        assert sum(groups, []) == list(range(n)) and max(sizes) <= 16 and max(sizes) - min(sizes) <= 1 and min(sizes) >= 2
        assert len(groups) <= 16
    assert model.decider.reasoner is not None
    for n in (17, 33, 256):                               # no group may be left with a single option
        options = {f"o{i}": f"Option number {i}" for i in range(n)}
        answer = model.decide(STATE, {"w": {"type": "choice", "instructions": "Which fits?", "criteria": options}})["w"]
        assert len(answer["probabilities"]) == n and sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    levels = [f"level {i}" for i in range(20)]
    answer = model.decide(STATE, {"s": {"type": "score", "instructions": "How much?", "criteria": levels}})["s"]
    assert 0 <= answer["score"] < 20
    too_many = {f"o{i}": "x" for i in range(257)}
    with pytest.raises(ValueError, match="2-256"):
        model.decide(STATE, {"wide": {"type": "choice", "instructions": "Which?", "criteria": too_many}})


def test_long_and_empty_inputs_are_filtered_or_refused(tiny):
    folder, _ = tiny("gemma", max_input_tokens=900)
    lines = ["LOG: the owner is the last assignee"] + [f"2026-01-{i % 28 + 1:02d} SUP-{1000 + i} note" for i in range(200)]
    lines.insert(120, "2026-02-02 SUP-605069 assigned to Bo")
    question = {"type": "choice", "instructions": "Who owns SUP-605069?", "criteria": {"bo": "Bo", "al": "Al"}}
    model = load(folder)
    assert model.decide("\n".join(lines), {"q": question})["q"]["choice"] in ("bo", "al")
    assert model.decider.last["shortened"] == "lines"
    assert model.decide("", {"q": question})["q"]["choice"] in ("bo", "al")
    strict = load(folder, long_input="refuse")
    with pytest.raises(ValueError, match="exceed limit"):
        strict.decide("\n".join(lines), {"q": question})
    with pytest.raises(ValueError, match="state"):
        strict.decide("", {"q": question})


def test_server_reports_output_tokens(tiny):
    folder, _ = tiny("qwen", "full")
    model = load(folder, reasoning=True, think_share=1.0)
    httpd = HTTPServer(("127.0.0.1", 0), make_handler(model, {"model": model.name}))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/systemone",
                                         data=json.dumps({"state": STATE, "questions": REQUEST}).encode())
        body = json.loads(urllib.request.urlopen(request, timeout=120).read())
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert body["runtime"]["reasoned"] == 3
    assert body["usage"]["output_tokens"] == model.decider.reasoner.served[1] and body["usage"]["input_tokens"] > 0


def test_warmup_is_not_counted(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0)
    model.warmup()
    assert model.decider.reasoner.served == [0, 0] and model.decider.reasoner is not None


def test_unsupported_attention_mode_is_refused(tmp_path):
    (tmp_path / "jeff_config.json").write_text(json.dumps({"name": "x", "bidirectional": "sideways"}))
    with pytest.raises(ValueError, match="bidirectional"):
        JeffModel(tmp_path, device="cpu")


def test_compiled_forward_gives_the_same_answers(tiny, monkeypatch):
    """--compile swaps in a compiled decision forward pass (here traced with the eager backend, so no C++ compiler is
    needed); the answers stay the same and the warm-up covers several input lengths."""
    folder, _ = tiny("gemma", "full")
    model = load(folder)
    row = model.row(STATE, REQUEST["route"], "route")
    want = model.decider.decide(row)[0]
    calls = []
    real = torch.compile
    monkeypatch.setattr(torch, "compile", lambda module, **kw: calls.append(kw) or real(module, backend="eager", **kw))
    fast = load(folder, compile=True)
    assert calls == [{"dynamic": True}] and fast.decider.forward is not fast.decider.base
    fast.warmup()
    assert fast.decider.decide(row)[0] == pytest.approx(want, abs=1e-5)


@pytest.mark.parametrize("kind", ["gemma", "qwen"])
def test_fast_reasoning_swaps_in_a_copy_of_the_base_weights(tiny, kind):
    """--fast-reasoning: inside base() the adapted layers hold bf16(W_released - s B A) (close to the base layers),
    outside it the released weights again; generation uses a static KV cache; the answers are still distributions."""
    folder, info = tiny(kind)
    model = load(folder, reasoning=True, think_share=1.0, fast_reasoning=True)
    reasoner = model.decider.reasoner
    assert model.fast_reasoning and reasoner.static_cache and not reasoner.delta.handles
    released, base = dict(model.decider.model.named_modules()), dict(info["base"].named_modules())
    name = next(iter(info["lora"]))
    before = released[name].weight.data_ptr()
    x = torch.randn(3, base[name].in_features, dtype=torch.bfloat16)
    with torch.no_grad(), reasoner.delta.base():
        off = released[name](x).float()
    assert released[name].weight.data_ptr() == before
    assert torch.allclose(off, base[name](x).float(), atol=0.05, rtol=0.02)
    answer = model.decide(STATE, {"route": REQUEST["route"]})["route"]
    assert model.decider.last["thought"] and sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_vllm_engine_writes_the_reasoning(tiny, monkeypatch):
    """--reasoning-engine vllm: the generator (vLLM, here a stand-in) is built before the decision model with the base
    model and the memory share, writes the reasoning (cut at the first stop token), and no adapter hooks are needed."""
    import jeff.reasoning as R
    calls = []

    class FakeVLLM:
        def __init__(self, base, revision, tokenizer, max_len, memory):
            calls.append((base, revision, max_len, memory))

        def generate(self, ids, budget, stops):
            return [ids[-1]] * 3 + [stops[0], 7]           # three tokens, a stop token, anything after it

    monkeypatch.setattr(R, "VLLMGenerator", FakeVLLM)
    folder, _ = tiny("gemma", "full")
    model = load(folder, reasoning=True, think_share=1.0, reasoning_engine="vllm", vllm_memory=0.4, vllm_base="base-x")
    assert calls == [("base-x", None, 8192 + 4 + 64, 0.4)]
    assert model.reasoning_engine == "vllm" and model.decider.reasoner.delta is None
    answer = model.decide(STATE, {"route": REQUEST["route"]})["route"]
    assert model.decider.last["thought"] and model.decider.last["output_tokens"] == 3
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    model.warmup()                                         # warms the generator too, not counted
    assert model.decider.reasoner.served == [1, 3]
    with pytest.raises(ValueError, match="reasoning_engine"):
        load(folder, reasoning=True, reasoning_engine="tgi")
