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
        assert model.decider.last == {"output_tokens": 0, "thought": False, "dropped": False, "shortened": False}


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
        probs, logits, _, finished = reasoner.think(ids, 3, "choice", [float(v) for v in torch.tensor(first).log()])
    assert finished
    reasoner.config["mix"] = False
    with torch.no_grad():
        read, _, _, _ = reasoner.think(ids, 3, "choice", [0.0, 0.0, 0.0])     # read alone (pooling off)
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
    assert model.reasoning["decision_seconds"] == R.JEVBENCH_DEFAULTS["decision_seconds"] == 8.5
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


def test_time_limit_stops_before_the_token_that_would_end_too_late():
    """TimeLimit (decision_seconds) on a fake clock: the prefill (time to the first token) gives the read estimate,
    the steps so far the time per token; it stops once the next token and the read would end after `end`."""
    now = [100.0]
    limit = R.TimeLimit(end=104.0, prompt_len=100, extra=4, clock=lambda: now[0])
    ids = lambda n: torch.zeros((1, 100 + n), dtype=torch.long)
    now[0] = 100.5            # prefill and first token 0.5 s; next token 0.5 s, read 1.5 x 0.5 x 106 / 100 + 0.05 s
    assert not limit(ids(1), None).item()
    now[0] = 101.0            # 0.5 s per token
    assert not limit(ids(2), None).item()
    now[0] = 102.7            # slower: 1.1 s per token, 102.7 + 1.1 + 0.86 > 104
    assert limit(ids(3), None).item()


def test_decision_seconds_drops_reasonings_that_cannot_finish(tiny):
    """No time to reason (decision_seconds tiny): every question keeps its one-pass answer, nothing is written, and
    the limits' counts stay balanced."""
    folder, _ = tiny("gemma")
    plain = load(folder)
    model = load(folder, reasoning=True, think_share=1.0, decision_seconds=1e-6)
    for key, question in REQUEST.items():
        assert model.decide(STATE, {key: question})[key] == plain.decide(STATE, {key: question})[key]
        assert model.decider.last["dropped"] and not model.decider.last["thought"]
        assert model.decider.last["output_tokens"] == 0
    assert model.decider.reasoner.served == [3, 0] and model.decider.reasoner.pending == [0, 0]
    with pytest.raises(ValueError, match="decision_seconds"):
        load(folder, reasoning=True, decision_seconds=0.0)


def test_decision_seconds_with_time_to_spare_changes_nothing(tiny):
    folder, _ = tiny("gemma")
    limited = load(folder, reasoning=True, think_share=1.0, decision_seconds=600.0)
    free = load(folder, reasoning=True, think_share=1.0)
    for key, question in REQUEST.items():
        assert limited.decide(STATE, {key: question}) == free.decide(STATE, {key: question})
        assert limited.decider.last["thought"] and not limited.decider.last["dropped"]
    assert limited.decider.reasoner.served == free.decider.reasoner.served


def test_a_reasoning_cut_by_the_time_limit_is_dropped(tiny, monkeypatch):
    """read_partial off: a generation the time limit stops is not read: the one-pass answer, with the written tokens
    counted."""
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0, decision_seconds=600.0, read_partial=False)
    reasoner = model.decider.reasoner
    ids = encode_prompt(model.decider.tokenizer, model.row(STATE, REQUEST["route"], "route"), 8192)
    with torch.no_grad():
        full, finished = reasoner.write([reasoner.prompt(ids)])[0]
    assert finished
    if len(full) < 2:
        pytest.skip("the tiny model stops at once")

    class OutOfTime(R.TimeLimit):                       # no time after the first token
        def __call__(self, input_ids, scores, **kwargs):
            return torch.ones(input_ids.shape[0], dtype=torch.bool)

    monkeypatch.setattr(R, "TimeLimit", OutOfTime)
    with torch.no_grad():
        gen, finished = reasoner.write([reasoner.prompt(ids)], [reasoner.decision_end()], [0.0])[0]
        assert not finished and gen == full[:1]
        assert reasoner.think(ids, 3, "choice", [0.0, 0.0, 0.0], reasoner.decision_end(), 0.0) == (None, None, 1, False)
    plain = load(folder)
    assert model.decide(STATE, {"route": REQUEST["route"]}) == plain.decide(STATE, {"route": REQUEST["route"]})
    assert model.decider.last["dropped"] and model.decider.last["output_tokens"] == 1
    assert reasoner.served == [1, 1]


def test_reasoning_config_replaces_the_released_settings(tiny, tmp_path):
    """--reasoning-config: budget, read, pooling, temperatures, shares and JevBench limits from a file; checked."""
    folder, _ = tiny("gemma")
    path = tmp_path / "jevbench.json"
    path.write_text(json.dumps({"budget": 2, "read": "head", "mix": False, "temperatures": {"choice": 2.5},
                                "share": {"noul": 1.0}, "jevbench_limits": {"decision_seconds": 7.5}}))
    model = load(folder, reasoning=True, jevbench_limits=True, reasoning_config=str(path))
    r = model.reasoning
    assert r["budget"] == 2 and r["read"] == "head" and r["mix"] is False and r["decision_seconds"] == 7.5
    assert r["temperatures"]["choice"] == 2.5 and r["temperatures"]["noul"] == R.json.loads(
        (folder / "reasoning.json").read_text())["temperatures"]["noul"]
    assert r["share"]["noul"] == 1.0 and r["tau"]["noul"] == 2.0          # share 1: every question reasons
    model.decide(STATE, {"urgent": REQUEST["urgent"]})
    assert model.decider.last["thought"] and model.decider.last["output_tokens"] <= 2
    for bad in ({"budgets": 3}, {"read": "lm"}, {"temperatures": {"choice": 0}}, {"share": {"choice": 2}}):
        with pytest.raises(ValueError):
            load(folder, reasoning=True, reasoning_config=bad)


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
    with pytest.raises(ValueError, match="exceed limit .*maximum context length"):
        strict.decide("\n".join(lines), {"q": question})
    wide = {"type": "choice", "instructions": "Which code?", "criteria": {f"c{i}": f"code {i}" for i in range(20)}}
    for empty in ("", {}, []):                       # refuse mode still sends empty states as text (since 1.3)
        assert strict.decide(empty, {"q": question})["q"]["choice"] in ("bo", "al")
        assert strict.decide(empty, {"w": wide})["w"]["choice"] in wide["criteria"]     # also with > 16 options


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
    assert body["runtime"]["reasoned"] == 3 and body["runtime"]["reasoning_dropped"] == 0
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


def test_shared_fast_reasoning_compiles_the_hooks_without_a_copy(tiny):
    """--fast-reasoning shared: no second copy (the hooks stay), a static KV cache; the reasoning is the one the plain
    hooks write; auto takes the copy when there is room (always on the CPU); unknown modes are refused."""
    folder, _ = tiny("gemma")
    shared = load(folder, reasoning=True, think_share=1.0, fast_reasoning="shared")
    plain = load(folder, reasoning=True, think_share=1.0)
    r = shared.decider.reasoner
    assert shared.fast_reasoning == "shared" and r.static_cache and r.delta.handles and not r.delta.copies
    ids = encode_prompt(shared.decider.tokenizer, shared.row(STATE, REQUEST["route"], "route"), 8192)
    with torch.no_grad():
        assert r.write([r.prompt(ids)]) == plain.decider.reasoner.write([plain.decider.reasoner.prompt(ids)])
    answer = shared.decide(STATE, {"route": REQUEST["route"]})["route"]
    assert shared.decider.last["thought"] and sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    auto = load(folder, reasoning=True, fast_reasoning="auto")
    assert auto.fast_reasoning == "copy" and auto.decider.reasoner.delta.copies
    assert load(folder, reasoning=True, fast_reasoning=True).fast_reasoning == "copy"
    with pytest.raises(ValueError, match="fast_reasoning"):
        load(folder, reasoning=True, fast_reasoning="fastest")


def test_fast_reasoning_uses_one_static_cache_size(tiny, monkeypatch):
    """--fast-reasoning: every reasoning uses one static cache size (compiled once, at warm-up); a prompt too long for
    it reasons with a dynamic cache (not compiled), and writes the same reasoning."""
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0, fast_reasoning="shared")
    r, budget = model.decider.reasoner, int(model.reasoning["budget"])
    assert r.static_len == R.STATIC_PROMPT_TOKENS + budget
    seen, real = [], model.decider.model.generate
    monkeypatch.setattr(model.decider.model, "generate",
                        lambda **kw: seen.append((kw.get("cache_implementation"), kw.get("max_cache_len"))) or real(**kw))
    prompt = r.prompt(encode_prompt(model.decider.tokenizer, model.row(STATE, REQUEST["route"], "route"), 8192))
    with torch.no_grad():
        short = r.write([prompt])
        r.static_len = len(prompt) + budget - 1          # one token too long for the static cache
        long = r.write([prompt])
    assert seen == [("static", R.STATIC_PROMPT_TOKENS + budget), (None, None)]
    assert short == long


def test_read_partial_reads_a_reasoning_the_time_limit_cut(tiny, monkeypatch):
    """read_partial (the default): a reasoning stopped by the time limit is read as it stands (prompt, its tokens,
    "Final answer:"), not dropped; without tokens there is nothing to read and the one pass stays. A reasoning config
    can turn it off, and the option overrides the config."""
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0, decision_seconds=600.0)
    reasoner = model.decider.reasoner
    assert model.reasoning["read_partial"] is True
    assert load(folder, reasoning=True, reasoning_config={"read_partial": False}).reasoning["read_partial"] is False
    assert load(folder, reasoning=True, reasoning_config={"read_partial": False},
                read_partial=True).reasoning["read_partial"] is True
    ids = encode_prompt(model.decider.tokenizer, model.row(STATE, REQUEST["route"], "route"), 8192)
    with torch.no_grad():
        full, _ = reasoner.write([reasoner.prompt(ids)])[0]
    if len(full) < 2:
        pytest.skip("the tiny model stops at once")

    class OutOfTime(R.TimeLimit):                       # no time after the first token
        def __call__(self, input_ids, scores, **kwargs):
            return torch.ones(input_ids.shape[0], dtype=torch.bool)

    monkeypatch.setattr(R, "TimeLimit", OutOfTime)
    first = [0.1, -0.2, 0.3]
    with torch.no_grad():
        probs, _, n, read = reasoner.think(ids, 3, "choice", first, reasoner.decision_end(), 0.0)
        want, _ = reasoner.read(model.decider.hidden(reasoner.read_ids(reasoner.prompt(ids), full[:1])), 3, "choice",
                                first)
    assert read and n == 1 and probs == pytest.approx(want, abs=1e-6)
    model.decide(STATE, {"route": REQUEST["route"]})
    assert model.decider.last["thought"] and not model.decider.last["dropped"]
    assert model.decider.last["output_tokens"] == 1
    off = load(folder, reasoning=True, think_share=1.0, decision_seconds=1e-6)
    off.decide(STATE, {"route": REQUEST["route"]})               # no time to start: nothing written, nothing read
    assert off.decider.last["dropped"] and off.decider.last["output_tokens"] == 0


def test_vllm_engine_writes_the_reasoning(tiny, monkeypatch):
    """--reasoning-engine vllm: the generator (vLLM, here a stand-in) is built before the decision model with the base
    model and the memory share, writes the reasoning (cut at the first stop token), and no adapter hooks are needed."""
    import jeff.reasoning as R
    calls = []

    class FakeVLLM:
        def __init__(self, base, revision, tokenizer, max_len, memory, max_seqs=1):
            calls.append((base, revision, max_len, memory))

        def generate_many(self, prompts, budget, stops):
            return [[ids[-1]] * 3 + [stops[0], 7] for ids in prompts]    # three tokens, a stop token, anything after

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


def test_vllm_engine_without_vllm_says_how_to_install(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "vllm", None)          # import vllm fails
    with pytest.raises(ImportError, match="pip install"):
        R.VLLMGenerator("base-x", None, "tok", 100, 0.4)
