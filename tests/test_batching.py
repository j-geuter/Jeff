"""1.3 serving on tiny random models (CPU, see conftest.py): batched forward passes, the batching server, reasoning in
batches and its limits under concurrency, thresholds and the hard share cap, and the base model rebuilt for vLLM."""
import json
import shutil
import threading
import time
import urllib.request
from collections import namedtuple

import pytest
import torch

from jeff import reasoning as R
from jeff.batching import Batcher
from jeff.model import JeffModel
from jeff.serve import BatchingServer, answer_batched, make_handler

STATE = {"ticket": "Refund not received after 14 days", "customer": "J. Doe"}
REQUEST = {"route": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Payments and refunds", "tech": "Technical problems", "sales": "Orders"}},
           "urgent": {"type": "noul", "instructions": "Is this urgent?"},
           "severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["low", "mid", "high"]}}
STATES = [STATE, {"ticket": "App crashes when I upload a photo"}, "Please cancel order 4411 before it ships.",
          {"note": "x" * 300}, {"ticket": "Where is my parcel?", "days": 9}, "Charged twice."]


def load(folder, **options) -> JeffModel:
    return JeffModel(folder, device="cpu", **options)


def serial(model, state) -> dict:
    return model.decide(state, REQUEST)


def concurrently(batcher, states) -> list:
    out = [None] * len(states)

    def run(i):
        out[i] = batcher.submit(states[i], REQUEST)
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(states))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    return out


def close(a: dict, b: dict, tol: float = 0.03) -> bool:
    for key in a:
        x, y = a[key], b[key]
        if "probabilities" in x:
            if any(abs(x["probabilities"][o] - y["probabilities"][o]) > tol for o in x["probabilities"]):
                return False
        elif abs(x["noul"] - y["noul"]) > tol:
            return False
    return True


@pytest.mark.parametrize("kind,bidirectional", [("gemma", None), ("gemma", "full"), ("qwen", None), ("qwen", "full")])
def test_batched_forward_pass_matches_single_ones(tiny, kind, bidirectional):
    folder, _ = tiny(kind, bidirectional)
    model = load(folder)
    rows = [model.decider.prepare(model.row(s, REQUEST["urgent"]))[1] for s in STATES]
    assert len({len(ids) for ids in rows}) > 1                      # different lengths: padding is exercised
    with torch.no_grad():
        alone = model.decider.hidden_batch(rows[:1])
        assert torch.equal(alone, model.decider.hidden(rows[0]))    # one input: exactly the single pass
        model.decider.model.float()                                 # in float32 the batch gives the same states
        single = torch.cat([model.decider.hidden(ids) for ids in rows])
        batched = model.decider.hidden_batch(rows)
    assert (batched - single).abs().max() / single.abs().mean() < 1e-4


def test_batcher_answers_like_serial_serving(tiny):
    folder, _ = tiny("gemma", "full")
    model = load(folder)
    expected = [serial(model, s) for s in STATES]
    batcher = Batcher(model, max_batch=8)
    sizes = []
    hidden_batch = model.decider.hidden_batch
    model.decider.hidden_batch = lambda batch: (sizes.append(len(batch)), hidden_batch(batch))[1]
    got = concurrently(batcher, STATES)
    for request, answers in zip(got, expected):
        assert set(request.answers) == set(REQUEST) and request.left == 0 and request.thought == 0
        assert close(request.answers, answers)
        assert request.tokens > 0 and request.generated == 0
    assert sum(sizes) == len(STATES) * len(REQUEST) and max(sizes) > 1        # some questions shared a batch
    for key, question in REQUEST.items():                                    # alone: exactly the serial answers
        assert batcher.submit(STATE, {key: question}).answers == model.decide(STATE, {key: question})


def test_token_cap_splits_batches(tiny):
    folder, _ = tiny("gemma")
    model = load(folder)
    longest = max(len(model.decider.prepare(model.row(s, q))[1]) for s in STATES for q in REQUEST.values())
    batcher = Batcher(model, max_batch=32, max_tokens=longest)      # room for one input of the longest kind
    sizes = []
    hidden_batch = model.decider.hidden_batch
    model.decider.hidden_batch = lambda batch: (sizes.append((len(batch), max(map(len, batch)))), hidden_batch(batch))[1]
    concurrently(batcher, STATES)
    assert all(n * width <= longest or n == 1 for n, width in sizes)


def test_reasoning_in_batches_keeps_the_limits(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0, reasoning_engine="transformers")
    batcher = Batcher(model, max_batch=8)
    got = concurrently(batcher, STATES)
    n = len(STATES) * len(REQUEST)
    assert sum(r.thought for r in got) == n and all(r.generated > 0 for r in got)
    reasoner = model.decider.reasoner
    assert reasoner.served == [n, sum(r.generated for r in got)] and reasoner.pending == [0, 0]
    # the cost limit holds with concurrent requests: budgets of questions still reasoning count as spent
    model = load(folder, reasoning=True, think_share=1.0, reasoning_engine="transformers", max_output_per_decision=1.0)
    batcher = Batcher(model, max_batch=8)
    got = concurrently(batcher, STATES)
    reasoner = model.decider.reasoner
    assert reasoner.served[1] / reasoner.served[0] <= 1.0 and reasoner.pending == [0, 0]
    assert 0 < sum(r.thought for r in got) < n


def test_vllm_reasonings_run_as_one_batch_beside_the_decisions(tiny, monkeypatch):
    calls = []

    class FakeVLLM:
        def __init__(self, base, revision, tokenizer, max_len, memory, max_seqs=1):
            self.max_seqs = max_seqs

        def generate_many(self, prompts, budget, stops):
            calls.append(len(prompts))
            time.sleep(0.2)                                         # writing takes a while: reasonings pile up
            return [[p[-1]] * 3 + [stops[0]] for p in prompts]

    monkeypatch.setattr(R, "VLLMGenerator", FakeVLLM)
    folder, _ = tiny("gemma", "full")
    model = load(folder, reasoning=True, think_share=1.0, reasoning_engine="vllm", vllm_base="base-x", vllm_seqs=8,
                 vllm_memory=0.4)
    batcher = Batcher(model, max_batch=8)
    assert batcher.parallel
    got = concurrently(batcher, STATES)
    n = len(STATES) * len(REQUEST)
    assert sum(r.thought for r in got) == n and sum(r.generated for r in got) == 3 * n
    assert sum(calls) == n and max(calls) > 1
    assert model.decider.reasoner.pending == [0, 0] and model.decider.reasoner.served == [n, 3 * n]
    # alone, a question gets the answer of serial serving with the same engine
    alone = batcher.submit(STATE, {"urgent": REQUEST["urgent"]}).answers
    assert alone == model.decide(STATE, {"urgent": REQUEST["urgent"]})


def test_batcher_errors_and_wide_questions(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, long_input="refuse")
    batcher = Batcher(model, max_batch=8)
    with pytest.raises(ValueError, match="unknown question type"):
        batcher.submit(STATE, {"q": {"type": "rank", "instructions": "x"}})
    with pytest.raises(ValueError, match="exceed limit"):
        batcher.submit({"note": "word " * 5000}, {"q": REQUEST["urgent"]})
    wide = {"type": "choice", "instructions": "Which code?", "criteria": {f"c{i}": f"code {i}" for i in range(20)}}
    answers = batcher.submit(STATE, {"wide": wide, "urgent": REQUEST["urgent"]}).answers
    assert sum(answers["wide"]["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert answers == model.decide(STATE, {"wide": wide, "urgent": REQUEST["urgent"]})


def test_thresholds_and_hard_share_cap(tiny):
    folder, _ = tiny("gemma")
    s = R.settings(folder, threshold={"noul": 0.75}, share={"score": 1.0})
    assert s["tau"]["noul"] == 0.75 and s["tau"]["score"] == 2.0 and s["share"]["noul"] is None
    assert R.settings(folder, threshold=0.5)["tau"] == {q: 0.5 for q in R.TYPES}
    with pytest.raises(ValueError, match="not both"):
        R.settings(folder, threshold={"noul": 0.7}, share=0.3)
    with pytest.raises(ValueError, match="between 0 and 1"):
        R.settings(folder, threshold=1.5)
    with pytest.raises(ValueError, match="unknown question types"):
        R.settings(folder, threshold={"rank": 0.5})
    model = load(folder, reasoning=True, think_threshold=1.0, max_think_share=0.5, reasoning_engine="transformers")
    reasoner = model.decider.reasoner
    reasoner.config["warmup"] = 2
    wanted = [reasoner.wants("noul", [0.5, 0.5]) for _ in range(40)]
    for w in wanted:
        reasoner.count(0, w)
    assert 0.4 <= sum(wanted) / len(wanted) <= 0.6 and reasoner.pending == [0, 0]


def test_concurrent_admissions_reserve_their_budget(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True, think_share=1.0, max_output_per_decision=2.0, reasoning_engine="transformers")
    reasoner = model.decider.reasoner                               # budget 4: one reasoning per two questions
    reasoner.count(0)
    assert reasoner.wants("noul", [0.5, 0.5]) is True               # (0 + 4) / 2 <= 2
    assert reasoner.wants("noul", [0.5, 0.5]) is False              # (0 + 4 + 4) / 3 > 2: the first still holds 4
    reasoner.count(1, True)                                         # it wrote 1 token
    reasoner.count(0)
    assert reasoner.wants("noul", [0.5, 0.5]) is True               # (1 + 4) / 4 <= 2


def test_auto_engine_uses_transformers_without_a_gpu(tiny):
    folder, _ = tiny("gemma")
    model = load(folder, reasoning=True)
    assert model.reasoning_engine == "transformers" and model.decider.reasoner.generator is None
    with pytest.raises(ValueError, match="reasoning_engine"):
        load(folder, reasoning=True, reasoning_engine="tgi")


@pytest.mark.parametrize("kind", ["gemma", "qwen"])
def test_base_model_rebuilt_from_the_release(tiny, kind, tmp_path):
    from safetensors.torch import load_file
    folder, info = tiny(kind)
    out = R.derive_base(folder, tmp_path / "cache", device="cpu")
    assert (out / ".complete").is_file() and (out / "tokenizer.json").is_file()
    assert json.loads((out / "config.json").read_text()) == json.loads((folder / "config.json").read_text())
    merged, rebuilt = {}, {}
    for f in folder.glob("model*.safetensors"):
        merged.update(load_file(str(f)))
    for f in out.glob("model*.safetensors"):
        rebuilt.update(load_file(str(f)))
    assert merged.keys() == rebuilt.keys()
    base = info["base"].state_dict()
    for key, w in merged.items():
        path = key.removesuffix(".weight")
        if path in info["lora"]:
            a, b = info["lora"][path]
            assert torch.equal(rebuilt[key], R.base_weight(w, a, b, info["scale"]))
            assert (rebuilt[key].float() - base[key].float()).abs().max() <= 0.02 * base[key].float().abs().max()
        else:
            assert torch.equal(rebuilt[key], w)
    stamp = (out / ".complete").stat().st_mtime_ns
    assert R.derive_base(folder, tmp_path / "cache", device="cpu") == out           # reused, not rebuilt
    assert (out / ".complete").stat().st_mtime_ns == stamp


def test_rebuilt_base_maps_architectures_vllm_lacks_and_checks_disk_space(tiny, tmp_path, monkeypatch):
    folder, _ = tiny("gemma")
    copy = tmp_path / "release"
    shutil.copytree(folder, copy)
    config = json.loads((copy / "config.json").read_text())
    (copy / "config.json").write_text(json.dumps({**config, "architectures": ["Gemma4UnifiedForCausalLM"],
                                                  "model_type": "gemma4_unified_text"}))
    out = R.derive_base(copy, tmp_path / "cache", device="cpu")
    rebuilt = json.loads((out / "config.json").read_text())
    assert rebuilt["architectures"] == ["Gemma4ForCausalLM"] and rebuilt["model_type"] == "gemma4_text"
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(R.shutil, "disk_usage", lambda path: Usage(10, 10, 0))
    with pytest.raises(OSError, match="JEFF_CACHE"):
        R.derive_base(folder, tmp_path / "other", device="cpu")


def test_batching_server_over_http(tiny):
    folder, _ = tiny("qwen", "full")
    model = load(folder)
    batcher = Batcher(model, max_batch=8)
    httpd = BatchingServer(("127.0.0.1", 0), make_handler(model, {"model": model.name}, answer_batched(batcher)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/v1/systemone"
    out = [None] * len(STATES)

    def call(i):
        body = json.dumps({"state": STATES[i], "questions": REQUEST}).encode()
        out[i] = json.loads(urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=60).read())
    threads = [threading.Thread(target=call, args=(i,)) for i in range(len(STATES))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    try:
        for i, response in enumerate(out):
            assert list(response["answers"]) == list(REQUEST)                # the request's own order
            assert close(response["answers"], serial(model, STATES[i]))
            assert response["usage"]["input_tokens"] > 0 and response["usage"]["output_tokens"] == 0
        request = urllib.request.Request(url, data=json.dumps({"state": "s", "questions": {"q": {"type": "x"}}}).encode())
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=30)
        assert error.value.code == 422
    finally:
        httpd.shutdown()
