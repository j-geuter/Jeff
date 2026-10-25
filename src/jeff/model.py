"""Load a released Jeff model and answer decisions: by default one forward pass per question, no generation.

A released model folder (or Hugging Face repo) holds the merged weights and tokenizer (standard transformers files),
the decision head (head.safetensors + head_config.json), the serving rule (serving_rule.json) and jeff_config.json.
Each question is rendered as a SemIf-format multiple-choice prompt (options labeled A, B, C, ...; chat template with
thinking disabled), the model reads it once, the head turns the last hidden state into one logit per option, and the
serving rule is applied to the softmax. This computes exactly what we evaluated the released models with.

Since 1.2 (all models):
- inputs over the model's limit (8,192 tokens) keep their question-relevant parts instead of being refused, and states
  the prompt format cannot hold (empty, a bare number) are sent as text (jeff.longinput; long_input="refuse" refuses
  long inputs instead; since 1.3 it still sends such states as text);
- questions with more than 16 options (up to 256) are answered in groups of 16 and a final round between the group
  winners (Decider.decide_wide).
Jeff 1.1 models also have
- bidirectional attention (jeff_config.json "bidirectional"; jeff.bidir), as they were trained;
- reasoning when unsure (reasoning.json and adapter/; jeff.reasoning), off by default (reasoning=True turns it on).

    from jeff import JeffModel
    model = JeffModel("jgeuter/Jeff-1.0-Large")
    model.decide({"ticket": "Refund not received after 14 days"},
                 {"route": {"type": "choice", "instructions": "Which team should handle this?",
                            "criteria": {"billing": "Payments and refunds", "tech": "Technical problems"}}})
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from . import typesafe as T
from .head import load_head
from .longinput import plain_state, shorten
from .prompting import MAX_OPTIONS, MAX_WIDE, PROMPT_VERSION, encode_prompt, render_prompt, validate_row

MAX_TOKENS = 8192          # as in all our evaluations; longer inputs are filtered (jeff.longinput) or refused
FAST_REASONING = ("off", "copy", "shared", "auto")
LONG_INPUT = ("filter", "refuse")
WARMUP = {"id": "warmup", "state": "The sky is blue.", "question": "Is the sky blue?", "qtype": "noul",
          "options": [{"id": "true", "description": "true: yes"}, {"id": "false", "description": "false: no"}]}
CUDNN_MIN_TOKENS = 8192    # cuDNN attention only for inputs at least this long (see disable_cudnn_attention)


def disable_cudnn_attention() -> None:
    """Turn off PyTorch's cuDNN attention kernel for this process.

    cuDNN attention builds a new execution plan for every new input shape, and served requests almost always have a
    new length. For Gemma 4 31B this costs about 2x per request on H200 (0.123 s vs 0.061 s median on our validation
    items) and up to 10x on H100 with torch 2.11 (0.72 s vs 0.07 s); answers do not change. The flash and
    memory-efficient kernels have no per-shape cost. Very long inputs are the exception, so the Decider turns cuDNN
    back on from CUDNN_MIN_TOKENS tokens.
    """
    import torch
    torch.backends.cuda.enable_cudnn_sdp(False)


def model_folder(ref: str | Path, revision: str | None = None) -> Path:
    """A local folder, or a snapshot of a Hugging Face model repo (downloaded once, then cached)."""
    path = Path(ref).expanduser()
    if path.is_dir():
        return path
    text = str(ref)
    if text.startswith((".", "/", "~")) or text.count("/") != 1:   # a path, not a Hugging Face repo id like org/name
        raise FileNotFoundError(f"model folder not found: {text}. Download the weights first, e.g. "
                                f"`hf download jgeuter/Jeff-1.1-Large --local-dir {text}`, or pass a repo id.")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(text, revision=revision))


def load_causal_model(folder: Path, device: str = "cuda", dtype: str = "bfloat16"):
    import torch
    import transformers
    common = {"local_files_only": True, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(folder, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(folder, **common)
    model, loading = transformers.AutoModelForCausalLM.from_pretrained(
        folder, config=config, dtype=getattr(torch, dtype), device_map={"": device}, low_cpu_mem_usage=True,
        output_loading_info=True, **common)
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    return model.eval(), tokenizer


class Decider:
    """One model + head; decide(row) -> option probabilities, option logits, input tokens, seconds. After each call,
    `last` holds {"output_tokens", "thought", "dropped", "shortened"} for that decision (dropped: a reasoning that
    could not finish within decision_seconds; the answer is the one pass)."""

    def __init__(self, model, tokenizer, head, max_tokens: int = MAX_TOKENS, bidirectional: str | None = None,
                 long_input: str = "filter"):
        import torch
        if long_input not in LONG_INPUT:
            raise ValueError(f"long_input must be one of {LONG_INPUT}, got {long_input!r}")
        self.model, self.tokenizer, self.max_tokens = model, tokenizer, max_tokens
        self.base = getattr(model, "model", model)
        self.device = next(model.parameters()).device
        self.head = head.to(self.device).eval()
        self._torch = torch
        self.cudnn_min_tokens = None   # set: cuDNN attention only for inputs this long (see disable_cudnn_attention)
        self.bidirectional = bidirectional
        if bidirectional:
            from . import bidir
            bidir.enable(self.base, bidirectional)
        self.long_input = long_input
        self.reasoner = None           # a jeff.reasoning.Reasoner when reasoning is on
        self.forward = self.base       # the decision forward pass; compile() swaps in a compiled one
        self.last = {"output_tokens": 0, "thought": False, "dropped": False, "shortened": False}

    def hidden(self, ids: list[int]):
        """Post-final-norm hidden state at the last position (with bidirectional masks for such models)."""
        torch = self._torch
        x = torch.tensor([ids], device=self.device)
        mask = None
        if self.bidirectional:
            from .bidir import attention_arg
            mask = attention_arg(self.base, x, torch.ones_like(x))
        return self.forward(input_ids=x, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state[:, -1]

    def hidden_batch(self, batch: list[list[int]]):
        """hidden() for several inputs in one forward pass [B, d]: right-padded, so every input keeps its positions and
        (with the padding masked) attends to what it attends to alone; the results differ from hidden() by numerical
        noise only (about the size of the effect of --compile). One input: exactly hidden()."""
        torch = self._torch
        if len(batch) == 1:
            return self.hidden(batch[0])
        width = max(len(ids) for ids in batch)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        x = torch.tensor([ids + [pad] * (width - len(ids)) for ids in batch], device=self.device)
        attention = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in batch], device=self.device)
        if self.bidirectional:
            from .bidir import attention_arg
            mask = attention_arg(self.base, x, attention)
        else:
            mask = attention
        out = self.forward(input_ids=x, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
        last = torch.tensor([len(ids) - 1 for ids in batch], device=self.device)
        return out[torch.arange(len(batch), device=self.device), last]

    def compile(self) -> None:
        """Compile the decision forward pass with torch.compile (one graph for all input lengths). About 30-40% less
        time per decision on an H200; probabilities change by numerical noise only (fused kernels round differently),
        of the same size as changing the input's padding. The first calls compile (a few minutes for the 31B)."""
        import torch._inductor.config as inductor
        inductor.emulate_precision_casts = True     # round intermediate results as eager mode does: closer answers
        self.forward = self._torch.compile(self.base, dynamic=True)

    def start_request(self) -> None:
        """A new request (the reasoning's per-request time limit starts now)."""
        if self.reasoner is not None:
            self.reasoner.start_request()

    def decide(self, row: dict) -> tuple[list[float], list[float], int, float]:
        if isinstance(row.get("options"), list) and len(row["options"]) > MAX_OPTIONS:
            return self.decide_wide(row)
        return self._decide(row)

    def decide_wide(self, row: dict) -> tuple[list[float], list[float], int, float]:
        """More options than the head has slots (JevBench allows up to 255): score the options in groups of at most 16
        (as equal in size as possible, so none has a single option), then the group winners against each other; an option's probability is P(its group's winner wins the final) x P(it
        within its group). Sums to 1. These questions do not reason."""
        row = plain_state(row)                    # an empty or numeric state as text, as in prepare()
        validate_row(row, MAX_WIDE)
        options = row["options"]
        groups = split_groups(len(options), MAX_OPTIONS)
        inner, winners, tokens, seconds = [], [], 0, 0.0
        reasoner, self.reasoner = self.reasoner, None
        try:
            for g in groups:
                probs, _, n, t = self._decide({**row, "options": [options[i] for i in g]})
                inner.append(probs)
                winners.append(g[max(range(len(g)), key=probs.__getitem__)])
                tokens, seconds = tokens + n, seconds + t
            final, _, n, t = self._decide({**row, "options": [options[i] for i in winners]})
        finally:
            self.reasoner = reasoner
        probs = [0.0] * len(options)
        for gi, g in enumerate(groups):
            for j, i in enumerate(g):
                probs[i] = final[gi] * inner[gi][j]
        self.last["groups"] = len(groups)
        return probs, [math.log(max(p, 1e-30)) for p in probs], tokens + n, seconds + t

    def prepare(self, row: dict) -> tuple[dict, list[int], str | bool]:
        """(row as read, its prompt token ids, how it was shortened or False): long inputs filtered, states the
        prompt cannot hold sent as text (long_input "filter")."""
        if self.reasoner is not None and row.get("qtype") not in ("choice", "noul", "score"):
            raise ValueError("reasoning needs the question type in the row (qtype: choice, noul or score)")
        row = plain_state(row)                    # in both modes: an empty or numeric state is sent as text
        try:
            return row, encode_prompt(self.tokenizer, row, self.max_tokens), False
        except ValueError as error:
            if self.long_input != "filter" or "exceed limit" not in str(error):
                raise
            row, how = shorten(self.tokenizer, row, self.max_tokens, render_prompt)
            return row, encode_prompt(self.tokenizer, row, self.max_tokens), how

    def cudnn_for(self, tokens: int) -> None:
        if self.cudnn_min_tokens is not None:
            self._torch.backends.cuda.enable_cudnn_sdp(tokens >= self.cudnn_min_tokens)

    def first(self, h, k: int) -> tuple[list[float], list[float]]:
        """(probabilities, logits) of the one-pass answer from the last hidden state h [1, d]."""
        torch = self._torch
        logits = self.head(h.to(torch.bfloat16).float())[0, :k]     # bfloat16 h, as in our evaluations
        return torch.softmax(logits, dim=-1).tolist(), logits.tolist()

    def _decide(self, row: dict) -> tuple[list[float], list[float], int, float]:
        torch = self._torch
        self.last = {"output_tokens": 0, "thought": False, "dropped": False, "shortened": False}
        reasoner = self.reasoner
        row, ids, self.last["shortened"] = self.prepare(row)
        self.cudnn_for(len(ids))
        start = time.perf_counter()
        k = len(row["options"])
        with torch.inference_mode():
            probs, logits = self.first(self.hidden(ids), k)
            if reasoner is not None:
                admitted = reasoner.wants(row["qtype"], probs)
                if admitted:
                    t1 = time.perf_counter() - start            # the one pass (its .tolist() waited for the GPU)
                    p, z, generated, finished = reasoner.think(ids, k, row["qtype"], logits, reasoner.decision_end(), t1)
                    if finished:
                        probs, logits = p, z
                    self.last.update(output_tokens=generated, thought=finished, dropped=not finished)
                reasoner.count(self.last["output_tokens"], admitted)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return probs, logits, len(ids), time.perf_counter() - start

    def batch_tokens(self, reserve: float = 0.05, min_reserve_bytes: int = 2 << 30) -> int:
        """How many padded input tokens (batch size x longest input) one forward pass can take on this GPU: the free
        memory (minus a reserve of 5% of the GPU, at least 2 GB) over the peak memory per token of a forward pass on
        an input of the model's full length. Measured once (the server calls it after loading, with vLLM's share
        already taken); never below one full-length input."""
        torch = self._torch
        if self.device.type != "cuda":
            return 4 * self.max_tokens
        ids = [self.tokenizer.pad_token_id or 0] * self.max_tokens
        torch.cuda.synchronize(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        before = torch.cuda.memory_allocated(self.device)
        with torch.inference_mode():
            self.hidden(ids)
        torch.cuda.synchronize(self.device)
        per_token = max(torch.cuda.max_memory_allocated(self.device) - before, 1) / len(ids)
        usable = free - max(reserve * total, min_reserve_bytes)
        return max(self.max_tokens, int(usable / per_token))


def split_groups(n: int, size: int) -> list[list[int]]:
    """Indices 0..n-1 in consecutive groups of at most `size`, as equal in size as possible (17 -> 9 + 8)."""
    count = -(-n // size)
    bounds = [round(i * n / count) for i in range(count + 1)]
    return [list(range(bounds[i], bounds[i + 1])) for i in range(count)]


def served_answer(qtype: str, option_ids: list[str], probabilities: list[float], rule: dict) -> dict:
    """TypeSafe answer object with the serving rule applied (identity without a rule)."""
    out = T.answer(qtype, option_ids, probabilities)
    if not rule:
        return out
    if qtype == "noul":
        return {"type": "noul", "noul": T.apply_rule("noul", {"yes": out["noul"], "no": 1 - out["noul"]}, rule)["yes"]}
    probs = T.apply_rule(qtype, out["probabilities"], rule)
    best = max(option_ids, key=lambda i: probs[i])
    return {**out, qtype: best if qtype == "choice" else int(best), "probabilities": probs}


class JeffModel:
    """commit (default True): the yes/no commitment of the released serving rule. JevBench counts a yes/no answer with
    0.2 < P(yes) < 0.8 as wrong, so the rule moves such answers to just outside the nearer edge (0.801 or 0.199).
    commit=False keeps the calibrated P(yes) (temperatures still apply).

    long_input: "filter" (default) or "refuse" (see the module docstring).

    reasoning (default False; Jeff 1.1 models): reason when unsure (jeff.reasoning). With it: jevbench_limits (default
    False) applies the limits we evaluate on JevBench with; think_share (a number, or {type: share}) and think_budget
    override the released share of questions that reason and the most tokens a reasoning may take; think_threshold (a
    number, or {type: threshold}) sets confidence thresholds directly; max_output_per_decision, max_think_share,
    request_seconds and decision_seconds set those limits on their own; reasoning_config (a dict or JSON file) replaces
    reasoning.json's budget, read, pooling, temperatures, shares or JevBench limits (jeff.reasoning.read_config);
    read_partial: read a reasoning cut by decision_seconds instead of dropping it (default: the config's, else on).

    reasoning_engine: what writes the reasoning. "transformers": this model, with the adapter removed by hooks (or a
    second weight copy, see fast_reasoning); "vllm": the base model in vLLM on the same GPU, rebuilt from this release
    once (jeff.reasoning.derive_base, under cache_dir) unless vllm_base names one; "auto" (default): vLLM if it is
    installed and the GPU can hold the model twice, else transformers. vllm_seqs: how many reasonings vLLM writes at
    once.

    fast_reasoning (transformers engine): "off" (False, default): hooks, a dynamic KV cache; "copy" (True): a second
    copy of the adapted layers' weights and a static KV cache, which transformers compiles (the hooks when the GPU has
    no room for the copy); "shared": the hooks with a static KV cache, compiled, so no second copy is needed; "auto":
    copy when there is room, else shared."""

    def __init__(self, ref: str | Path, revision: str | None = None, device: str = "cuda",
                 rule: dict | None = None, cudnn_attention: bool = False, commit: bool = True,
                 long_input: str = "filter", reasoning: bool = False, jevbench_limits: bool = False,
                 think_share: float | dict | None = None, think_budget: int | None = None,
                 max_output_per_decision: float | None = None, request_seconds: float | None = None,
                 compile: bool = False, fast_reasoning: bool | str = False, reasoning_engine: str = "auto",
                 vllm_memory: float | None = None, vllm_base: str | None = None, vllm_seqs: int = 1,
                 think_threshold: float | dict | None = None, max_think_share: float | None = None,
                 cache_dir: str | Path | None = None, decision_seconds: float | None = None,
                 reasoning_config: str | Path | dict | None = None, read_partial: bool | None = None):
        from . import bidir
        folder = model_folder(ref, revision)
        self.folder = folder
        self.config = json.loads((folder / "jeff_config.json").read_text())
        self.name = self.config.get("name", folder.name)
        for key, supported in (("prompt_version", PROMPT_VERSION), ("noul_mapping", "tf-true-first")):
            if self.config.get(key, supported) != supported:
                raise ValueError(f"{folder}: {key} {self.config[key]!r} is not supported (only {supported!r})")
        if self.config.get("bidirectional") not in bidir.MODES:
            raise ValueError(f"{folder}: bidirectional {self.config['bidirectional']!r} is not supported")
        if long_input not in LONG_INPUT:
            raise ValueError(f"long_input must be one of {LONG_INPUT}, got {long_input!r}")
        self.reasoning, self.fast_reasoning = None, False
        if reasoning:   # read the settings before loading the weights, so a model without them fails at once
            from .reasoning import settings
            self.reasoning = settings(folder, share=think_share, budget=think_budget, jevbench_limits=jevbench_limits,
                                      threshold=think_threshold, max_output_per_decision=max_output_per_decision,
                                      max_share=max_think_share, request_seconds=request_seconds,
                                      decision_seconds=decision_seconds, config=reasoning_config,
                                      read_partial=read_partial)
        self.rule = json.loads((folder / "serving_rule.json").read_text()) if rule is None else rule
        if not commit:
            self.rule = {k: v for k, v in self.rule.items() if k != "noul_edge"}
        if not cudnn_attention:
            disable_cudnn_attention()
        if reasoning_engine not in ("auto", "transformers", "vllm"):
            raise ValueError(f"reasoning_engine must be auto, transformers or vllm, got {reasoning_engine!r}")
        fast = {True: "copy", False: "off", None: "off"}.get(fast_reasoning, fast_reasoning)
        if fast not in FAST_REASONING:
            raise ValueError(f"fast_reasoning must be one of {FAST_REASONING}, got {fast_reasoning!r}")
        generator = None
        if self.reasoning is not None and reasoning_engine == "auto":
            from .reasoning import vllm_usable
            why = "--fast-reasoning asks for transformers" if fast != "off" else vllm_usable(folder, device)
            reasoning_engine = "transformers" if why else "vllm"
            if why:
                print(f"reasoning with transformers ({why})", flush=True)
        if self.reasoning is not None and reasoning_engine == "vllm":   # before the decision model: see VLLMGenerator
            from . import reasoning as R
            base = vllm_base or str(R.derive_base(folder, cache_dir))
            generator = R.VLLMGenerator(
                base, None, str(folder),
                int(self.config.get("max_input_tokens", MAX_TOKENS)) + int(self.reasoning["budget"]) + 64,
                vllm_memory or R.vllm_memory(folder), max_seqs=vllm_seqs)
        self.reasoning_engine = reasoning_engine if self.reasoning is not None else None
        model, tokenizer = load_causal_model(folder, device=device)
        self.decider = Decider(model, tokenizer, load_head(folder), int(self.config.get("max_input_tokens", MAX_TOKENS)),
                               bidirectional=self.config.get("bidirectional"), long_input=long_input)
        if not cudnn_attention:
            self.decider.cudnn_min_tokens = CUDNN_MIN_TOKENS
        if compile:
            self.decider.compile()
        if generator is not None:
            from .reasoning import Reasoner
            self.decider.reasoner = Reasoner(self.decider, self.reasoning, None, self.rule, generator=generator)
        elif self.reasoning is not None:
            from .reasoning import LoraDelta, Reasoner
            copy = fast in ("copy", "auto") and LoraDelta.copy_fits(model, folder / "adapter")
            shared = fast == "shared" or (fast == "auto" and not copy)
            if fast in ("copy", "auto") and not copy:
                print(f"--fast-reasoning: not enough GPU memory for a second copy of the weights; using the hooks"
                      f"{' (compiled, shared weights)' if shared else ''}", flush=True)
            self.fast_reasoning = "copy" if copy else ("shared" if shared else False)
            self.decider.reasoner = Reasoner(self.decider, self.reasoning, LoraDelta(model, folder / "adapter", copy=copy),
                                             self.rule, static_cache=copy or shared)

    def warmup(self, times: int = 3, batch: bool = False) -> None:
        """A few forward passes (kernels, allocator; with compile: the compiled graphs, for short and long inputs, and
        with batch also for batches), without reasoning and not counted by its limits."""
        reasoner, self.decider.reasoner = self.decider.reasoner, None
        try:
            for i in range(times):
                self.decider.decide(WARMUP)
            if self.decider.forward is not self.decider.base:
                for words in (60, 600, 1500):
                    self.decider.decide({**WARMUP, "state": "The sky is blue. " * words})
            if batch:
                rows = [self.decider.prepare({**WARMUP, "state": "The sky is blue. " * words})[1] for words in (5, 60)]
                with self.decider._torch.inference_mode():
                    for n in (2, 3):
                        self.decider.hidden_batch((rows * 2)[:n])
            if reasoner is not None and (reasoner.static_cache or reasoner.generator is not None):
                # --fast-reasoning: transformers compiles the decoding step on first use (minutes for the 31B; once:
                # one static cache size for all prompts, STATIC_PROMPT_TOKENS); do it now, not on the first request
                for words in (5, 120, 600):
                    try:
                        ids = encode_prompt(self.decider.tokenizer, {**WARMUP, "state": "The sky is blue. " * words},
                                            self.decider.max_tokens)
                    except ValueError:                     # longer than this model's input limit: skip this length
                        continue
                    with self.decider._torch.inference_mode():
                        reasoner.think(ids, 2, "noul", [0.0, 0.0])
        finally:
            self.decider.reasoner = reasoner

    def row(self, state, question: dict, key: str = "q") -> dict:
        return T.question_row(state, question, key)

    def answer(self, qtype: str, option_ids: list[str], probabilities: list[float]) -> dict:
        return served_answer(qtype, option_ids, probabilities, self.rule)

    def decide(self, state, questions: dict) -> dict:
        """TypeSafe request: state + {name: {type, instructions, criteria}} -> {name: answer}."""
        self.decider.start_request()
        answers = {}
        for key, question in questions.items():
            row = self.row(state, question, key)
            probs, _, _, _ = self.decider.decide(row)
            answers[key] = self.answer(question["type"], [o["id"] for o in row["options"]], probs)
        return answers
