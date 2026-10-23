"""Load a released Jeff model and answer decisions: by default one forward pass per question, no generation.

A released model folder (or Hugging Face repo) holds the merged weights and tokenizer (standard transformers files),
the decision head (head.safetensors + head_config.json), the serving rule (serving_rule.json) and jeff_config.json.
Each question is rendered as a SemIf-format multiple-choice prompt (options labeled A, B, C, ...; chat template with
thinking disabled), the model reads it once, the head turns the last hidden state into one logit per option, and the
serving rule is applied to the softmax. This computes exactly what we evaluated the released models with.

Since 1.2 (all models):
- inputs over the model's limit (8,192 tokens) keep their question-relevant parts instead of being refused, and states
  the prompt format cannot hold (empty, a bare number) are sent as text (jeff.longinput; long_input="refuse" turns
  both off);
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
    `last` holds {"output_tokens", "thought", "shortened"} for that decision."""

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
        self.last = {"output_tokens": 0, "thought": False, "shortened": False}

    def hidden(self, ids: list[int]):
        """Post-final-norm hidden state at the last position (with bidirectional masks for such models)."""
        torch = self._torch
        x = torch.tensor([ids], device=self.device)
        mask = None
        if self.bidirectional:
            from .bidir import attention_arg
            mask = attention_arg(self.base, x, torch.ones_like(x))
        return self.forward(input_ids=x, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state[:, -1]

    def compile(self) -> None:
        """Compile the decision forward pass with torch.compile (one graph for all input lengths). About 30-40% less
        time per decision on an H200; probabilities change by numerical noise only (fused kernels round differently),
        of the same size as changing the input's padding. The first calls compile (a few minutes for the 31B)."""
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

    def _decide(self, row: dict) -> tuple[list[float], list[float], int, float]:
        torch = self._torch
        self.last = {"output_tokens": 0, "thought": False, "shortened": False}
        reasoner = self.reasoner
        if reasoner is not None and row.get("qtype") not in ("choice", "noul", "score"):
            raise ValueError("reasoning needs the question type in the row (qtype: choice, noul or score)")
        if self.long_input == "filter":
            row = plain_state(row)
        try:
            ids = encode_prompt(self.tokenizer, row, self.max_tokens)
        except ValueError as error:
            if self.long_input != "filter" or "exceed limit" not in str(error):
                raise
            row, how = shorten(self.tokenizer, row, self.max_tokens, render_prompt)
            ids = encode_prompt(self.tokenizer, row, self.max_tokens)
            self.last["shortened"] = how
        if self.cudnn_min_tokens is not None:
            torch.backends.cuda.enable_cudnn_sdp(len(ids) >= self.cudnn_min_tokens)
        start = time.perf_counter()
        k = len(row["options"])
        with torch.inference_mode():
            h = self.hidden(ids).to(torch.bfloat16).float()  # as in our evaluations
            logits = self.head(h)[0, :k]
            probs = torch.softmax(logits, dim=-1).tolist()
            logits = logits.tolist()
            if reasoner is not None:
                if reasoner.wants(row["qtype"], probs):
                    probs, logits, generated = reasoner.think(ids, k, row["qtype"], logits)
                    self.last.update(output_tokens=generated, thought=True)
                reasoner.count(self.last["output_tokens"])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return probs, logits, len(ids), time.perf_counter() - start


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
    override the released share of questions that reason and the most tokens a reasoning may take;
    max_output_per_decision and request_seconds set those limits on their own."""

    def __init__(self, ref: str | Path, revision: str | None = None, device: str = "cuda",
                 rule: dict | None = None, cudnn_attention: bool = False, commit: bool = True,
                 long_input: str = "filter", reasoning: bool = False, jevbench_limits: bool = False,
                 think_share: float | dict | None = None, think_budget: int | None = None,
                 max_output_per_decision: float | None = None, request_seconds: float | None = None,
                 compile: bool = False, fast_reasoning: bool = False):
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
                                      max_output_per_decision=max_output_per_decision,
                                      request_seconds=request_seconds)
        self.rule = json.loads((folder / "serving_rule.json").read_text()) if rule is None else rule
        if not commit:
            self.rule = {k: v for k, v in self.rule.items() if k != "noul_edge"}
        if not cudnn_attention:
            disable_cudnn_attention()
        model, tokenizer = load_causal_model(folder, device=device)
        self.decider = Decider(model, tokenizer, load_head(folder), int(self.config.get("max_input_tokens", MAX_TOKENS)),
                               bidirectional=self.config.get("bidirectional"), long_input=long_input)
        if not cudnn_attention:
            self.decider.cudnn_min_tokens = CUDNN_MIN_TOKENS
        if compile:
            self.decider.compile()
        if self.reasoning is not None:
            from .reasoning import LoraDelta, Reasoner
            copy = fast_reasoning and LoraDelta.copy_fits(model, folder / "adapter")
            if fast_reasoning and not copy:
                print("--fast-reasoning: not enough GPU memory for a second copy of the weights; using the hooks",
                      flush=True)
            self.fast_reasoning = copy
            self.decider.reasoner = Reasoner(self.decider, self.reasoning, LoraDelta(model, folder / "adapter", copy=copy),
                                             self.rule, static_cache=copy)

    def warmup(self, times: int = 3) -> None:
        """A few forward passes (kernels, allocator; with compile: the compiled graphs, for short and long inputs),
        without reasoning and not counted by its limits."""
        reasoner, self.decider.reasoner = self.decider.reasoner, None
        try:
            for i in range(times):
                self.decider.decide(WARMUP)
            if self.decider.forward is not self.decider.base:
                for words in (60, 600, 1500):
                    self.decider.decide({**WARMUP, "state": "The sky is blue. " * words})
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
