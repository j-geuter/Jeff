"""Reasoning when unsure (Jeff 1.1 models; off by default, `--reasoning on`).

Every question is first answered in one forward pass, as without reasoning. If the calibrated confidence of that answer
(its largest probability at the serving rule's temperature) is below a threshold for the question's type, the question
gets a short reasoning:
1. the base model writes up to `budget` tokens greedily, with causal attention, after the prompt and the opener
   "Let me check the deciding facts first.". The base model is the released weights minus the LoRA adapter, which
   ships in adapter/ for this (LoraDelta);
2. the decision model (the released weights) reads the prompt, the reasoning and "\\n\\nFinal answer:", and the answer
   is read from its letter logits at that position (the LM's logits of the option letters, bare and with a space);
3. that distribution is pooled with the one-pass answer (mean of the log-probabilities) and calibrated with its own
   temperature per type. It replaces the one-pass answer; the yes/no commitment of the serving rule still applies.
The generated tokens are reported as output tokens.

Which questions reason: for a think share s of a type, the threshold is the s-quantile of the calibrated one-pass
confidences on our own validation questions of that type (reasoning.json "tau_table", steps of 0.01), so about a share
s of similar questions reason. The released shares are the ones we selected on held-out data; `--think-share` picks
others (0: never, 1: always).

JevBench limits (`--jevbench-limits on`; off by default): the limits we evaluate on JevBench with, so that the model
stays within the board's cost and latency caps:
- max_output_per_decision L: a question reasons only if (output tokens so far + budget) / (questions so far + 1) <= L,
  counted over everything this server has answered. The mean output per question therefore never exceeds L, whatever
  the share of unsure questions;
- max_share: no new reasoning while more than this share of the last `window` questions reasoned (after `warmup`
  questions); this keeps the median latency at one forward pass;
- request_seconds: no new reasoning once a request has taken this long, so a request with many questions stays bounded.
Without the limits, every question below its threshold reasons. --max-output-per-decision and --request-seconds also
work on their own.
"""
from __future__ import annotations

import collections
import contextlib
import json
import math
import time
from pathlib import Path

TYPES = ("choice", "noul", "score")
LIMITS = ("max_output_per_decision", "max_share", "request_seconds")


def settings(folder, share=None, budget: int | None = None, jevbench_limits: bool = False, **limits) -> dict:
    """The reasoning settings of a model folder (reasoning.json), with overrides.

    share: None (the released shares), one number for all types, or {type: share}. limits: values for LIMITS (None =
    not given); with jevbench_limits, the released value of every limit that is not given."""
    path = Path(folder) / "reasoning.json"
    if not path.is_file():
        raise ValueError(f"{folder}: this model has no reasoning settings (reasoning.json); Jeff 1.0 models answer in "
                         "one forward pass only")
    config = json.loads(path.read_text())
    shares = dict(config["share"])
    if share is not None:
        shares.update(share if isinstance(share, dict) else {q: share for q in TYPES})
    unknown = set(shares) - set(TYPES)
    if unknown:
        raise ValueError(f"think share for unknown question types {sorted(unknown)}")
    tau = {}
    for q in TYPES:
        s = float(shares[q])
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"think share must be between 0 and 1, got {s} for {q}")
        table = config["tau_table"][q]
        tau[q] = table[round(s * (len(table) - 1))]
    out = {key: config[key] for key in ("budget", "opener", "suffix", "read", "mix", "temperatures")}
    out.update(tau=tau, share=shares, window=config["jevbench_limits"]["window"],
               warmup=config["jevbench_limits"]["warmup"])
    if budget is not None:
        if int(budget) < 1:
            raise ValueError("the think budget must be at least 1 token")
        out["budget"] = int(budget)
    for key in LIMITS:
        value = limits.get(key)
        out[key] = value if value is not None else (config["jevbench_limits"][key] if jevbench_limits else None)
    return out


class LoraDelta:
    """The released LoRA adapter (adapter/, PEFT format) as forward hooks on the adapted linear layers. Inside base(),
    each adapted layer's output loses the adapter's term s B A x (computed in float32, as PEFT computes it), so a layer
    computes bf16(W + s B A) x - s B A x: the base model, up to one bfloat16 rounding of its weights. Outside base() the
    hooks change nothing: the released (merged) weights. No PEFT dependency."""

    def __init__(self, model, folder: Path):
        import torch
        from safetensors.torch import load_file
        folder = Path(folder)
        config = json.loads((folder / "adapter_config.json").read_text())
        if config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern") or config.get("lora_bias"):
            raise ValueError(f"{folder}: unsupported adapter options (DoRA, rank or alpha patterns, LoRA bias)")
        r, alpha = int(config["r"]), float(config["lora_alpha"])
        self.scale = alpha / math.sqrt(r) if config.get("use_rslora") else alpha / r
        tensors = load_file(str(folder / "adapter_model.safetensors"))
        names = sorted({k[: -len(".lora_A.weight")] for k in tensors if k.endswith(".lora_A.weight")})
        if not names or len(tensors) != 2 * len(names):
            raise ValueError(f"{folder}: expected one lora_A and one lora_B weight per adapted layer")
        modules = dict(model.named_modules())
        device = next(model.parameters()).device
        self.active, self.handles = False, []
        for name in names:
            path = name.removeprefix("base_model.model.")
            module = modules.get(path)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"{folder}: adapted layer {path} is not a linear layer of this model")
            a = tensors[name + ".lora_A.weight"].float().to(device)
            b = tensors[name + ".lora_B.weight"].float().to(device)
            if a.shape[1] != module.in_features or b.shape[0] != module.out_features or a.shape[0] != b.shape[1]:
                raise ValueError(f"{folder}: adapter shapes do not match layer {path}")
            self.handles.append(module.register_forward_hook(self._hook(a, b)))
        self.layers = len(names)

    def _hook(self, a, b):
        def hook(module, inputs, output):
            if not self.active:
                return None
            x = inputs[0]
            return output - (((x.to(a.dtype) @ a.T) @ b.T) * self.scale).to(output.dtype)
        return hook

    @contextlib.contextmanager
    def base(self):
        self.active = True
        try:
            yield
        finally:
            self.active = False


class Reasoner:
    """Decides which questions reason (wants), writes and reads the reasoning (think), and keeps the counts the
    limits need. `decider` is the jeff.model.Decider it serves; `rule` the serving rule."""

    def __init__(self, decider, config: dict, delta: LoraDelta, rule: dict):
        import torch
        from .prompting import LETTERS
        self.decider, self.config, self.delta, self.rule = decider, dict(config), delta, dict(rule or {})
        tok, head = decider.tokenizer, decider.head
        self._torch = torch
        self._opener = tok.encode(self.config["opener"], add_special_tokens=False)
        self._suffix = tok.encode(self.config["suffix"], add_special_tokens=False)
        # the answer letters as the LM writes them after the suffix: bare (the head's letter rows) and with a space
        weight = decider.model.get_output_embeddings().weight
        spaced = []
        for letter in LETTERS[: head.k_slots]:
            ids = tok.encode(self.config["suffix"] + " " + letter, add_special_tokens=False)
            ok = len(ids) == len(self._suffix) + 1 and ids[: len(self._suffix)] == self._suffix
            spaced.append(ids[-1] if ok else None)
        self._spaced_ok = [i is not None for i in spaced]
        with torch.no_grad():
            self._spaced_rows = torch.stack([weight[i] if i is not None else torch.zeros_like(weight[0])
                                             for i in spaced]).float().to(decider.device)
        stops = set([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else (tok.eos_token_id or []))
        for name in ("<end_of_turn>", "<turn|>", "<eos>"):
            if name in tok.get_vocab():
                stops.add(tok.convert_tokens_to_ids(name))
        self._stops = sorted(stops)
        self._pad = tok.pad_token_id if tok.pad_token_id is not None else self._stops[0]
        self._recent = collections.deque(maxlen=int(self.config["window"]))
        self.served = [0, 0]                    # questions answered and output tokens generated since the start
        self.deadline = None                    # time.perf_counter() after which no question of this request reasons

    def start_request(self) -> None:
        seconds = self.config.get("request_seconds")
        self.deadline = time.perf_counter() + float(seconds) if seconds is not None else None

    def wants(self, qtype: str, probs: list[float]) -> bool:
        """Unsure (calibrated confidence below tau[type]) and the limits allow it."""
        c = self.config
        temp = float(self.rule.get(f"{qtype}_t", 1.0) or 1.0)
        logs = [math.log(max(p, 1e-12)) / temp for p in probs]
        top = max(logs)
        confidence = 1.0 / sum(math.exp(v - top) for v in logs)       # largest calibrated probability
        ok = confidence < float(c["tau"][qtype])
        if ok and self.deadline is not None and time.perf_counter() > self.deadline:
            ok = False                                    # this request took request_seconds: answer the rest at once
        if ok and c.get("max_share") is not None and len(self._recent) >= int(c["warmup"]) and self._recent:
            ok = sum(self._recent) / len(self._recent) < float(c["max_share"])
        if ok and c.get("max_output_per_decision") is not None:
            n, spent = self.served
            ok = (spent + int(c["budget"])) / (n + 1) <= float(c["max_output_per_decision"])
        self._recent.append(1 if ok else 0)
        return ok

    def count(self, output_tokens: int) -> None:
        self.served[0] += 1
        self.served[1] += int(output_tokens)

    def think(self, ids: list[int], k: int, qtype: str, first: list[float]) -> tuple[list[float], list[float], int]:
        """(probabilities, logits, generated tokens) after reasoning; `first` = the one-pass option logits."""
        torch = self._torch
        from . import bidir
        decider, c = self.decider, self.config
        prompt = ids + self._opener
        x = torch.tensor([prompt], device=decider.device)
        with self.delta.base(), bidir.causal(decider.base):
            out = decider.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                                         max_new_tokens=int(c["budget"]), eos_token_id=self._stops,
                                         pad_token_id=self._pad, use_cache=True)
        gen = []
        for token in out[0, len(prompt):].tolist():
            if token in self._stops:
                break
            gen.append(token)
        h = decider.hidden(prompt + gen + self._suffix).float()
        head = decider.head
        if c.get("read", "letters") == "head":           # the trained head at the position after "Final answer:"
            z = head(h.to(torch.bfloat16).float())[0, :k]
        else:                                             # the LM's letter logits there, bare and spaced letters
            bare = head.cap(h @ head.base_rows.T)[0, :k]
            if all(self._spaced_ok[:k]):
                z = torch.logsumexp(torch.stack([bare, head.cap(h @ self._spaced_rows.T)[0, :k]]), dim=0)
            else:
                z = bare
        z = torch.log_softmax(z, dim=-1)
        if c.get("mix"):                                  # pooled with the one pass (mean of log-probabilities)
            z = (z + torch.log_softmax(torch.tensor(first, device=z.device, dtype=z.dtype), dim=-1)) / 2
        # The serving rule later divides the log-probabilities by its temperature for this type; scaling by
        # rule_t / think_t here makes the served distribution softmax(z / think_t), as fitted.
        think_t = float(c["temperatures"].get(qtype, 1.0))
        rule_t = float(self.rule.get(f"{qtype}_t", 1.0) or 1.0)
        logits = z * (rule_t / think_t)
        return torch.softmax(logits, dim=-1).tolist(), logits.tolist(), len(gen)
