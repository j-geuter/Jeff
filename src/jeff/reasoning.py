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
s of similar questions reason (on other data the share can differ). The released shares are the ones we selected on
held-out data; `--think-share` picks others (0: never, 1: always). `--think-threshold` sets the confidence threshold
of a type directly instead (a question reasons when its calibrated confidence is below it).

Limits (all off unless set; `--jevbench-limits on` sets the ones we evaluate on JevBench with, so that the model stays
within the board's cost and latency caps):
- max_output_per_decision L: a question reasons only if (output tokens so far + budget) / (questions so far + 1) <= L,
  counted over everything this server has answered. The mean output per question therefore never exceeds L, whatever
  the share of unsure questions. Questions still reasoning count with the full budget until they finish;
- max_share (`--max-think-share`): no new reasoning while more than this share of the last `window` questions reasoned
  (after `warmup` questions); this keeps the median latency at one forward pass;
- request_seconds: no new reasoning once a request has taken this long, so a request with many questions stays bounded.
Without limits, every question below its threshold reasons.
"""
from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

TYPES = ("choice", "noul", "score")
LIMITS = ("max_output_per_decision", "max_share", "request_seconds")


def per_type(value, name: str) -> dict:
    """{type: value} from one number (every type) or a {type: number} dict."""
    out = dict(value) if isinstance(value, dict) else {q: value for q in TYPES}
    unknown = set(out) - set(TYPES)
    if unknown:
        raise ValueError(f"{name} for unknown question types {sorted(unknown)}")
    for q, v in out.items():
        if not 0.0 <= float(v) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {v} for {q}")
    return {q: float(v) for q, v in out.items()}


def settings(folder, share=None, budget: int | None = None, jevbench_limits: bool = False, threshold=None,
             **limits) -> dict:
    """The reasoning settings of a model folder (reasoning.json), with overrides.

    share: None (the released shares), one number for all types, or {type: share}. threshold: None, one number or
    {type: confidence threshold}, used directly instead of a share's threshold (a type cannot have both). limits:
    values for LIMITS (None = not given); with jevbench_limits, the released value of every limit that is not given."""
    path = Path(folder) / "reasoning.json"
    if not path.is_file():
        raise ValueError(f"{folder}: this model has no reasoning settings (reasoning.json); Jeff 1.0 models answer in "
                         "one forward pass only")
    config = json.loads(path.read_text())
    given = per_type(share, "think share") if share is not None else {}
    fixed = per_type(threshold, "think threshold") if threshold is not None else {}
    both = sorted(set(given) & set(fixed))
    if both:
        raise ValueError(f"set either a think share or a think threshold for a question type, not both: {both}")
    shares = {**config["share"], **given}
    tau = {}
    for q in TYPES:
        table = config["tau_table"][q]
        tau[q] = fixed[q] if q in fixed else table[round(float(shares[q]) * (len(table) - 1))]
    for q in fixed:
        shares[q] = None                                  # set by its threshold, not by a share
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


def base_weight(w, a, b, scale: float):
    """bf16(W - s B A): a base-model weight from a released (merged) one, in float32 as PEFT merges."""
    return (w.float() - scale * (b.float() @ a.float())).to(w.dtype)


def read_adapter(folder: Path) -> tuple[float, dict]:
    """(scale s, {module path: (A, B)}) of a released LoRA adapter (adapter/, PEFT format)."""
    from safetensors.torch import load_file
    folder = Path(folder)
    config = json.loads((folder / "adapter_config.json").read_text())
    if config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern") or config.get("lora_bias"):
        raise ValueError(f"{folder}: unsupported adapter options (DoRA, rank or alpha patterns, LoRA bias)")
    r, alpha = int(config["r"]), float(config["lora_alpha"])
    scale = alpha / math.sqrt(r) if config.get("use_rslora") else alpha / r
    tensors = load_file(str(folder / "adapter_model.safetensors"))
    names = sorted({k[: -len(".lora_A.weight")] for k in tensors if k.endswith(".lora_A.weight")})
    if not names or len(tensors) != 2 * len(names):
        raise ValueError(f"{folder}: expected one lora_A and one lora_B weight per adapted layer")
    return scale, {name.removeprefix("base_model.model."): (tensors[name + ".lora_A.weight"],
                                                           tensors[name + ".lora_B.weight"]) for name in names}


class LoraDelta:
    """The released LoRA adapter (adapter/, PEFT format) as forward hooks on the adapted linear layers. Inside base(),
    each adapted layer's output loses the adapter's term s B A x (computed in float32, as PEFT computes it), so a layer
    computes bf16(W + s B A) x - s B A x: the base model, up to one bfloat16 rounding of its weights. Outside base() the
    hooks change nothing: the released (merged) weights. No PEFT dependency.

    copy=True (--fast-reasoning): instead of hooks, a second copy of the adapted layers' weights, bf16(W - s B A), is
    swapped in inside base() (a pointer swap). Generation then runs at the speed of a plain model (about 1.5x faster on
    the 31B), at the cost of the adapted layers' memory once more; the base model is again exact up to bfloat16
    rounding, rounded once more than with the hooks."""

    def __init__(self, model, folder: Path, copy: bool = False):
        import torch
        self.scale, lora = read_adapter(folder)
        modules = dict(model.named_modules())
        device = next(model.parameters()).device
        self.active, self.handles, self.copies = False, [], []
        for path, (a, b) in lora.items():
            module = modules.get(path)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"{folder}: adapted layer {path} is not a linear layer of this model")
            a, b = a.float().to(device), b.float().to(device)
            if a.shape[1] != module.in_features or b.shape[0] != module.out_features or a.shape[0] != b.shape[1]:
                raise ValueError(f"{folder}: adapter shapes do not match layer {path}")
            if copy:
                with torch.no_grad():
                    self.copies.append((module, base_weight(module.weight, a, b, self.scale)))
            else:
                self.handles.append(module.register_forward_hook(self._hook(a, b)))
        self.layers = len(lora)

    def _hook(self, a, b):
        def hook(module, inputs, output):
            if not self.active:
                return None
            x = inputs[0]
            return output - (((x.to(a.dtype) @ a.T) @ b.T) * self.scale).to(output.dtype)
        return hook

    @contextlib.contextmanager
    def base(self):
        released = [m.weight.data for m, _ in self.copies]
        for m, w in self.copies:
            m.weight.data = w
        self.active = True
        try:
            yield
        finally:
            self.active = False
            for (m, _), w in zip(self.copies, released):
                m.weight.data = w

    @staticmethod
    def copy_fits(model, folder: Path, spare_bytes: int = 8 << 30) -> bool:
        """Whether the GPU has room for a second copy of the adapted layers (plus spare_bytes)."""
        import torch
        from safetensors import safe_open
        with safe_open(str(Path(folder) / "adapter_model.safetensors"), framework="pt") as f:
            names = {k[len("base_model.model."):-len(".lora_A.weight")] for k in f.keys() if k.endswith(".lora_A.weight")}
        modules = dict(model.named_modules())
        need = sum(modules[n].weight.numel() * modules[n].weight.element_size() for n in names if n in modules)
        device = next(model.parameters()).device
        return device.type != "cuda" or torch.cuda.mem_get_info(device)[0] > need + spare_bytes


# Release architectures vLLM has no class for, and the vLLM text class that runs the same model (the 12B "unified"
# Gemma 4 has the language model of Gemma 4; vLLM only registers its multimodal class)
VLLM_ARCHITECTURES = {"Gemma4UnifiedForCausalLM": ("Gemma4ForCausalLM", "gemma4_text")}
VLLM_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "special_tokens_map.json", "tokenizer.model", "model.safetensors.index.json")


def cache_dir(cache: str | Path | None = None) -> Path:
    """Where rebuilt base weights go: `cache`, else $JEFF_CACHE, else ~/.cache/jeff."""
    return Path(cache or os.environ.get("JEFF_CACHE") or Path.home() / ".cache" / "jeff").expanduser()


def derive_base(folder: Path, cache: str | Path | None = None, device: str | None = None) -> Path:
    """A folder with the base model rebuilt from a release, for vLLM: the released (merged) weights minus the LoRA
    adapter (adapter/), each adapted weight bf16(W - s B A) as with --fast-reasoning, all other files as released.
    Written once under cache_dir(cache) (as large as the model's weights), then reused; no download of the original
    base model is needed."""
    import torch
    from safetensors.torch import load_file, save_file
    folder = Path(folder)
    index_file = folder / "model.safetensors.index.json"
    if index_file.is_file():                              # sharded checkpoint
        shards = sorted(set(json.loads(index_file.read_text())["weight_map"].values()))
    else:                                                 # one file
        shards = [f.name for f in sorted(folder.glob("model*.safetensors"))]
    if not shards:
        raise FileNotFoundError(f"{folder}: no model*.safetensors weights")
    digest = hashlib.sha256(json.dumps([[f, (folder / f).stat().st_size] for f in shards]).encode())
    if index_file.is_file():
        digest.update(index_file.read_bytes())
    digest.update((folder / "adapter" / "adapter_config.json").read_bytes())
    with open(folder / "adapter" / "adapter_model.safetensors", "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    name = json.loads((folder / "jeff_config.json").read_text()).get("name", folder.name)
    out = cache_dir(cache) / f"{name}-base-{digest.hexdigest()[:16]}"
    if (out / ".complete").is_file():
        return out
    need = sum((folder / shard).stat().st_size for shard in shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < need * 1.05:
        raise OSError(f"rebuilding the base model for vLLM needs {need / 1e9:.0f} GB free in {out.parent}; set "
                      f"JEFF_CACHE (or --cache-dir) to a larger disk, or pass the original base model with --vllm-base")
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    scale, lora = read_adapter(folder / "adapter")
    done = set()
    started = time.perf_counter()
    for shard in shards:
        tensors = load_file(str(folder / shard))
        for key, w in tensors.items():
            path = key.removesuffix(".weight")
            if key.endswith(".weight") and path in lora:
                a, b = lora[path]
                tensors[key] = base_weight(w.to(device), a.to(device), b.to(device), scale).cpu()
                done.add(path)
        save_file(tensors, str(tmp / shard), metadata={"format": "pt"})
    missing = sorted(set(lora) - done)
    if missing:
        shutil.rmtree(tmp, ignore_errors=True)
        raise ValueError(f"{folder}: adapted layers without a weight in the checkpoint: {missing[:3]}")
    for file in VLLM_FILES:
        if (folder / file).is_file() and file != "config.json":
            shutil.copy2(folder / file, tmp / file)
    config = json.loads((folder / "config.json").read_text())
    arch = (config.get("architectures") or [""])[0]
    if arch in VLLM_ARCHITECTURES:
        config["architectures"], config["model_type"] = [VLLM_ARCHITECTURES[arch][0]], VLLM_ARCHITECTURES[arch][1]
    (tmp / "config.json").write_text(json.dumps(config, indent=2))
    (tmp / ".complete").write_text(json.dumps({"release": str(folder), "layers": len(done)}) + "\n")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()                          # vLLM starts next and needs this memory free
    try:
        tmp.rename(out)
    except OSError:                                       # another process finished first
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"rebuilt the base model for vLLM in {out} ({need / 1e9:.0f} GB, {time.perf_counter() - started:.0f} s)",
          flush=True)
    return out


class VLLMGenerator:
    """--reasoning-engine vllm: the base model in vLLM writes the reasoning (greedy, as the transformers path); the
    decision passes stay in this server. vLLM runs its engine in its own process on the same GPU. It must start before
    the decision model is loaded (it checks that its share of the GPU memory is free) and takes `memory` of the GPU's
    memory: the base model's weights, the KV cache and CUDA graphs for up to max_seqs sequences at once. `base` is a
    folder or repo id of the base model: by default the one rebuilt from the release (derive_base), or the original
    one (--vllm-base)."""

    def __init__(self, base: str, revision: str | None, tokenizer: str, max_len: int, memory: float, max_seqs: int = 1):
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError("--reasoning-engine vllm needs vLLM: pip install \".[vllm]\" (vllm==0.30.0)") from e
        sizes = sorted({n for n in (1, 2, 4, 8, 16, 32, 64) if n <= max_seqs} | {max_seqs})
        # max_num_batched_tokens = one full-length prompt: vLLM then reserves the same working memory for several
        # sequences as for one, which leaves the KV cache of a full-length sequence in the same memory share
        kwargs = dict(model=base, revision=revision, tokenizer=tokenizer, dtype="bfloat16", max_model_len=max_len,
                      gpu_memory_utilization=memory, max_num_seqs=max_seqs, max_num_batched_tokens=max_len,
                      enable_prefix_caching=False, seed=0, compilation_config={"cudagraph_capture_sizes": sizes})
        try:
            self.llm = LLM(**kwargs, language_model_only=True)     # multimodal checkpoints: the text model only
        except (TypeError, ValueError):
            self.llm = LLM(**kwargs)

    def generate_many(self, prompts: list[list[int]], budget: int, stops: list[int]) -> list[list[int]]:
        """The written tokens for each prompt, in order (vLLM runs them as one batch)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        params = SamplingParams(temperature=0.0, max_tokens=budget, stop_token_ids=stops, detokenize=False,
                                skip_special_tokens=False)
        out = self.llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in out]

    def generate(self, ids: list[int], budget: int, stops: list[int]) -> list[int]:
        return self.generate_many([ids], budget, stops)[0]


def vllm_memory(folder: Path, spare_bytes: int = 12 << 30) -> float:
    """vLLM's share of the GPU memory: the base model's weights (as large as the released ones) plus spare_bytes for
    the KV cache of one full-length sequence (vLLM reserves it at start: 7.4 GB for the 31B), activations and the CUDA
    graphs. Raises ValueError if the GPU cannot hold the model twice."""
    import torch
    weights = sum(f.stat().st_size for f in Path(folder).glob("model*.safetensors"))
    total = torch.cuda.mem_get_info()[1]
    if 2 * weights + 2 * spare_bytes > total:
        raise ValueError(f"--reasoning-engine vllm needs the model twice on the GPU ({2 * weights / 1e9:.0f} GB of "
                         f"weights); this GPU has {total / 1e9:.0f} GB")
    return round((weights + spare_bytes) / total, 3)


def vllm_usable(folder: Path, device: str = "cuda") -> str | None:
    """Why --reasoning-engine auto cannot use vLLM here (None: it can)."""
    import importlib.util
    if not str(device).startswith("cuda"):
        return "not on a GPU"
    if importlib.util.find_spec("vllm") is None:
        return "vLLM is not installed (pip install \".[vllm]\")"
    try:
        vllm_memory(folder)
    except ValueError as error:
        return str(error)
    return None


class Reasoner:
    """Decides which questions reason (wants), writes and reads the reasoning (think, or its steps prompt, write,
    read_ids and read for batches), and keeps the counts the limits need. `decider` is the jeff.model.Decider it serves;
    `rule` the serving rule."""

    def __init__(self, decider, config: dict, delta: LoraDelta | None, rule: dict, static_cache: bool = False,
                 generator: VLLMGenerator | None = None):
        import torch
        from .prompting import LETTERS
        self.decider, self.config, self.delta, self.rule = decider, dict(config), delta, dict(rule or {})
        self.static_cache = static_cache          # --fast-reasoning: a static KV cache (with the weight copy)
        self.generator = generator                # --reasoning-engine vllm: the base model in vLLM writes
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
        self.pending = [0, 0]                   # questions still reasoning, and their budgets (counted as spent)
        self.deadline = None                    # time.perf_counter() after which no question of this request reasons

    def start_request(self) -> None:
        self.deadline = self.request_deadline()

    def request_deadline(self) -> float | None:
        seconds = self.config.get("request_seconds")
        return time.perf_counter() + float(seconds) if seconds is not None else None

    def wants(self, qtype: str, probs: list[float], deadline: float | None | str = "request") -> bool:
        """Unsure (calibrated confidence below tau[type]) and the limits allow it. A question that reasons counts with
        its full budget until count() reports its real output. deadline: the question's request deadline (default:
        the one start_request() set)."""
        c = self.config
        deadline = self.deadline if deadline == "request" else deadline
        temp = float(self.rule.get(f"{qtype}_t", 1.0) or 1.0)
        logs = [math.log(max(p, 1e-12)) / temp for p in probs]
        top = max(logs)
        confidence = 1.0 / sum(math.exp(v - top) for v in logs)       # largest calibrated probability
        ok = confidence < float(c["tau"][qtype])
        if ok and deadline is not None and time.perf_counter() > deadline:
            ok = False                                    # this request took request_seconds: answer the rest at once
        if ok and c.get("max_share") is not None and len(self._recent) >= int(c["warmup"]) and self._recent:
            ok = sum(self._recent) / len(self._recent) < float(c["max_share"])
        if ok and c.get("max_output_per_decision") is not None:
            n, spent = self.served
            ok = ((spent + self.pending[1] + int(c["budget"])) / (n + self.pending[0] + 1)
                  <= float(c["max_output_per_decision"]))
        self._recent.append(1 if ok else 0)
        if ok:
            self.pending[0] += 1
            self.pending[1] += int(c["budget"])
        return ok

    def count(self, output_tokens: int, reasoned: bool = False) -> None:
        """A question is answered (reasoned: it was admitted by wants())."""
        self.served[0] += 1
        self.served[1] += int(output_tokens)
        if reasoned:
            self.pending[0] -= 1
            self.pending[1] -= int(self.config["budget"])

    def prompt(self, ids: list[int]) -> list[int]:
        return ids + self._opener

    def write(self, prompts: list[list[int]]) -> list[list[int]]:
        """The reasoning of each prompt (base model, greedy, up to the budget; cut before the first stop token): one
        vLLM batch, or one transformers generation after another."""
        torch = self._torch
        from . import bidir
        budget, decider = int(self.config["budget"]), self.decider
        if self.generator is not None:
            outs = self.generator.generate_many(prompts, budget, self._stops)
        else:
            outs = []
            for prompt in prompts:
                x = torch.tensor([prompt], device=decider.device)
                with self.delta.base(), bidir.causal(decider.base):
                    out = decider.model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                                                 max_new_tokens=budget, eos_token_id=self._stops,
                                                 pad_token_id=self._pad, use_cache=True,
                                                 **({"cache_implementation": "static"} if self.static_cache else {}))
                outs.append(out[0, len(prompt):].tolist())
        cut = []
        for written in outs:
            gen = []
            for token in written:
                if token in self._stops:
                    break
                gen.append(token)
            cut.append(gen)
        return cut

    def read_ids(self, prompt: list[int], gen: list[int]) -> list[int]:
        """What the decision model reads after the reasoning: prompt, reasoning, "Final answer:"."""
        return prompt + gen + self._suffix

    def read(self, h, k: int, qtype: str, first: list[float]) -> tuple[list[float], list[float]]:
        """(probabilities, logits) from the decision model's hidden state h [1, d] after read_ids; `first` = the
        one-pass option logits."""
        torch = self._torch
        c, head = self.config, self.decider.head
        h = h.float()
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
        return torch.softmax(logits, dim=-1).tolist(), logits.tolist()

    def think(self, ids: list[int], k: int, qtype: str, first: list[float]) -> tuple[list[float], list[float], int]:
        """(probabilities, logits, generated tokens) after reasoning; `first` = the one-pass option logits."""
        prompt = self.prompt(ids)
        gen = self.write([prompt])[0]
        probs, logits = self.read(self.decider.hidden(self.read_ids(prompt, gen)), k, qtype, first)
        return probs, logits, len(gen)
