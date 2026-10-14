"""Load a released Jeff model and answer decisions: one forward pass per question, no generation.

A released model folder (or Hugging Face repo) holds the merged weights and tokenizer (standard transformers files),
the decision head (head.safetensors + head_config.json), the serving rule (serving_rule.json) and jeff_config.json.
Each question is rendered as a SemIf-format multiple-choice prompt (options labeled A, B, C, ...; chat template with
thinking disabled), the model reads it once, the head turns the last hidden state into one logit per option, and the
serving rule is applied to the softmax. This is the code path we evaluated with (jevhead.jeff), copied unchanged except
for the loader, which reads the released folder.

    from jeff import JeffModel
    model = JeffModel("jgeuter/Jeff-1.0-Large")
    model.decide({"ticket": "Refund not received after 14 days"},
                 {"route": {"type": "choice", "instructions": "Which team should handle this?",
                            "criteria": {"billing": "Payments and refunds", "tech": "Technical problems"}}})
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import typesafe as T
from .head import DecisionHead, collect_letter_rows, load_head_safetensors, model_softcap
from .prompting import encode_prompt

MAX_TOKENS = 8192          # as in all our evaluations; longer inputs are refused (JevBench counts them as wrong)
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
                                f"`hf download jgeuter/Jeff-1.0-Large --local-dir {text}`, or pass a repo id.")
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
    """One model + head; decide(row) -> option probabilities, option logits, input tokens, seconds."""

    def __init__(self, model, tokenizer, head=None, max_tokens: int = MAX_TOKENS):
        import torch
        self.model, self.tokenizer, self.max_tokens = model, tokenizer, max_tokens
        self.base = getattr(model, "model", model)
        self.device = next(model.parameters()).device
        if head is None:  # zero-shot letter readout = slice-initialized head
            rows = collect_letter_rows(model, tokenizer, 16)
            head = DecisionHead(rows.shape[1], rows.shape[0], abstain=False, kind="linear",
                                softcap=model_softcap(model)).float()  # Gemma 4: LM logit capping
            head.slice_init_(rows)
        self.readout = head.to(self.device).eval()
        self._torch = torch
        self.cudnn_min_tokens = None   # set: cuDNN attention only for inputs this long (see disable_cudnn_attention)

    def decide(self, row: dict, check_boundary: bool = False) -> tuple[list[float], list[float], int, float]:
        torch = self._torch
        ids, _, _ = encode_prompt(self.tokenizer, row, self.max_tokens, check_boundary=check_boundary)
        if self.cudnn_min_tokens is not None:
            torch.backends.cuda.enable_cudnn_sdp(len(ids) >= self.cudnn_min_tokens)
        start = time.perf_counter()
        with torch.inference_mode():
            hidden = self.base(input_ids=torch.tensor([ids], device=self.device), use_cache=False,
                               return_dict=True).last_hidden_state[:, -1]
            h = hidden.to(torch.bfloat16).float()  # as in our evaluations
            k = torch.tensor([len(row["options"])], device=self.device)
            logits, _ = self.readout.option_logits(h, k)
            logits = logits[0, : len(row["options"])]
            probs = torch.softmax(logits, dim=-1).tolist()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return probs, logits.tolist(), len(ids), time.perf_counter() - start


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
    commit=False keeps the calibrated P(yes) (temperatures still apply)."""

    def __init__(self, ref: str | Path, revision: str | None = None, device: str = "cuda",
                 rule: dict | None = None, cudnn_attention: bool = False, commit: bool = True):
        folder = model_folder(ref, revision)
        self.folder = folder
        self.config = json.loads((folder / "jeff_config.json").read_text())
        self.name = self.config.get("name", folder.name)
        self.rule = json.loads((folder / "serving_rule.json").read_text()) if rule is None else rule
        if not commit:
            self.rule = {k: v for k, v in self.rule.items() if k != "noul_edge"}
        self.noul_mapping = self.config.get("noul_mapping", "tf-true-first")
        if not cudnn_attention:
            disable_cudnn_attention()
        model, tokenizer = load_causal_model(folder, device=device)
        head = load_head_safetensors(folder).float()
        self.decider = Decider(model, tokenizer, head, int(self.config.get("max_input_tokens", MAX_TOKENS)))
        if not cudnn_attention:
            self.decider.cudnn_min_tokens = CUDNN_MIN_TOKENS

    def row(self, state, question: dict, key: str = "q") -> dict:
        return T.question_row_mapped(state, question, key, self.noul_mapping)

    def answer(self, qtype: str, option_ids: list[str], probabilities: list[float]) -> dict:
        return served_answer(qtype, option_ids, probabilities, self.rule)

    def decide(self, state, questions: dict) -> dict:
        """TypeSafe request: state + {name: {type, instructions, criteria}} -> {name: answer}."""
        answers = {}
        for key, question in questions.items():
            row = self.row(state, question, key)
            probs, _, _, _ = self.decider.decide(row)
            answers[key] = self.answer(question["type"], [o["id"] for o in row["options"]], probs)
        return answers
