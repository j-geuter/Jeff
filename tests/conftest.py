"""Tiny random Jeff release folders (CPU, no downloads): a Gemma 4 or Qwen3.5 text model with a few layers, a
character tokenizer, a random LoRA merged into the weights (and shipped in adapter/), a decision head, the serving rule
and, for 1.1 folders, reasoning.json. They have the files and formats of the real releases."""
import json

import pytest
import torch

RULE = {"choice_t": 1.3, "noul_t": 1.2, "score_t": 1.1, "noul_edge": 0.0}
TARGETS = {"gemma": ["q_proj", "v_proj", "gate_proj", "down_proj"],
           "qwen": ["q_proj", "v_proj", "in_proj_qkv", "out_proj", "gate_proj"]}
SPECIAL = ["<pad>", "<eos>", "<unk>", "<turn|>"]
TEMPLATE = ("{% for m in messages %}<|{{ m.role }}>{{ m.content }}<turn|>{% endfor %}"
            "{% if add_generation_prompt %}<|model>{% endif %}")
REASONING = {"budget": 4, "opener": "Let me check the deciding facts first.\n", "suffix": "\n\nFinal answer:",
             "read": "letters", "mix": True, "temperatures": {"choice": 1.4, "noul": 1.1, "score": 1.2},
             "share": {"choice": 0.0, "noul": 0.2, "score": 0.6},
             "jevbench_limits": {"max_output_per_decision": 2.0, "max_share": 0.45, "window": 1000, "warmup": 100,
                                 "request_seconds": 30.0}}


def char_tokenizer():
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast
    chars = [chr(i) for i in range(32, 127)] + ["\n", "ü", "€"]
    vocab = {tok: i for i, tok in enumerate(SPECIAL + chars)}
    core = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    core.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>",
                                  additional_special_tokens=["<turn|>"])
    tok.chat_template = TEMPLATE
    return tok


def tiny_config(kind: str, vocab: int):
    import transformers
    common = dict(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, max_position_embeddings=8192, eos_token_id=1, pad_token_id=0,
                  bos_token_id=None)
    if kind == "gemma":
        return transformers.AutoConfig.for_model(
            "gemma4_text", global_head_dim=32, sliding_window=8, final_logit_softcapping=30.0,
            layer_types=["sliding_attention", "full_attention"] * 2, hidden_size_per_layer_input=16,
            vocab_size_per_layer_input=vocab, **common)
    return transformers.AutoConfig.for_model(
        "qwen3_5_text", layer_types=["linear_attention", "full_attention"] * 2, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, **common)


def build(folder, kind: str = "gemma", bidirectional: str | None = None, reasoning: bool = True,
          max_input_tokens: int = 8192, seed: int = 0) -> dict:
    """Write a release folder; returns {"base": the base model (bfloat16), "lora": {layer: (A, B)}, "scale": s}."""
    import transformers
    from safetensors.torch import save_file
    tok = char_tokenizer()
    torch.manual_seed(seed)
    base = transformers.AutoModelForCausalLM.from_config(tiny_config(kind, len(tok))).to(torch.bfloat16).eval()
    merged = transformers.AutoModelForCausalLM.from_config(tiny_config(kind, len(tok))).to(torch.bfloat16).eval()
    merged.load_state_dict(base.state_dict())
    r, alpha = 4, 8
    scale, lora, tensors = alpha / r, {}, {}
    modules = dict(merged.named_modules())
    for name, module in base.named_modules():
        if isinstance(module, torch.nn.Linear) and name.split(".")[-1] in TARGETS[kind]:
            a = torch.randn(r, module.in_features) * 0.05      # s B A about half the size of W, as in training
            b = torch.randn(module.out_features, r) * 0.05
            lora[name] = (a, b)
            tensors[f"base_model.model.{name}.lora_A.weight"] = a.contiguous()
            tensors[f"base_model.model.{name}.lora_B.weight"] = b.contiguous()
            with torch.no_grad():                         # as PEFT merges: bf16(W + s B A)
                w = modules[name].weight
                w.copy_((w.float() + scale * (b @ a)).to(w.dtype))
    folder.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(folder)
    tok.save_pretrained(folder)
    (folder / "adapter").mkdir()
    save_file(tensors, str(folder / "adapter" / "adapter_model.safetensors"))
    (folder / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": r, "lora_alpha": alpha, "target_modules": TARGETS[kind], "use_rslora": False}))
    letters = [tok.convert_tokens_to_ids(c) for c in "ABCDEFGHIJKLMNOP"]
    rows = merged.get_output_embeddings().weight.detach()[letters].float()
    d = rows.shape[1]
    torch.manual_seed(seed + 1)
    save_file({"base_rows": rows.contiguous(), "delta.weight": torch.randn(16, d) * 0.05, "delta.bias": torch.zeros(16),
               "mu": torch.randn(d) * 0.1, "sigma": torch.ones(d)}, str(folder / "head.safetensors"))
    softcap = merged.config.final_logit_softcapping if kind == "gemma" else None
    (folder / "head_config.json").write_text(json.dumps({"kind": "residual", "d_model": d, "k_slots": 16,
                                                         "softcap": softcap}))
    (folder / "serving_rule.json").write_text(json.dumps(RULE))
    config = {"name": f"tiny-{kind}", "prompt_version": "direct-options-v1", "noul_mapping": "tf-true-first",
              "max_input_tokens": max_input_tokens}
    if bidirectional:
        config["bidirectional"] = bidirectional
    (folder / "jeff_config.json").write_text(json.dumps(config))
    if reasoning:
        # thresholds 0.3 .. 0.99 for shares 0.01 .. 0.99; share 0: never (-1), share 1: always (2)
        table = [-1.0] + [0.3 + 0.69 * i / 99 for i in range(1, 100)] + [2.0]
        (folder / "reasoning.json").write_text(json.dumps({**REASONING, "tau_table": {q: table for q in RULE_TYPES}}))
    return {"base": base, "lora": lora, "scale": scale}


RULE_TYPES = ("choice", "noul", "score")


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    """tiny(kind, bidirectional, reasoning=True, max_input_tokens=8192) -> (folder, build info); built once each."""
    cache = {}

    def get(kind="gemma", bidirectional=None, reasoning=True, max_input_tokens=8192):
        key = (kind, bidirectional, reasoning, max_input_tokens)
        if key not in cache:
            folder = tmp_path_factory.mktemp("tiny") / f"{kind}-{bidirectional}-{reasoning}-{max_input_tokens}"
            cache[key] = (folder, build(folder, kind, bidirectional, reasoning, max_input_tokens))
        return cache[key]
    return get
