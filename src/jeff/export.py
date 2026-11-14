"""Average trained runs (jeff.train) into one model and write it as a model folder that jeff.model serves.

    python -m jeff.export --name Jeff-1.1-Large --runs runs/Large-s1 runs/Large-s2 runs/Large-s3 --out models/Jeff-1.1-Large

Each released Jeff 1.1 model averages three runs (seeds 1, 2 and 3):
- the adapters become one LoRA adapter of rank 3r and alpha 3 alpha, with the runs' A matrices stacked and their B
  matrices divided by 3, so it adds exactly the mean of the three runs' weight changes;
- the head is the mean of the three heads (every tensor, including mu and sigma).
The adapter is then merged into the backbone's bf16 weights. The folder also keeps the adapter (adapter/), which the
optional reasoning mode uses. serving_rule.json gets the per-type temperatures of the released model (SERVING_RULES; we
fitted them on our own held-out data) and the yes/no commit (noul_edge 0.0).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from .prompting import MAX_OPTIONS, PROMPT_VERSION
from .train import load_base

SERVING_RULES = {
    "Jeff-1.1-Large": {"choice_t": 1.4933825289366305, "noul_t": 1.568051655383462, "score_t": 1.1143850361606575,
                       "noul_edge": 0.0},
    "Jeff-1.1-Qwen-27B": {"choice_t": 1.4222690751777431, "noul_t": 1.228609502367125, "score_t": 1.0107800781502563,
                          "noul_edge": 0.0},
    "Jeff-1.1-Medium": {"choice_t": 1.4222690751777431, "noul_t": 1.3545419763597555, "score_t": 1.061319082057769,
                        "noul_edge": 0.0},
    "Jeff-1.1-Qwen-4B": {"choice_t": 1.2900399774854812, "noul_t": 1.228609502367125, "score_t": 1.1701042879686905,
                         "noul_edge": 0.0},
}


def average(runs: list[Path]) -> tuple[dict, dict, dict]:
    """(adapter tensors, adapter config, head tensors) of the averaged model."""
    from safetensors.torch import load_file
    n = len(runs)
    configs = [json.loads((run / "train_config.json").read_text()) for run in runs]
    if any({k: v for k, v in c.items() if k != "seed"} != {k: v for k, v in configs[0].items() if k != "seed"}
           for c in configs):
        raise ValueError("the runs differ in more than the seed")
    adapters = [load_file(str(run / "adapter" / "adapter_model.safetensors")) for run in runs]
    tensors = {}
    for name in adapters[0]:
        parts = [a[name].float() for a in adapters]
        if ".lora_A." in name:
            tensors[name] = torch.cat(parts, dim=0).contiguous()
        elif ".lora_B." in name:
            tensors[name] = torch.cat([p / n for p in parts], dim=1).contiguous()
        else:
            raise ValueError(f"unexpected adapter tensor {name}")
    adapter_config = json.loads((runs[0] / "adapter" / "adapter_config.json").read_text())
    adapter_config.update(r=n * int(adapter_config["r"]), lora_alpha=n * int(adapter_config["lora_alpha"]))
    heads = [load_file(str(run / "head.safetensors")) for run in runs]
    head = {k: sum(h[k].float() for h in heads) / n for k in heads[0]}
    return tensors, adapter_config, head


def text_model_keys(out: Path) -> None:
    """Store Gemma 4 weights under the text model's own names (model.*), so plain transformers loads them.

    The loader reads the text part of the multimodal checkpoint (model.language_model.* -> model.*), and
    save_pretrained writes it back as model.language_model.*, while config.json describes the text model, which expects
    model.*. This renames the keys in every shard and in the index, and checks them against the model of config.json."""
    import transformers
    from safetensors import safe_open
    from safetensors.torch import save_file
    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:                                                   # small models: one file, no index
        with safe_open(str(out / "model.safetensors"), framework="pt") as f:
            index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in f.keys()}}
    rename = lambda k: "model." + k[len("model.language_model."):] if k.startswith("model.language_model.") else k  # noqa: E731
    for shard in sorted(set(index["weight_map"].values())):
        path = out / shard
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            if all(k == rename(k) for k in keys):
                continue
            tensors = {rename(k): f.get_tensor(k) for k in keys}
            meta = f.metadata() or {"format": "pt"}
        temporary = path.with_suffix(".tmp")
        save_file(tensors, str(temporary), metadata=meta)
        temporary.replace(path)
    index["weight_map"] = {rename(k): v for k, v in index["weight_map"].items()}
    if index_path.exists():
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    config = transformers.AutoConfig.from_pretrained(out, local_files_only=True)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config)
    expected = {k for k in model.state_dict() if not (k == "lm_head.weight" and config.tie_word_embeddings)}
    if expected != set(index["weight_map"]):
        raise RuntimeError(f"saved weights do not match {config.model_type}: "
                           f"missing {sorted(expected - set(index['weight_map']))[:5]}, "
                           f"extra {sorted(set(index['weight_map']) - expected)[:5]}")


def export(name: str, runs: list[Path], out: Path, device: str = "cuda") -> Path:
    from peft import PeftModel
    from safetensors.torch import save_file
    if name not in SERVING_RULES:
        raise ValueError(f"name must be one of {sorted(SERVING_RULES)} (it selects the released temperatures)")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    config = json.loads((runs[0] / "train_config.json").read_text())
    tensors, adapter_config, head = average(runs)
    base = config["model"]
    adapter_config.update(base_model_name_or_path=base["source"], revision=base.get("revision"))
    (out / "adapter").mkdir(parents=True)
    save_file(tensors, str(out / "adapter" / "adapter_model.safetensors"))
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n")
    model, tokenizer = load_base(base["source"], base.get("revision"), device)
    model = PeftModel.from_pretrained(model, str(out / "adapter")).merge_and_unload().eval()
    model.save_pretrained(str(out), max_shard_size="5GB")
    tokenizer.save_pretrained(str(out))
    text_model_keys(out)
    save_file({k: v.contiguous() for k, v in head.items()}, str(out / "head.safetensors"))
    head_config = json.loads((runs[0] / "head_config.json").read_text())
    (out / "head_config.json").write_text(json.dumps(head_config, indent=1) + "\n")
    (out / "serving_rule.json").write_text(json.dumps(SERVING_RULES[name], indent=1) + "\n")
    (out / "jeff_config.json").write_text(json.dumps({
        "name": name, "version": "1.1", "base_model": base["source"], "base_revision": base.get("revision"),
        "bidirectional": config.get("bidirectional"), "prompt_version": PROMPT_VERSION,
        "max_input_tokens": 8192, "max_options": MAX_OPTIONS, "noul_mapping": "tf-true-first", "dtype": "bfloat16",
    }, indent=1) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, choices=sorted(SERVING_RULES))
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders of jeff.train (seeds 1-3)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(export(args.name, args.runs, args.out, args.device))


if __name__ == "__main__":
    main()
