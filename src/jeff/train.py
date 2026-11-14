"""Train one Jeff 1.1 run: LoRA adapters on the backbone and the decision head (jeff.head), trained together on
DecisionBank-260k. The released models average three such runs (seeds 1, 2, 3; see jeff.export).

    torchrun --nproc_per_node=4 -m jeff.train --config configs/train/Jeff-1.1-Large.json --seed 1 --out runs/Large-s1

Every GPU holds a full bf16 copy of the model and reads its own share of the rows (row i goes to GPU i mod #GPUs).
Gradients are averaged over the GPUs after every step. The full training state (weights, optimizer, data position and
random states) is saved every `checkpoint_every` steps; started again with the same command and number of GPUs, a run
continues from its latest saved state. At the end, the run folder holds adapter/ (the LoRA adapter, PEFT format) and the
decision head (head.safetensors, head_config.json).

Training starts exactly at the backbone's own answer: the LoRA B matrices start at zero, the head's letter rows are the
LM-head rows of the letters A-P and its correction starts at zero. The loss is the cross-entropy against the row's target
distribution. Each time a row is used, its options (but not the ordered levels of score questions) are shuffled.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import bidir
from .head import DecisionHead
from .prompting import LETTERS, MAX_OPTIONS, encode_prompt

DATASET, SUBSET = "jgeuter/DecisionBank", "DecisionBank-260k"
STATS_ROWS = 2048          # rows for the head's feature statistics (mu, sigma)


# ----------------------------------------------------------------------------------------------------------- model

def load_base(source: str, revision: str | None, device: str = "cuda"):
    """The backbone in bf16, text part only: a Hugging Face repo at a pinned commit, or a local folder."""
    import transformers
    local = Path(source).is_dir()
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("Hugging Face models need a pinned 40-character commit revision")
    common = {"revision": None if local else revision, "local_files_only": local, "trust_remote_code": False}
    config = transformers.AutoConfig.from_pretrained(source, **common)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = transformers.AutoModelForCausalLM
    text_only = {"qwen3_5": "Qwen3_5ForCausalLM", "gemma4": "Gemma4ForCausalLM"}
    if config.model_type in text_only:      # the checkpoints are multimodal: load the language model only
        cls = getattr(transformers, text_only[config.model_type])
        config = config.get_text_config()
        if config.model_type.startswith("gemma4"):
            # Gemma 4 stores the text model under model.language_model.*; transformers has this rule for Qwen3.5 only
            from transformers.conversion_mapping import (get_checkpoint_conversion_mapping,
                                                         register_checkpoint_conversion_mapping)
            from transformers.core_model_loading import PrefixChange
            if get_checkpoint_conversion_mapping(config.model_type) is None:
                register_checkpoint_conversion_mapping(
                    config.model_type, [PrefixChange(prefix_to_remove="language_model", model_prefix="model")])
    model, loading = cls.from_pretrained(source, config=config, dtype=torch.bfloat16, device_map={"": str(device)},
                                         low_cpu_mem_usage=True, output_loading_info=True, **common)
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    return model.eval(), tokenizer


def letter_rows(model, tokenizer) -> torch.Tensor:
    """[16, d] fp32 LM-head rows of the letters A-P; each letter must be one token."""
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"answer letter {letter!r} is not one token")
        ids.append(encoded[0])
    return model.get_output_embeddings().weight.detach()[ids].float()


def softcap_of(model) -> float | None:
    """The LM's final-logit soft-capping (Gemma 4: 30.0), or None (Qwen)."""
    value = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    return float(value) if value else None


def decoder_of(model):
    """The decoder stack without the LM head, through the PEFT wrapper."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return getattr(inner, "model", inner)


def last_hidden(decoder, items: list[dict], pad_id: int, device) -> torch.Tensor:
    """[B, d] fp32 post-final-norm hidden states at each prompt's last token (right padding)."""
    width = max(len(it["ids"]) for it in items)
    ids = torch.full((len(items), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(items), width), dtype=torch.long)
    for j, it in enumerate(items):
        ids[j, : len(it["ids"])] = torch.tensor(it["ids"])
        attention[j, : len(it["ids"])] = 1
    ids, attention = ids.to(device), attention.to(device)
    out = decoder(input_ids=ids, attention_mask=bidir.attention_arg(decoder, ids, attention), use_cache=False)
    last = torch.tensor([len(it["ids"]) - 1 for it in items], device=device)
    return out.last_hidden_state[torch.arange(len(items), device=device), last].float()


def option_logits(head: DecisionHead, h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Head logits with the slots after each row's k options at -inf."""
    return head(h).masked_fill(torch.arange(head.k_slots, device=h.device)[None, :] >= k[:, None], float("-inf"))


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over rows of -sum_j t_j log p_j (slots with t_j = 0 are left out, so -inf logits give no NaN)."""
    log_p = F.log_softmax(logits, dim=-1).masked_fill(target <= 0, 0.0)
    return -(target * log_p).sum(dim=1).mean()


# ------------------------------------------------------------------------------------------------------------ data

def load_rows(source: str | None = None) -> list[dict]:
    """DecisionBank-260k in its file order: a .jsonl file, a folder of .jsonl shards, or (None) the Hugging Face
    dataset. JSON states (state_is_json) are parsed back into objects, as the models saw them."""
    if source is None:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=f"data/{SUBSET}/*.jsonl")) / "data" / SUBSET
    path = Path(source)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no .jsonl files in {path}")
    rows = []
    for file in files:
        with open(file) as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("state_is_json"):
                    row["state"] = json.loads(row["state"])
                rows.append(row)
    return rows


def shuffled(row: dict, rng: random.Random) -> dict:
    """The row with its options in a random order (label and target follow); score rows keep their level order."""
    if row["qtype"] == "score":
        return row
    order = list(range(len(row["options"])))
    rng.shuffle(order)
    out = dict(row, options=[row["options"][i] for i in order], label=order.index(row["label"]))
    if row.get("target_distribution") is not None:
        out["target_distribution"] = [row["target_distribution"][i] for i in order]
    return out


def encode(tokenizer, row: dict, max_tokens: int) -> dict | None:
    """Prompt ids, option count and target (the normalized target distribution, else the label); None for rows
    longer than max_tokens, which are left out."""
    try:
        ids = encode_prompt(tokenizer, row, max_tokens)
    except ValueError:
        return None
    target = torch.zeros(MAX_OPTIONS)
    dist = row.get("target_distribution")
    if dist is None:
        target[row["label"]] = 1.0
    else:
        target[: len(dist)] = torch.tensor(dist, dtype=torch.float32) / sum(dist)
    return {"ids": ids, "k": len(row["options"]), "t": target}


def length_batches(items: list[dict], token_budget: int, rng: random.Random, chunk: int = 64) -> list[list[int]]:
    """Batches of rows of similar length whose padded size stays within token_budget, in random order: rows are
    shuffled, sorted by length within blocks of chunk * 16, cut into batches, and the batches shuffled."""
    order = list(range(len(items)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), chunk * 16):
        batch, width = [], 0
        for i in sorted(order[start: start + chunk * 16], key=lambda i: len(items[i]["ids"])):
            n = len(items[i]["ids"])
            if batch and max(width, n) * (len(batch) + 1) > token_budget:
                batches.append(batch)
                batch, width = [], 0
            batch.append(i)
            width = max(width, n)
        if batch:
            batches.append(batch)
    rng.shuffle(batches)
    return batches


def lr_scale(step: int, config: dict) -> float:
    """Warmup-stable-decay: linear warmup, constant, then a cosine decay from decay_start to max_steps."""
    warmup, start, end = int(config["warmup"]), int(config["decay_start"]), int(config["max_steps"])
    if not 0 <= warmup <= start < end:
        raise ValueError("need 0 <= warmup <= decay_start < max_steps")
    scale = min(1.0, step / max(warmup, 1))
    if step > start:
        scale *= 0.5 * (1 + math.cos(math.pi * min(1.0, (step - start) / (end - start))))
    return max(scale, 1e-3)


# ---------------------------------------------------------------------------------------------- saved training state

def trainable(model) -> dict:
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def atomic_save(value, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def latest_state(out: Path, world: int) -> Path | None:
    done = sorted(marker.parent for marker in (out / "state").glob("step-*/complete.json"))
    if done and json.loads((done[-1] / "complete.json").read_text())["world"] != world:
        raise ValueError(f"{done[-1]} was saved with another number of GPUs")
    return done[-1] if done else None


def save_state(out: Path, step: int, model, head, optimizer, config: dict, local: dict, world: int, rank: int,
               keep: int = 2) -> None:
    folder = out / "state" / f"step-{step:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    atomic_save(local, folder / f"rank-{rank}.pt")
    if rank == 0:
        atomic_save({"config": config, "model": trainable(model), "head": head.state_dict(),
                     "optimizer": optimizer.state_dict()}, folder / "shared.pt")
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        (folder / "complete.json").write_text(json.dumps({"step": step, "world": world}) + "\n")
        done = sorted(marker.parent for marker in (out / "state").glob("step-*/complete.json"))
        for old in done[:-keep]:
            shutil.rmtree(old)
    if world > 1:
        torch.distributed.barrier()


def load_state(folder: Path, model, head, optimizer, config: dict, rank: int) -> dict:
    """Restore weights, head and optimizer; returns this GPU's data position and random states."""
    shared = torch.load(folder / "shared.pt", map_location="cpu", weights_only=False)   # files this code wrote
    if shared["config"] != config:
        raise ValueError(f"{folder} was saved with another configuration")
    with torch.no_grad():
        for name, p in trainable(model).items():
            p.copy_(shared["model"][name])
    head.load_state_dict(shared["head"])
    optimizer.load_state_dict(shared["optimizer"])
    return torch.load(folder / f"rank-{rank}.pt", map_location="cpu", weights_only=False)


# -------------------------------------------------------------------------------------------------------- training

def train(config: dict, seed: int, out: Path, data: str | None = None) -> Path:
    from peft import LoraConfig, get_peft_model
    config = {**config, "seed": seed}
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank() if world > 1 else 0
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', '0')}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    log = print if rank == 0 else (lambda *a, **k: None)
    torch.manual_seed(seed)
    rng = random.Random(seed * 1000 + rank)                # this GPU's option orders and batch order
    if not config.get("cudnn_attention", False):
        torch.backends.cuda.enable_cudnn_sdp(False)        # cuDNN attention plans again for every new input shape

    model, tokenizer = load_base(config["model"]["source"], config["model"].get("revision"), device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows_w0, softcap = letter_rows(model, tokenizer), softcap_of(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                             target_modules=lora["target_modules"], bias="none"))
    decoder = decoder_of(model)
    if config.get("bidirectional"):
        bidir.enable(decoder, config["bidirectional"])

    max_tokens, budget = int(config["max_tokens"]), int(config["batch_tokens"])
    rows = load_rows(data)
    head = DecisionHead(decoder.config.hidden_size, MAX_OPTIONS, softcap).to(device)
    with torch.no_grad():
        head.base_rows.copy_(rows_w0)
        head.delta.weight.zero_()
        head.delta.bias.zero_()
        sample = [it for it in (encode(tokenizer, r, max_tokens)
                                for r in random.Random(seed).sample(rows, min(STATS_ROWS, len(rows)))) if it]
        h = torch.cat([last_hidden(decoder, [sample[i] for i in batch], pad_id, device)
                       for batch in length_batches(sample, budget, random.Random(0))])
        head.mu.copy_(h.mean(0))
        head.sigma.copy_(h.std(0).clamp(min=1e-6))
    lora_params, head_params = list(trainable(model).values()), list(head.delta.parameters())
    if world > 1:                                          # every GPU starts from GPU 0's adapter and head
        for t in [*lora_params, *head_params, head.mu, head.sigma]:
            torch.distributed.broadcast(t.data, src=0)
    optimizer = torch.optim.AdamW([{"params": lora_params, "lr": config["lr_lora"], "base_lr": config["lr_lora"]},
                                   {"params": head_params, "lr": config["lr_head"], "base_lr": config["lr_head"]}],
                                  weight_decay=0.0)
    mine = rows[rank::world]
    log(f"{len(rows):,} rows, {world} GPUs; trainable: LoRA {sum(p.numel() for p in lora_params):,}, "
        f"head {sum(p.numel() for p in head_params):,}", flush=True)

    def new_epoch() -> list[list[dict]]:                   # fresh option orders, encoded and batched
        items = [it for it in (encode(tokenizer, shuffled(r, rng), max_tokens) for r in mine) if it]
        return [[items[i] for i in batch] for batch in length_batches(items, budget, rng)]

    step, batches, cursor, epoch_rng = 0, [], 0, None
    resume = latest_state(out, world) if rank == 0 else None
    if world > 1:
        shared = [resume]
        torch.distributed.broadcast_object_list(shared, src=0)
        resume = shared[0]
    if resume is not None:
        local = load_state(resume, model, head, optimizer, config, rank)
        step, cursor, epoch_rng = local["step"], local["cursor"], local["epoch_rng"]
        if epoch_rng is not None:                          # rebuild the current epoch's batches
            rng.setstate(epoch_rng)
            batches = new_epoch()
        rng.setstate(local["rng"])
        torch.set_rng_state(local["torch_rng"])
        if local["cuda_rng"] is not None:
            torch.cuda.set_rng_state(local["cuda_rng"], device)
        log(f"resumed at step {step} from {resume}", flush=True)
    elif rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_config.json").write_text(json.dumps(config, indent=1) + "\n")

    model.train()
    head.train()
    grad_accum, every = int(config["grad_accum"]), int(config["checkpoint_every"])
    started, losses = time.time(), []
    while step < int(config["max_steps"]):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            if cursor >= len(batches):                     # next pass over this GPU's rows
                epoch_rng = rng.getstate()
                batches, cursor = new_epoch(), 0
            chunk = batches[cursor]
            cursor += 1
            k = torch.tensor([it["k"] for it in chunk], device=device)
            target = torch.stack([it["t"] for it in chunk]).to(device)
            loss = soft_ce(option_logits(head, last_hidden(decoder, chunk, pad_id, device), k), target)
            (loss / grad_accum).backward()
            losses.append(float(loss.detach()))
        if world > 1:
            for p in lora_params + head_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                torch.distributed.all_reduce(p.grad)
                p.grad /= world
        norm = float(torch.nn.utils.clip_grad_norm_(lora_params + head_params, 1.0))
        if not math.isfinite(norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step + 1}; the update was not applied")
        step += 1
        scale = lr_scale(step, config)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        optimizer.step()
        if step % 10 == 0:
            log(f"step {step}: loss {sum(losses) / len(losses):.4f} (GPU 0, last 10 steps), lr scale {scale:.3f}, "
                f"{time.time() - started:.0f} s", flush=True)
            losses = []
        if step % every == 0 or step == int(config["max_steps"]):
            local = {"step": step, "cursor": cursor, "epoch_rng": epoch_rng, "rng": rng.getstate(),
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
            save_state(out, step, model, head, optimizer, config, local, world, rank)
    if rank == 0:
        from safetensors.torch import save_file
        model.save_pretrained(str(out / "adapter"))
        save_file({k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()}, str(out / "head.safetensors"))
        (out / "head_config.json").write_text(json.dumps({"kind": "residual", "d_model": head.d_model,
                                                          "k_slots": head.k_slots, "softcap": head.softcap}) + "\n")
        log(f"done: {out}", flush=True)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="configs/train/<model>.json")
    parser.add_argument("--seed", type=int, required=True, help="the released models average seeds 1, 2 and 3")
    parser.add_argument("--out", type=Path, required=True, help="run folder (a saved state there is resumed)")
    parser.add_argument("--data", help=f"DecisionBank-260k .jsonl file or folder (default: {DATASET} from Hugging Face)")
    args = parser.parse_args()
    train(json.loads(args.config.read_text()), args.seed, args.out, args.data)


if __name__ == "__main__":
    main()
