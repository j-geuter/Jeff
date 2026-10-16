# Jeff: Jev-style decision models from open LLMs

Serving code for the **Jeff 1.0** Jev-style decision models: a state, a typed question, and its options go in, and
one forward pass returns a probability for each option.

| Model | Base model (+ LoRA, merged) | Download | GPU memory | Tested revision |
|---|---|---|---|---|
| [jgeuter/Jeff-1.0-Large](https://huggingface.co/jgeuter/Jeff-1.0-Large) | Gemma 4 31B | 61 GB | 80 GB | `275325a5eab52cce4f109a09ee3832f2eb89fe2f` |
| [jgeuter/Jeff-1.0-Qwen-27B](https://huggingface.co/jgeuter/Jeff-1.0-Qwen-27B) | Qwen3.6-27B | 54 GB | 80 GB | `eb99f94481c493f5a1e9f61d371168215a4b422d` |
| [jgeuter/Jeff-1.0-Medium](https://huggingface.co/jgeuter/Jeff-1.0-Medium) | Gemma 4 12B | 24 GB | 40 GB | `1fcdb4d648eec28c4f95f2a1cf2fd0e456a556cb` |
| [jgeuter/Jeff-1.0-Qwen-4B](https://huggingface.co/jgeuter/Jeff-1.0-Qwen-4B) | Qwen3.5-4B | 8 GB | 16 GB | `5f441c5f385ab8fe4889dc597fa48bd1d372e19e` |

All four are served the same way; the examples below use Jeff-1.0-Large, swap in another repo id and revision from the table.

## How it works

Each question is rendered as a multiple-choice prompt. A decision head turns the last hidden state $h$ into one
logit per option, and a softmax over the $K$ options gives the probabilities:

$$z = c\tanh(W_0 h / c) + A \mathrm{std}(h) + b, \qquad p = \mathrm{softmax}(z_1, \dots, z_K),$$

where $W_0$ are the language-model-head rows of the answer letters (A, B, C, ...), $c = 30$ is Gemma's final-logit
soft-capping (Qwen models have none), $\mathrm{std}(h)$ standardizes each feature with statistics computed once on training data, and the
learned correction $A$, $b$ starts at zero, so training starts exactly at the model's own letter readout.

Question types: `choice` (2 to 16 options), `noul` (yes/no; the answer is `P(yes)`) and `score` (ordered levels,
2 to 16). Inputs up to 8,192 tokens.

## Run without Docker

```bash
pip install -r requirements.txt && pip install .
jeff-serve jgeuter/Jeff-1.0-Large --revision 275325a5eab52cce4f109a09ee3832f2eb89fe2f --host 127.0.0.1 --port 8013
```

On the first start this downloads the weights (about 61 GB) into the Hugging Face cache, which is `~/.cache/huggingface`
unless you set `HF_HOME` (set it to a disk with enough space). `--revision` pins the version we tested; leave it out to
get the newest. Instead of the repo id you can also pass a folder that already holds the weights.

## Run with Docker

The container has no network access, so download the weights first, to any folder with about 61 GB free
(`hf` comes with `pip install huggingface_hub`):

```bash
export JEFF_DIR=$HOME/models/Jeff-1.0-Large          # any folder you like
hf download jgeuter/Jeff-1.0-Large --revision 275325a5eab52cce4f109a09ee3832f2eb89fe2f --local-dir "$JEFF_DIR"
docker build -t jeff .
docker run --gpus '"device=0"' -v "$JEFF_DIR":/model:ro -p 8013:8013 jeff
python scripts/smoke_test.py --endpoint http://127.0.0.1:8013      # check it
```

The container reads the weights from `/model`. It needs one GPU with the memory given in the table (for the 31B and
27B models e.g. H100 80GB, H200 or RTX PRO 6000 96GB).

## Requirements

Tested with Python 3.12, torch 2.13.0 (CUDA 13.0), transformers 5.17.0 and accelerate 1.15.0 (`requirements.txt`).
Without Docker, the machine needs a C compiler (e.g. `gcc`; the image has one): Triton, which torch uses for some GPU
operations, compiles a small helper with it on the first request. The GPU driver must support CUDA 13.0 (R580 or
newer). The Qwen models need nothing extra: without `flash-linear-attention` and `causal-conv1d`, transformers runs their
linear-attention layers with a slower PyTorch implementation (same results; we tested this way).

## Request/response format

```
POST /v1/systemone
{"state": {...} or "text",
 "model": "Jeff-1.0-Large",
 "questions": {
   "route":    {"type": "choice", "instructions": "Which team should handle this ticket?",
                "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"}},
   "urgent":   {"type": "noul", "instructions": "Does this need a reply today?",
                "criteria": {"true": "Reply today", "false": "Can wait"}},
   "severity": {"type": "score", "instructions": "How severe is the problem?",
                "criteria": ["trivial", "minor", "moderate", "serious", "critical"]}}}
->
{"answers": {"route":    {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "technical": 0.03}},
             "urgent":   {"type": "noul", "noul": 0.801},
             "severity": {"type": "score", "score": 2, "probabilities": {"0": 0.01, "1": 0.2, ...}}},
 "model": "Jeff-1.0-Large", "usage": {"input_tokens": 612, "output_tokens": 0}, "runtime": {...}}
```

Each question is one forward pass; questions of one request are answered one after another.

Yes/no answers: JevBench counts $0.2 < P(\text{yes}) < 0.8$ as wrong, so by default the server commits such answers to
0.199 or 0.801 (whichever edge is nearer, as `"urgent"` above). Start it with `--no-commit` (Python: `commit=False`) to get
the calibrated $P(\text{yes})$ instead.

## Python

```python
from jeff import JeffModel
model = JeffModel("jgeuter/Jeff-1.0-Large")      # or a local folder with the weights; commit=False: calibrated P(yes)
model.decide({"ticket": "Refund not received after 14 days"},
             {"route": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Payments and refunds", "tech": "Technical problems"}}})
```

## Licence

The code is Apache-2.0 (`LICENSE`); it contains code derived from SemIf and JevBench (MIT), see `NOTICE`. The model
weights have their own licence, given in the model card.
