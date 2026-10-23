# Jeff: Jev-style decision models from open LLMs

Serving code for the **Jeff 1.1** and **Jeff 1.0** Jev-style decision models: a state, a typed question, and its
options go in, and one forward pass returns a probability for each option. Jeff 1.1 models can also reason before
answering when they are unsure (off by default, see [Reasoning](#reasoning-jeff-11)).

| Model | Base model (+ LoRA, merged) | Download | GPU memory | Tested revision |
|---|---|---|---|---|
| [jgeuter/Jeff-1.1-Large](https://huggingface.co/jgeuter/Jeff-1.1-Large) | Gemma 4 31B | 64 GB | 80 GB | `REVISION-LARGE` |
| [jgeuter/Jeff-1.1-Qwen-27B](https://huggingface.co/jgeuter/Jeff-1.1-Qwen-27B) | Qwen3.6-27B | 56 GB | 80 GB | `REVISION-QWEN27B` |
| [jgeuter/Jeff-1.1-Medium](https://huggingface.co/jgeuter/Jeff-1.1-Medium) | Gemma 4 12B | 26 GB | 40 GB | `REVISION-MEDIUM` |
| [jgeuter/Jeff-1.1-Qwen-4B](https://huggingface.co/jgeuter/Jeff-1.1-Qwen-4B) | Qwen3.5-4B | 9 GB | 16 GB | `REVISION-QWEN4B` |
| [jgeuter/Jeff-1.0-Large](https://huggingface.co/jgeuter/Jeff-1.0-Large) | Gemma 4 31B | 61 GB | 80 GB | `275325a5eab52cce4f109a09ee3832f2eb89fe2f` |
| [jgeuter/Jeff-1.0-Qwen-27B](https://huggingface.co/jgeuter/Jeff-1.0-Qwen-27B) | Qwen3.6-27B | 54 GB | 80 GB | `eb99f94481c493f5a1e9f61d371168215a4b422d` |
| [jgeuter/Jeff-1.0-Medium](https://huggingface.co/jgeuter/Jeff-1.0-Medium) | Gemma 4 12B | 24 GB | 40 GB | `1fcdb4d648eec28c4f95f2a1cf2fd0e456a556cb` |
| [jgeuter/Jeff-1.0-Qwen-4B](https://huggingface.co/jgeuter/Jeff-1.0-Qwen-4B) | Qwen3.5-4B | 8 GB | 16 GB | `5f441c5f385ab8fe4889dc597fa48bd1d372e19e` |

All are served the same way; the examples below use Jeff-1.1-Large, swap in another repo id and revision from the
table. Jeff 1.0 models are served as before (the same answers as with v1.1.1), except that they now also answer long
inputs and questions with more than 16 options (see below).

## How it works

Each question is rendered as a multiple-choice prompt. A decision head turns the last hidden state $h$ into one
logit per option, and a softmax over the $K$ options gives the probabilities:

$$z = c\tanh(W_0 h / c) + A \mathrm{std}(h) + b, \qquad p = \mathrm{softmax}(z_1 / T, \dots, z_K / T),$$

where $W_0$ are the language-model-head rows of the answer letters (A, B, C, ...), $c = 30$ is Gemma's final-logit
soft-capping (Qwen models have none), $\mathrm{std}(h)$ standardizes each feature with statistics computed once on training data, and the
learned correction $A$, $b$ starts at zero, so training starts exactly at the model's own letter readout. $T$ is a
temperature per question type (choice, yes/no, score), fitted on our validation data and stored in the model's
`serving_rule.json` (for Jeff-1.1-Large: 1.49 for choice, 1.57 for yes/no, 1.11 for score). Jeff 1.1 models other
than Jeff-1.1-Large read the prompt with bidirectional attention in their full-attention layers, as they were trained
(`"bidirectional": "full"` in `jeff_config.json`).

Question types: `choice`, `noul` (yes/no; the answer is `P(yes)`) and `score` (ordered levels). The head has 16 answer
slots; questions with more options (up to 256) are answered in groups of 16 and a final round between the group
winners (an option's probability is P(its group's winner wins the final) x P(it wins its group)). The model reads up
to 8,192 tokens. Longer inputs keep the parts that mention the question (lines, sentences or JSON entries with its
identifiers and words, in their original order, with a note that the rest was left out; `--long-input refuse` refuses
them instead). States that the prompt cannot hold (empty, a bare number) are sent as text.

## Reasoning (Jeff 1.1)

With `--reasoning on`, a question whose calibrated confidence is below a threshold for its type gets a short reasoning
before the answer:

1. the base model (the released weights minus the LoRA adapter in `adapter/`, causal attention) writes up to 512
   tokens (256 for Jeff-1.1-Qwen-4B) greedily after "Let me check the deciding facts first.";
2. the decision model reads the prompt, the reasoning and "Final answer:", and its letter logits there give a second
   distribution over the options;
3. that distribution is averaged with the one-pass answer (mean of the log-probabilities) and calibrated with its own
   temperature per type.

The thresholds come from our validation data: by default the share of questions that reason is the one we selected
on held-out data (for Jeff-1.1-Large: no choice questions, 20% of yes/no and 60% of score questions; see
`reasoning.json`). `--think-share 0.3` (or `--think-share choice=0.2,noul=0.3,score=0.6`) sets other shares, from 0
(never) to 1 (always); `--think-budget` sets the most tokens per reasoning. The generated tokens are reported as
`output_tokens`.

`--jevbench-limits on` adds the limits we evaluate on JevBench with, which keep the model within the board's cost and
latency caps:

- the mean output tokens per question, over everything the server answered, never exceed a limit (for Jeff-1.1-Large
  35): a question reasons only if (output tokens so far + budget) / (questions so far + 1) is at most the limit;
- no new reasoning while more than 45% of the last 1,000 questions reasoned;
- no new reasoning once a request has taken 30 seconds.

`--max-output-per-decision` and `--request-seconds` also work on their own. For JevBench:

```bash
jeff-serve jgeuter/Jeff-1.1-Large --revision REVISION-LARGE --reasoning on --jevbench-limits on --port 8013
```

## Faster serving (optional)

These options make the server faster on large GPUs. They change the probabilities by numerical noise only (about 0.005
on average, at most about 0.15 on single questions; the same size as changing how the input is padded), and we measured
the released scores with them as well (see the model cards). With `--reasoning-engine vllm`, the reasoning text itself
can differ in a few words from the default (vLLM rounds differently), which changes the answers of the questions that
reason a little more; on OpenDecisions, Capability stayed within 0.5 points of the default for all four models.

- `--compile`: compiles the decision forward pass with `torch.compile`. On an H200, the median time per decision drops
  from 60 to 43 ms for Jeff-1.1-Large and from 32 to 19 ms for Jeff-1.1-Medium. Startup takes a few minutes longer.
- `--fast-reasoning` (with `--reasoning on`): the base model that writes the reasoning is kept as a second copy of the
  weights, and generation uses a static KV cache, which transformers compiles. Jeff-1.1-Large then writes about 35
  tokens per second instead of 15 (28 instead of 69 ms per token on an H200). It needs the model's memory twice (an H200
  for the 31B and 27B models); without enough free memory the server falls back to the default. The compilation runs at
  startup (several minutes).
- For the Qwen models, `pip install flash-linear-attention==0.5.2`: transformers then runs the linear-attention layers
  with these kernels instead of its PyTorch fallback (Jeff-1.1-Qwen-27B: 93 to 61 ms per decision, 53 ms with
  `--compile`). Use it together with `--compile`: without it, compiling the Qwen models takes hours.

- `--reasoning-engine vllm` (with `--reasoning on`; `pip install ".[vllm]"`, or the Docker image built with
  `--build-arg EXTRAS="vllm==0.30.0"`): the original base model, run by vLLM on the same GPU, writes the reasoning; the
  decisions stay in this server. vLLM starts first and takes the base model's size plus 12 GB of the GPU memory
  (`--vllm-memory` to change it); the GPU must hold the model twice (an H200 for the 31B and 27B models). On an H200,
  Jeff-1.1-Large writes about 53 tokens per second (19 ms per token, against 69 by default and 31 with
  `--fast-reasoning`). vLLM needs the base model's own weights (`base_model` and `base_revision` in the model's
  `jeff_config.json`; for Jeff-1.1-Large about 62 GB). By default they come from the Hugging Face cache (downloaded on
  the first start); `--vllm-base <folder>` uses a folder instead. With Docker, mount that folder too:

  ```bash
  hf download google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --local-dir "$BASE_DIR"
  docker build --build-arg EXTRAS="vllm==0.30.0" -t jeff-vllm .
  docker run --gpus '"device=0"' -v "$JEFF_DIR":/model:ro -v "$BASE_DIR":/base:ro -p 127.0.0.1:8013:8013 jeff-vllm \
    /model --host 0.0.0.0 --port 8013 --compile --reasoning on --reasoning-engine vllm --vllm-base /base
  ```

With `--compile --fast-reasoning`, the server needs about 8 to 9 minutes to start on an H200 (compilation). The compiled
code is cached in `/tmp`; with Singularity/Apptainer, bind a folder on a local disk there (`-B <folder>:/tmp`), since its
default `/tmp` is too small (and vLLM fails to delete its temporary folders on NFS).

## Run without Docker

```bash
pip install -r requirements.txt && pip install .
jeff-serve jgeuter/Jeff-1.1-Large --revision REVISION-LARGE --host 127.0.0.1 --port 8013
```

On the first start this downloads the weights (about 64 GB) into the Hugging Face cache, which is `~/.cache/huggingface`
unless you set `HF_HOME` (set it to a disk with enough space). `--revision` pins the version we tested; leave it out to
get the newest. Instead of the repo id you can also pass a folder that already holds the weights.

## Run with Docker

The image holds the code and its dependencies, but not the weights. Download the weights first, to any folder with
about 64 GB free (`hf` comes with `pip install huggingface_hub`):

```bash
export JEFF_DIR=$HOME/models/Jeff-1.1-Large          # any folder you like
hf download jgeuter/Jeff-1.1-Large --revision REVISION-LARGE --local-dir "$JEFF_DIR"
docker build -t jeff .
docker run --gpus '"device=0"' -v "$JEFF_DIR":/model:ro -p 127.0.0.1:8013:8013 jeff
python scripts/smoke_test.py --endpoint http://127.0.0.1:8013      # check it
```

Options go after the model folder, e.g. with reasoning and the JevBench limits:

```bash
docker run --gpus '"device=0"' -v "$JEFF_DIR":/model:ro -p 127.0.0.1:8013:8013 jeff \
  /model --host 0.0.0.0 --port 8013 --reasoning on --jevbench-limits on
```

The container reads the weights from `/model` and needs no network access (it also runs with `--network none`). It
needs one GPU with the memory given in the table (for the 31B and 27B models e.g. H100 80GB, H200 or RTX PRO 6000
96GB); `--gpus` needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host. `-p 127.0.0.1:8013:8013` makes the server reachable only from this machine. The server has no
authentication, so do not expose it to the internet.

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
 "model": "Jeff-1.1-Large",
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
 "model": "Jeff-1.1-Large", "usage": {"input_tokens": 612, "output_tokens": 0}, "runtime": {...}}
```

Each question is one forward pass (plus a reasoning, if it is on and the question is unsure); questions of one request
are answered one after another. Invalid requests (and, with `--long-input refuse`, inputs over 8,192 tokens) get HTTP
422; a failure during the forward pass (for example, out of GPU memory) gets HTTP 500. Both come with
`{"error": "..."}`.

Yes/no answers: JevBench counts $0.2 < P(\text{yes}) < 0.8$ as wrong, so by default the server commits such answers to
0.199 or 0.801 (whichever edge is nearer, as `"urgent"` above). Start it with `--no-commit` (Python: `commit=False`) to get
the calibrated $P(\text{yes})$ instead.

## Python

```python
from jeff import JeffModel
model = JeffModel("jgeuter/Jeff-1.1-Large")      # or a local folder with the weights; commit=False: calibrated P(yes)
# reasoning=True, jevbench_limits=True, think_share=0.3, think_budget=256, long_input="refuse": as the server options
model.decide({"ticket": "Refund not received after 14 days"},
             {"route": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Payments and refunds", "tech": "Technical problems"}}})
```

## Tests

```bash
pip install ".[test]" && pytest                                  # CPU tests, with tiny random models (no download)
python scripts/smoke_test.py --endpoint http://127.0.0.1:8013   # a running server's answers vs. ours
```

This test checks the outputs on three example questions against the probabilities we got for the same model
(`scripts/smoke_expected.json`).

## Licence

The code is Apache-2.0 (`LICENSE`); it contains code derived from SemIf and JevBench (MIT), see `NOTICE`. The model
weights are CC BY-NC 4.0, as given in the model cards.

## Citation

If you find this repository helpful, please consider citing our paper:

```bibtex
@misc{geuter2026releasinginnerjevsimple,
      title={Releasing the Inner Jev: A Simple Recipe for Open Decision Models}, 
      author={Jonathan Geuter and Andrew Cheng and David Alvarez-Melis},
      year={2026},
      eprint={2610.17205},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.17205}, 
}
```
