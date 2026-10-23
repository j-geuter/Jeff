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

Which questions reason. By default the thresholds come from our validation data: the share of questions that
reason is the one we selected on held-out data (for Jeff-1.1-Large: no choice questions, 20% of yes/no and 60% of score
questions; see `reasoning.json`). You can change this in two ways:

- `--think-share 0.3` (or `--think-share choice=0.2,noul=0.3,score=0.6`): the thresholds at which this share of our
  validation questions reasons, from 0 (never) to 1 (always). On your data the share can come out higher or lower;
- `--think-threshold 0.8` (or `--think-threshold noul=0.8,score=0.7`): a question reasons when its calibrated
  confidence (its largest probability) is below this value.

`--think-budget` sets the most tokens per reasoning. The generated tokens are reported as `output_tokens`.

Limits on reasoning (none by default; each is optional):

- `--max-output-per-decision L`: the mean output tokens per question, over everything the server has answered, never
  exceed L. A question reasons only if (output tokens so far + budget) / (questions so far + 1) is at most L; questions
  still reasoning count with their full budget;
- `--max-think-share S`: no new reasoning while more than a share S of the last 1,000 questions reasoned (after the
  first 100). Unlike `--think-share`, this is a hard cap on the questions this server actually receives;
- `--request-seconds T`: no new reasoning once a request has taken T seconds.

`--jevbench-limits on` sets the values we evaluate JevBench with (Jeff-1.1-Large: 35 tokens, 45%, 30 s). Our JevBench
submission (reasoning on, these limits, reasoning written by transformers, no batching) is the separate repository
[Jeff-JevBench](https://github.com/j-geuter/Jeff-JevBench), which pins the release we tested (v1.2.0).

What writes the reasoning (`--reasoning-engine`):

- `auto` (default): vLLM if it is installed (`pip install ".[vllm]"`) and the GPU can hold the model twice, otherwise
  transformers;
- `vllm`: the base model in vLLM, on the same GPU, while the decisions stay in this server. The base model is rebuilt
  once from the release itself (the released weights minus the LoRA adapter in `adapter/`), so nothing else is
  downloaded. It is written to `--cache-dir` (default `$JEFF_CACHE` or `~/.cache/jeff`; as large as the model's
  weights) and reused at the next start. `--vllm-base` takes a base model folder or repo id instead. vLLM starts first
  and takes the model's size plus 12 GB of GPU memory (`--vllm-memory` to change it): an H200 for the 31B and 27B
  models. On an H200, Jeff-1.1-Large writes about 50 tokens per second (20 ms per token);
- `transformers`: the decision model itself, with the adapter removed by hooks while it writes. It needs no extra
  memory, and writes about 15 tokens per second on the 31B (69 ms per token; `--fast-reasoning` keeps a second copy of
  the base weights and gets 31 ms).

vLLM rounds differently, so the reasoning text can differ in a few words between the engines. On OpenDecisions, the
Capability with each engine stayed within 0.5 points of the other for all four models.

## Faster serving (optional)

These options make the server faster on large GPUs. They change the probabilities by numerical noise only (about 0.005
on average, at most about 0.15 on single questions; the same size as changing how the input is padded), and we measured
the released scores with them as well (see the model cards).

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

With `--compile --fast-reasoning`, the server needs about 8 to 9 minutes to start on an H200 (compilation). The compiled
code is cached in `/tmp`; with Singularity/Apptainer, bind a folder on a local disk there (`-B <folder>:/tmp`), since its
default `/tmp` is too small (and vLLM fails to delete its temporary folders on NFS).

## Batching

Requests that arrive at the same time are answered together. The server takes all waiting questions, sorts them by
length and runs questions of similar length through the model in one forward pass (at most `--max-batch` questions,
default 32). It never waits for more requests to arrive: a request that finds the server idle is answered at once, on
its own. With vLLM writing the reasoning, all waiting reasonings run as one vLLM batch while the server goes on
answering other questions; with transformers, they run one after another. `--batching off` serves one request at a
time and its questions one after another.

Memory: a batch takes at most `--max-batch-tokens` padded input tokens (questions x longest input). By default this cap
is measured at start for your model and GPU: the free GPU memory after loading (and after vLLM took its share), minus
5% of the GPU (at least 2 GB), divided by the memory one full-length (8,192-token) forward pass needs per token. On an
H200 this gives about 300,000 tokens for Jeff-1.1-Large without vLLM.

Reproducibility: the numbers we report (paper, model cards, JevBench) were measured with one request at a time. With
one request at a time, batching changes nothing: on 500 OpenDecisions questions, the answers of Jeff-1.1-Large were
identical to those of `--batching off`. When questions share a batch (several requests at once, or several questions
in one request), their inputs are padded to the same length and the probabilities change by numerical noise. With 8
clients sending OpenDecisions questions at once to Jeff-1.1-Large, a question's probabilities changed by 0.004 on
average (99% of questions: under 0.045, at most 0.125), and 6 of 2,389 yes/no answers near P(yes) = 0.5 moved to the
other side of the 0.2-0.8 band. With vLLM writing the reasoning, reasonings that run together can also come out with
different words than alone (vLLM's batched kernels round differently), so the answer of a single question that
reasons can change a lot; over OpenDecisions, Capability with 8 clients stayed within 0.3 points of one request at a
time for the 27B, 12B and 4B models. Use `--batching off` to reproduce our numbers exactly.

Speed: batching raises throughput most for the smaller models and for reasoning. A forward pass of Jeff-1.1-Large on a
typical question (300 tokens) already keeps an H200 busy, so batching its one-pass answers adds little. With 8 clients
sending OpenDecisions questions at once to one H200, compared with one request at a time:

| Model | One pass (questions per second) | With reasoning, vLLM (questions per second) |
|---|---|---|
| Jeff-1.1-Large | 9.9 -> 13.0 (1.3x) | |
| Jeff-1.1-Qwen-27B | | 1.7 -> 3.0 (1.8x) |
| Jeff-1.1-Medium | 15.9 -> 26.8 (1.7x) | 0.9 -> 2.3 (2.5x) |
| Jeff-1.1-Qwen-4B | 12.8 -> 31.8 (2.5x) | 4.6 -> 6.9 (1.5x) |

Each request then waits for the others: the median time per request of Jeff-1.1-Large went from 0.06 to 0.4 s.

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

For reasoning written by vLLM, build the image with `--build-arg EXTRAS="vllm==0.30.0"` and mount a writable folder at
`/cache` (the base model is rebuilt there once, as large as the model; without the mount it is rebuilt at every start):

```bash
docker build --build-arg EXTRAS="vllm==0.30.0" -t jeff-vllm .
docker run --gpus '"device=0"' -v "$JEFF_DIR":/model:ro -v "$HOME/.cache/jeff":/cache -p 127.0.0.1:8013:8013 jeff-vllm \
  /model --host 0.0.0.0 --port 8013 --reasoning on
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
