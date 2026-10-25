# Jeff: Jev-style decision models from open LLMs

Serving code for the **Jeff** decision models: a state, a typed question, and its
options go in, and one forward pass returns a probability for each option. Jeff 1.1 models can also reason before
answering when they are unsure (off by default, see [Reasoning](#reasoning-jeff-11)).

- [New 1.1 suite](https://huggingface.co/collections/jgeuter/jeff-11):
    - [jgeuter/Jeff-1.1-Large](https://huggingface.co/jgeuter/Jeff-1.1-Large) (Backbone: [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it))
    - [jgeuter/Jeff-1.1-Qwen-27B](https://huggingface.co/jgeuter/Jeff-1.1-Qwen-27B) (Backbone: [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B))
    - [jgeuter/Jeff-1.1-Medium](https://huggingface.co/jgeuter/Jeff-1.1-Medium) (Backbone: [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it))
    - [jgeuter/Jeff-1.1-Qwen-4B](https://huggingface.co/jgeuter/Jeff-1.1-Qwen-4B) (Backbone: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B))
- [1.0 suite](https://huggingface.co/collections/jgeuter/jeff-10):
    - [jgeuter/Jeff-1.0-Large](https://huggingface.co/jgeuter/Jeff-1.0-Large) (Backbone: [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it))
    - [jgeuter/Jeff-1.0-Qwen-27B](https://huggingface.co/jgeuter/Jeff-1.0-Qwen-27B) (Backbone: [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B))
    - [jgeuter/Jeff-1.0-Medium](https://huggingface.co/jgeuter/Jeff-1.0-Medium) (Backbone: [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it))
    - [jgeuter/Jeff-1.0-Qwen-4B](https://huggingface.co/jgeuter/Jeff-1.0-Qwen-4B) (Backbone: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B))
- Technical Report: [Releasing the Inner Jev: A Simple Recipe for Open Decision Models](https://arxiv.org/abs/2610.17205)

## How it works

Each question is rendered as a multiple-choice prompt. A decision head turns the last token's hidden state $h$ into one
logit per option, and a softmax over the $K$ options gives the probabilities:

$$z = c\tanh(W_0 h / c) + A \mathrm{std}(h) + b, \qquad p = \mathrm{softmax}(z_1 / T, \dots, z_K / T),$$

where $W_0$ are the language-model-head rows of the answer letters (A, B, C, ...), $c = 30$ is Gemma's final-logit
soft-capping (Qwen models have none), $\mathrm{std}(h)$ standardizes each feature with statistics computed once on training data, and the
learned correction $A$, $b$ starts at zero, so training starts exactly at the backbone's own letter readout. $T$ is a
temperature per question type (choice, yes/no, score), fitted on our validation data and stored in the model's
`serving_rule.json` (for Jeff-1.1-Large: 1.49 for choice, 1.57 for yes/no, 1.11 for score). Jeff 1.1 models other
than Jeff-1.1-Large read the prompt with bidirectional attention in their full-attention layers, as they were trained
(`"bidirectional": "full"` in `jeff_config.json`).

Question types: `choice`, `noul` (yes/no; the answer is `P(yes)`) and `score` (ordered levels). The head has 16 answer
slots; questions with more options (up to 256) are answered in groups of 16 and a final round between the group
winners. The model reads up to 8,192 tokens. Longer inputs are condensed to at most 8192 tokens; `--long-input refuse` refuses
them instead.

## Reasoning (Jeff 1.1)

With `--reasoning on`, a question whose calibrated confidence is below a threshold gets a short reasoning
before the answer:

1. the base model (with LoRA off) writes up to 512 tokens (256 for Jeff-1.1-Qwen-4B) of reasoning;
2. the decision model reads the prompt, the reasoning and "Final answer:", and makes a second prediction;
3. that distribution is averaged with the one-pass answer (mean of the log-probabilities) and calibrated with its own temperature.

By default, reasoning is off. If turned on, the share of questions that reasons was selected on held-out data (for Jeff-1.1-Large: 0% of choice questions, 20% of yes/no, and 60% of score
questions; see `reasoning.json`). You can change this in two ways:

- `--think-share 0.3` (or `--think-share choice=0.2,noul=0.3,score=0.6`): the thresholds at which this share of our
  validation questions reasons, from 0 (never) to 1 (always);
- `--think-threshold 0.8` (or `--think-threshold noul=0.8,score=0.7`): a question reasons when its calibrated
  confidence (its largest probability) is below this value.

Limits on reasoning (optional):

- `--think-budget`: sets the maximum tokens per reasoning. The generated tokens are reported as `output_tokens`. Default 512 (256 for Jeff-1.1-Qwen-4B).
- `--max-output-per-decision L`: the mean output tokens per question, over everything the server has answered, never
  exceed L. A question reasons only if (output tokens so far + budget) / (questions so far + 1) is at most L;
- `--max-think-share S`: no new reasoning while more than a share S of the last 1,000 questions reasoned (after the
  first 100). Unlike `--think-share`, this is a hard cap on the questions this server actually receives;
- `--request-seconds T`: no new reasoning once a request has taken T seconds.

The reasoning engine (`--reasoning-engine`) can be set to the following values:

- `auto` (default): vLLM if it is installed (`pip install ".[vllm]"`) and the GPU can hold the model twice, otherwise
  transformers;
- `vllm`: the base model in vLLM, on the same GPU, while the merged model (base+LoRA) runs in `transformers`. The base model is written to `--cache-dir` (default `$JEFF_CACHE` or `~/.cache/jeff`) and reused at the next start. `--vllm-base` takes a base model folder or repo id instead. vLLM starts first
  and takes the model's size plus 12 GB of GPU memory (`--vllm-memory` to change it). Note you need to fit both the base weights and merged weights in GPU memory;
- `transformers`: no vLLM required. It needs no extra
  memory, and writes about 15 tokens per second on the 31B (69 ms per token; `--fast-reasoning` keeps a second copy of
  the base weights and gets 28 ms).

Note that vLLM rounds differently, so the reasoning text can differ slightly.

## Faster serving (optional)

These options make the server faster on large GPUs.

- `--compile`: compiles the decision forward pass with `torch.compile`. Startup takes a few minutes longer.
- `--fast-reasoning` (with `--reasoning on`): the base model that writes the reasoning is kept as a second copy of the
  weights, and generation uses a static KV cache, which transformers compiles. Jeff-1.1-Large then writes about 35
  tokens per second instead of 15 (on an H200). It needs the model's memory twice (e.g. an H200
  for the 31B and 27B models); without enough free memory the server falls back to the default. Compiles at
  startup (several minutes).
- For Qwen models: `pip install flash-linear-attention==0.5.2`: transformers then runs the linear-attention layers
  with these kernels. Use it together with `--compile`.

With `--compile --fast-reasoning`, the server needs about 8 to 9 minutes to start on an H200. The compiled
code is cached in `/tmp`; with Singularity/Apptainer, bind a folder on a local disk there (`-B <folder>:/tmp`), since its
default `/tmp` is too small (and vLLM fails to delete its temporary folders on NFS).

## Batching

Requests that arrive at the same time are answered together: the server sorts the waiting questions by length and runs
questions of similar length in one forward pass (up to `--max-batch` questions, default 32). It never waits for more
requests. With vLLM, waiting reasonings also run as one
batch. `--batching off` answers one request at a time.

- Memory: a batch holds at most `--max-batch-tokens` padded input tokens (questions x longest input). By default this
  is set at start from the free GPU memory (about 300,000 tokens for Jeff-1.1-Large on an H200).
- Reproducibility: Padding in batches can change the probabilities by numerical noise (0.004 on average with 8 clients), and vLLM can slightly change probabilities as well. On OpenDecisions, Capability with 8 clients stayed within 0.6 points of
  one request at a time. Use `--batching off` to reproduce our numbers exactly.
- Speed: batching helps most for the smaller models and for reasoning. Throughput with 8 clients on one H200, compared
  with one request at a time:

| Model | One pass | With reasoning (vLLM) |
|---|---|---|
| Jeff-1.1-Large | 1.3x | 2.2x |
| Jeff-1.1-Qwen-27B | 1.5x | 1.8x |
| Jeff-1.1-Medium | 1.7x | 2.5x |
| Jeff-1.1-Qwen-4B | 2.5x | 1.5x |

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

Options go after the model folder, e.g. with reasoning on:

```bash
docker run --gpus '"device=0"' -v "$JEFF_DIR":/model:ro -p 127.0.0.1:8013:8013 jeff \
  /model --host 0.0.0.0 --port 8013 --reasoning on
```

For reasoning written by vLLM, build the image with `--build-arg EXTRAS="vllm==0.30.0"` and mount a writable folder at
`/cache` (the base model is rebuilt there once, as large as the model; without the mount it is rebuilt at every start):

```bash
docker build --build-arg EXTRAS="vllm==0.30.0" -t jeff-vllm .
docker run --gpus '"device=0"' -v "$JEFF_DIR":/model:ro -v "$HOME/.cache/jeff":/cache -p 127.0.0.1:8013:8013 jeff-vllm \
  /model --host 0.0.0.0 --port 8013 --reasoning on
```

The container reads the weights from `/model` and needs no network access (it also runs with `--network none`). It
needs one GPU with enough memory (for the 31B and 27B models e.g. H100 80GB, or H200 when reasoning with vLLM); `--gpus` needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
on the host.

## Requirements

Tested with Python 3.12, torch 2.13.0 (CUDA 13.0), transformers 5.17.0 and accelerate 1.15.0 (`requirements.txt`).
Without Docker, the machine needs a C compiler (e.g. `gcc`; the image has one): Triton, which torch uses for some GPU
operations, compiles a small helper with it on the first request. The GPU driver must support CUDA 13.0 (R580 or
newer). The Qwen models need nothing extra: without `flash-linear-attention` and `causal-conv1d`, transformers runs their
linear-attention layers with a slower PyTorch implementation.

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

Each question is one forward pass (plus a reasoning, if it is on and the question is unsure); the questions of one
request are batched together (one after another with `--batching off`). Instructions and option descriptions that are
not text (objects, lists) are sent to the model as JSON text; an empty state as "(no state was given)". Invalid
requests (and, with `--long-input refuse`, inputs over 8,192 tokens, with a message naming the maximum context length)
get HTTP 422; a failure during the forward pass (for example, out of GPU memory) gets HTTP 500. Both come with
`{"error": "..."}`.

Yes/no answers: JevBench counts $0.2 < P(\text{yes}) < 0.8$ as wrong, so by default the server commits such answers to
0.199 or 0.801 (whichever edge is nearer, as `"urgent"` above). Start it with `--commit off` (Python: `commit=False`) to get
the calibrated $P(\text{yes})$ instead.

## Python

```python
from jeff import JeffModel
model = JeffModel("jgeuter/Jeff-1.1-Large")      # or a local folder with the weights; commit=False: calibrated P(yes)
# reasoning=True, think_share=0.3, think_budget=256, long_input="refuse": as the server options
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
