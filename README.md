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

## Run without Docker

```bash
pip install -r requirements.txt && pip install .
jeff-serve jgeuter/Jeff-1.1-Large --revision 970c21af246a44c5527f7109fca129ab6015c941 --host 127.0.0.1 --port 8013
```

On the first start this downloads the weights (about 64 GB) into the Hugging Face cache, which is `~/.cache/huggingface`
unless you set `HF_HOME` (set it to a disk with enough space). `--revision` pins the version we tested; leave it out to
get the newest. Instead of the repo id you can also pass a folder that already holds the weights.

## Run with Docker

The image holds the code and its dependencies, but not the weights. Download the weights first, to any folder with
about 64 GB free (`hf` comes with `pip install huggingface_hub`):

```bash
export JEFF_DIR=$HOME/models/Jeff-1.1-Large          # any folder you like
hf download jgeuter/Jeff-1.1-Large --revision 970c21af246a44c5527f7109fca129ab6015c941 --local-dir "$JEFF_DIR"
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

## Python

```python
from jeff import JeffModel
model = JeffModel("jgeuter/Jeff-1.1-Large")      # or a local folder with the weights; commit=False: calibrated P(yes)
# reasoning=True, think_share=0.3, think_budget=256, long_input="refuse": as the server options
model.decide({"ticket": "Refund not received after 14 days"},
             {"route": {"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {"billing": "Payments and refunds", "tech": "Technical problems"}}})
```

## Requirements

Tested with Python 3.12, torch 2.13.0 (CUDA 13.0), transformers 5.17.0 and accelerate 1.15.0 (`requirements.txt`).
Without Docker, the machine needs a C compiler (e.g. `gcc`; the image has one): Triton, which torch uses for some GPU
operations, compiles a small helper with it on the first request. The GPU driver must support CUDA 13.0 (R580 or
newer). The Qwen models need nothing extra: without `flash-linear-attention` and `causal-conv1d`, transformers runs their
linear-attention layers with a slower PyTorch implementation.

## Reasoning (Jeff 1.1)

With `--reasoning on`, a question whose calibrated confidence is below a threshold gets a short reasoning
before the answer:

1. the base model (with LoRA off) writes up to 512 tokens (256 for Jeff-1.1-Qwen-4B) of reasoning;
2. the decision model reads the prompt, the reasoning and "Final answer:", and makes a second prediction;
3. that distribution is averaged with the one-pass answer (mean of the log-probabilities) and calibrated with its own temperature.

By default, reasoning is off. If turned on, you can set the share of questions that reasons in two ways:

- `--think-share 0.3` (or `--think-share choice=0.2,noul=0.3,score=0.6`): the thresholds at which this share of questions reasons (measured on validation data), from 0 (never) to 1 (always);
- `--think-threshold 0.8` (or `--think-threshold noul=0.8,score=0.7`): a question reasons when its calibrated
  confidence is below this value.

Limits on reasoning (optional):

- `--think-budget`: sets the maximum tokens per reasoning. Default 512 (256 for Jeff-1.1-Qwen-4B).
- `--max-output-per-decision L`: the mean output tokens per question, over everything the server has answered.
- `--max-think-share S`: no new reasoning while more than a share S of the last 1,000 questions reasoned (after the
  first 100).
- `--request-seconds T`: no new reasoning once a request has taken T seconds.
- `--decision-seconds S`: every question is answered within about S seconds (approximate with vLLM). A reasoning
  that runs out of time stops early and is read as it stands; `--read-partial off` drops it instead.

`--reasoning-config FILE` replaces the model's reasoning
settings with the ones in a JSON file.

The reasoning engine (`--reasoning-engine`) can be set to the following values:

- `auto` (default): vLLM if it is installed (`pip install ".[vllm]"`) and the GPU can hold the model twice, otherwise
  transformers;
- `vllm`: the base model in vLLM, on the same GPU, while the merged model (base+LoRA) runs in `transformers`. `--vllm-base` takes a base model folder or repo id instead. Note you need to fit both the base weights and merged weights in GPU memory;
- `transformers`: no vLLM required. It needs no extra
  memory, and writes about 15 tokens per second on the 31B (69 ms per token; `--fast-reasoning` keeps a second copy of
  the base weights and gets 28 ms).

Note that vLLM rounds differently, so the reasoning text can differ slightly.

## Faster serving (optional)

These options make the server faster on large GPUs.

- `--compile`: compiles the decision forward pass with `torch.compile`. Startup takes a few minutes longer.
- `--fast-reasoning MODE` (with `--reasoning on`): generation uses a static KV cache, which transformers compiles at
  startup. `copy` (the default when no mode is given): the base model that writes the reasoning is kept as a second
  copy of the weights. Roughly doubles reasoning throughput, while also doubling memory. `shared`: no second copy (a
  few GB more memory), somewhat slower than `copy`. `auto`: `copy` when there is room, else `shared`. Prompts over
  4,096 tokens reason uncompiled, so nothing compiles while serving.
- For Qwen models: `pip install flash-linear-attention==0.5.2`: transformers then runs the linear-attention layers
  with these kernels. Use it together with `--compile`.
  

## Batching

Requests that arrive at the same time are answered together (up to `--max-batch` questions, default 32). With vLLM, waiting reasonings also run as one
batch. `--batching off` answers one request at a time.

- Memory: a batch holds at most `--max-batch-tokens` padded input tokens (questions x longest input). By default this
  is set at start from the free GPU memory (about 300,000 tokens for Jeff-1.1-Large on an H200).
- Reproducibility: Padding in batches can change the probabilities by numerical noise (0.004 on average with 8 clients), and vLLM can slightly change probabilities as well. Use `--batching off` to reproduce our numbers exactly.
- Speed: batching helps most for the smaller models and for reasoning. Throughput with 8 clients on one H200, compared
  with one request at a time:

| Model | One pass | With reasoning (vLLM) |
|---|---|---|
| Jeff-1.1-Large | 1.3x | 2.2x |
| Jeff-1.1-Qwen-27B | 1.5x | 1.8x |
| Jeff-1.1-Medium | 1.7x | 2.5x |
| Jeff-1.1-Qwen-4B | 2.5x | 1.5x |

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
