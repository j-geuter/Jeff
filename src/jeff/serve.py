"""Serve a Jeff model over the TypeSafe "System One" wire format, as JevBench's stock `typesafe` adapter calls it.

    POST /v1/systemone  {"state": ..., "model": ..., "questions": {name: {type, instructions, criteria}}}
    -> {"answers": {name: answer}, "model": ..., "usage": {"input_tokens", "output_tokens"}, "runtime": {...}}
    GET  /             health check: {"status": "ok", "model": ...}

Answers: choice -> {"type": "choice", "choice": best option, "probabilities": {...}}; noul -> {"type": "noul",
"noul": P(yes)}; score -> {"type": "score", "score": best level, "probabilities": {...}}. Requests that arrive together
are answered in batches (jeff.batching; --batching off: one request at a time, its questions one after another).
Invalid requests get HTTP 422; a failure in the forward pass (e.g. out of GPU memory) gets HTTP 500. Both come with
{"error": message}. Yes/no answers between 0.2 and 0.8 are committed to 0.199 / 0.801 unless --no-commit is given.

By default every question is one forward pass (output_tokens 0). Inputs over the model's limit (8,192 tokens) keep
their question-relevant parts (--long-input refuse: HTTP 422 instead); questions with more than 16 options (up to
256) are answered in groups of 16.

Jeff 1.1 models can reason when unsure (jeff.reasoning): --reasoning on, with limits of your choice (--think-share,
--think-threshold, --max-think-share, --max-output-per-decision, --request-seconds) or the ones we evaluate JevBench
with (--jevbench-limits on):

    python -m jeff.serve jgeuter/Jeff-1.0-Large --host 0.0.0.0 --port 8013
    python -m jeff.serve jgeuter/Jeff-1.1-Large --reasoning on --jevbench-limits on --port 8013
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

from .model import JeffModel


def answer_serially(model: JeffModel, state, questions: dict) -> tuple[dict, int, int, float, int]:
    """(answers, input tokens, output tokens, forward seconds, questions that reasoned), one question after another."""
    answers, tokens, forward, generated, thought = {}, 0, 0.0, 0, 0
    model.decider.start_request()
    for key, question in questions.items():
        row = model.row(state, question, key)
        probs, _, n, seconds = model.decider.decide(row)
        answers[key] = model.answer(question["type"], [o["id"] for o in row["options"]], probs)
        tokens, forward = tokens + n, forward + seconds
        generated += int(model.decider.last.get("output_tokens", 0))
        thought += int(bool(model.decider.last.get("thought")))
    return answers, tokens, generated, forward, thought


def answer_batched(batcher) -> callable:
    def answer(model: JeffModel, state, questions: dict) -> tuple[dict, int, int, float, int]:
        request = batcher.submit(state, questions)
        answers = {key: request.answers[key] for key in request.keys}
        return answers, request.tokens, request.generated, request.seconds, request.thought
    return answer


class BatchingServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


def make_handler(model: JeffModel, meta: dict, answer=answer_serially):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"status": "ok", "model": model.name})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                questions = body["questions"]
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a nonempty object")
                answers, tokens, generated, forward, thought = answer(model, body["state"], questions)
            except (KeyError, TypeError, ValueError) as error:
                return self._send(422, {"error": f"{type(error).__name__}: {error}"})
            except Exception as error:  # e.g. out of GPU memory: answer with the error instead of dropping the connection
                traceback.print_exc()
                return self._send(500, {"error": f"{type(error).__name__}: {error}"})
            self._send(200, {"answers": answers, "model": model.name,
                             "usage": {"input_tokens": tokens, "output_tokens": generated},
                             "runtime": {**meta, "forward_seconds": forward, "input_tokens": tokens,
                                         "reasoned": thought}})
    return Handler


def on_off(text: str) -> bool:
    if text not in ("on", "off"):
        raise argparse.ArgumentTypeError("expected on or off")
    return text == "on"


def think_share(text: str) -> float | dict:
    """0.3 (every type) or choice=0.2,noul=0.3,score=0.6 (the types not named keep their released setting)."""
    try:
        if "=" not in text:
            return float(text)
        return {key.strip(): float(value) for key, value in (part.split("=") for part in text.split(","))}
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a number like 0.3 or choice=0.2,noul=0.3: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face repo id or local folder of a released Jeff model")
    parser.add_argument("--revision", default=None, help="Hugging Face revision (commit) to pin")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--ready-file", type=Path, default=None, help="written once the server listens")
    parser.add_argument("--no-commit", action="store_true",
                        help="return the calibrated P(yes) for yes/no questions instead of committing answers between "
                             "0.2 and 0.8 to 0.199 / 0.801 (the commitment is on by default; JevBench counts answers "
                             "in that band as wrong)")
    parser.add_argument("--compile", action="store_true",
                        help="compile the decision forward pass with torch.compile: about 30-40%% faster per decision "
                             "on an H200, numerically slightly different; startup takes a few minutes longer")
    parser.add_argument("--long-input", choices=("filter", "refuse"), default="filter",
                        help="inputs over the model's limit: keep their question-relevant parts (default) or refuse "
                             "them with HTTP 422")
    parser.add_argument("--batching", type=on_off, default=True, metavar="on|off",
                        help="answer requests that arrive together in batches (default on); off: one request at a "
                             "time, exactly as in our evaluations")
    parser.add_argument("--max-batch", type=int, default=32, help="most questions per batch (default 32)")
    parser.add_argument("--max-batch-tokens", type=int, default=None,
                        help="most padded input tokens per batch (batch size x longest input); default: measured at "
                             "start from the GPU's free memory and the model's memory per token")
    group = parser.add_argument_group("reasoning (Jeff 1.1 models)")
    group.add_argument("--reasoning", type=on_off, default=False, metavar="on|off",
                       help="reason before answering when unsure (default off)")
    group.add_argument("--jevbench-limits", type=on_off, default=False, metavar="on|off",
                       help="the limits we evaluate on JevBench with: mean output tokens per question, share of "
                            "questions that reason, seconds per request (default off)")
    group.add_argument("--fast-reasoning", action="store_true",
                       help="write the reasoning with a second copy of the base weights and a static KV cache: about "
                            "2.4x faster generation on the 31B, needs the model's memory twice (e.g. an H200 for the 31B)")
    group.add_argument("--reasoning-engine", choices=("auto", "transformers", "vllm"), default="auto",
                       help="what writes the reasoning: the base model in vLLM on the same GPU (pip install "
                            "\".[vllm]\"; about 3.5x faster generation on the 31B, reasonings batched; the GPU must "
                            "hold the model twice, e.g. an H200 for the 31B), or transformers (this model with the "
                            "adapter removed; no extra memory). Default auto: vLLM if installed and it fits")
    group.add_argument("--vllm-memory", type=float, default=None,
                       help="vLLM's share of the GPU memory (default: the base model's size plus 12 GB)")
    group.add_argument("--vllm-base", default=None,
                       help="the base model for vLLM (repo id or folder; default: rebuilt from the release once, "
                            "under --cache-dir)")
    group.add_argument("--cache-dir", default=None,
                       help="where the rebuilt base model is kept (as large as the model; default $JEFF_CACHE or "
                            "~/.cache/jeff)")
    group.add_argument("--think-share", type=think_share, default=None,
                       help="share of questions that reason (by their confidence on our validation data), e.g. 0.3 "
                            "or choice=0.2,noul=0.3,score=0.6; default: the released shares")
    group.add_argument("--think-threshold", type=think_share, default=None,
                       help="reason when the calibrated confidence is below this, e.g. 0.8 or noul=0.8,score=0.7 "
                            "(instead of a share for these types)")
    group.add_argument("--think-budget", type=int, default=None, help="most tokens per reasoning (default: released)")
    group.add_argument("--max-output-per-decision", type=float, default=None,
                       help="limit on the mean output tokens per question, over everything served")
    group.add_argument("--max-think-share", type=float, default=None,
                       help="no new reasoning while more than this share of the last 1,000 questions reasoned")
    group.add_argument("--request-seconds", type=float, default=None,
                       help="no new reasoning once a request has taken this long")
    args = parser.parse_args()

    import torch
    started = time.perf_counter()
    model = JeffModel(args.model, revision=args.revision, commit=not args.no_commit, long_input=args.long_input,
                      reasoning=args.reasoning, jevbench_limits=args.jevbench_limits, think_share=args.think_share,
                      think_budget=args.think_budget, max_output_per_decision=args.max_output_per_decision,
                      request_seconds=args.request_seconds, compile=args.compile,
                      fast_reasoning=args.fast_reasoning, reasoning_engine=args.reasoning_engine,
                      vllm_memory=args.vllm_memory, vllm_base=args.vllm_base,
                      vllm_seqs=min(args.max_batch, 16) if args.batching else 1, think_threshold=args.think_threshold,
                      max_think_share=args.max_think_share, cache_dir=args.cache_dir)
    model.warmup(batch=args.batching)
    batcher = None
    if args.batching:
        from .batching import Batcher
        batcher = Batcher(model, args.max_batch, args.max_batch_tokens)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    reasoning = model.reasoning and {k: model.reasoning[k] for k in (
        "share", "tau", "budget", "temperatures", "max_output_per_decision", "max_share", "request_seconds")}
    if batcher:
        print(f"batching: up to {batcher.max_batch} questions and {batcher.max_tokens} padded input tokens per batch",
              flush=True)
    print(f"loaded {model.name} from {model.folder} in {time.perf_counter() - started:.0f}s on {gpu}; "
          f"serving rule {model.rule}; reasoning {reasoning or 'off'}", flush=True)
    meta = {"model": model.name, "gpu": gpu, "dtype": "bfloat16", "readout": "decision head",
            "probability_origin": "softmax over the options + serving rule", "serving_rule": model.rule,
            "bidirectional": model.config.get("bidirectional"), "long_input": args.long_input, "compile": args.compile,
            "reasoning": reasoning or None, "reasoning_engine": model.reasoning_engine,
            "batching": {"max_batch": batcher.max_batch, "max_tokens": batcher.max_tokens} if batcher else None,
            "torch": torch.__version__, "cudnn_attention": torch.backends.cuda.cudnn_sdp_enabled()}
    if batcher:
        httpd = BatchingServer((args.host, args.port), make_handler(model, meta, answer_batched(batcher)))
    else:
        httpd = HTTPServer((args.host, args.port), make_handler(model, meta))
    if args.ready_file:
        args.ready_file.write_text(json.dumps({"port": args.port, **meta}) + "\n")
    print(f"serving {model.name} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
