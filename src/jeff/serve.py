"""Serve a Jeff model over the TypeSafe "System One" wire format, as JevBench's stock `typesafe` adapter calls it.

    POST /v1/systemone  {"state": ..., "model": ..., "questions": {name: {type, instructions, criteria}}}
    -> {"answers": {name: answer}, "model": ..., "usage": {"input_tokens", "output_tokens": 0}, "runtime": {...}}
    GET  /             health check: {"status": "ok", "model": ...}

Answers: choice -> {"type": "choice", "choice": best option, "probabilities": {...}}; noul -> {"type": "noul",
"noul": P(yes)}; score -> {"type": "score", "score": best level, "probabilities": {...}}. One question per forward pass,
requests served one at a time. Invalid requests, questions with more than 16 options and inputs above the model's
input limit (8,192 tokens) get HTTP 422; a failure in the forward pass (e.g. out of GPU memory) gets HTTP 500. Both
come with {"error": message}. Yes/no answers between 0.2 and 0.8 are committed to 0.199 / 0.801 unless --no-commit is
given.

    python -m jeff.serve jgeuter/Jeff-1.0-Large --host 0.0.0.0 --port 8013
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .model import JeffModel

WARMUP = {"id": "warmup", "state": "The sky is blue.", "question": "Is the sky blue?",
          "options": [{"id": "true", "description": "true: yes"}, {"id": "false", "description": "false: no"}]}


def make_handler(model: JeffModel, meta: dict):
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
                answers, tokens, forward = {}, 0, 0.0
                for key, question in questions.items():
                    row = model.row(body["state"], question, key)
                    probs, _, n, seconds = model.decider.decide(row)
                    answers[key] = model.answer(question["type"], [o["id"] for o in row["options"]], probs)
                    tokens, forward = tokens + n, forward + seconds
            except (KeyError, TypeError, ValueError) as error:
                return self._send(422, {"error": f"{type(error).__name__}: {error}"})
            except Exception as error:  # e.g. out of GPU memory: answer with the error instead of dropping the connection
                traceback.print_exc()
                return self._send(500, {"error": f"{type(error).__name__}: {error}"})
            self._send(200, {"answers": answers, "model": model.name,
                             "usage": {"input_tokens": tokens, "output_tokens": 0},
                             "runtime": {**meta, "forward_seconds": forward, "input_tokens": tokens}})
    return Handler


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
    args = parser.parse_args()

    import torch
    started = time.perf_counter()
    model = JeffModel(args.model, revision=args.revision, commit=not args.no_commit)
    for _ in range(3):  # warm-up: kernels, allocator
        model.decider.decide(WARMUP)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"loaded {model.name} from {model.folder} in {time.perf_counter() - started:.0f}s on {gpu}; "
          f"serving rule {model.rule}", flush=True)
    meta = {"model": model.name, "gpu": gpu, "dtype": "bfloat16", "readout": "decision head",
            "probability_origin": "softmax over the options + serving rule", "serving_rule": model.rule,
            "torch": torch.__version__, "cudnn_attention": torch.backends.cuda.cudnn_sdp_enabled()}
    httpd = HTTPServer((args.host, args.port), make_handler(model, meta))
    if args.ready_file:
        args.ready_file.write_text(json.dumps({"port": args.port, **meta}) + "\n")
    print(f"serving {model.name} on http://{args.host}:{args.port}/v1/systemone", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
