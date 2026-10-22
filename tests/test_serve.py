"""The HTTP server: answers, and errors as HTTP status codes (with a stand-in model, so no GPU is needed)."""
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import pytest

from jeff import typesafe as T
from jeff.model import served_answer
from jeff.serve import make_handler

QUESTIONS = {"route": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Payments", "tech": "Bugs"}},
             "urgent": {"type": "noul", "instructions": "Urgent?"},
             "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "mid", "high"]}}


class StandInModel:
    """JeffModel's interface: every option gets probability 1/k, or decide() raises `error`."""
    name = "stand-in"

    def __init__(self, error: Exception | None = None):
        self.error, self.rule, self.decider, self.last = error, {}, self, {}

    def start_request(self):
        pass

    def row(self, state, question, key="q"):
        return T.question_row(state, question, key)

    def decide(self, row):
        if self.error:
            raise self.error
        k = len(row["options"])
        return [1 / k] * k, [0.0] * k, 100, 0.01

    def answer(self, qtype, option_ids, probabilities):
        return served_answer(qtype, option_ids, probabilities, self.rule)


@pytest.fixture
def start():
    servers = []

    def run(model) -> str:
        httpd = HTTPServer(("127.0.0.1", 0), make_handler(model, {"model": model.name}))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}"
    yield run
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


def call(url: str, body=None, path: str = "/v1/systemone") -> tuple[int, dict]:
    data = None if body is None else body if isinstance(body, bytes) else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url + path, data=data), timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_answers(start):
    status, body = call(start(StandInModel()), {"state": "Charged twice", "questions": QUESTIONS})
    assert status == 200
    assert body["answers"]["route"] == {"type": "choice", "choice": "billing",
                                        "probabilities": {"billing": 0.5, "tech": 0.5}}
    assert body["answers"]["urgent"] == {"type": "noul", "noul": 0.5}
    assert body["answers"]["severity"]["score"] == 0
    assert body["usage"] == {"input_tokens": 300, "output_tokens": 0}


def test_health_check_and_unknown_path(start):
    url = start(StandInModel())
    assert call(url, path="/") == (200, {"status": "ok", "model": "stand-in"})
    assert call(url, {"state": "s", "questions": QUESTIONS}, path="/v1/other")[0] == 404


@pytest.mark.parametrize("body", [b"not json", {"state": "s"}, {"state": "s", "questions": {}},
                                  {"state": "s", "questions": {"q": {"type": "rank", "instructions": "?"}}}])
def test_bad_requests_get_422(start, body):
    status, out = call(start(StandInModel()), body)
    assert status == 422 and "error" in out


def test_invalid_rows_get_422(start):
    model = StandInModel(ValueError("Row q: 9000 input tokens exceed limit 8192; no truncation allowed"))
    status, out = call(start(model), {"state": "s", "questions": QUESTIONS})
    assert status == 422 and "exceed limit" in out["error"]


def test_forward_pass_failure_gets_500(start):
    status, out = call(start(StandInModel(RuntimeError("CUDA out of memory"))), {"state": "s", "questions": QUESTIONS})
    assert (status, out) == (500, {"error": "RuntimeError: CUDA out of memory"})
