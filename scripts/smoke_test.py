"""Send three example decisions (choice, yes/no, score) to a running server, print the answers, and compare the
probabilities with the answers we got for the same model (smoke_expected.json next to this script).

    python scripts/smoke_test.py --endpoint http://127.0.0.1:8013

Other GPUs and drivers can shift the probabilities slightly; the check allows --tolerance (default 0.05; across
A100, H100 and RTX PRO 6000 GPUs we saw differences up to about 0.03 on 99% of questions).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REQUEST = {
    "state": {"ticket": "I was charged twice for order #4411 and want the second charge refunded.",
              "customer_tier": "gold"},
    "model": "Jeff-1.0-Large",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this ticket?",
                  "criteria": {"billing": "Payments, charges and refunds", "technical": "Bugs and outages",
                               "shipping": "Delivery problems"}},
        "urgent": {"type": "noul", "instructions": "Does this ticket need a reply today?",
                   "criteria": {"true": "A reply is needed today", "false": "It can wait"}},
        "severity": {"type": "score", "instructions": "How severe is the customer's problem?",
                     "criteria": ["trivial", "minor", "moderate", "serious", "critical"]},
    },
}
EXPECTED = Path(__file__).with_name("smoke_expected.json")


def probabilities(answers: dict) -> dict:
    """All probabilities of the answers: {"route/billing": p, ..., "urgent": P(yes), ...}."""
    out = {}
    for key, answer in answers.items():
        if answer["type"] == "noul":
            out[key] = answer["noul"]
        else:
            out.update({f"{key}/{option}": p for option, p in answer["probabilities"].items()})
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:8013")
    p.add_argument("--tolerance", type=float, default=0.05, help="largest allowed difference of a probability")
    a = p.parse_args()
    req = urllib.request.Request(a.endpoint.rstrip("/") + "/v1/systemone", data=json.dumps(REQUEST).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read())
    print(json.dumps(body["answers"], indent=1))
    print(f"input tokens {body['usage']['input_tokens']}, round trip {time.perf_counter() - t0:.3f} s")
    assert set(body["answers"]) == set(REQUEST["questions"]), "missing answers"

    commit = "noul_edge" in body.get("runtime", {}).get("serving_rule", {})
    mode = "commit" if commit else "no-commit"
    reference = json.loads(EXPECTED.read_text()).get(body["model"], {}).get(mode)
    if reference is None:
        print(f"no reference answers for {body['model']} ({mode}); not checked")
        sys.exit(0)
    got, want = probabilities(body["answers"]), probabilities(reference)
    if set(got) != set(want):
        sys.exit(f"FAIL: the answers have other options than our reference: {sorted(set(got) ^ set(want))}")
    worst = max(abs(got[k] - want[k]) for k in want)
    if worst > a.tolerance:
        sys.exit(f"FAIL: a probability differs from our reference by {worst:.4f} (allowed: {a.tolerance})")
    print(f"OK: matches our answers for {body['model']} ({mode}); largest difference {worst:.4f}")
