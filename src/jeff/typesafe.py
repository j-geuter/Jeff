"""JevBench / TypeSafe wire format <-> decision rows, and the serving rule.

- question_row(): a TypeSafe question (state + typed question with instructions and criteria) becomes one decision row,
  the way JevBench's own `semif_direct` adapter maps it: yes/no -> options "true", "false"; choice -> one option per
  criteria key, in request order; score -> options "0".."K-1"; every description is prefixed with "<id>: ";
- answer() turns the option distribution back into a TypeSafe answer object;
- apply_rule() is the serving rule: per-type temperatures, then yes/no answers with 0.2 < P(yes) < 0.8 moved to just
  outside the nearer edge (0.801 or 0.199), because JevBench counts answers inside that band as abstentions (wrong).
"""

from __future__ import annotations

import json
import math

NOUL_YES, NOUL_NO = 0.80, 0.20      # JevBench's yes/no band
BAND_YES, BAND_NO = 0.801, 0.199    # where a committed answer moves to


def _text(value) -> str:
    """An option description as text: strings as they are, structured values (objects, lists, numbers) as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def question_options(question: dict) -> list[dict]:
    """Options of one TypeSafe question, as JevBench's semif_direct adapter builds them (descriptions that are not
    strings, e.g. lists of colors or move objects, as JSON text)."""
    qtype, criteria = question["type"], question.get("criteria")
    if qtype == "noul":
        options = [{"id": key, "description": _text((criteria or {}).get(key, f"The proposition is {key}."))}
                   for key in ("true", "false")]
    elif qtype == "choice":
        if isinstance(criteria, dict):
            options = [{"id": key, "description": _text(value) if value not in (None, "", [], {}) else key}
                       for key, value in criteria.items()]
        else:  # a bare list of option ids
            options = [{"id": str(key), "description": str(key)} for key in criteria]
    elif qtype == "score":
        options = [{"id": str(i), "description": _text(level)} for i, level in enumerate(criteria)]
    else:
        raise ValueError(f"unknown question type {qtype!r}")
    return [{"id": o["id"], "description": o["id"] + ": " + o["description"]} for o in options]


def question_text(instructions) -> str:
    """The question's instructions as text: structured instructions (an object or array) as JSON, none as a note."""
    if isinstance(instructions, str) and instructions:
        return instructions
    if instructions is None or instructions == "" or instructions == {} or instructions == []:
        return "(no instructions were given)"
    return json.dumps(instructions, ensure_ascii=False)


def question_row(state, question: dict, row_id: str = "jevbench") -> dict:
    """One decision row (prompting.messages input) for a TypeSafe question."""
    return {"id": row_id, "state": state, "question": question_text(question["instructions"]),
            "options": question_options(question), "qtype": question["type"]}


def answer(qtype: str, option_ids: list[str], probabilities: list[float]) -> dict:
    """TypeSafe answer object from our distribution over the row's options."""
    probs = {i: float(p) for i, p in zip(option_ids, probabilities)}
    if qtype == "noul":
        return {"type": "noul", "noul": probs["true"]}
    best = max(option_ids, key=lambda i: probs[i])
    if qtype == "choice":
        return {"type": "choice", "choice": best, "probabilities": probs}
    return {"type": "score", "score": int(best), "probabilities": probs}


# ------------------------------------------------------------------ serving rule

def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def noul_temperature(p: float, temp: float) -> float:
    return 1 / (1 + math.exp(-_logit(p) / temp))


def noul_edge(p: float, margin: float = 0.0) -> float:
    """Inside the abstention band and at least `margin` from 0.5: move to the band edge."""
    if NOUL_NO < p < NOUL_YES and abs(p - 0.5) >= margin and p != 0.5:
        return BAND_YES if p > 0.5 else BAND_NO
    return p


def score_temperature(probs: list[float], temp: float) -> list[float]:
    logs = [math.log(max(p, 1e-12)) / temp for p in probs]
    top = max(logs)
    exp = [math.exp(v - top) for v in logs]
    total = sum(exp)
    return [v / total for v in exp]


def apply_rule(qtype: str, probs: dict, rule: dict) -> dict:
    """rule: {"noul_t": T, "noul_edge": margin or None, "score_t": T, "choice_t": T} (missing = identity).
    noul probs {"yes", "no"}; score probs {"0": ..}; choice probs {label: p}."""
    if qtype == "choice" and rule.get("choice_t", 1.0) != 1.0:
        keys = list(probs)
        return dict(zip(keys, score_temperature([probs[k] for k in keys], rule["choice_t"])))
    if qtype == "noul":
        p = probs["yes"]
        if rule.get("noul_t", 1.0) != 1.0:
            p = noul_temperature(p, rule["noul_t"])
        if rule.get("noul_edge") is not None:
            p = noul_edge(p, rule["noul_edge"])
        return {"yes": p, "no": 1.0 - p}
    if qtype == "score" and rule.get("score_t", 1.0) != 1.0:
        keys = sorted(probs, key=int)
        return dict(zip(keys, score_temperature([probs[k] for k in keys], rule["score_t"])))
    return probs
