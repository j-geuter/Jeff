"""TypeSafe questions -> rows, answer objects and the serving rule."""
import math

import pytest

from jeff import typesafe as T


def test_yes_no_question():
    q = {"type": "noul", "instructions": "Urgent?", "criteria": {"true": "Reply today", "false": "Can wait"}}
    assert T.question_row("state", q, "urgent") == {
        "id": "urgent", "state": "state", "question": "Urgent?",
        "options": [{"id": "true", "description": "true: Reply today"},
                    {"id": "false", "description": "false: Can wait"}]}


def test_yes_no_question_without_criteria():
    options = T.question_options({"type": "noul", "instructions": "Urgent?"})
    assert [o["description"] for o in options] == ["true: The proposition is true.", "false: The proposition is false."]


def test_choice_keeps_request_order():
    q = {"type": "choice", "instructions": "Team?", "criteria": {"tech": "Bugs", "billing": ""}}
    assert T.question_options(q) == [{"id": "tech", "description": "tech: Bugs"},
                                     {"id": "billing", "description": "billing: billing"}]
    assert [o["id"] for o in T.question_options({"type": "choice", "criteria": ["x", "y"]})] == ["x", "y"]


def test_score_levels_are_numbered():
    q = {"type": "score", "instructions": "Severity?", "criteria": ["low", "high"]}
    assert T.question_options(q) == [{"id": "0", "description": "0: low"}, {"id": "1", "description": "1: high"}]


def test_unknown_type_is_refused():
    with pytest.raises(ValueError):
        T.question_options({"type": "rank", "criteria": ["a", "b"]})


def test_answer_objects():
    assert T.answer("choice", ["a", "b"], [0.3, 0.7]) == {"type": "choice", "choice": "b",
                                                          "probabilities": {"a": 0.3, "b": 0.7}}
    assert T.answer("noul", ["true", "false"], [0.8, 0.2]) == {"type": "noul", "noul": 0.8}
    assert T.answer("score", ["0", "1", "2"], [0.2, 0.5, 0.3])["score"] == 1


def softmax(z: list[float]) -> list[float]:
    e = [math.exp(v - max(z)) for v in z]
    return [v / sum(e) for v in e]


@pytest.mark.parametrize("qtype, ids, key", [("choice", ["a", "b", "c"], "choice_t"), ("score", ["0", "1", "2"], "score_t")])
def test_temperature_divides_the_logits(qtype, ids, key):
    z = [1.0, 2.5, -0.3]
    out = T.apply_rule(qtype, dict(zip(ids, softmax(z))), {key: 1.5})
    assert [out[i] for i in ids] == pytest.approx(softmax([v / 1.5 for v in z]))


def test_yes_no_temperature():
    out = T.apply_rule("noul", {"yes": 1 / (1 + math.exp(-2.0)), "no": 0.0}, {"noul_t": 2.0})
    assert out["yes"] == pytest.approx(1 / (1 + math.exp(-1.0)))


@pytest.mark.parametrize("p, committed", [(0.6, 0.801), (0.3, 0.199), (0.9, 0.9), (0.1, 0.1), (0.5, 0.5)])
def test_commitment(p, committed):
    assert T.apply_rule("noul", {"yes": p, "no": 1 - p}, {"noul_edge": 0.0})["yes"] == committed


def test_empty_rule_changes_nothing():
    probs = {"a": 0.25, "b": 0.75}
    assert T.apply_rule("choice", probs, {}) == probs
