"""The prompt: SemIf's format, row checks and the input limit."""
import json

import pytest

from jeff.prompting import DIRECT_SYSTEM, LETTERS, encode_prompt, messages, render_prompt, validate_row

ROW = {"id": "route", "state": {"ticket": "Refund not received"}, "question": "Which team should handle this?",
       "options": [{"id": "billing", "description": "billing: Payments"}, {"id": "tech", "description": "tech: Bugs"}]}


def changed(**change) -> dict:
    return {**ROW, **change}


def options(n: int) -> list[dict]:
    return [{"id": f"o{i}", "description": f"o{i}: option {i}"} for i in range(n)]


def test_semif_messages():
    system, user = messages(ROW)
    assert system == {"role": "system", "content": DIRECT_SYSTEM}
    assert user == {"role": "user", "content":
                    '{"evidence": {"ticket": "Refund not received"}, "criterion": "Which team should handle this?", '
                    '"options": [{"letter": "A", "description": "billing: Payments"}, '
                    '{"letter": "B", "description": "tech: Bugs"}]}'}


def test_text_is_not_escaped():
    _, user = messages(changed(state="Rückerstattung über 50 €"))
    assert "Rückerstattung über 50 €" in user["content"]


def test_sixteen_options_get_letters_a_to_p():
    payload = json.loads(messages(changed(options=options(16)))[1]["content"])
    assert [o["letter"] for o in payload["options"]] == list(LETTERS)


@pytest.mark.parametrize("change", [
    {"options": options(1)}, {"options": options(17)}, {"options": [ROW["options"][0]] * 2},
    {"options": [{"id": "a"}, {"id": "b"}]}, {"state": ""}, {"state": {"x": float("nan")}}, {"question": ""},
])
def test_invalid_rows_are_refused(change):
    with pytest.raises(ValueError):
        validate_row(changed(**change))


def test_missing_field_is_refused():
    with pytest.raises(ValueError, match="missing"):
        validate_row({k: v for k, v in ROW.items() if k != "state"})


class FakeTokenizer:
    """Records the chat-template arguments; one token per character."""

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "".join(m["content"] for m in messages)

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def test_chat_template_without_thinking():
    tokenizer = FakeTokenizer()
    assert render_prompt(tokenizer, ROW).startswith(DIRECT_SYSTEM)
    assert tokenizer.kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}


def test_long_inputs_are_refused_not_truncated():
    tokenizer = FakeTokenizer()
    n = len(encode_prompt(tokenizer, ROW, max_tokens=10_000))
    assert len(encode_prompt(tokenizer, ROW, max_tokens=n)) == n
    with pytest.raises(ValueError, match="exceed limit"):
        encode_prompt(tokenizer, ROW, max_tokens=n - 1)
