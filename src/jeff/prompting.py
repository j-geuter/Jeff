"""Prompts for decision rows, in the format of SemIf (github.com/TheoLeeCJ/SemIf, MIT).

A row is {"id": str, "state": str | dict | list, "question": str, "options": [{"id": str, "description": str}, ...]}
with 2 to 16 options (jeff.model answers questions with up to 256 options in groups of 16). Score questions use the
same format: the ordered levels are the options.

The prompt is SemIf's: a system message, then the row as a JSON object {evidence, criterion, options} with the options
labeled A, B, C, ..., rendered with the model's chat template with thinking disabled. Jeff models were trained and
evaluated with exactly this prompt (PROMPT_VERSION). It is byte-identical to SemIf's; we checked this against the
prompt hashes (prompt_sha256) in SemIf's published prediction files.
"""

from __future__ import annotations

import json

LETTERS = "ABCDEFGHIJKLMNOP"
MAX_OPTIONS = len(LETTERS)  # the decision head has one slot per letter
MAX_WIDE = MAX_OPTIONS * MAX_OPTIONS  # most options of a question answered in groups (jeff.model.Decider.decide_wide)
PROMPT_VERSION = "direct-options-v1"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def validate_row(row: dict, max_options: int = MAX_OPTIONS) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= max_options:
        raise ValueError(f"options must contain 2-{max_options} entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def messages(row: dict) -> list[dict]:
    """Chat messages of a row: SemIf's system message, then the row as JSON."""
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def render_prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode_prompt(tokenizer, row: dict, max_tokens: int) -> list[int]:
    """Token ids of a row's prompt. Rows longer than max_tokens are rejected, not truncated (as in SemIf)."""
    ids = tokenizer.encode(render_prompt(tokenizer, row), add_special_tokens=False)
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens} (the maximum context length); "
                         "no truncation allowed")
    return ids
