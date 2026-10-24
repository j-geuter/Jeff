"""Inputs longer than the model's limit (on by default; --long-input refuse turns it off).

JevBench sends every item to every system, including a few very long states (logs and ledgers of about 80k tokens),
and counts a refused item as wrong. Instead of refusing, the server keeps the parts of the state that matter to the
question. On the long items of our development data this answers far more of them correctly than cutting the state
to its start, and also more than reading the whole input.

No model is involved. The state is split into units and each unit is scored against the question: identifier-like
words of the question (a digit, or an inner . _ - @, e.g. SUP-605069, ACCT-70151, daria.romero, 2026-02-13) count 10
per unit they appear in (as a whole word: ACCT-7015 does not match ACCT-70151), other question words of 5+ letters
count 1; words found in more than a quarter of the units (COMMON) count 0. The highest-scoring units are kept up to
the token budget, in their original order, with a short note that the rest was left out. Units by state shape:
  text with line breaks       lines (the first two lines, usually the header with the rules, are always kept)
  long text without breaks    sentences (the opening ones, up to 400 tokens, always kept); sentences that are still
                              long are split before repeated dates (2026-02-13, records of a flat log) or else into
                              windows of 32 words
  JSON (object or array,      the entries of its largest list or object, or the units of its largest string, going
  or a string holding JSON)   down while one entry holds most of the size; repeated for the next largest part while
                              the state does not fit; all other fields are kept as they are
If no unit mentions the question, or the prompt still does not fit, the state is cut at the token level, keeping its
start. If the question and its options alone do not fit, their texts are shortened too. So a request is never refused
for its length.

plain_state() applies in both modes (since 1.3): a state the prompt format would refuse (empty, a number, NaN inside)
is sent as text, so that every question gets an answer.
"""
from __future__ import annotations

import json
import re

NOTE = "[Only the parts that mention {keys} are shown; {kept} of {total} {unit} kept.]"
CUT = "\n[... cut: the input is too long ...]"
DATE = r"\d{4}-\d{2}-\d{2}"


def question_keys(question: str) -> tuple[set[str], set[str]]:
    """(identifier-like words, other content words) of a question."""
    words = [re.sub(r"'s$", "", w.strip(".,;:?!()[]{}'\"`")) for w in re.findall(r"\S+", question)]
    keys = {w for w in words if len(w) >= 3 and (re.search(r"\d", w) or re.search(r"\w[._@-]\w", w))}
    content = {w.lower() for w in words if len(w) >= 5 and w not in keys and w.isalpha()}
    return keys, content


def _ntok(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _prompt_tokens(tokenizer, row, render) -> int:
    return _ntok(tokenizer, render(tokenizer, row))


def _scorer(question: str):
    keys, content = question_keys(question)
    pats = [re.compile(r"(?<![0-9A-Za-z])" + re.escape(k) + r"(?![0-9A-Za-z])") for k in keys]

    def matches(unit: str) -> tuple[list[bool], list[bool]]:
        low = unit.lower()
        return [bool(p.search(unit)) for p in pats], [c in low for c in content]
    return matches


# Question words found in more than this share of the units (e.g. "2026" in a ledger of 2026) select nothing.
COMMON: float | None = 0.25


def select(units: list[str], question: str, tokenizer, budget: int, always: int = 0) -> list[int]:
    """Indices of the units to keep (in order): the first `always` units, then the best-scoring ones within budget."""
    matches = _scorer(question)
    keep = list(range(min(always, len(units))))
    used = sum(_ntok(tokenizer, units[i]) + 1 for i in keep)
    hits = {i: matches(units[i]) for i in range(len(keep), len(units))}
    n_keys = len(next(iter(hits.values()))[0]) if hits else 0
    n_words = len(next(iter(hits.values()))[1]) if hits else 0
    key_w, word_w = [10] * n_keys, [1] * n_words
    if COMMON is not None and hits:
        limit = COMMON * len(hits)
        key_w = [0 if sum(h[0][j] for h in hits.values()) > limit else 10 for j in range(n_keys)]
        word_w = [0 if sum(h[1][j] for h in hits.values()) > limit else 1 for j in range(n_words)]
    scored = []
    for i, (k, w) in hits.items():
        s = sum(a * b for a, b in zip(k, key_w)) + sum(a * b for a, b in zip(w, word_w))
        if s > 0:
            scored.append((-s, i))
    for _, i in sorted(scored):
        n = _ntok(tokenizer, units[i]) + 1
        if used + n > budget:
            continue
        keep.append(i)
        used += n
    return sorted(keep)


def _note(question: str, kept: int, total: int, unit: str) -> str:
    keys, _ = question_keys(question)
    return NOTE.format(keys=", ".join(sorted(keys)) or "the question", kept=kept, total=total, unit=unit)


def _flat_units(text: str, tokenizer, budget: int) -> list[str]:
    """Units of a text without line breaks: sentences; long ones split before repeated dates, else into word windows."""
    cap = max(64, budget // 16)
    out = []
    for sentence in (u for u in re.split(r"(?<=[.!?;])\s+", text) if u):
        if _ntok(tokenizer, sentence) <= cap:
            out.append(sentence)
            continue
        parts = re.split(r"\s+(?=" + DATE + r"\b)", sentence) if len(re.findall(DATE, sentence)) >= 8 else [sentence]
        for part in parts:
            if _ntok(tokenizer, part) <= cap:
                out.append(part)
            else:
                words = part.split(" ")
                out += [" ".join(words[i:i + 32]) for i in range(0, len(words), 32)]
    return out


def _text(text: str, question: str, tokenizer, budget: int) -> tuple[str, int]:
    """(filtered text, number of units kept for matching the question)."""
    if "\n" in text.strip():
        units, sep, always, unit = text.split("\n"), "\n", 2, "lines"
    else:                                   # one long line: sentences; the opening ones (the rules/header) are kept
        units, sep, unit = _flat_units(text, tokenizer, budget), " ", "parts"
        always, used = 1, _ntok(tokenizer, units[0]) if units else 0
        while always < len(units) and used + _ntok(tokenizer, units[always]) <= min(400, budget // 8):
            used += _ntok(tokenizer, units[always])
            always += 1
    if sum(_ntok(tokenizer, units[i]) for i in range(min(always, len(units)))) > budget // 2:
        always = 0                          # a giant header would leave no room for the matching units
    note = _note(question, 0, len(units), unit)
    keep = select(units, question, tokenizer, budget - _ntok(tokenizer, note) - 2, always)
    head = [units[i] for i in keep[:always]]
    rest = [units[i] for i in keep[always:]]
    return sep.join(head + [_note(question, len(keep), len(units), unit)] + rest), len(rest)


# ----------------------------------------------------------------- JSON states

def _dumps(node) -> str:
    return json.dumps(node, ensure_ascii=False)


def _children(node):
    return list(node.items()) if isinstance(node, dict) else list(enumerate(node)) if isinstance(node, list) else []


def _candidates(node, path=()):
    """(size, path) of the parts a filter can shorten: strings, and lists/objects with 2+ entries where no single entry
    holds most of the size (otherwise the filter goes down into that entry)."""
    size = len(_dumps(node))
    out = []
    if isinstance(node, str):
        return [(size, path)]
    kids = _children(node)
    if len(kids) >= 2 and max(len(_dumps(v)) for _, v in kids) < size / 2:
        out.append((size, path))
    for key, child in kids:
        if isinstance(child, (dict, list, str)) and len(_dumps(child)) > 200:
            out += _candidates(child, path + (key,))
    return out


def _get(node, path):
    for key in path:
        node = node[key]
    return node


def _set(node, path, value):
    if not path:
        return value
    _get(node, path[:-1])[path[-1]] = value
    return node


def _filter_part(target, question: str, tokenizer, room: int):
    """(filtered list / object / string, number of entries kept for matching)."""
    if isinstance(target, str):
        return _text(target, question, tokenizer, room)
    kids = _children(target)
    units = [_dumps({k: v}) if isinstance(target, dict) else _dumps(v) for k, v in kids]
    note = _note(question, 0, len(units), "entries")
    keep = select(units, question, tokenizer, room - _ntok(tokenizer, note) - 4)
    note = _note(question, len(keep), len(units), "entries")
    if isinstance(target, dict):
        return {"_note": note, **{kids[i][0]: kids[i][1] for i in keep}}, len(keep)
    return [note] + [kids[i][1] for i in keep], len(keep)


def _json(state, question: str, tokenizer, budget: int) -> tuple[object, int]:
    """(the JSON state with its largest parts filtered until it fits, number of entries kept for matching)."""
    state = json.loads(_dumps(state))                     # private copy
    matched, tried = 0, set()
    for _ in range(8):
        size = _ntok(tokenizer, _dumps(state))
        if size <= budget:
            break
        options = [(s, p) for s, p in _candidates(state) if p not in tried]
        if not options:
            break
        _, path = max(options)
        tried.add(path)
        target = _get(state, path)
        tsize = _ntok(tokenizer, _dumps(target))
        room = max(budget - (size - tsize), budget * tsize // max(1, size))   # what is left, or a fair share
        new, n = _filter_part(target, question, tokenizer, room)
        if n:                                             # nothing matched: leave it, try a smaller part
            state = _set(state, path, new)
            matched += n
    return state, matched


# ----------------------------------------------------------------- fallbacks

def truncate_start(tokenizer, row: dict, limit: int, render, margin: int = 32) -> dict:
    """Last resort: the state as text, cut at the token level (start kept) so that the prompt fits."""
    text = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    overhead = _prompt_tokens(tokenizer, {**row, "state": "x"}, render)
    ids = tokenizer.encode(text, add_special_tokens=False)
    keep = max(1, limit - overhead - margin)
    out = {**row, "state": tokenizer.decode(ids[:keep]) + CUT}
    while _prompt_tokens(tokenizer, out, render) > limit and keep > 1:
        keep = int(keep * 0.95)
        out = {**row, "state": tokenizer.decode(ids[:keep]) + CUT}
    return out


def squeeze_question(tokenizer, row: dict, limit: int, render) -> dict:
    """The question and option texts cut until they leave at least a quarter of the limit for the state: options keep
    their start (the option id comes first), the question its start and its end (where the actual ask usually is)."""
    def cut(text, n, end=False):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= n:
            return text
        if end:
            return tokenizer.decode(ids[:n // 2]) + " [...] " + tokenizer.decode(ids[-(n // 2):])
        return tokenizer.decode(ids[:n]) + " [...]"
    n_q, n_o = 2048, 512
    out = row
    while _prompt_tokens(tokenizer, {**out, "state": "x"}, render) > 0.75 * limit and n_o >= 4:
        out = {**row, "question": cut(row["question"], n_q, end=True),
               "options": [{**o, "description": cut(o["description"], n_o)} for o in row["options"]]}
        n_q, n_o = n_q // 2, n_o // 2
    return out


def shorten(tokenizer, row: dict, limit: int, render, margin: int = 96) -> tuple[dict, str]:
    """(row whose rendered prompt fits `limit` tokens, how: "lines" | "parts" | "json" | "cut", plus "+question" when
    the question or option texts were cut too; "question" when only they were).

    The units are counted one by one, but the prompt holds the state JSON-escaped, so the filtered prompt can come out
    a little longer than planned; then the filter runs again with the budget lowered by the overshoot (up to 3 times)."""
    extra = ""
    if _prompt_tokens(tokenizer, {**row, "state": "x"}, render) > 0.75 * limit:
        row, extra = squeeze_question(tokenizer, row, limit, render), "+question"
        if _prompt_tokens(tokenizer, row, render) <= limit:
            return row, "question"                        # the state was short; only the question was too long
    overhead = _prompt_tokens(tokenizer, {**row, "state": "x"}, render)
    budget = limit - overhead - margin
    state = row["state"]
    if isinstance(state, str) and state.lstrip()[:1] in ("{", "["):
        try:
            state = json.loads(state)                     # a string that holds JSON: filtered as JSON, sent as text
        except ValueError:
            pass
    as_text = isinstance(row["state"], str)
    for _ in range(4):
        if budget <= 0:
            break
        if isinstance(state, (dict, list)):
            new, matched = _json(state, row["question"], tokenizer, budget)
            how = "json"
            if as_text:
                new = _dumps(new)
        else:
            text = state if isinstance(state, str) else str(state)
            new, matched = _text(text, row["question"], tokenizer, budget)
            how = "lines" if "\n" in text.strip() else "parts"
        if not matched:
            break
        out = {**row, "state": new}
        n = _prompt_tokens(tokenizer, out, render)
        if n <= limit:
            return out, how + extra
        budget -= (n - limit) + max(16, budget // 20)
    return truncate_start(tokenizer, row, limit, render), "cut" + extra


def plain_state(row: dict) -> dict:
    """A state that the prompt format would refuse (empty, a number, NaN inside) as text."""
    state = row.get("state")
    if isinstance(state, (str, dict, list)) and state:
        try:
            json.dumps(state, allow_nan=False)
            return row
        except ValueError:
            return {**row, "state": json.dumps(state, ensure_ascii=False)}
    if state is None or state == "" or state == [] or state == {}:
        return {**row, "state": "(no state was given)"}
    return {**row, "state": json.dumps(state, ensure_ascii=False)}
