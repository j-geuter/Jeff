"""The long-input filter (jeff.longinput): what is kept, in which order, and that the result fits."""
import json

from jeff import longinput


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


RENDER = lambda tok, r: (r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])) + " " + r["question"]  # noqa: E731


def test_shorten_lines_keeps_the_question_lines_in_order():
    lines = ["SUPPORT LOG header: the owner is set by the most recent assignment",
             "2026-01-01 | SUP-1 | assigned to A"] + [f"2026-01-{i % 28 + 1:02d} | SUP-{1000 + i} | note" for i in range(3000)]
    lines.insert(1700, "2026-05-05 | SUP-605069 | reassigned to Bo")
    lines.insert(400, "2026-02-02 | SUP-605069 | created, assigned to Al")
    row = {"id": "x", "state": "\n".join(lines), "question": "Who owns ticket SUP-605069 at the end of the log?"}
    render = lambda tok, r: r["state"] + " " + r["question"]          # noqa: E731
    out = longinput.shorten(WordTokenizer(), row, 400, render)[0]
    kept = out["state"].split("\n")
    assert kept[0] == lines[0]
    assert [x for x in kept if "SUP-605069 |" in x] == ["2026-02-02 | SUP-605069 | created, assigned to Al",
                                                         "2026-05-05 | SUP-605069 | reassigned to Bo"]
    assert len(render(None, out).split()) <= 400


def test_shorten_json_list_keeps_the_question_entries():
    entries = [{"ticket": f"SUP-{1000 + i}", "event": "note", "by": "x"} for i in range(3000)]
    entries[2100] = {"ticket": "SUP-605069", "event": "reassigned to Bo", "by": "lead"}
    entries[300] = {"ticket": "SUP-605069", "event": "created, assigned to Al", "by": "desk"}
    state = {"rules": "the owner is the most recent assignee", "log": entries}
    row = {"id": "x", "state": state, "question": "Who owns ticket SUP-605069 at the end of the log?"}
    out, how = longinput.shorten(WordTokenizer(), row, 400, RENDER)
    assert how == "json" and out["state"]["rules"] == state["rules"] and out["state"]["log"][0].startswith("[Only")
    assert [e["event"] for e in out["state"]["log"][1:] if e["ticket"] == "SUP-605069"] == [
        "created, assigned to Al", "reassigned to Bo"]
    assert len(RENDER(None, out).split()) <= 400 and row["state"]["log"] is entries     # input left as it was


def test_shorten_json_top_level_list_and_dict_of_strings():
    entries = [f"2026-01-01 ACCT-{i} paid" for i in range(2000)] + ["2026-03-03 ACCT-70151 refunded"]
    out, how = longinput.shorten(WordTokenizer(), {"state": entries, "question": "Was ACCT-70151 refunded?"}, 300, RENDER)
    assert how == "json" and out["state"][0].startswith("[Only") and entries[-1] in out["state"]
    big = "\n".join(entries)
    out, how = longinput.shorten(WordTokenizer(), {"state": {"title": "ledger", "body": big},
                                                   "question": "Was ACCT-70151 refunded?"}, 300, RENDER)
    assert how == "json" and entries[-1] in out["state"]["body"] and out["state"]["title"] == "ledger"


def test_shorten_single_line_text_uses_sentences():
    text = "Rule: the last status wins. " + " ".join(f"Order ORD-{i} shipped." for i in range(3000)) + \
        " Order ORD-777777 was cancelled."
    out, how = longinput.shorten(WordTokenizer(), {"state": text, "question": "Is ORD-777777 cancelled?"}, 300, RENDER)
    assert how == "parts" and out["state"].startswith("Rule:") and "ORD-777777 was cancelled." in out["state"]
    assert len(RENDER(None, out).split()) <= 300


def test_shorten_flat_log_without_sentences_splits_at_dates():
    text = "AUTH LOG (UTC). " + " ".join(f"2026-01-{i % 28 + 1:02d} 10:00 user=u{i} result=SUCCESS" for i in range(3000))
    text = text.replace("user=u1700 result=SUCCESS", "user=daria.romero result=FAILED")
    out, how = longinput.shorten(WordTokenizer(), {"state": text, "question": "Did daria.romero fail to sign in?"},
                                 300, RENDER)
    assert how == "parts" and "user=daria.romero result=FAILED" in out["state"] and out["state"].startswith("AUTH LOG")


def test_shorten_dict_of_records_and_json_in_a_string():
    records = {f"SUP-{1000 + i}": {"owner": f"agent {i % 7}", "status": "open"} for i in range(2000)}
    records["SUP-605069"] = {"owner": "Bo", "status": "closed"}
    row = {"state": {"meta": {"source": "desk"}, "tickets": records}, "question": "Who owns SUP-605069?"}
    out, how = longinput.shorten(WordTokenizer(), row, 300, RENDER)
    assert how == "json" and out["state"]["tickets"]["SUP-605069"]["owner"] == "Bo" and "_note" in out["state"]["tickets"]
    out, how = longinput.shorten(WordTokenizer(), {**row, "state": json.dumps(row["state"])}, 300, RENDER)
    assert how == "json" and isinstance(out["state"], str) and json.loads(out["state"])["tickets"]["SUP-605069"]


def test_shorten_cuts_huge_option_texts():
    row = {"state": "short state", "question": "Context: " + "please " * 3000 + "Which plan fits?",
           "options": [{"id": f"o{i}", "description": f"o{i}: " + "long text " * 400} for i in range(4)]}
    render = lambda tok, r: RENDER(tok, r) + " " + " ".join(o["description"] for o in r["options"])  # noqa: E731
    out, how = longinput.shorten(WordTokenizer(), row, 1000, render)
    assert how == "question" and len(render(None, out).split()) <= 1000 and out["state"] == "short state"
    assert [o["description"].split(":")[0] for o in out["options"]] == ["o0", "o1", "o2", "o3"]
    assert out["question"].startswith("Context:") and out["question"].endswith("Which plan fits?")
    big = {**row, "state": "\n".join(f"line {i}" for i in range(5000)) + "\nplan facts"}
    out, how = longinput.shorten(WordTokenizer(), big, 1000, render)
    assert how.endswith("+question") and len(render(None, out).split()) <= 1000


def test_shorten_falls_back_to_a_cut():
    text = " ".join(f"w{i}" for i in range(5000))                        # one giant unit, nothing matches
    out, how = longinput.shorten(WordTokenizer(), {"state": text, "question": "Is it fine?"}, 300, RENDER)
    assert how == "cut" and out["state"].startswith("w0 w1 ") and len(RENDER(None, out).split()) <= 300


def test_plain_state():
    assert longinput.plain_state({"state": ""})["state"] == "(no state was given)"
    assert longinput.plain_state({"state": 42})["state"] == "42"
    assert longinput.plain_state({"state": {"x": float("nan")}})["state"] == '{"x": NaN}'
    assert longinput.plain_state({"state": {"x": 1}})["state"] == {"x": 1}
