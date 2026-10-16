"""Model folder lookup, the serving rule on answers, and refusing model folders this code cannot serve."""
import json

import pytest

from jeff.model import JeffModel, model_folder, served_answer

RULE = {"choice_t": 1.5, "noul_t": 1.0, "score_t": 1.0, "noul_edge": 0.0}


def test_local_folder(tmp_path):
    assert model_folder(tmp_path) == tmp_path


@pytest.mark.parametrize("ref", ["./missing", "/no/such/folder", "a/b/c"])
def test_missing_folder_has_a_clear_error(ref):
    with pytest.raises(FileNotFoundError, match="hf download"):
        model_folder(ref)


def test_rule_commits_yes_no_answers():
    assert served_answer("noul", ["true", "false"], [0.6, 0.4], RULE) == {"type": "noul", "noul": 0.801}
    no_commit = {k: v for k, v in RULE.items() if k != "noul_edge"}
    assert served_answer("noul", ["true", "false"], [0.6, 0.4], no_commit)["noul"] == pytest.approx(0.6)


def test_rule_tempers_choices():
    out = served_answer("choice", ["a", "b"], [0.2, 0.8], RULE)
    assert out["choice"] == "b" and 0.2 < out["probabilities"]["a"] < 0.5


@pytest.mark.parametrize("config", [{"prompt_version": "another-format"}, {"noul_mapping": "another-layout"}])
def test_unsupported_model_folders_are_refused(tmp_path, config):
    (tmp_path / "jeff_config.json").write_text(json.dumps({"name": "x", **config}))
    with pytest.raises(ValueError, match="not supported"):
        JeffModel(tmp_path, device="cpu")
