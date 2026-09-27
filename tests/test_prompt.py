import json

import pytest

from loupe.api import Question
from loupe.backend_fake import _OPTION_LINE
from loupe.prompt import SYSTEM, build_messages, render_question, render_state


def test_render_state_passes_strings_and_pretty_prints_json():
    assert render_state("hello") == "hello"
    text = render_state({"b": 1, "a": [1, 2]})
    assert json.loads(text) == {"b": 1, "a": [1, 2]}
    assert "\n" in text


def test_render_choice_lists_labels_and_options():
    q = Question(type="choice", instructions="Which team?", criteria=["billing", "technical"])
    text = render_question(q, ["A", "B"])
    assert text.startswith("Which team?")
    assert "A) billing" in text and "B) technical" in text
    assert text.endswith("Answer with the label only.")


def test_render_score_marks_levels_ordered():
    q = Question(type="score", instructions="Severity", criteria=["low", "high"])
    text = render_question(q, ["A", "B"])
    assert "lowest to highest" in text
    assert text.index("A) low") < text.index("B) high")


def test_render_noul_asks_yes_or_no():
    q = Question(type="noul", instructions={"check": "is it spam"})
    text = render_question(q, [])
    assert '"check": "is it spam"' in text
    assert text.endswith("Answer yes or no.")


def test_render_rejects_label_count_mismatch():
    q = Question(type="choice", instructions="x", criteria=["a", "b", "c"])
    with pytest.raises(ValueError):
        render_question(q, ["A", "B"])


def test_build_messages_orders():
    first = build_messages("STATE", "QUESTION", "state_first")
    assert first[0] == {"role": "system", "content": SYSTEM}
    assert first[1]["role"] == "user"
    assert first[1]["content"].index("STATE") < first[1]["content"].index("QUESTION")
    second = build_messages("STATE", "QUESTION", "rubric_first")
    assert second[1]["content"].index("QUESTION") < second[1]["content"].index("STATE")


def test_render_noul_lists_criteria_as_guidance():
    q = Question(type="noul", instructions="Is it fraud?", criteria=["large amount", "new device"])
    text = render_question(q, [])
    assert text == "Is it fraud?\nCriteria:\n- large amount\n- new device\nAnswer yes or no."
    assert _OPTION_LINE.findall(text) == []


def test_compact_style_drops_the_system_message_and_answer_instruction(monkeypatch):
    from loupe.api import Question
    from loupe.prompt import build_messages, render_question

    q = Question.model_validate({"type": "choice", "instructions": "Pick one.", "criteria": {"a": "first", "b": "second"}})
    n = Question.model_validate({"type": "noul", "instructions": "Is it?", "criteria": {"true": "yes", "false": "no"}})
    assert render_question(q, ["A", "B"]).endswith("Answer with the label only.")
    monkeypatch.setenv("LOUPE_PROMPT_STYLE", "compact")
    assert render_question(q, ["A", "B"]).splitlines()[-1] == "B) b: second"
    assert "Answer" not in render_question(n, ["yes", "no"])
    assert [m["role"] for m in build_messages("S", "Q", "state_first")] == ["user"]


def test_system_prompt_can_be_replaced(monkeypatch):
    from loupe.prompt import build_messages

    monkeypatch.setenv("LOUPE_SYSTEM_PROMPT", "Decide.")
    assert build_messages("S", "Q", "state_first")[0] == {"role": "system", "content": "Decide."}
