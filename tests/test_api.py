import pytest
from pydantic import ValidationError

from loupe.api import (
    ChoiceAnswer,
    ContextBudgetExceeded,
    DecideRequest,
    DecideResponse,
    NoulAnswer,
    Question,
    RequestRejected,
    ScoreAnswer,
    Usage,
)


def choice(n):
    return {"type": "choice", "instructions": "pick", "criteria": [f"opt{i}" for i in range(n)]}


def test_choice_needs_two_to_255_options():
    Question(**choice(2))
    Question(**choice(255))
    with pytest.raises(ValidationError):
        Question(**choice(1))
    with pytest.raises(ValidationError):
        Question(**choice(256))


def test_noul_criteria_optional():
    q = Question(type="noul", instructions="is it spam?")
    assert q.criteria is None


def test_score_needs_criteria():
    with pytest.raises(ValidationError):
        Question(type="score", instructions="rate")


def test_option_keys_prefer_caller_labels():
    q = Question(type="choice", instructions="team", criteria=[{"text": "Billing team", "label": "billing"}, "technical"])
    assert q.option_texts() == ["Billing team", "technical"]
    assert q.option_keys() == ["billing", "technical"]


def test_duplicate_options_rejected():
    with pytest.raises(ValidationError):
        Question(type="choice", instructions="x", criteria=["a", "a"])
    with pytest.raises(ValidationError):
        Question(type="choice", instructions="x", criteria=[{"text": "a", "label": "k"}, {"text": "b", "label": "k"}])


def test_request_defaults():
    r = DecideRequest(state={"x": 1}, questions={"q": choice(3)})
    assert r.model == "loupe-1.0"
    assert r.policy.permutations == 1
    assert r.policy.escalate_below is None
    assert r.policy.order == "state_first"


def test_request_rejects_unknown_fields_and_empty_questions():
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={})
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"q": choice(2)}, extra=1)


def test_policy_values():
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"q": choice(2)}, policy={"permutations": 3})
    with pytest.raises(ValidationError):
        DecideRequest(state="s", questions={"q": choice(2)}, policy={"escalate_below": 1.5})


def test_context_budget_error_message():
    err = ContextBudgetExceeded(9000, 8192)
    assert "9000" in str(err) and "8192" in str(err)


def test_usage_output_tokens_defaults_to_zero_and_counts_thinking():
    assert Usage(input_tokens=5).output_tokens == 0
    assert Usage(input_tokens=5, output_tokens=12).output_tokens == 12


def test_option_text_and_labels_must_be_single_line():
    with pytest.raises(ValidationError, match="single-line"):
        Question(type="choice", instructions="x", criteria=["a\nb", "c"])
    with pytest.raises(ValidationError, match="single-line"):
        Question(type="choice", instructions="x", criteria=[{"text": "a", "label": "k\r1"}, "c"])


def test_noul_criteria_are_guidance_strings():
    q = Question(type="noul", instructions="fraud?", criteria=["large amount", "new device"])
    assert q.option_texts() == ["large amount", "new device"]
    with pytest.raises(ValidationError):
        Question(type="noul", instructions="fraud?", criteria=[""])


def test_answers_round_trip_as_a_discriminated_union():
    response = DecideResponse(
        model="loupe-1.0",
        answers={
            "a": NoulAnswer(noul=0.7),
            "b": ChoiceAnswer(choice="x", probabilities={"x": 0.6, "y": 0.4}, confidence=0.2),
            "c": ScoreAnswer(
                score="high",
                legend={"A": "low", "B": "high"},
                probabilities={"low": 0.3, "high": 0.7},
                confidence=0.4,
            ),
        },
        usage=Usage(input_tokens=1),
        policy_used={"a": "one_pass", "b": "one_pass", "c": "one_pass"},
    )
    again = DecideResponse.model_validate(response.model_dump())
    assert isinstance(again.answers["a"], NoulAnswer)
    assert isinstance(again.answers["b"], ChoiceAnswer)
    assert isinstance(again.answers["c"], ScoreAnswer)
    answers = DecideResponse.model_json_schema()["properties"]["answers"]["additionalProperties"]
    assert answers["discriminator"]["propertyName"] == "type"


def test_confidence_is_a_probability():
    with pytest.raises(ValidationError):
        ChoiceAnswer(choice="x", probabilities={"x": 1.0}, confidence=1.5)
    with pytest.raises(ValidationError):
        ScoreAnswer(score="x", legend={"A": "x"}, probabilities={"x": 1.0}, confidence=-0.1)


def test_request_rejected_carries_path_and_message():
    err = RequestRejected("model", "unknown model 'gpt'")
    assert err.path == "model"
    assert err.message == "unknown model 'gpt'"
    assert str(err) == "unknown model 'gpt'"


def test_choice_criteria_dict_becomes_labelled_options():
    q = Question(type="choice", instructions="Which intent?", criteria={"track_order": "Where is my order", "cancel_order": "Cancel it"})
    assert q.option_keys() == ["track_order", "cancel_order"]
    assert q.option_texts() == ["track_order: Where is my order", "cancel_order: Cancel it"]


def test_choice_criteria_dict_with_empty_description_uses_label_as_text():
    q = Question(type="choice", instructions="x", criteria={"a": "", "b": "B things"})
    assert q.option_texts() == ["a", "b: B things"]


def test_noul_criteria_dict_becomes_guidance():
    q = Question(type="noul", instructions="Permitted?", criteria={"false": "A condition is missing.", "true": "All conditions hold."})
    assert q.option_texts() == ["no: A condition is missing.", "yes: All conditions hold."]


def test_noul_criteria_dict_rejects_other_keys():
    with pytest.raises(ValidationError):
        Question(type="noul", instructions="x", criteria={"maybe": "y", "true": "z"})


def test_score_criteria_dict_rejected():
    with pytest.raises(ValidationError):
        Question(type="score", instructions="x", criteria={"0": "low", "1": "high"})


def test_score_keys_are_level_indices_unless_labelled():
    q = Question(type="score", instructions="Severity", criteria=["low", "medium", "high"])
    assert q.option_keys() == ["0", "1", "2"]
    labelled = Question(type="score", instructions="Severity", criteria=[{"text": "low", "label": "sev_low"}, "high"])
    assert labelled.option_keys() == ["sev_low", "1"]
