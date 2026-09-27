import math

from loupe.api import DecideRequest
from loupe.backend_fake import FakeBackend
from loupe.calibration import Calibration, softmax
from loupe.engine import Engine

CHOICE = {"type": "choice", "instructions": "Which?", "criteria": ["x", "y"]}
NOUL = {"type": "noul", "instructions": "Is it?"}
SCORE = {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]}
FIVE = {"type": "choice", "instructions": "Which?", "criteria": ["a", "b", "c", "d", "e"]}


def decide(backend, questions, policy):
    return Engine(backend).decide(DecideRequest(state="s", questions=questions, policy=policy))


def test_no_policy_when_threshold_unset():
    backend = FakeBackend(scripted={"x": 0.0, "y": 0.0}, position_bias=1.0)
    response = decide(backend, {"q": CHOICE}, {"permutations": 4})
    assert response.policy_used == {"q": "one_pass"}
    assert len(backend.calls) == 1


def test_confident_answer_skips_policy():
    backend = FakeBackend(scripted={"x": 5.0, "y": 0.0})
    response = decide(backend, {"q": CHOICE}, {"permutations": 4, "escalate_below": 0.3})
    assert response.policy_used == {"q": "one_pass"}
    assert len(backend.calls) == 1


def test_permutation_averaging_cancels_position_bias():
    backend = FakeBackend(scripted={"x": 0.0, "y": 0.8}, position_bias=1.0)
    one_pass = decide(backend, {"q": CHOICE}, {"permutations": 1})
    assert one_pass.answers["q"].choice == "x"
    permuted = decide(backend, {"q": CHOICE}, {"permutations": 2, "escalate_below": 0.2})
    assert permuted.policy_used == {"q": "permuted"}
    assert permuted.answers["q"].choice == "y"
    assert len(backend.calls) == 3
    assert permuted.usage.input_tokens > one_pass.usage.input_tokens


def test_escalation_when_still_uncertain():
    backend = FakeBackend(scripted={"x": 0.0, "y": 0.0}, think_scripted={"x": -3.0, "y": 3.0})
    response = decide(backend, {"q": CHOICE}, {"permutations": 2, "escalate_below": 0.5})
    assert response.policy_used == {"q": "escalated"}
    assert response.answers["q"].choice == "y"
    assert len(backend.think_calls) == 1
    assert "A) x" in backend.think_calls[0][1]["content"]


def test_noul_never_permutes_but_can_escalate():
    backend = FakeBackend(scripted={"yes": 0.0, "no": 0.0}, think_scripted={"yes": 4.0, "no": 0.0})
    response = decide(backend, {"q": NOUL}, {"permutations": 4, "escalate_below": 0.5})
    assert response.policy_used == {"q": "escalated"}
    assert len(backend.calls) == 1
    assert math.isclose(response.answers["q"].noul, math.exp(4) / (math.exp(4) + 1))


def test_policy_is_per_question():
    backend = FakeBackend(scripted={"x": 5.0, "y": 0.0, "p": 0.0, "q": 0.0}, think_scripted={"p": 2.0, "q": 0.0})
    tie = {"type": "choice", "instructions": "Tie?", "criteria": ["p", "q"]}
    response = decide(backend, {"sure": CHOICE, "tie": tie}, {"permutations": 1, "escalate_below": 0.5})
    assert response.policy_used == {"sure": "one_pass", "tie": "escalated"}


def test_score_never_permutes_and_escalates_instead():
    backend = FakeBackend(scripted={"low": 0.0, "high": 0.0}, think_scripted={"low": 0.0, "high": 4.0})
    response = decide(backend, {"sev": SCORE}, {"permutations": 4, "escalate_below": 0.5})
    assert response.policy_used == {"sev": "escalated"}
    assert len(backend.calls) == 1
    assert len(backend.think_calls) == 1


def test_choice_permutes_once_over_the_remaining_orders():
    backend = FakeBackend(scripted={"a": 0.0, "b": 2.0, "c": 0.0, "d": 0.0, "e": 0.0}, position_bias=1.5)
    response = decide(backend, {"q": FIVE}, {"permutations": 4, "escalate_below": 0.3})
    assert response.policy_used == {"q": "permuted"}
    assert len(backend.calls) == 2
    assert len(backend.calls[1]) == 3
    assert backend.think_calls == []
    assert response.answers["q"].choice == "b"


def test_escalated_probabilities_use_the_escalated_temperature():
    backend = FakeBackend(scripted={"x": 0.0, "y": 0.0}, think_scripted={"x": 0.0, "y": 3.0})
    engine = Engine(backend, calibration=Calibration({"choice": 4.0, "escalated": 2.0}))
    response = engine.decide(DecideRequest(state="s", questions={"q": CHOICE}, policy={"escalate_below": 0.5}))
    expected = softmax([0.0, 3.0], 2.0)
    assert response.policy_used == {"q": "escalated"}
    assert math.isclose(response.answers["q"].confidence, expected[1] - expected[0])


def test_escalation_reports_the_thinking_tokens_it_spent():
    backend = FakeBackend(scripted={"x": 0.0, "y": 0.0}, think_scripted={"x": 0.0, "y": 3.0})
    response = decide(backend, {"q": CHOICE}, {"escalate_below": 0.5})
    assert backend.think_budgets == [512]
    assert response.usage.output_tokens == 7


def test_input_tokens_count_every_prompt_on_the_permute_then_escalate_path():
    backend = FakeBackend(scripted={"x": 0.0, "y": 0.0}, think_scripted={"x": 1.0, "y": 0.0})
    response = decide(backend, {"q": CHOICE}, {"permutations": 2, "escalate_below": 0.5})
    assert response.policy_used == {"q": "escalated"}
    scored = [messages for call in backend.calls for messages in call] + backend.think_calls
    assert len(scored) == 3
    assert response.usage.input_tokens == sum(backend.count_tokens(m) for m in scored)


def test_escalation_cap_stops_after_the_first_question():
    tie = {"type": "choice", "instructions": "Tie?", "criteria": ["p", "q"]}
    backend = FakeBackend(scripted={"x": 0.0, "y": 0.0, "p": 0.0, "q": 0.0}, think_scripted={"x": 3.0, "p": 3.0})
    engine = Engine(backend, max_escalations=1)
    response = engine.decide(
        DecideRequest(state="s", questions={"first": CHOICE, "second": tie}, policy={"escalate_below": 0.5})
    )
    assert response.policy_used == {"first": "escalated", "second": "one_pass"}
    assert len(backend.think_calls) == 1
    assert response.answers["second"].confidence == 0.0


def test_escalation_cap_keeps_a_permuted_answer_permuted():
    backend = FakeBackend(scripted={"a": 0.0, "b": 2.0, "c": 0.0, "d": 0.0, "e": 0.0}, position_bias=1.5)
    engine = Engine(backend, max_escalations=0)
    response = engine.decide(
        DecideRequest(state="s", questions={"five": FIVE}, policy={"permutations": 4, "escalate_below": 0.9})
    )
    assert response.policy_used == {"five": "permuted"}
    assert backend.think_calls == []
    assert response.answers["five"].choice == "b"
    assert response.usage.output_tokens == 0
