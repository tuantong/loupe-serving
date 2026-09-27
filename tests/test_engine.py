import math

import pytest

from loupe.api import HARNESS_MODEL_NAME, MODEL_NAME, ContextBudgetExceeded, DecideRequest, RequestRejected
from loupe.backend_fake import FakeBackend
from loupe.calibration import Calibration
from loupe.engine import Engine


def request(questions, policy=None, state="the app crashes on login", model=MODEL_NAME):
    return DecideRequest(model=model, state=state, questions=questions, policy=policy or {})


def nest(depth):
    value = {"leaf": 1}
    for _ in range(depth):
        value = {"nested": value}
    return value


CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical", "sales"]}
SCORE = {"type": "score", "instructions": "Severity", "criteria": ["low", "medium", "high"]}
NOUL = {"type": "noul", "instructions": "Is the user angry?"}


def test_choice_answer_probabilities_and_confidence():
    backend = FakeBackend(scripted={"billing": 0.0, "technical": 2.0, "sales": -1.0})
    engine = Engine(backend)
    response = engine.decide(request({"team": CHOICE}))
    answer = response.answers["team"]
    assert answer.type == "choice"
    assert answer.choice == "technical"
    assert math.isclose(sum(answer.probabilities.values()), 1.0, rel_tol=1e-9)
    top = sorted(answer.probabilities.values(), reverse=True)
    assert math.isclose(answer.confidence, top[0] - top[1])
    assert response.policy_used == {"team": "one_pass"}
    assert response.usage.output_tokens == 0
    assert response.usage.input_tokens == backend.count_tokens(backend.calls[0][0])


def test_choice_keys_use_caller_labels():
    q = {"type": "choice", "instructions": "team", "criteria": [{"text": "Billing team", "label": "billing"}, "technical"]}
    backend = FakeBackend(scripted={"Billing team": 3.0, "technical": 0.0})
    answer = Engine(backend).decide(request({"t": q})).answers["t"]
    assert answer.choice == "billing"
    assert set(answer.probabilities) == {"billing", "technical"}


def test_score_answer_keyed_by_level_index():
    backend = FakeBackend(scripted={"low": -1.0, "medium": 0.0, "high": 2.0})
    answer = Engine(backend).decide(request({"sev": SCORE})).answers["sev"]
    assert answer.type == "score"
    assert answer.score == "2"
    assert set(answer.probabilities) == {"0", "1", "2"}
    assert answer.legend == {"A": "0", "B": "1", "C": "2"}


def test_choice_dict_criteria_round_trip_keys():
    q = {"type": "choice", "instructions": "Which intent?", "criteria": {"track_order": "Where is my order", "cancel_order": "Cancel it"}}
    backend = FakeBackend(scripted={"track_order: Where is my order": 3.0, "cancel_order: Cancel it": 0.0})
    answer = Engine(backend).decide(request({"intent": q})).answers["intent"]
    assert answer.choice == "track_order"
    assert set(answer.probabilities) == {"track_order", "cancel_order"}


def test_noul_dict_criteria_are_rendered_as_guidance():
    q = {"type": "noul", "instructions": "Permitted?", "criteria": {"false": "missing", "true": "holds"}}
    backend = FakeBackend(scripted={"yes": 1.0, "no": 0.0})
    Engine(backend).decide(request({"ok": q}))
    content = backend.calls[0][0][1]["content"]
    assert "- no: missing" in content and "- yes: holds" in content


def test_noul_answer_is_probability_of_yes():
    backend = FakeBackend(scripted={"yes": 2.0, "no": 0.0})
    answer = Engine(backend).decide(request({"angry": NOUL})).answers["angry"]
    assert answer.type == "noul"
    assert math.isclose(answer.noul, math.exp(2) / (math.exp(2) + 1))


def test_all_questions_scored_in_one_backend_call():
    backend = FakeBackend()
    response = Engine(backend).decide(request({"a": CHOICE, "b": SCORE, "c": NOUL}))
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == 3
    assert set(response.answers) == {"a", "b", "c"}


def test_temperature_is_applied_per_primitive():
    backend = FakeBackend(scripted={"billing": 0.0, "technical": 2.0, "sales": -1.0})
    sharp = Engine(backend).decide(request({"t": CHOICE})).answers["t"].confidence
    flat = Engine(backend, calibration=Calibration({"choice": 10.0})).decide(request({"t": CHOICE})).answers["t"].confidence
    assert flat < sharp


def test_rubric_first_order_puts_question_before_state():
    backend = FakeBackend()
    Engine(backend).decide(request({"t": CHOICE}, policy={"order": "rubric_first"}))
    content = backend.calls[0][0][1]["content"]
    assert content.index("Which team?") < content.index("the app crashes")


def test_context_budget_is_enforced_before_scoring():
    backend = FakeBackend()
    engine = Engine(backend, context_budget=20)
    with pytest.raises(ContextBudgetExceeded):
        engine.decide(request({"t": CHOICE}, state="word " * 50))
    assert backend.calls == []


def test_model_name_echoed():
    response = Engine(FakeBackend(), model_name="loupe-test").decide(request({"t": CHOICE}))
    assert response.model == "loupe-test"


def test_score_answers_are_keyed_by_caller_labels():
    q = {
        "type": "score",
        "instructions": "Severity",
        "criteria": [{"text": "a scratch", "label": "low"}, {"text": "a fire", "label": "high"}],
    }
    backend = FakeBackend(scripted={"a scratch": 0.0, "a fire": 2.0})
    answer = Engine(backend).decide(request({"sev": q})).answers["sev"]
    assert answer.score == "high"
    assert answer.legend == {"A": "low", "B": "high"}
    assert set(answer.probabilities) == {"low", "high"}
    mixed = {"type": "score", "instructions": "Severity", "criteria": [{"text": "a scratch", "label": "low"}, "a fire"]}
    answer = Engine(backend).decide(request({"sev": mixed})).answers["sev"]
    assert answer.score == "1"
    assert answer.legend == {"A": "low", "B": "1"}


def test_alphabet_is_probed_with_the_backend_label_prefix():
    assert Engine(FakeBackend(label_prefix="")).alphabet.prefix == ""
    assert Engine(FakeBackend()).alphabet.prefix == " "


def test_unknown_model_is_rejected():
    engine = Engine(FakeBackend(), model_name="loupe-test")
    with pytest.raises(RequestRejected) as raised:
        engine.decide(request({"t": CHOICE}, model="gpt-6"))
    assert raised.value.path == "model"
    assert "gpt-6" in raised.value.message
    assert engine.decide(request({"t": CHOICE}, model=MODEL_NAME)).model == "loupe-test"
    assert engine.decide(request({"t": CHOICE}, model=HARNESS_MODEL_NAME)).model == "loupe-test"
    assert engine.decide(request({"t": CHOICE}, model="loupe-test")).model == "loupe-test"


def test_deeply_nested_state_is_rejected():
    with pytest.raises(RequestRejected) as raised:
        Engine(FakeBackend()).decide(request({"t": CHOICE}, state=nest(5000)))
    assert raised.value.path == "state"
    assert raised.value.message == "state is nested too deeply"


def test_deeply_nested_instructions_are_rejected():
    question = {"type": "choice", "instructions": nest(5000), "criteria": ["a", "b"]}
    with pytest.raises(RequestRejected) as raised:
        Engine(FakeBackend()).decide(request({"t": question}))
    assert raised.value.path == "questions.t.instructions"


def test_budget_reserves_thinking_tokens_when_escalation_is_enabled():
    backend = FakeBackend()
    engine = Engine(backend, context_budget=200, max_thinking_tokens=512)
    engine.decide(request({"t": CHOICE}))
    with pytest.raises(ContextBudgetExceeded) as raised:
        engine.decide(request({"t": CHOICE}, policy={"escalate_below": 0.5}))
    assert raised.value.tokens == backend.count_tokens(backend.calls[0][0]) + 512


def test_too_many_options_for_the_backend_is_rejected_with_path():
    def only_letters(text):
        label = text.strip()
        return [1] if len(label) == 1 or label in ("yes", "no") else [1, 2]

    backend = FakeBackend()
    backend.encode = only_letters
    engine = Engine(backend)
    q = {"type": "choice", "instructions": "x", "criteria": [f"opt{i}" for i in range(27)]}
    with pytest.raises(RequestRejected) as info:
        engine.decide(request({"big": q}))
    assert info.value.path == "questions.big.criteria"


def test_escalation_gate_reads_the_uncalibrated_margin():
    backend = FakeBackend(scripted={"billing": 2.0, "technical": 0.0, "sales": -1.0}, think_scripted={"billing": 0.0, "technical": 5.0, "sales": 0.0})
    flat = Engine(backend, calibration=Calibration({"choice": 8.0}))
    response = flat.decide(request({"t": CHOICE}, policy={"escalate_below": 0.4}))
    assert response.policy_used == {"t": "one_pass"}, "raw margin is about 0.72, so a flattening temperature must not trigger thinking"
    probs = response.answers["t"].probabilities
    top = sorted(probs.values(), reverse=True)
    assert top[0] - top[1] < 0.4, "the returned probabilities are the calibrated ones"
    assert backend.think_calls == []


def test_escalation_still_happens_when_the_raw_margin_is_small():
    backend = FakeBackend(scripted={"billing": 0.2, "technical": 0.0, "sales": -3.0}, think_scripted={"billing": 0.0, "technical": 5.0, "sales": 0.0})
    engine = Engine(backend, calibration=Calibration({"choice": 0.1}))
    response = engine.decide(request({"t": CHOICE}, policy={"escalate_below": 0.4}))
    assert response.policy_used == {"t": "escalated"}, "raw margin is about 0.10; the sharpened margin (about 0.76) must not skip thinking"
    assert response.answers["t"].choice == "technical"


def test_long_prompt_uses_the_long_temperature():
    backend = FakeBackend(scripted={"billing": 2.0, "technical": 0.0, "sales": -1.0})
    short_len = backend.count_tokens([{"role": "user", "content": "x"}])
    cal = Calibration({"choice": 1.0, "choice_long": 10.0, "long_tokens": short_len + 40})
    short = Engine(backend, calibration=cal).decide(request({"t": CHOICE}, state="short")).answers["t"]
    long_ = Engine(backend, calibration=cal).decide(request({"t": CHOICE}, state="word " * 400)).answers["t"]
    assert max(short.probabilities.values()) > 0.6
    assert max(long_.probabilities.values()) < 0.45, "the flattening long temperature applies to the long prompt"


def test_capped_trace_uses_the_capped_temperature():
    backend = FakeBackend(scripted={"billing": 0.0, "technical": 0.0, "sales": 0.0}, think_scripted={"billing": 4.0, "technical": 0.0, "sales": 0.0})
    backend.think_tokens = 64
    cal = Calibration({"escalated": 1.0, "escalated_capped": 8.0})
    capped = Engine(backend, calibration=cal, max_thinking_tokens=64).decide(request({"t": CHOICE}, policy={"escalate_below": 0.5})).answers["t"]
    closed = Engine(backend, calibration=cal, max_thinking_tokens=128).decide(request({"t": CHOICE}, policy={"escalate_below": 0.5})).answers["t"]
    assert max(capped.probabilities.values()) < max(closed.probabilities.values())


def test_questions_asking_for_probabilities_use_the_probability_temperature():
    backend = FakeBackend(scripted={"billing": 0.0, "technical": 2.0, "sales": -1.0})
    engine = Engine(backend, calibration=Calibration({"probability": 10.0}))
    asks = dict(CHOICE, instructions="Which team? Give probabilities that reflect the evidence.")
    flat = engine.decide(request({"t": asks})).answers["t"].confidence
    sharp = engine.decide(request({"t": CHOICE})).answers["t"].confidence
    assert flat < sharp


class ScratchBackend(FakeBackend):
    """Writes a scratch that makes the last option certain, and states probabilities when asked to."""

    scratch_marker_tokens = 5

    def __init__(self, work="", **kw):
        super().__init__(**kw)
        self.work = work
        self.scratched = []

    def scratch_then_slice_many(self, batch, slice_ids, max_tokens):
        self.scratched.append((batch, max_tokens))
        return [([-5.0] * (len(ids) - 1) + [5.0], 42, self.work) for ids in slice_ids]


LONG = "the app crashes on login " * 60


def test_scratch_replaces_thinking_on_long_prompts_and_is_billed_as_output():
    backend = ScratchBackend(scripted={"billing": 0.1, "technical": 0.0})
    engine = Engine(backend, scratch_min_tokens=100, max_scratch_tokens=64)
    out = engine.decide(request({"q": {"type": "choice", "instructions": "Route.", "criteria": {"billing": "b", "technical": "t"}}}, policy={"escalate_below": 0.5}, state=LONG))
    assert out.policy_used["q"] == "escalated" and out.answers["q"].choice == "technical"
    assert out.usage.output_tokens == 42 and backend.scratched[0][1] == 64 and not backend.think_calls
    one_pass = engine.decide(request({"q": {"type": "choice", "instructions": "Route.", "criteria": {"billing": "b", "technical": "t"}}}, state=LONG))
    assert out.usage.input_tokens == one_pass.usage.input_tokens + 5


def test_short_prompts_keep_the_one_pass_answer_when_scratch_is_enabled():
    backend = ScratchBackend(scripted={"billing": 0.1, "technical": 0.0})
    engine = Engine(backend, scratch_min_tokens=10_000)
    out = engine.decide(request({"q": {"type": "choice", "instructions": "Route.", "criteria": {"billing": "b", "technical": "t"}}}, policy={"escalate_below": 0.5}))
    assert out.policy_used["q"] == "one_pass" and not backend.scratched and not backend.think_calls


def test_probability_questions_return_the_stated_distribution():
    backend = ScratchBackend(work="11 of 16\nProbabilities: yes 0.69, no 0.31", scripted={"yes": 0.0, "no": 0.1})
    engine = Engine(backend, scratch_min_tokens=100)
    q = {"type": "noul", "instructions": "Will it be paid late? Give probabilities that reflect the evidence.", "criteria": {"true": "late", "false": "on time"}}
    out = engine.decide(request({"q": q}, policy={"escalate_below": 0.5}, state=LONG))
    assert abs(out.answers["q"].noul - 0.69) < 1e-9
