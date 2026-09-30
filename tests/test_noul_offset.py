import math

from loupe.api import DecideRequest
from loupe.backend_fake import FakeBackend
from loupe.calibration import Calibration
from loupe.engine import Engine

NOUL = {"type": "noul", "instructions": "Is the invoice overdue?"}
CHOICE = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical"]}


def answer(calibration, question, scripted):
    engine = Engine(FakeBackend(scripted=scripted), calibration=calibration)
    return engine.decide(DecideRequest(state="Invoice 7 was due on 3 May.", questions={"q": question})).answers["q"]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def test_a_noul_offset_shifts_the_log_odds_of_yes():
    got = answer(Calibration({"noul": 2.0, "noul_offset": 0.7}), NOUL, {"yes": 0.4, "no": -0.1}).noul
    assert math.isclose(got, sigmoid(0.5 / 2.0 + 0.7), rel_tol=1e-12)


def test_without_an_offset_the_answer_is_unchanged():
    got = answer(Calibration({"noul": 2.0}), NOUL, {"yes": 0.4, "no": -0.1}).noul
    assert math.isclose(got, sigmoid(0.25), rel_tol=1e-12)


def test_choice_answers_ignore_the_noul_offset():
    plain = answer(Calibration(), CHOICE, {"billing": 1.0, "technical": 0.0}).probabilities
    shifted = answer(Calibration({"noul_offset": 3.0}), CHOICE, {"billing": 1.0, "technical": 0.0}).probabilities
    assert plain == shifted


def test_the_long_offset_applies_from_long_tokens_and_falls_back_to_the_plain_one():
    c = Calibration({"noul_offset": 0.5, "noul_long_offset": -1.0, "long_tokens": 10})
    assert c.offset("noul", 9) == 0.5 and c.offset("noul", 10) == -1.0 and c.offset("choice", 50) == 0.0
    assert Calibration({"noul_offset": 0.5, "long_tokens": 10}).offset("noul", 50) == 0.5
    assert Calibration().offset("noul", 5) == 0.0


def test_offsets_survive_save_and_load(tmp_path):
    path = tmp_path / "calibration.json"
    Calibration({"noul_offset": 0.3, "noul_long_offset": -0.2, "long_tokens": 600}).save(path)
    back = Calibration.load(path)
    assert back.offset("noul", 700) == -0.2 and back.offset("noul", 10) == 0.3
