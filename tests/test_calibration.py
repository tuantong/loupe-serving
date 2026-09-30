import json
import math

import pytest

from loupe.calibration import Calibration, fit_temperature, softmax


def test_softmax_sums_to_one_and_temperature_flattens():
    p = softmax([2.0, 0.0, -1.0])
    assert math.isclose(sum(p), 1.0, rel_tol=1e-9)
    assert p[0] > p[1] > p[2]
    flat = softmax([2.0, 0.0, -1.0], temperature=100.0)
    assert max(flat) - min(flat) < 0.05


def test_softmax_is_stable_for_large_logits():
    p = softmax([1000.0, 999.0])
    assert math.isclose(sum(p), 1.0, rel_tol=1e-9)


def test_default_temperatures_and_roundtrip(tmp_path):
    c = Calibration({"choice": 1.7})
    assert c.temperature("choice") == 1.7
    assert c.temperature("noul") == 1.0
    path = tmp_path / "calibration.json"
    c.save(path)
    assert json.loads(path.read_text())["choice"] == 1.7
    assert Calibration.load(path).temperature("choice") == 1.7


def test_fit_temperature_recovers_overconfidence():
    logits = [[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]
    targets = [[0.8, 0.2], [0.2, 0.8], [0.8, 0.2], [0.2, 0.8]]
    t = fit_temperature(logits, targets)
    assert 2.5 < t < 3.3


def test_fit_temperature_near_one_when_calibrated():
    logits = [[math.log(0.7), math.log(0.3)], [math.log(0.3), math.log(0.7)]]
    targets = [[0.7, 0.3], [0.3, 0.7]]
    assert abs(fit_temperature(logits, targets) - 1.0) < 0.05


def test_fit_temperature_accepts_rows_of_different_lengths():
    logits = [[4.0, 0.0], [0.0, 4.0, 1.0], [2.0, 0.0, -2.0, 1.0]]
    targets = [softmax(row, temperature=2.0) for row in logits]
    assert abs(fit_temperature(logits, targets) - 2.0) < 0.05


def test_escalation_has_its_own_temperature():
    assert Calibration().temperature("escalated") == 1.0
    assert Calibration({"escalated": 2.5}).temperature("escalated") == 2.5


def test_unknown_temperature_keys_are_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"unknown temperature keys: \['thinking'\]"):
        Calibration({"thinking": 2.0})
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"choice": 1.5, "thinking": 2.0}))
    with pytest.raises(ValueError, match="thinking"):
        Calibration.load(path)


def test_long_prompts_and_capped_traces_use_their_own_temperatures():
    from loupe.calibration import Calibration

    cal = Calibration({"choice": 1.0, "choice_long": 2.5, "escalated": 4.0, "escalated_capped": 6.0, "long_tokens": 800})
    assert cal.temperature("choice", 799) == 1.0
    assert cal.temperature("choice", 800) == 2.5
    assert cal.temperature("choice") == 1.0
    assert cal.temperature("noul", 5000) == 1.0, "a missing long key falls back to the plain one"
    assert cal.temperature("escalated") == 4.0
    assert cal.temperature("escalated", capped=True) == 6.0


def test_save_and_load_keep_the_long_threshold(tmp_path):
    from loupe.calibration import Calibration

    path = tmp_path / "cal.json"
    Calibration({"choice_long": 3.0, "long_tokens": 1200}).save(path)
    back = Calibration.load(path)
    assert back.long_tokens == 1200 and back.temperature("choice", 1500) == 3.0 and back.temperature("choice", 1000) == 1.0


def test_probability_questions_use_their_own_temperature():
    cal = Calibration({"noul": 1.0, "noul_long": 2.0, "probability": 3.0, "escalated": 4.0, "long_tokens": 600})
    assert cal.temperature("noul", 100, probability=True) == 3.0
    assert cal.temperature("noul", 900, probability=True) == 3.0
    assert cal.temperature("noul", 900) == 2.0
    assert cal.temperature("escalated", probability=True) == 4.0
    assert Calibration({"noul": 1.5}).temperature("noul", 100, probability=True) == 1.5


def test_a_calibration_file_can_name_the_model_it_defines(tmp_path):
    path = tmp_path / "calibration.json"
    Calibration({"noul": 0.5, "noul_offset": -1.0, "model_name": "loupe-1.1"}).save(path)
    loaded = Calibration.load(path)
    assert loaded.model_name == "loupe-1.1" and loaded.temperature("noul") == 0.5 and loaded.offset("noul") == -1.0
    plain = tmp_path / "plain.json"
    Calibration({"noul": 0.5}).save(plain)
    assert Calibration.load(plain).model_name is None and "model_name" not in json.loads(plain.read_text())
