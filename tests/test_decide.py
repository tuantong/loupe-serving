import json

import pytest

from loupe.backend_fake import FakeBackend
from loupe.decide import decide, default_engine, main
from loupe.engine import Engine


@pytest.fixture(autouse=True)
def fresh_default_engine():
    default_engine.cache_clear()
    yield
    default_engine.cache_clear()


def test_decide_in_process_with_explicit_engine():
    engine = Engine(FakeBackend(scripted={"yes": 3.0, "no": 0.0}))
    out = decide("late payment", {"risk": {"type": "noul", "instructions": "fraud?"}}, engine=engine)
    assert out["answers"]["risk"]["noul"] > 0.9
    assert out["policy_used"] == {"risk": "one_pass"}


def test_decide_uses_default_fake_engine(monkeypatch):
    monkeypatch.setenv("LOUPE_BACKEND", "fake")
    out = decide("s", {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}})
    assert set(out["answers"]["q"]["probabilities"]) == {"a", "b"}


def test_cli_prints_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LOUPE_BACKEND", "fake")
    path = tmp_path / "req.json"
    path.write_text(json.dumps({"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"]}}}))
    main([str(path)])
    printed = json.loads(capsys.readouterr().out)
    assert printed["answers"]["q"]["type"] == "choice"


def test_default_engine_reads_backend_from_env(monkeypatch):
    monkeypatch.setenv("LOUPE_BACKEND", "nope")
    with pytest.raises(ValueError, match="nope"):
        default_engine()
