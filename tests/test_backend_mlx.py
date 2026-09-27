import os

import pytest

pytest.importorskip("mlx_lm")
if not os.environ.get("LOUPE_MLX_MODEL"):
    pytest.skip("set LOUPE_MLX_MODEL to run the MLX integration tests", allow_module_level=True)

from loupe.api import DecideRequest
from loupe.backend_mlx import MLXBackend
from loupe.engine import Engine

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine():
    return Engine(MLXBackend.from_env(os.environ))


def test_obvious_noul(engine):
    out = engine.decide(DecideRequest(state="The sky is blue today.", questions={"q": {"type": "noul", "instructions": "Is the sky blue?"}}))
    assert out.answers["q"].noul > 0.9


def test_obvious_choice_with_dict_criteria(engine):
    q = {"type": "choice", "instructions": "Which intent does the message express?", "criteria": {"track_order": "Where is my order", "cancel_order": "Cancel my order"}}
    out = engine.decide(DecideRequest(state="Where is my package? It was due yesterday.", questions={"q": q}))
    assert out.answers["q"].choice == "track_order"
    assert out.usage.input_tokens > 20


def test_escalation_returns_thinking_tokens(engine):
    q = {"type": "choice", "instructions": "Pick the larger number.", "criteria": ["seventeen", "four"]}
    out = engine.decide(DecideRequest(state="Numbers only.", questions={"q": q}, policy={"escalate_below": 1.0}))
    assert out.policy_used["q"] == "escalated"
    assert 0 < out.usage.output_tokens <= 512
    assert out.answers["q"].choice == "seventeen"
