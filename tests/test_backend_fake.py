from loupe.backend_fake import FakeBackend
from loupe.prompt import build_messages


def prompt(lines):
    return build_messages("s", "\n".join(lines), "state_first")


def test_encode_is_whitespace_tokens_and_labels_are_single():
    b = FakeBackend()
    assert len(b.encode(" A")) == 1
    assert len(b.encode(" AB")) == 1
    assert len(b.encode(" yes")) == 1
    assert len(b.encode("two words")) == 2


def test_count_tokens_is_deterministic_and_grows_with_text():
    b = FakeBackend()
    short = b.count_tokens(prompt(["A) x"]))
    long = b.count_tokens(prompt(["A) x", "B) y y y"]))
    assert short == b.count_tokens(prompt(["A) x"]))
    assert long > short


def test_scripted_logits_follow_option_text_not_position():
    b = FakeBackend(scripted={"billing": 2.0, "technical": -1.0})
    forward = b.slice_logits([prompt(["A) billing", "B) technical"])], [[1, 2]])[0]
    backward = b.slice_logits([prompt(["A) technical", "B) billing"])], [[2, 1]])[0]
    assert forward == [2.0, -1.0]
    assert backward == [-1.0, 2.0]


def test_position_bias_favours_first_listed_option():
    b = FakeBackend(scripted={"x": 0.0, "y": 0.0}, position_bias=3.0)
    assert b.slice_logits([prompt(["A) x", "B) y"])], [[1, 2]])[0] == [3.0, 0.0]


def test_noul_uses_yes_no_keys():
    b = FakeBackend(scripted={"yes": 1.5, "no": -1.5})
    assert b.slice_logits([prompt(["Answer yes or no."])], [[7, 8]])[0] == [1.5, -1.5]


def test_unscripted_options_are_deterministic():
    b = FakeBackend()
    first = b.slice_logits([prompt(["A) p", "B) q", "C) r"])], [[1, 2, 3]])[0]
    second = b.slice_logits([prompt(["A) p", "B) q", "C) r"])], [[1, 2, 3]])[0]
    assert first == second
    assert len(first) == 3


def test_think_then_slice_ignores_position_bias_and_reports_thinking_tokens():
    b = FakeBackend(scripted={"x": 0.0, "y": 1.0}, position_bias=5.0, think_scripted={"x": -2.0, "y": 2.0})
    messages = prompt(["A) x", "B) y"])
    assert b.think_then_slice(messages, [1, 2], 512) == ([-2.0, 2.0], 7)
    assert b.think_then_slice(messages, [1, 2], 3) == ([-2.0, 2.0], 3)
    assert b.think_calls == [messages, messages]
    assert b.think_budgets == [512, 3]
    assert b.calls == []


def test_label_prefix_is_a_leading_space_by_default():
    assert FakeBackend().label_prefix == " "
    assert FakeBackend(label_prefix="", think_tokens=2).think_then_slice(prompt(["A) x", "B) y"]), [1, 2], 9)[1] == 2
