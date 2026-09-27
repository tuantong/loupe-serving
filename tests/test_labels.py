import pytest

from loupe.labels import NO, YES, LabelAlphabet, candidate_labels


def split_encode(text):
    return [hash(w) & 0x7FFFFFFF for w in text.split()]


def test_candidates_start_with_letters_then_pairs():
    c = candidate_labels()
    assert c[:3] == ["A", "B", "C"]
    assert c[26] == "AA" and c[27] == "AB"
    assert len(c) == 26 + 26 * 26


def test_build_keeps_single_token_labels_only():
    def encode(text):
        if text.strip() in {"B", "AB"}:
            return [1, 2]
        return split_encode(text)

    alphabet = LabelAlphabet.build(encode, need=100)
    assert "A" in alphabet.labels
    assert "B" not in alphabet.labels
    assert "AB" not in alphabet.labels
    assert alphabet.token_id(YES) == split_encode(" yes")[0]
    assert alphabet.token_id(NO) == split_encode(" no")[0]


def test_build_fails_when_too_few_labels():
    def encode(text):
        label = text.strip()
        return [1, 2] if len(label) == 2 and label.isupper() else split_encode(text)

    with pytest.raises(ValueError, match="single-token labels"):
        LabelAlphabet.build(encode, need=255)


def test_build_fails_when_yes_is_multi_token():
    def encode(text):
        return [1, 2] if text == " yes" else split_encode(text)

    with pytest.raises(ValueError, match="not a single token"):
        LabelAlphabet.build(encode, need=10)


def test_assign_and_token_ids():
    alphabet = LabelAlphabet.build(split_encode)
    assert alphabet.assign(3) == ["A", "B", "C"]
    assert alphabet.assign(2, offset=26) == ["AA", "AB"]
    assert alphabet.token_id("A") == split_encode(" A")[0]
    with pytest.raises(ValueError):
        alphabet.assign(0)
    with pytest.raises(ValueError):
        alphabet.assign(10, offset=len(alphabet.labels) - 5)


def test_prefix_is_what_gets_probed():
    spaced, bare = [], []

    def record(seen):
        def encode(text):
            seen.append(text)
            return split_encode(text)

        return encode

    LabelAlphabet.build(record(spaced), need=10)
    alphabet = LabelAlphabet.build(record(bare), need=10, prefix="")
    assert spaced[0] == " A"
    assert bare[0] == "A"
    assert " A" not in bare
    assert bare[-2:] == [YES, NO]
    assert alphabet.prefix == ""


def test_build_keeps_all_single_token_labels_with_a_small_floor():
    def only_letters(text):
        label = text.strip()
        return [1] if len(label) == 1 or label in ("yes", "no") else [1, 2]

    alphabet = LabelAlphabet.build(only_letters)
    assert alphabet.labels == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    with pytest.raises(ValueError, match="single-token labels"):
        LabelAlphabet.build(only_letters, need=27)
