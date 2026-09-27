import string
from collections.abc import Callable

Encode = Callable[[str], list[int]]
YES = "yes"
NO = "no"


def candidate_labels() -> list[str]:
    letters = string.ascii_uppercase
    return list(letters) + [a + b for a in letters for b in letters]


class LabelAlphabet:
    def __init__(self, labels: list[str], token_ids: dict[str, int], prefix: str):
        self.labels = labels
        self.prefix = prefix
        self._ids = token_ids

    @classmethod
    def build(cls, encode: Encode, need: int = 26, prefix: str = " ") -> "LabelAlphabet":
        labels: list[str] = []
        ids: dict[str, int] = {}
        for label in candidate_labels():
            tokens = encode(prefix + label)
            if len(tokens) == 1:
                labels.append(label)
                ids[label] = tokens[0]
        for label in (YES, NO):
            tokens = encode(prefix + label)
            if len(tokens) != 1:
                raise ValueError(f"{label!r} is not a single token in this tokenizer")
            ids[label] = tokens[0]
        if len(labels) < need:
            raise ValueError(f"only {len(labels)} single-token labels, need {need}")
        return cls(labels, ids, prefix)

    def assign(self, n: int, offset: int = 0) -> list[str]:
        if n < 1 or offset < 0 or offset + n > len(self.labels):
            raise ValueError(f"cannot assign {n} labels at offset {offset} from {len(self.labels)}")
        return self.labels[offset : offset + n]

    def token_id(self, label: str) -> int:
        return self._ids[label]
