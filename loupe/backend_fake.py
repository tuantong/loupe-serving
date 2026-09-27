import hashlib
import re

from loupe.labels import NO, YES
from loupe.prompt import Messages

_OPTION_LINE = re.compile(r"^([A-Z]{1,2})\) (.*)$", re.MULTILINE)
_NOUL_LINE = f"Answer {YES} or {NO}."


def _token(word: str) -> int:
    return int(hashlib.blake2b(word.encode(), digest_size=4).hexdigest(), 16)


def _pseudo_logit(text: str) -> float:
    return (_token(text) % 2001 - 1000) / 1000


class FakeBackend:
    def __init__(
        self,
        scripted: dict[str, float] | None = None,
        position_bias: float = 0.0,
        think_scripted: dict[str, float] | None = None,
        label_prefix: str = " ",
        think_tokens: int = 7,
    ):
        self.scripted = scripted or {}
        self.position_bias = position_bias
        self.think_scripted = think_scripted or {}
        self.label_prefix = label_prefix
        self.think_tokens = think_tokens
        self.calls: list[list[Messages]] = []
        self.think_calls: list[Messages] = []
        self.think_budgets: list[int] = []

    def encode(self, text: str) -> list[int]:
        return [_token(word) for word in text.split()]

    def count_tokens(self, messages: Messages) -> int:
        return sum(len(self.encode(m["content"])) + 4 for m in messages)

    def _option_texts(self, messages: Messages) -> list[str]:
        content = messages[-1]["content"]
        if _NOUL_LINE in content:
            return [YES, NO]
        return [m.group(2) for m in _OPTION_LINE.finditer(content)]

    def _logits(self, messages: Messages, n: int, scripted: dict[str, float], bias: float) -> list[float]:
        texts = self._option_texts(messages)
        if len(texts) != n:
            raise ValueError(f"prompt lists {len(texts)} options but slice has {n} ids")
        logits = [scripted.get(t, _pseudo_logit(t)) for t in texts]
        logits[0] += bias
        return logits

    def slice_logits(self, batch: list[Messages], slice_ids: list[list[int]]) -> list[list[float]]:
        self.calls.append(batch)
        return [self._logits(m, len(ids), self.scripted, self.position_bias) for m, ids in zip(batch, slice_ids)]

    def think_then_slice(
        self, messages: Messages, slice_ids: list[int], max_thinking_tokens: int
    ) -> tuple[list[float], int]:
        self.think_calls.append(messages)
        self.think_budgets.append(max_thinking_tokens)
        logits = self._logits(messages, len(slice_ids), self.think_scripted or self.scripted, 0.0)
        return logits, min(self.think_tokens, max_thinking_tokens)
