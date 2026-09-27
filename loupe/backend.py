from typing import Protocol

from loupe.prompt import Messages


# Contract a real backend must meet, beyond the signatures:
# - count_tokens counts the fully templated prompt: chat template applied, the
#   assistant turn opened, the empty think block included. The engine enforces
#   the context budget on this number.
# - slice_logits runs one batched forward pass and returns one row per batch
#   element in input order; each row has one logit per entry of that element's
#   slice_ids, in the same order, read at the answer position.
# - think_then_slice runs the served model, adapter included, lets it think for
#   at most max_thinking_tokens, closes the think block itself if the model has
#   not, then reads the same slice at the answer position. It returns the
#   logits and the count of thinking tokens it generated. Rows may be logits or
#   log-probabilities over the slice: the engine only applies a softmax, which
#   is shift invariant.
# - label_prefix is the string placed before a label when probing the tokenizer:
#   " " when the answer token carries a leading space, "" when it does not. The
#   backend decides; the engine, prompt, and label modules do not assume.
class Backend(Protocol):
    label_prefix: str

    def encode(self, text: str) -> list[int]: ...

    def count_tokens(self, messages: Messages) -> int: ...

    def slice_logits(self, batch: list[Messages], slice_ids: list[list[int]]) -> list[list[float]]: ...

    def think_then_slice(
        self, messages: Messages, slice_ids: list[int], max_thinking_tokens: int
    ) -> tuple[list[float], int]: ...
