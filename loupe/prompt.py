import json
import os

from loupe.api import Order, Question
from loupe.labels import NO, YES

SYSTEM = (
    "You are a decision engine. Read the state, then answer the question "
    "with a single label and nothing else."
)
Messages = list[dict[str, str]]


def compact() -> bool:
    """LOUPE_PROMPT_STYLE=compact drops the system message and the closing answer instruction, about 33
    tokens per prompt, which the cost axis bills. Adapters must be trained and served with the same style."""
    return os.environ.get("LOUPE_PROMPT_STYLE", "default") == "compact"


def _dump(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_state(state) -> str:
    return _dump(state)


def render_question(question: Question, labels: list[str]) -> str:
    lines = [_dump(question.instructions)]
    if question.type == "noul":
        if question.criteria:
            lines.append("Criteria:")
            lines.extend(f"- {text}" for text in question.option_texts())
        if not compact():
            lines.append(f"Answer {YES} or {NO}.")
        return "\n".join(lines)
    texts = question.option_texts()
    if len(labels) != len(texts):
        raise ValueError(f"{len(texts)} options but {len(labels)} labels")
    lines.append("Options:" if question.type == "choice" else "Levels, lowest to highest:")
    lines.extend(f"{label}) {text}" for label, text in zip(labels, texts))
    if not compact():
        lines.append("Answer with the label only.")
    return "\n".join(lines)


def build_messages(state_text: str, question_text: str, order: Order) -> Messages:
    if order == "state_first":
        user = f"State:\n{state_text}\n\nQuestion:\n{question_text}"
    else:
        user = f"Question:\n{question_text}\n\nState:\n{state_text}"
    if compact():
        return [{"role": "user", "content": user}]
    # LOUPE_SYSTEM_PROMPT replaces the system message, for serving another adapter that was trained with its own.
    return [{"role": "system", "content": os.environ.get("LOUPE_SYSTEM_PROMPT", SYSTEM)}, {"role": "user", "content": user}]
