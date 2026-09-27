"""The short-scratch text format shared by training and serving: the markers around the work, and the
"Probabilities:" line a scratch may end with, which serving can return as the answer's distribution."""

import re

WORK_OPEN = "Work:\n"
WORK_CLOSE = "\nAnswer:\n"
PROBABILITIES = "Probabilities:"


def parse_probabilities(text: str, labels: list[str]) -> list[float] | None:
    """The distribution stated on the work's last "Probabilities:" line, in label order, or None."""
    lines = [l for l in text.splitlines() if l.strip().startswith(PROBABILITIES)]
    if not lines:
        return None
    found = dict(re.findall(r"(\S+)\s+([01](?:\.\d+)?)", lines[-1][len(PROBABILITIES) :].replace(",", " ")))
    if set(found) != set(labels):
        return None
    probs = [float(found[label]) for label in labels]
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else None
