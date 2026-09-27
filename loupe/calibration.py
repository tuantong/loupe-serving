import json
from pathlib import Path

import numpy as np

TEMPERATURE_KEYS = ("noul", "choice", "score", "escalated")
# Optional difficulty-aware keys: a one-pass answer to a prompt of at least `long_tokens` tokens uses
# "<type>_long", a thinking trace that ran out of budget uses "escalated_capped", and a one-pass answer
# to a question that asks for probabilities uses "probability" (the benchmark scores those against
# exact distributions), and an answer read after a short scratch uses "scratch". Missing keys fall back
# to the plain ones (1.0 for "scratch"), so existing calibration files keep
# their meaning.
OPTIONAL_KEYS = ("noul_long", "choice_long", "score_long", "escalated_capped", "probability", "scratch")
DEFAULT_LONG_TOKENS = 800


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max()
    e = np.exp(z)
    return (e / e.sum()).tolist()


class Calibration:
    def __init__(self, temperatures: dict[str, float] | None = None, long_tokens: int = DEFAULT_LONG_TOKENS):
        given = dict(temperatures or {})
        long_tokens = int(given.pop("long_tokens", long_tokens))
        unknown = set(given) - set(TEMPERATURE_KEYS) - set(OPTIONAL_KEYS)
        if unknown:
            raise ValueError(f"unknown temperature keys: {sorted(unknown)}")
        self.temperatures = {key: 1.0 for key in TEMPERATURE_KEYS}
        self.temperatures.update(given)
        self.long_tokens = long_tokens

    def temperature(self, qtype: str, prompt_tokens: int | None = None, capped: bool = False, probability: bool = False) -> float:
        if qtype == "scratch":
            return self.temperatures.get("scratch", 1.0)
        if qtype == "escalated":
            return self.temperatures.get("escalated_capped", self.temperatures["escalated"]) if capped else self.temperatures["escalated"]
        if probability and "probability" in self.temperatures:
            return self.temperatures["probability"]
        if prompt_tokens is not None and prompt_tokens >= self.long_tokens:
            return self.temperatures.get(f"{qtype}_long", self.temperatures[qtype])
        return self.temperatures[qtype]

    @classmethod
    def load(cls, path) -> "Calibration":
        return cls(json.loads(Path(path).read_text()))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({**self.temperatures, "long_tokens": self.long_tokens}, indent=2) + "\n")


def _pad(rows: list[list[float]], fill: float) -> np.ndarray:
    width = max(len(row) for row in rows)
    out = np.full((len(rows), width), fill, dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-(np.where(targets > 0, log_probs, 0.0) * targets).sum(axis=1).mean())


def fit_temperature(logits_list: list[list[float]], targets: list[list[float]]) -> float:
    logits = _pad(logits_list, -np.inf)
    target = _pad(targets, 0.0)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    losses = [_nll(logits, target, t) for t in grid]
    best = int(np.argmin(losses))
    lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    fine = np.linspace(lo, hi, 200)
    return float(fine[int(np.argmin([_nll(logits, target, t) for t in fine]))])
