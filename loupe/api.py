from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

QuestionType = Literal["noul", "choice", "score"]
Order = Literal["state_first", "rubric_first"]
PolicyUsed = Literal["one_pass", "permuted", "escalated"]
MAX_OPTIONS = 255


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    label: str | None = Field(default=None, min_length=1)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: QuestionType
    instructions: str | dict | list
    criteria: list[Annotated[str, Field(min_length=1)] | Option] | None = None

    def option_texts(self) -> list[str]:
        return [c if isinstance(c, str) else c.text for c in self.criteria or []]

    def option_keys(self) -> list[str]:
        keys = []
        for i, c in enumerate(self.criteria or []):
            if isinstance(c, Option) and c.label:
                keys.append(c.label)
            elif self.type == "score":
                keys.append(str(i))
            else:
                keys.append(c if isinstance(c, str) else c.text)
        return keys

    @model_validator(mode="before")
    @classmethod
    def normalise_criteria(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get("criteria"), dict):
            return data
        raw = data["criteria"]
        kind = data.get("type")
        if kind == "choice":
            options = [{"text": f"{label}: {text}" if text else label, "label": label} for label, text in raw.items()]
        elif kind == "noul":
            if set(raw) != {"false", "true"}:
                raise ValueError("noul criteria dict must have exactly the keys false and true")
            options = [f"no: {raw['false']}", f"yes: {raw['true']}"]
        else:
            raise ValueError("score criteria must be an ordered list of levels")
        return {**data, "criteria": options}

    @model_validator(mode="after")
    def check_criteria(self):
        if self.criteria is not None:
            for value in (*self.option_texts(), *self.option_keys()):
                if "\n" in value or "\r" in value:
                    raise ValueError("option text and labels must be single-line")
        if self.type == "noul":
            return self
        if self.criteria is None:
            raise ValueError(f"{self.type} questions need criteria")
        n = len(self.criteria)
        if n < 2 or n > MAX_OPTIONS:
            raise ValueError(f"criteria must have 2 to {MAX_OPTIONS} entries, got {n}")
        texts, keys = self.option_texts(), self.option_keys()
        if len(set(texts)) != n:
            raise ValueError("option texts must be unique")
        if len(set(keys)) != n:
            raise ValueError("option labels must be unique")
        return self


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    permutations: Literal[1, 2, 4] = 1
    escalate_below: float | None = Field(default=None, ge=0.0, le=1.0)
    order: Order = "state_first"


MODEL_NAME = "loupe-1.0"
# the name the public decision benchmark's harness sends; accepted as an alias of MODEL_NAME
HARNESS_MODEL_NAME = "jev-latest"


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = MODEL_NAME
    state: str | dict | list
    questions: dict[str, Question] = Field(min_length=1, max_length=64)
    policy: Policy = Policy()


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: str
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0


class DecideResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
    policy_used: dict[str, PolicyUsed]


class RequestRejected(Exception):
    def __init__(self, path: str, message: str):
        super().__init__(message)
        self.path = path
        self.message = message


class ContextBudgetExceeded(Exception):
    def __init__(self, tokens: int, budget: int):
        super().__init__(f"prompt is {tokens} tokens, budget is {budget}")
        self.tokens = tokens
        self.budget = budget
