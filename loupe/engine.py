from dataclasses import dataclass

from loupe.api import (
    Answer,
    ChoiceAnswer,
    ContextBudgetExceeded,
    DecideRequest,
    DecideResponse,
    HARNESS_MODEL_NAME,
    MODEL_NAME,
    NoulAnswer,
    Order,
    Policy,
    PolicyUsed,
    Question,
    RequestRejected,
    ScoreAnswer,
    Usage,
)
from loupe.backend import Backend
from loupe.calibration import Calibration, softmax
from loupe.labels import NO, YES, LabelAlphabet
from loupe.policy import average_over_orders, cyclic_orders, margin
from loupe.prompt import Messages, build_messages, render_question, render_state
from loupe.work import parse_probabilities


@dataclass
class _Plan:
    key: str
    question: Question
    keys: list[str]
    texts: list[str]
    labels: list[str]
    slice_ids: list[int]


def asks_for_probabilities(question: Question) -> bool:
    """The benchmark's probability items say so ("give probabilities that reflect the evidence") and are
    scored against exact distributions, so they get their own temperature."""
    return "probabilit" in (question.instructions or "").lower()


class Engine:
    def __init__(
        self,
        backend: Backend,
        alphabet: LabelAlphabet | None = None,
        calibration: Calibration | None = None,
        context_budget: int = 32768,
        model_name: str = MODEL_NAME,
        max_thinking_tokens: int = 512,
        max_escalations: int = 8,
        scratch_min_tokens: int | None = None,
        max_scratch_tokens: int = 256,
    ):
        self.backend = backend
        self.alphabet = alphabet or LabelAlphabet.build(backend.encode, prefix=backend.label_prefix)
        self.calibration = calibration or Calibration()
        self.context_budget = context_budget
        self.model_name = model_name
        self.max_thinking_tokens = max_thinking_tokens
        self.max_escalations = max_escalations
        # With a scratch threshold set, escalation writes a short scratch instead of thinking, and only on
        # prompts of at least that many tokens; shorter prompts keep the one-pass answer.
        self.scratch_min_tokens = scratch_min_tokens
        self.max_scratch_tokens = max_scratch_tokens

    def decide(self, request: DecideRequest) -> DecideResponse:
        if request.model not in (MODEL_NAME, HARNESS_MODEL_NAME, self.model_name):
            raise RequestRejected("model", f"unknown model {request.model!r}")
        try:
            state_text = render_state(request.state)
        except RecursionError:
            raise RequestRejected("state", "state is nested too deeply") from None
        order = request.policy.order
        plans = [self._plan(key, q) for key, q in request.questions.items()]
        batch = [self._render_or_reject(p, state_text, order) for p in plans]
        tokens = [self.backend.count_tokens(m) for m in batch]
        reserve = self.max_thinking_tokens if request.policy.escalate_below is not None else 0
        if max(tokens) + reserve > self.context_budget:
            raise ContextBudgetExceeded(max(tokens) + reserve, self.context_budget)
        logits = self.backend.slice_logits(batch, [p.slice_ids for p in plans])
        answers: dict[str, Answer] = {}
        used: dict[str, PolicyUsed] = {}
        extra, thinking, escalations = 0, 0, 0
        for plan, row, prompt_tokens in zip(plans, logits, tokens):
            temperature = self.calibration.temperature(plan.question.type, prompt_tokens, probability=asks_for_probabilities(plan.question))
            scaled = [x / temperature for x in row]
            if plan.question.type == "noul":
                scaled[0] += self.calibration.offset("noul", prompt_tokens)
            probs = softmax(scaled)
            probs, policy_used, spent, thought = self._refine(
                plan, state_text, order, probs, request.policy, escalations < self.max_escalations, gate=softmax(row, 1.0), prompt_tokens=prompt_tokens
            )
            if policy_used == "escalated":
                escalations += 1
            extra += spent
            thinking += thought
            answers[plan.key] = self._answer(plan, probs)
            used[plan.key] = policy_used
        return DecideResponse(
            model=self.model_name,
            answers=answers,
            usage=Usage(input_tokens=sum(tokens) + extra, output_tokens=thinking),
            policy_used=used,
        )

    def _plan(self, key: str, question: Question) -> _Plan:
        if question.type == "noul":
            labels = [YES, NO]
            return _Plan(key, question, labels, labels, labels, [self.alphabet.token_id(x) for x in labels])
        texts = question.option_texts()
        if len(texts) > len(self.alphabet.labels):
            raise RequestRejected(f"questions.{key}.criteria", f"{len(texts)} options but this backend supports {len(self.alphabet.labels)}")
        labels = self.alphabet.assign(len(texts))
        keys = question.option_keys()
        return _Plan(key, question, keys, texts, labels, [self.alphabet.token_id(x) for x in labels])

    def _render(self, plan: _Plan, state_text: str, order: Order, order_indices: list[int] | None = None) -> Messages:
        question = plan.question
        if order_indices is not None:
            question = question.model_copy(update={"criteria": [question.criteria[i] for i in order_indices]})
        return build_messages(state_text, render_question(question, plan.labels), order)

    def _render_or_reject(self, plan: _Plan, state_text: str, order: Order) -> Messages:
        try:
            return self._render(plan, state_text, order)
        except RecursionError:
            path = f"questions.{plan.key}.instructions"
            raise RequestRejected(path, "instructions are nested too deeply") from None

    def _refine(
        self,
        plan: _Plan,
        state_text: str,
        order: Order,
        probs: list[float],
        policy: Policy,
        may_escalate: bool,
        gate: list[float] | None = None,
        prompt_tokens: int = 0,
    ) -> tuple[list[float], PolicyUsed, int, int]:
        """`gate` is the uncalibrated one-pass distribution. The escalation decision reads its margin, so
        refitting a temperature changes the probabilities returned but not which questions think."""
        threshold = policy.escalate_below
        if threshold is None or margin(gate if gate is not None else probs) >= threshold:
            return probs, "one_pass", 0, 0
        spent = 0
        reached: PolicyUsed = "one_pass"
        if policy.permutations > 1 and plan.question.type == "choice":
            orders = cyclic_orders(len(plan.texts), policy.permutations)
            batch = [self._render(plan, state_text, order, o) for o in orders[1:]]
            spent += sum(self.backend.count_tokens(m) for m in batch)
            rows = self.backend.slice_logits(batch, [plan.slice_ids] * len(batch))
            temperature = self.calibration.temperature(plan.question.type)
            per_order = [probs] + [softmax(r, temperature) for r in rows]
            probs = average_over_orders(per_order, orders)
            reached = "permuted"
            if margin(probs) >= threshold:
                return probs, reached, spent, 0
        if not may_escalate:
            return probs, reached, spent, 0
        if self.scratch_min_tokens is not None:
            if prompt_tokens < self.scratch_min_tokens:
                return probs, reached, spent, 0
            messages = self._render(plan, state_text, order)
            row, written, work = self.backend.scratch_then_slice_many([messages], [plan.slice_ids], self.max_scratch_tokens)[0]
            stated = parse_probabilities(work, plan.labels) if asks_for_probabilities(plan.question) else None
            # Billed: the scratch as output, the two markers as input; the prompt itself is served from the prefix cache.
            return stated or softmax(row, self.calibration.temperature("scratch")), "escalated", spent + self.backend.scratch_marker_tokens, written
        messages = self._render(plan, state_text, order)
        spent += self.backend.count_tokens(messages)
        row, thought = self.backend.think_then_slice(messages, plan.slice_ids, self.max_thinking_tokens)
        capped = thought >= self.max_thinking_tokens
        return softmax(row, self.calibration.temperature("escalated", capped=capped)), "escalated", spent, thought

    def _answer(self, plan: _Plan, probs: list[float]) -> Answer:
        if plan.question.type == "noul":
            return NoulAnswer(noul=probs[0])
        best = max(range(len(probs)), key=probs.__getitem__)
        confidence = margin(probs)
        if plan.question.type == "choice":
            return ChoiceAnswer(
                choice=plan.keys[best],
                probabilities=dict(zip(plan.keys, probs)),
                confidence=confidence,
            )
        return ScoreAnswer(
            score=plan.keys[best],
            legend=dict(zip(plan.labels, plan.keys)),
            probabilities=dict(zip(plan.keys, probs)),
            confidence=confidence,
        )
