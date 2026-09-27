import sys
import types
from dataclasses import dataclass, field

import pytest

CLOSE = 999


class FakeParams:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.SamplingParams = FakeParams
    monkeypatch.setitem(sys.modules, "vllm", mod)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text.startswith("</think>", i):
                ids.append(CLOSE)
                i += len("</think>")
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tokenize):
        return [1, 2, 3] + ([7] if enable_thinking else [8])


@dataclass
class Logprob:
    logprob: float


@dataclass
class Completion:
    token_ids: list = field(default_factory=list)
    logprobs: list | None = None
    finish_reason: str = "stop"
    text: str = ""


@dataclass
class Output:
    outputs: list


class FakeLLM:
    def __init__(self, label_logprobs: dict[int, float], trace: list[int], finish_reason: str):
        self.label_logprobs = label_logprobs
        self.trace = trace
        self.finish_reason = finish_reason
        self.calls = []

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        params = params if isinstance(params, list) else [params] * len(prompts)
        self.calls.append((prompts, params))
        outs = []
        for p in params:
            if p.kw.get("max_tokens") == 1:
                allowed = p.kw["allowed_token_ids"]
                step = {t: Logprob(self.label_logprobs[t]) for t in allowed if t in self.label_logprobs}
                outs.append(Output([Completion(logprobs=[step])]))
            else:
                outs.append(Output([Completion(token_ids=list(self.trace), finish_reason=self.finish_reason)]))
        return outs


def backend(llm):
    from loupe.backend_vllm import VLLMBackend

    return VLLMBackend(llm=llm)


A, B, C = ord("A"), ord("B"), ord("C")


def test_slice_logits_restricts_to_labels_and_keeps_order():
    llm = FakeLLM({A: -0.2, B: -1.9, C: -3.0}, [], "stop")
    rows = backend(llm).slice_logits([[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]], [[B, A], [C, B, A]])
    assert rows == [[-1.9, -0.2], [-3.0, -1.9, -0.2]]
    _, params = llm.calls[-1]
    assert params[0].kw == {"max_tokens": 1, "temperature": 0.0, "allowed_token_ids": [B, A], "logprobs": 2}
    assert llm.calls[-1][0][0] == {"prompt_token_ids": [1, 2, 3, 8]}


def test_missing_label_gets_a_very_low_logprob():
    from loupe.backend_vllm import MISSING_LOGPROB

    llm = FakeLLM({A: -0.1}, [], "stop")
    assert backend(llm).slice_logits([[{"role": "user", "content": "x"}]], [[A, B]]) == [[-0.1, MISSING_LOGPROB]]


def test_closed_trace_reads_the_slice_after_the_think_block():
    llm = FakeLLM({A: -0.5, B: -0.9}, [50, 51, CLOSE], "stop")
    row, thought = backend(llm).think_then_slice([{"role": "user", "content": "x"}], [A, B], 64)
    assert row == [-0.5, -0.9] and thought == 2
    think_prompts, think_params = llm.calls[-2]
    assert think_prompts == [{"prompt_token_ids": [1, 2, 3, 7]}]
    assert think_params[0].kw["max_tokens"] == 64 and think_params[0].kw["temperature"] == 0.6
    assert think_params[0].kw["stop_token_ids"] == [CLOSE]
    read_prompt = llm.calls[-1][0][0]["prompt_token_ids"]
    assert read_prompt == [1, 2, 3, 7, 50, 51, CLOSE, 10, 10]


def test_budget_exhausted_trace_is_closed_before_reading():
    llm = FakeLLM({A: -0.3, B: -1.2}, [50, 51, 52, 53], "length")
    row, thought = backend(llm).think_then_slice([{"role": "user", "content": "x"}], [A, B], 4)
    assert thought == 4
    assert llm.calls[-1][0][0]["prompt_token_ids"] == [1, 2, 3, 7, 50, 51, 52, 53, CLOSE, 10, 10]


class FakeLoRARequest:
    def __init__(self, name, lora_id, path):
        self.name = name


@pytest.fixture
def fake_lora(monkeypatch):
    lora_mod = types.ModuleType("vllm.lora.request")
    lora_mod.LoRARequest = FakeLoRARequest
    monkeypatch.setitem(sys.modules, "vllm.lora", types.ModuleType("vllm.lora"))
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_mod)


class RecordingLLM(FakeLLM):
    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.loras = getattr(self, "loras", []) + [lora_request]
        return super().generate(prompts, params, lora_request, use_tqdm)


def test_one_pass_uses_the_adapter_and_thinking_uses_the_base_model(fake_lora):
    from loupe.backend_vllm import VLLMBackend

    llm = RecordingLLM({A: -0.5, B: -0.9}, [50, CLOSE], "stop")
    b = VLLMBackend(llm=llm, adapter_path="/tmp/adapter")
    b.slice_logits([[{"role": "user", "content": "x"}]], [[A, B]])
    b.think_then_slice([{"role": "user", "content": "x"}], [A, B], 16)
    assert llm.loras[0] is b.lora and b.lora is not None
    assert llm.loras[1] is None and llm.loras[2] is None


def test_thinking_can_opt_into_the_adapter(fake_lora):
    from loupe.backend_vllm import VLLMBackend

    llm = RecordingLLM({A: -0.5, B: -0.9}, [50, CLOSE], "stop")
    b = VLLMBackend(llm=llm, adapter_path="/tmp/adapter", think_with_adapter=True)
    b.think_then_slice([{"role": "user", "content": "x"}], [A, B], 16)
    assert llm.loras == [b.lora, b.lora]


def test_think_many_batches_generation_and_reads_every_slice():
    llm = FakeLLM({A: -0.2, B: -1.0}, [50, CLOSE], "stop")
    out = backend(llm).think_then_slice_many([[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]], [[A, B], [B, A]], 32)
    assert out == [([-0.2, -1.0], 1), ([-1.0, -0.2], 1)]
    think_prompts, _ = llm.calls[-2]
    assert len(think_prompts) == 2


MSG = [{"role": "user", "content": "x"}]


class ChunkLLM(FakeLLM):
    """Continues `trace` from wherever the prompt leaves off, `max_tokens` at a time. The label read
    favours A strongly once `settle_after` trace tokens precede it, and B mildly before that."""

    def __init__(self, trace: list[int], settle_after: int):
        super().__init__({}, trace, "length")
        self.settle_after = settle_after

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        params = params if isinstance(params, list) else [params] * len(prompts)
        self.calls.append((prompts, params))
        outs = []
        for prompt, p in zip(prompts, params):
            done = sum(50 <= t < 100 for t in prompt["prompt_token_ids"])
            if p.kw.get("max_tokens") == 1:
                settled = done >= self.settle_after
                outs.append(Output([Completion(logprobs=[{A: Logprob(-0.01 if settled else -2.0), B: Logprob(-5.0 if settled else -0.2)}])]))
            else:
                new = self.trace[done : done + p.kw["max_tokens"]]
                if CLOSE in new:
                    new = new[: new.index(CLOSE) + 1]
                outs.append(Output([Completion(token_ids=new, finish_reason="stop" if CLOSE in new else "length")]))
        return outs


def test_presence_penalty_reaches_the_thinking_sampler():
    from loupe.backend_vllm import VLLMBackend

    llm = FakeLLM({A: -0.5, B: -0.9}, [50, CLOSE], "stop")
    VLLMBackend(llm=llm, think_presence_penalty=1.5).think_then_slice(MSG, [A, B], 64)
    assert llm.calls[-2][1][0].kw["presence_penalty"] == 1.5


def test_capped_trace_reads_after_the_wrap_up_instruction():
    from loupe.backend_vllm import VLLMBackend

    llm = FakeLLM({A: -0.3, B: -1.2}, [50, 51, 52, 53], "length")
    VLLMBackend(llm=llm, wrap_up="W").think_then_slice(MSG, [A, B], 4)
    assert llm.calls[-1][0][0]["prompt_token_ids"] == [1, 2, 3, 7, 50, 51, 52, 53, ord("W"), CLOSE, 10, 10]


def test_trajectories_probe_after_every_chunk_until_the_budget():
    from loupe.backend_vllm import VLLMBackend

    llm = ChunkLLM([50, 51, 52, 53, 54, 55], settle_after=99)
    [t] = VLLMBackend(llm=llm, wrap_up="W").think_trajectories_many([MSG], [[A, B]], 6, chunk=2)
    assert [n for n, _ in t["probes"]] == [2, 4, 6]
    assert set(t["probes"][0][1]) == {"plain", "wrap"}
    assert t["closed"] is False and t["thought"] == 6 and t["final"] is None


def test_trajectories_stop_probing_when_the_trace_closes():
    from loupe.backend_vllm import VLLMBackend

    llm = ChunkLLM([50, 51, 52, CLOSE], settle_after=99)
    [t] = VLLMBackend(llm=llm).think_trajectories_many([MSG], [[A, B]], 8, chunk=2)
    assert [n for n, _ in t["probes"]] == [2]
    assert t["closed"] is True and t["thought"] == 3 and t["final"] is not None
    assert llm.calls[-1][0][0]["prompt_token_ids"] == [1, 2, 3, 7, 50, 51, 52, CLOSE, 10, 10]


def test_early_exit_once_two_probes_agree_confidently():
    from loupe.backend_vllm import VLLMBackend

    llm = ChunkLLM(list(range(50, 58)), settle_after=2)
    row, thought = VLLMBackend(llm=llm, probe_every=2, exit_confidence=0.9).think_then_slice(MSG, [A, B], 8)
    assert thought == 4 and row[0] > row[1]
    assert sum(1 for _, params in llm.calls if params[0].kw.get("max_tokens") != 1) == 2


def test_no_early_exit_below_the_confidence_bar():
    from loupe.backend_vllm import VLLMBackend

    llm = ChunkLLM(list(range(50, 58)), settle_after=99)
    _, thought = VLLMBackend(llm=llm, probe_every=2, exit_confidence=0.9).think_then_slice(MSG, [A, B], 8)
    assert thought == 8


def test_empty_batch_thinks_about_nothing():
    llm = FakeLLM({A: -0.5}, [50, CLOSE], "stop")
    assert backend(llm).think_then_slice_many([], [], 64) == []
    assert llm.calls == []
