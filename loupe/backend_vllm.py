"""vLLM backend: one engine serves the one-pass slice read and the thinking escalation.

The one-pass read asks for a single next token restricted to the option labels
(`allowed_token_ids`) with `logprobs_mode="processed_logprobs"`, so the returned
logprobs are the model's distribution renormalised over the labels. Softmax is
shift invariant, so those logprobs stand in for the raw slice logits the engine
expects, at any temperature.

Escalation samples a thinking trace with the settings measured on the hard
tier (temperature 0.6, top-p 0.95, top-k 20), closes the think block if the
budget ran out, and reads the same slice at the answer position. It runs on
the base model by default, because that is what the 86.5% hard-tier
measurement used: the adapter only shapes the one-pass read.

Three optional guards against overthinking: a presence penalty in the
thinking sampler, Qwen's stop-thinking instruction before the forced close
when the budget runs out, and early exit, which thinks in chunks, reads the
answer after each one and stops once two reads in a row agree confidently.
"""

import json
import os

from loupe.calibration import softmax
from loupe.prompt import Messages
from loupe.work import WORK_CLOSE, WORK_OPEN

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
THINK_CLOSE = "</think>"
MISSING_LOGPROB = -1e4
WRAP_UP = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"


class VLLMBackend:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        adapter_path: str | None = None,
        label_prefix: str = "",
        max_model_len: int = 40960,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
        think_temperature: float = 0.6,
        think_top_p: float = 0.95,
        think_top_k: int = 20,
        think_with_adapter: bool = False,
        think_presence_penalty: float = 0.0,
        wrap_up: str | None = None,
        probe_every: int | None = None,
        exit_confidence: float = 0.9,
        quantization: str | None = None,
        kv_cache_dtype: str = "auto",
        speculative_config: dict | None = None,
        engine_args: dict | None = None,
        llm=None,
    ):
        from vllm import SamplingParams

        self._params = SamplingParams
        if llm is None:
            from vllm import LLM

            extra = {"max_lora_rank": 64}
            if quantization:
                extra["quantization"] = quantization
            if speculative_config:
                extra["speculative_config"] = speculative_config
            extra.update(engine_args or {})
            llm = LLM(
                model=model_path,
                dtype="bfloat16",
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=enforce_eager,
                logprobs_mode="processed_logprobs",
                enable_lora=bool(adapter_path),
                kv_cache_dtype=kv_cache_dtype,
                **extra,
            )
        self.llm = llm
        self.lora = None
        if adapter_path:
            from vllm.lora.request import LoRARequest

            self.lora = LoRARequest("adapter", 1, os.path.abspath(adapter_path))
        self.think_lora = self.lora if think_with_adapter else None
        self.tokenizer = llm.get_tokenizer()
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.label_prefix = label_prefix
        self.think = dict(temperature=think_temperature, top_p=think_top_p, top_k=think_top_k)
        if think_presence_penalty:
            self.think["presence_penalty"] = think_presence_penalty
        self.close_ids = self.encode(THINK_CLOSE)
        self.after_close_ids = self.encode(THINK_CLOSE + "\n\n")[len(self.close_ids) :]
        self.wrap_up_ids = self.encode(wrap_up) if wrap_up else []
        self.probe_every = probe_every
        self.exit_confidence = exit_confidence
        for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if len(self.encode(label_prefix + label)) != 1:
                raise ValueError(f"label {label!r} with prefix {label_prefix!r} is not a single token for {model_path}")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "VLLMBackend":
        env = env or os.environ
        return cls(
            env.get("LOUPE_VLLM_MODEL", DEFAULT_MODEL),
            adapter_path=env.get("LOUPE_VLLM_ADAPTER") or None,
            max_model_len=int(env.get("LOUPE_VLLM_MAX_LEN", "40960")),
            gpu_memory_utilization=float(env.get("LOUPE_VLLM_MEM", "0.85")),
            enforce_eager=env.get("LOUPE_VLLM_EAGER") == "1",
            think_with_adapter=env.get("LOUPE_VLLM_THINK_ADAPTER") == "1",
            think_presence_penalty=float(env.get("LOUPE_VLLM_PRESENCE", "0")),
            wrap_up=WRAP_UP if env.get("LOUPE_VLLM_WRAP_UP") == "1" else None,
            probe_every=int(env["LOUPE_VLLM_PROBE_EVERY"]) if env.get("LOUPE_VLLM_PROBE_EVERY") else None,
            exit_confidence=float(env.get("LOUPE_VLLM_EXIT_CONFIDENCE", "0.9")),
            quantization=env.get("LOUPE_VLLM_QUANT") or None,
            kv_cache_dtype=env.get("LOUPE_VLLM_KV_DTYPE", "auto"),
            speculative_config=json.loads(env["LOUPE_VLLM_SPEC"]) if env.get("LOUPE_VLLM_SPEC") else None,
            engine_args=json.loads(env["LOUPE_VLLM_EXTRA"]) if env.get("LOUPE_VLLM_EXTRA") else None,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prompt_ids(self, messages: Messages, thinking: bool) -> list[int]:
        out = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=True)
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        return list(out) if not isinstance(out, str) else self.encode(out)

    def count_tokens(self, messages: Messages) -> int:
        return len(self._prompt_ids(messages, False))

    def _slice_many(self, prompts: list[list[int]], slice_ids: list[list[int]], lora="default") -> list[list[float]]:
        lora = self.lora if lora == "default" else lora
        params = [self._params(max_tokens=1, temperature=0.0, allowed_token_ids=list(ids), logprobs=len(ids)) for ids in slice_ids]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=lora, use_tqdm=False)
        rows = []
        for out, ids in zip(outs, slice_ids):
            step = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            rows.append([step[t].logprob if t in step else MISSING_LOGPROB for t in ids])
        return rows

    def slice_logits(self, batch: list[Messages], slice_ids: list[list[int]]) -> list[list[float]]:
        return self._slice_many([self._prompt_ids(m, False) for m in batch], slice_ids)

    def think_then_slice(self, messages: Messages, slice_ids: list[int], max_thinking_tokens: int) -> tuple[list[float], int]:
        return self.think_then_slice_many([messages], [slice_ids], max_thinking_tokens)[0]

    def think_then_slice_many(
        self, batch: list[Messages], slice_ids: list[list[int]], max_thinking_tokens: int
    ) -> list[tuple[list[float], int]]:
        """Think on every prompt in one batched generate call, then read each slice. With
        `probe_every` set, think in chunks of that size and stop early once the answer settles."""
        if not batch:
            return []
        if self.probe_every:
            traces = self._think_in_chunks(batch, slice_ids, max_thinking_tokens, self.probe_every, stop_early=True)
            return [(t["final"], t["thought"]) for t in traces]
        prompts = [self._prompt_ids(m, True) for m in batch]
        stop_ids = self.close_ids if len(self.close_ids) == 1 else []
        params = self._params(max_tokens=max_thinking_tokens, stop_token_ids=stop_ids, stop=[THINK_CLOSE], **self.think)
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=self.think_lora, use_tqdm=False)
        reads, thoughts = [], []
        for prompt, out in zip(prompts, outs):
            completion = out.outputs[0]
            generated = list(completion.token_ids)
            ends_closed = generated[-len(self.close_ids) :] == self.close_ids
            closed = ends_closed or completion.finish_reason == "stop"
            if closed and not ends_closed:
                generated += self.close_ids
            tail = self.after_close_ids if closed else self.wrap_up_ids + self.close_ids + self.after_close_ids
            reads.append(prompt + generated + tail)
            thoughts.append(len(generated) - (len(self.close_ids) if closed else 0))
        rows = self._slice_many(reads, [list(ids) for ids in slice_ids], lora=self.think_lora)
        return list(zip(rows, thoughts))

    @property
    def scratch_marker_tokens(self) -> int:
        return len(self.encode(WORK_OPEN)) + len(self.encode(WORK_CLOSE))

    def scratch_then_slice_many(self, batch: list[Messages], slice_ids: list[list[int]], max_tokens: int) -> list[tuple[list[float], int, str]]:
        """Write a short worked computation after the one-pass prompt, then read the answer slice.
        The caller decides which prompts get a scratch; WORK_OPEN is appended here, the work is written
        greedily with the adapter and ends at WORK_CLOSE or the token limit. Returns the slice row, the
        number of generated tokens and the work text."""
        if not batch:
            return []
        stop = WORK_CLOSE.rstrip("\n")
        open_ids, close_ids = self.encode(WORK_OPEN), self.encode(WORK_CLOSE)
        prompts = [self._prompt_ids(m, False) + open_ids for m in batch]
        params = self._params(max_tokens=max_tokens, temperature=0.0, stop=[stop])
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], params, lora_request=self.lora, use_tqdm=False)
        texts = [out.outputs[0].text.split(stop)[0] for out in outs]
        rows = self._slice_many([p + self.encode(t) + close_ids for p, t in zip(prompts, texts)], [list(ids) for ids in slice_ids])
        return [(row, len(out.outputs[0].token_ids), text) for row, out, text in zip(rows, outs, texts)]

    def think_trajectories_many(
        self, batch: list[Messages], slice_ids: list[list[int]], max_thinking_tokens: int, chunk: int
    ) -> list[dict]:
        """Think in chunks and read the answer after each one as if thinking stopped there, with a
        plain close and, when configured, after the wrap-up instruction. Nothing stops early, so
        any exit rule can be measured offline. `final` is the read after a natural close, else None."""
        return self._think_in_chunks(batch, slice_ids, max_thinking_tokens, chunk, stop_early=False)

    def _think_in_chunks(
        self, batch: list[Messages], slice_ids: list[list[int]], budget: int, chunk: int, stop_early: bool
    ) -> list[dict]:
        prompts = [self._prompt_ids(m, True) for m in batch]
        stop_ids = self.close_ids if len(self.close_ids) == 1 else []
        tails = {"plain": self.close_ids + self.after_close_ids}
        if self.wrap_up_ids:
            tails["wrap"] = self.wrap_up_ids + self.close_ids + self.after_close_ids
        answer = "wrap" if self.wrap_up_ids else "plain"
        generated: list[list[int]] = [[] for _ in batch]
        traces = [{"probes": [], "final": None, "thought": 0, "closed": False, "exited": False} for _ in batch]
        active = list(range(len(batch)))
        while active:
            params = [
                self._params(max_tokens=min(chunk, budget - len(generated[i])), stop_token_ids=stop_ids, stop=[THINK_CLOSE], **self.think)
                for i in active
            ]
            outs = self.llm.generate([{"prompt_token_ids": prompts[i] + generated[i]} for i in active], params, lora_request=self.think_lora, use_tqdm=False)
            closing, probing = [], []
            for i, out in zip(active, outs):
                completion = out.outputs[0]
                generated[i] += list(completion.token_ids)
                ends_closed = generated[i][-len(self.close_ids) :] == self.close_ids
                if ends_closed or completion.finish_reason == "stop":
                    if not ends_closed:
                        generated[i] += self.close_ids
                    closing.append(i)
                else:
                    probing.append(i)
            if closing:
                rows = self._slice_many([prompts[i] + generated[i] + self.after_close_ids for i in closing], [slice_ids[i] for i in closing], lora=self.think_lora)
                for i, row in zip(closing, rows):
                    traces[i].update(final=row, closed=True, thought=len(generated[i]) - len(self.close_ids))
            if not probing:
                break
            reads = {
                name: self._slice_many([prompts[i] + generated[i] + tail for i in probing], [slice_ids[i] for i in probing], lora=self.think_lora)
                for name, tail in tails.items()
            }
            active = []
            for k, i in enumerate(probing):
                probe = {name: rows[k] for name, rows in reads.items()}
                traces[i]["probes"].append((len(generated[i]), probe))
                traces[i]["thought"] = len(generated[i])
                if stop_early and self._settled(traces[i]["probes"], answer):
                    traces[i].update(final=probe[answer], exited=True)
                elif len(generated[i]) >= budget:
                    if stop_early:
                        traces[i]["final"] = probe[answer]
                else:
                    active.append(i)
        return traces

    def _settled(self, probes: list[tuple[int, dict]], key: str) -> bool:
        if len(probes) < 2:
            return False
        before, now = softmax(probes[-2][1][key]), softmax(probes[-1][1][key])
        top = max(range(len(now)), key=now.__getitem__)
        return top == max(range(len(before)), key=before.__getitem__) and now[top] >= self.exit_confidence
