import os

import mlx.core as mx
from mlx.utils import tree_unflatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from loupe.prompt import Messages

DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-bf16"
THINK_CLOSE = "</think>"
PREFILL_CHUNK = 512


class MLXBackend:
    def __init__(self, model_path: str = DEFAULT_MODEL, label_prefix: str = "", adapter_path: str | None = None):
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        if adapter_path:
            fused = [(name, module.fuse()) for name, module in self.model.named_modules() if hasattr(module, "fuse")]
            self.model.update_modules(tree_unflatten(fused))
            mx.eval(self.model.parameters())
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.label_prefix = label_prefix
        self.eos_ids = set(self.tokenizer.eos_token_ids or [self.tokenizer.eos_token_id])
        self.close_ids = self.encode(THINK_CLOSE)
        self.after_close_ids = self.encode(THINK_CLOSE + "\n\n")[len(self.close_ids) :]
        for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if len(self.encode(label_prefix + label)) != 1:
                raise ValueError(f"label {label!r} with prefix {label_prefix!r} is not a single token for {model_path}")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "MLXBackend":
        env = env or os.environ
        return cls(env.get("LOUPE_MLX_MODEL", DEFAULT_MODEL), adapter_path=env.get("LOUPE_MLX_ADAPTER") or None)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prompt_ids(self, messages: Messages, thinking: bool) -> list[int]:
        out = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking)
        return out if isinstance(out, list) else self.encode(out)

    def count_tokens(self, messages: Messages) -> int:
        return len(self._prompt_ids(messages, False))

    def _last_logits(self, ids: list[int], cache) -> mx.array:
        for start in range(0, len(ids) - 1, PREFILL_CHUNK):
            mx.eval(self.model(mx.array([ids[start : start + PREFILL_CHUNK][: len(ids) - 1 - start]]), cache=cache))
        last = self.model(mx.array([ids[-1:]]), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(last)
        return last

    def slice_logits(self, batch: list[Messages], slice_ids: list[list[int]]) -> list[list[float]]:
        rows = []
        for messages, ids in zip(batch, slice_ids):
            last = self._last_logits(self._prompt_ids(messages, False), make_prompt_cache(self.model))
            rows.append(last[mx.array(ids)].tolist())
            mx.clear_cache()
        return rows

    def think_then_slice(self, messages: Messages, slice_ids: list[int], max_thinking_tokens: int) -> tuple[list[float], int]:
        cache = make_prompt_cache(self.model)
        last = self._last_logits(self._prompt_ids(messages, True), cache)
        generated: list[int] = []
        closed = False
        while len(generated) < max_thinking_tokens:
            token = int(mx.argmax(last))
            if token in self.eos_ids:
                break
            generated.append(token)
            last = self._last_logits([token], cache)
            if generated[-len(self.close_ids) :] == self.close_ids:
                closed = True
                break
        tail = self.after_close_ids if closed else self.close_ids + self.after_close_ids
        last = self._last_logits(tail, cache)
        row = last[mx.array(slice_ids)].tolist()
        mx.clear_cache()
        return row, len(generated)
