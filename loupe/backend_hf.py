import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from loupe.prompt import Messages
from loupe.adapters import lora_deltas, merge

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
THINK_CLOSE = "</think>"


def resolve_adapter(adapter_path: str) -> Path:
    local = Path(adapter_path)
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(adapter_path, allow_patterns=["adapters.safetensors", "adapter_model.safetensors", "adapter_config.json", "calibration.json"]))


def apply_adapter(model, adapter_dir: Path):
    if (adapter_dir / "adapter_model.safetensors").exists():
        from peft import PeftModel

        return PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    deltas = lora_deltas(adapter_dir)
    applied = merge(model, deltas)
    if applied != len(deltas):
        raise ValueError(f"adapter {adapter_dir}: applied {applied} of {len(deltas)} matrices")
    return model


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFBackend:
    def __init__(self, model_path: str = DEFAULT_MODEL, label_prefix: str = "", device: str | None = None, adapter_path: str | None = None):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        if adapter_path:
            self.model = apply_adapter(self.model, resolve_adapter(adapter_path))
        self.model = self.model.to(self.device).eval()
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.label_prefix = label_prefix
        self.max_batch = int(os.environ.get("LOUPE_HF_MAX_BATCH", "16"))
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos if eos is not None else self.tokenizer.eos_token_id])
        self.close_ids = self.encode(THINK_CLOSE)
        self.after_close_ids = self.encode(THINK_CLOSE + "\n\n")[len(self.close_ids) :]
        for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if len(self.encode(label_prefix + label)) != 1:
                raise ValueError(f"label {label!r} with prefix {label_prefix!r} is not a single token for {model_path}")

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "HFBackend":
        env = env or os.environ
        return cls(env.get("LOUPE_HF_MODEL", DEFAULT_MODEL), device=env.get("LOUPE_HF_DEVICE") or None, adapter_path=env.get("LOUPE_HF_ADAPTER") or None)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prompt_ids(self, messages: Messages, thinking: bool) -> list[int]:
        out = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=thinking, tokenize=True)
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        return list(out) if not isinstance(out, str) else self.encode(out)

    def count_tokens(self, messages: Messages) -> int:
        return len(self._prompt_ids(messages, False))

    @torch.inference_mode()
    def _last_logits(self, ids: list[int], past=None):
        inputs = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=inputs, past_key_values=past, use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float(), out.past_key_values

    @torch.inference_mode()
    def _batched_last_logits(self, prompts: list[list[int]]) -> torch.Tensor:
        width = max(len(p) for p in prompts)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ids = torch.tensor([p + [pad] * (width - len(p)) for p in prompts], device=self.device)
        attention = torch.tensor([[1] * len(p) + [0] * (width - len(p)) for p in prompts], device=self.device)
        last = torch.tensor([len(p) - 1 for p in prompts], device=self.device)
        hidden = self.model.model(input_ids=ids, attention_mask=attention).last_hidden_state
        rows = torch.arange(len(prompts), device=self.device)
        return self.model.get_output_embeddings()(hidden[rows, last]).float()

    def slice_logits(self, batch: list[Messages], slice_ids: list[list[int]]) -> list[list[float]]:
        prompts = [self._prompt_ids(messages, False) for messages in batch]
        return self.slice_logits_from_ids(prompts, slice_ids)

    def slice_logits_from_ids(self, prompts: list[list[int]], slice_ids: list[list[int]]) -> list[list[float]]:
        if len(prompts) == 1:
            last, _ = self._last_logits(prompts[0])
            return [last[slice_ids[0]].tolist()]
        rows = []
        for start in range(0, len(prompts), self.max_batch):
            chunk = prompts[start : start + self.max_batch]
            logits = self._batched_last_logits(chunk)
            for i, ids in enumerate(slice_ids[start : start + self.max_batch]):
                rows.append(logits[i, ids].tolist())
        return rows

    def think_then_slice(self, messages: Messages, slice_ids: list[int], max_thinking_tokens: int) -> tuple[list[float], int]:
        last, past = self._last_logits(self._prompt_ids(messages, True))
        generated: list[int] = []
        closed = False
        while len(generated) < max_thinking_tokens:
            token = int(torch.argmax(last))
            if token in self.eos_ids:
                break
            generated.append(token)
            last, past = self._last_logits([token], past)
            if generated[-len(self.close_ids) :] == self.close_ids:
                closed = True
                break
        tail = self.after_close_ids if closed else self.close_ids + self.after_close_ids
        last, _ = self._last_logits(tail, past)
        return last[slice_ids].tolist(), len(generated)
