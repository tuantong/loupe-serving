"""Fold a LoRA adapter into the base weights: mlx-lm format (lora_a, lora_b and a scale) as deltas added in place."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def lora_deltas(adapter_dir: Path) -> dict[str, torch.Tensor]:
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = float(config["lora_parameters"]["scale"])
    weights = load_file(str(adapter_dir / "adapters.safetensors"))
    deltas = {}
    for key, a in weights.items():
        if not key.endswith(".lora_a"):
            continue
        b = weights[key[: -len(".lora_a")] + ".lora_b"]
        module = key[: -len(".lora_a")].split("layers.", 1)[1]
        deltas["layers." + module] = scale * (a.float() @ b.float()).T
    return deltas


def merge(model, deltas: dict[str, torch.Tensor]) -> int:
    applied = 0
    for name, module in model.named_modules():
        for suffix, delta in deltas.items():
            if name.endswith(suffix) and hasattr(module, "weight"):
                if tuple(module.weight.shape) != tuple(delta.shape):
                    raise ValueError(f"{name}: weight {tuple(module.weight.shape)} vs delta {tuple(delta.shape)}")
                module.weight.data += delta.to(module.weight.dtype)
                applied += 1
    return applied
