import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from loupe.adapters import lora_deltas, merge


def test_lora_delta_matches_mlx_forward_and_merges_by_suffix(tmp_path):
    a, b = torch.randn(8, 2), torch.randn(2, 6)
    save_file({"language_model.model.layers.3.self_attn.q_proj.lora_a": a, "language_model.model.layers.3.self_attn.q_proj.lora_b": b}, str(tmp_path / "adapters.safetensors"))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"scale": 20.0}}))
    deltas = lora_deltas(tmp_path)
    x = torch.randn(1, 8)
    assert torch.allclose(x @ deltas["layers.3.self_attn.q_proj"].T, 20.0 * (x @ a) @ b, atol=1e-4)

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 6, bias=False)

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
    model.model.layers[3].self_attn = Attn()
    before = model.model.layers[3].self_attn.q_proj.weight.clone()
    assert merge(model, deltas) == 1
    assert torch.allclose(model.model.layers[3].self_attn.q_proj.weight, before + deltas["layers.3.self_attn.q_proj"])
