# Loupe

Loupe is a System-1 decision engine: it answers typed decision questions about a document in one forward
pass, the way a person makes a quick, practised judgment rather than working a problem out step by step.
It reads a document and one or more questions, each with an explicit option list, and returns a
calibrated probability for every option. It does not generate text: the answer is read from the
next-token probabilities of the option letters, so a decision costs one prefill and no decoding.

The model is a LoRA adapter for Qwen/Qwen3.5-4B, published at
[huggingface.co/tuantc/loupe-1.1](https://huggingface.co/tuantc/loupe-1.1) (`MODEL_CARD.md`) and
[huggingface.co/tuantc/loupe-1.0](https://huggingface.co/tuantc/loupe-1.0); this repository is its server.

## Install

```bash
pip install -e .                                   # torch 2.10, transformers 5.17, peft 0.21, flash-linear-attention 0.5.2
pip install --no-build-isolation causal-conv1d     # fused convolution kernels for the CUDA-graph backend (needs nvcc)
pip install -e ".[vllm]"                           # optional: the vLLM backend
```

The versions are pinned to the ones Loupe 1.0 was verified with. torch 2.10 is required on Blackwell GPUs
(RTX 50-series, RTX PRO 6000): earlier Triton builds cannot compile the linear-attention kernels for sm_120.

## Serve

```bash
hf download tuantc/loupe-1.1 --local-dir loupe-1.1
LOUPE_HF_MODEL=Qwen/Qwen3.5-4B LOUPE_HF_ADAPTER=loupe-1.1 \
  LOUPE_PROMPT_STYLE=compact LOUPE_CALIBRATION=loupe-1.1/calibration.json LOUPE_GRAPH_MAX_TOKENS=1024 \
  python -m loupe.serve --backend graph --port 8000
```

The adapter is merged into the base weights at load. `--backend graph` replays the whole forward pass as
CUDA graphs, captured at start-up for every 32-token length up to `LOUPE_GRAPH_MAX_TOKENS` (about 14 ms per
decision on an RTX PRO 6000; longer prompts take the plain path); `--backend hf` is the plain PyTorch path. The
server listens on 127.0.0.1 by default and refuses any other host unless `LOUPE_API_TOKEN` is set. In a
container:

```bash
docker build -t loupe . && docker run --gpus all -p 8000:8000 -v "$PWD/loupe-1.1:/model" \
  -e LOUPE_API_TOKEN=<token> -e LOUPE_HF_ADAPTER=/model -e LOUPE_CALIBRATION=/model/calibration.json \
  -e LOUPE_PROMPT_STYLE=compact loupe
```

Useful variables:

| Variable | Meaning |
|---|---|
| `LOUPE_HF_MODEL`, `LOUPE_HF_ADAPTER` | base model and adapter for the PyTorch backends |
| `LOUPE_VLLM_MODEL`, `LOUPE_VLLM_ADAPTER` | the same for `--backend vllm` |
| `LOUPE_CALIBRATION` | `calibration.json` from the model repository; it may also name the model |
| `LOUPE_PROMPT_STYLE` | `compact` for Loupe 1.0 and 1.1, which were trained without a system message |
| `LOUPE_MODEL_NAME` | the model name the server reports and accepts; defaults to the calibration's, else `loupe-1.0` |
| `LOUPE_API_TOKEN` | required to listen on a non-loopback host; requests then send `Authorization: Bearer <token>` |
| `LOUPE_GRAPH_MAX_TOKENS` | longest prompt served from a captured CUDA graph (default 2048) |
| `LOUPE_CONTEXT_BUDGET` | longest prompt accepted, in tokens |

## Ask

```bash
curl -s http://127.0.0.1:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "loupe-1.1",
  "state": "Invoice INV-7 for EUR 12,400 was due on 2026-10-09 and paid on 2026-10-14.",
  "questions": {"late": {"type": "noul", "instructions": "Was the invoice paid after its due date?",
                         "criteria": {"true": "paid late", "false": "paid on time"}}}
}'
```

A `noul` question returns the probability of yes; a `choice` question returns a distribution over its
option keys and the most likely key; a `score` question returns a distribution over ordered levels.

## Test

```bash
pip install -e ".[test]" && pytest -m "not slow"
```

## Licence

The code is licensed under Apache-2.0 (`LICENSE`). The base model, Qwen/Qwen3.5-4B, is Apache-2.0. The
adapter weights and their terms are described in `MODEL_CARD.md`.
