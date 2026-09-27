# Loupe 1.0

A LoRA adapter for [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) that makes typed decisions
about a document in one forward pass and returns calibrated probabilities over the options: a System-1
decision engine, fast judgment rather than generated reasoning. Served with the code at
[github.com/tuantong/loupe-serving](https://github.com/tuantong/loupe-serving), which reads the answer from
the next-token probabilities of the option labels; nothing is generated.

## Adapter

An exact average (model soup) of LoRA adapters on the attention, linear-attention and MLP projections of
all 32 layers (three seeds of one recipe, a continuation of that recipe on written rationales, and a run whose
targets include a second teacher), stored as one rank-96 adapter. Merge it into the base weights for
serving. It was trained with the compact prompt template, so serve it with `LOUPE_PROMPT_STYLE=compact`.
`calibration.json` reads answers to prompts of 600 tokens or more at temperature 0.7 and shorter ones at
1; the temperature was fitted on our own held-out cases, not on benchmark items.

## Training (summary; training code and data are not released)

- Supervised targets on generated decision families (rules, judgments, routing, probability items with
  exact distributions) and on authored case files written for this project by DeepSeek V4.1 Flash.
- On authored cases, the targets are a larger model's distribution (Qwen3.8-27B) after reasoning, used
  only where that reasoning reached the verified answer; elsewhere the base model's own distribution.
- In one ingredient, where DeepSeek V4.1 Flash's stated distribution (after its own reasoning) put the
  verified answer first, the target is half the existing target and half that distribution.
- No output of a Claude or OpenAI model was used as a training case or target.
- No evaluation item or paraphrase was used. Every authored case was checked for shared passages with
  the benchmark's public items and dropped when it had one.

## Evaluation (our runs, not official scores)

Public tiers of a System-1 decision benchmark (typed decisions over documents, many with a tempting
wrong reading):

| Tier | Items | Accuracy |
|---|---|---|
| Easy | 48 | 100.0% |
| Standard | 72 | 100.0% |
| Hard | 111 | 78.4% |

Hard-tier ECE 0.060; mean total variation distance to the exact distributions of the probability items
0.162. Serial loopback latency on an RTX PRO 6000 Blackwell with the CUDA-graph backend and the adapter
merged: p50 13.9 ms, p95 15.8 ms.

## Limitations

English only. It decides from what the document says and does not look anything up. Temporal
arithmetic over long documents, ranked-rule tradeoffs and long policies remain its weakest families.

## Licence

Apache-2.0, like the base model (Qwen/Qwen3.5-4B) and the serving code. The training data was written by
DeepSeek V4.1 Flash, with targets from Qwen3.8-27B (Apache-2.0) and DeepSeek V4.1 Flash.
