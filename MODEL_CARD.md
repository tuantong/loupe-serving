# Loupe 1.1

A LoRA adapter for [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) that makes typed decisions
about a document in one forward pass and returns probabilities over the options: a System-1 decision
engine, fast judgment rather than generated reasoning. Served with the code at
[github.com/tuantong/loupe-serving](https://github.com/tuantong/loupe-serving) (tag `loupe-1.1` or later),
which reads the answer from the next-token probabilities of the option labels; nothing is generated.

## Serving

Merge the adapter into the base weights at load and serve it with `LOUPE_PROMPT_STYLE=compact`, the
template it was trained with. Point `LOUPE_CALIBRATION` at this repository's `calibration.json`, which maps
the label probabilities to the reported distribution and names the model (`loupe-1.1`).

## Limitations

English only. It decides from what the document says and does not look anything up. Temporal
arithmetic over long documents, ranked-rule tradeoffs and long policies remain its weakest families.

## Licence

Apache-2.0, like the base model (Qwen/Qwen3.5-4B) and the serving code. The training data was written by
DeepSeek V4.1 Flash, with targets from Qwen3.8-27B (Apache-2.0) and DeepSeek V4.1 Flash.
