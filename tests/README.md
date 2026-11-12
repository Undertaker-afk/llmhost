# tests

Correctness tests for llmhost:

- Layer-by-layer and logit comparisons against Hugging Face reference dumps
  (see `golden/` and [`../tools/ref_dump.py`](../tools/ref_dump.py)).
- Tokenizer and chat-template checks.
- KV-cache, prefill/decode split, and speculative-decoding equivalence checks.
- API conformance with the official OpenAI and Anthropic SDKs.

Tolerances per milestone are listed in [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Golden dumps

Reference `.npz` files are large and are not committed. Each model directory
under `golden/` has a `manifest.json` with the Hugging Face revision, library
versions, prompts, and the sha256 of every file. Regenerate them with:

```
python tools/ref_dump.py --model LiquidAI/LFM2-700M --out tests/golden/lfm2-700m
```

The reference dumps are produced with Liquid AI's LFM models, which are
licensed under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-700M/blob/main/LICENSE).
The model weights and the dumps are not distributed with this repository.
