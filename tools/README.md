# tools

## `ref_dump.py`

Writes Hugging Face reference dumps (one `.npz` per prompt plus
`manifest.json`) for the correctness tests in [`../tests`](../tests). It runs
`transformers` on CPU in fp32 with eager attention, pinned to the revision
recorded in the manifest.

```
python3 -m venv .venv-ref
.venv-ref/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv-ref/bin/pip install transformers safetensors numpy huggingface_hub accelerate
.venv-ref/bin/python tools/ref_dump.py --model LiquidAI/LFM2-700M --out tests/golden/lfm2-700m
```

The `.npz` files are git-ignored; `manifest.json` keeps their sha256.
