# tools

- `ref_dump.py`: writes Hugging Face reference dumps (`.npz` per prompt plus
  `manifest.json`) for the correctness tests in `../tests`. It runs
  `transformers` on CPU in fp32 with eager attention, pinned to the HF revision
  recorded in the manifest.

  ```
  python3 -m venv /workspace/.venv-ref
  /workspace/.venv-ref/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
  /workspace/.venv-ref/bin/pip install transformers safetensors numpy huggingface_hub accelerate
  /workspace/.venv-ref/bin/python tools/ref_dump.py --model LiquidAI/LFM2-700M --out tests/golden/lfm2-700m
  ```

  The `.npz` files are git-ignored; `manifest.json` keeps their sha256.
