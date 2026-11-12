# llmhost

From-scratch LLM inference host for Liquid AI LFM2 and LFM3 models (text and
vision). Two ways to use it: an embeddable library (Rust crate + C ABI) and a
standalone server with OpenAI- and Anthropic-compatible APIs.

This is not a wrapper around llama.cpp, ggml, or ONNX Runtime. Small utility
crates (safetensors, tokenizers, axum, and so on) are allowed.

The primary speed target is the CPU path on the reference machine: 8 Xeon
vCPUs with AVX-512/AMX. The goal is **40 tok/s** decode at batch size 1 with
Q4 weights. See [ARCHITECTURE.md](ARCHITECTURE.md) and
[ROADMAP.md](ROADMAP.md).

## Status

Milestone 1 (fp32 LFM2-700M, checked against Hugging Face reference dumps)
is in progress. Later milestones add quantized CPU kernels, the HTTP APIs,
a portable GPU backend, vision, and native CUDA/Metal paths.

## Crates

| Crate | Role |
| --- | --- |
| `llm-core` | Tensor and shape types, `Backend` / `Block` traits, `ModelConfig` |
| `llm-cpu` | Runtime CPU dispatch and SIMD kernels |
| `llm-quant` | Q8_0 and Q4 block formats |
| `llm-engine` | Sessions, sampling, KV / conv state |
| `llm-capi` | C ABI (`llmhost_version` for now) |
| `llm-server` | Axum server, binds `127.0.0.1` by default |
| `llm-gpu-wgpu` | Future wgpu / Vulkan / Metal / DX12 backend |

Correctness tests live in `tests/` and the bench harness in `bench/`; both
are maintained by the maintainers.

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Do not add `-C target-cpu=native`. Kernels are selected at runtime so the
same binary can run on the reference machine (AVX-512 / AMX) and on an
x86-64-v3 Zen 2 host (AVX2 only).

The server listens on `127.0.0.1:8080` unless you pass `--addr`.

Golden comparison tests look for dumps under `tests/golden/` or
`LLMHOST_GOLDEN_DIR`, and for weights in `LLMHOST_MODEL_DIR` or a local
Hugging Face Hub cache. They skip cleanly when those files are absent.
The dumps and weights are not committed.

## License

The **code** in this repository is licensed under either of

- Apache License, Version 2.0, or
- MIT license

at your option.

**LFM model weights** are not included. They are distributed by Liquid AI
under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-700M/blob/main/LICENSE).
Free commercial use is allowed only if your organization's annual revenue
is under $10 million. Reference dumps produced from those weights are
likewise not distributed here.
