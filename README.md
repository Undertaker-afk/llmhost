# llmhost

From-scratch LLM inference host for Liquid AI LFM2 and LFM3 models (text and
vision). Two ways to use it: an embeddable library (Rust crate + C ABI) and a
standalone server with OpenAI- and Anthropic-compatible APIs.

This is not a wrapper around llama.cpp, ggml, or ONNX Runtime. Small utility
crates (safetensors, tokenizers, axum, and so on) are allowed.

The primary speed target is the CPU path on the shared development box
(8 Xeon vCPUs, AVX-512 VNNI, AMX, 15 GB RAM, no GPU): **40 tok/s** decode at
batch size 1 with Q4 weights. See [ARCHITECTURE.md](ARCHITECTURE.md) and
[ROADMAP.md](ROADMAP.md).

## Crates

| Crate | Role |
| --- | --- |
| `llm-core` | Tensor and shape types, `Backend` / `Block` traits, `ModelConfig` |
| `llm-cpu` | Runtime CPU dispatch and SIMD kernels |
| `llm-quant` | Q8_0 and Q4 block formats |
| `llm-engine` | Sessions, sampling, KV / conv state (skeleton: sampler config) |
| `llm-capi` | C ABI (`llmhost_version` for now) |
| `llm-server` | Axum server, binds `127.0.0.1` by default |
| `llm-gpu-wgpu` | Future wgpu / Vulkan / Metal / DX12 backend |

`tests/` and `bench/` are owned by SWE.

## Build

```bash
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Do not add `-C target-cpu=native`. Kernels are selected at runtime so the
same binary can run on this box (AVX-512 / AMX) and on an x86-64-v3 host
such as the Hetzner AX41 (AVX2 only).

The server listens on `127.0.0.1:8080` unless you pass `--addr`.

## Source control

Local git only. Do not push to GitHub or any other remote.
