# Architecture

llmhost is a from-scratch inference engine for Liquid AI LFM2 and LFM3
(including vision variants). The design keeps the two families on one graph
and treats the CPU as the primary backend.

## Config-driven layer graph

`ModelConfig` is parsed from a Hugging Face `config.json`. The file is turned
into a `Vec<LayerKind>` (`Conv`, `Attention`, or `Other(String)` for types
that are not wired yet, such as an LFM3 MoE block).

The graph is a sequence of `Block` implementations chosen from that list, so
LFM2 and LFM3 share the engine. A new family only needs new block types and
a mapping from its config; it does not need a second runtime.

## CPU first

The 40 tok/s target is measured on the development box (8 Xeon vCPUs with
AVX-512 VNNI / AMX, 15 GB RAM, no GPU). Decode is memory-bandwidth bound, so
the kernels exist to cut bytes per token:

- **Decode:** AVX-512 VNNI GEMV of Q4 weights (32-element blocks, f16
  scales) with activations quantized to int8.
- **Prefill:** AMX int8 tiles.
- **Fallback:** AVX2 kernels, chosen at runtime together with the paths
  above. A scalar f32 reference stays around for correctness tests.
- **Builds** stay on an x86-64-v3 baseline. There is no
  `-C target-cpu=native` flag, so the same artifact can run on this box and
  on a Zen 2 host (Hetzner AX41) that has AVX2 only. Kernel selection is
  always at runtime.

A wgpu GPU backend remains in scope for later milestones, but it cannot be
benchmarked here (lavapipe can only check correctness).

## State

- **Attention layers** keep a paged KV cache (blocks of 16 tokens, f16 or
  optional Q8) with prefix reuse across multi-turn turns.
- **Conv layers** keep a small rolling state of kernel-size minus one per
  layer, not a KV cache.

## Crates

- `llm-core` — shapes, tensors, `Backend`, `Block`, `ModelConfig`, placeholder `Model`.
- `llm-cpu` — feature detection, `KernelSet`, CPU kernels.
- `llm-quant` — Q8_0 and Q4 block formats, plus the future `.lfmb` container.
- `llm-engine` — sessions, cache, batching, sampling, tokenizer / chat template.
- `llm-capi` — C ABI (cbindgen header in a later milestone).
- `llm-server` — axum, `127.0.0.1` by default, OpenAI and Anthropic routes.
- `llm-gpu-wgpu` — default GPU path (Vulkan / Metal / DX12 via WGSL).
- `llm-cuda` / `llm-metal` — native fast paths, added later behind features.

`tests/` and `bench/` are owned by SWE. Official 40 tok/s numbers need a quiet
machine; the harness records load and pinned cores.

## Milestones

Each milestone has a test gate.

| | Goal | Gate |
| --- | --- | --- |
| **M1** | Run 700M on the CPU in fp32 | vs Hugging Face logits, 1e-3 overall, 1e-4 per layer |
| **M2** | Q8 and Q4 on the CPU with SIMD | SWE bench harness, decode tok/s |
| **M3** | Engine plus server | OpenAI and Anthropic routes with streaming, official SDK tests |
| **M4** | wgpu backend | logits match the CPU path |
| **M5** | LFM2-VL / LFM3-VL | vision encoder, projector, image inputs in both APIs |
| **M6** | 2B-class models, CUDA and Metal fast paths, C ABI and docs | correctness + benches on real GPUs |

The CPU throughput targets and the later GPU target (RTX 5090-class, 27B)
are in [ROADMAP.md](ROADMAP.md).
