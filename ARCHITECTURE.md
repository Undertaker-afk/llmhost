# Architecture

llmhost is a from-scratch inference engine for Liquid AI LFM2 and LFM3
(including vision variants). The design keeps the two families on one graph
and treats the CPU as the primary backend.

## Config-driven layer graph

`ModelConfig` is parsed from a Hugging Face `config.json` into a
`Vec<LayerKind>`. Vision configs (`lfm2_vl`, `lfm3_vl`) are read from the
nested `text_config` object. A new family only needs new block types and a
mapping from its config; it does not need a second runtime.

`LayerKind` is:

- `Conv { kernel }` — `kernel` is `conv_L_cache` (3 on LFM2, 7 on LFM3). The
  rolling conv state is `kernel - 1`.
- `Attention { sliding_window, rope: RopeParams { theta, partial_rotary_factor } }`
  — `sliding_window` is `Some` for LFM3 `sliding_attention` layers and `None`
  for full attention.
- `Moe` — reserved for a mixture-of-experts block.
- `Other(String)` — unknown type names, so a future layer can be described
  before its kernels exist.

Layer lists come from the config as follows:

- **LFM2:** `layer_types` when present (e.g. 2.6B). Smaller LFM2 checkpoints
  (700M / 1.2B / 350M) omit that list and use `full_attn_idxs` plus
  `num_hidden_layers` (those indices are full attention; the rest are conv).
  RoPE is the top-level `rope_theta` (or `theta`, or a flat
  `rope_parameters` object).
- **LFM3:** `layer_types` of `conv`, `full_attention`, and
  `sliding_attention`, with a global `sliding_window` (256) and per-type
  `rope_parameters` (`rope_theta` and optional `partial_rotary_factor` for
  `full_attention` and `sliding_attention`).

## CPU first

The 40 tok/s target is measured on the reference machine: 8 Xeon vCPUs with
AVX-512/AMX. Decode is memory-bandwidth bound, so the kernels exist to cut
bytes per token:

- **Decode:** AVX-512 VNNI GEMV of Q4 weights (32-element blocks, f16
  scales) with activations quantized to int8.
- **Prefill:** AMX int8 tiles.
- **Fallback:** AVX2 kernels, chosen at runtime together with the paths
  above. A scalar f32 reference stays around for correctness tests.
- **Builds** stay on an x86-64-v3 baseline. There is no
  `-C target-cpu=native` flag, so the same artifact can run on the
  reference machine and on an x86-64-v3 Zen 2 host (AVX2 only). Kernel
  selection is always at runtime.

Q4 weights are packed at load into 16-row VNNI tiles (unsigned nibbles plus
f32 scales). When AMX-INT8 is available the matching B panels are also
pre-packed in the 8×64 signed layout the tiles expect, and pages are
touched so the first run is not dominated by faults. Decode GEMV uses a
process-wide pinned spin pool: each thread keeps a fixed row range of every
matrix. Prefill GEMM reuses each weight tile across the token axis and, on
AMX, a 2×2 tile kernel (32 tokens × 32 outputs per step).

A wgpu GPU backend remains in scope for later milestones. On machines
without a GPU, software Vulkan (lavapipe) can be used to check
correctness, not speed.

## Formats

`WeightFormat` is `F32`, `F16`, `BF16`, `Q8_0`, `Q4_32`, or `Nvfp4`.
`Nvfp4` is reserved for the later GPU path and returns `Unsupported` if
selected. `KvDtype` is `F16`, `Q8`, or `K8V4` (8-bit keys, 4-bit values).

## Batch-first engine

The engine runs a `ForwardBatch` of `SeqInput`s (sequence id, tokens,
start position, block table, optional tree) and returns logits for **every**
token position, not just the last.

A `tree` is a parent-index mask, one entry per token, with `-1` as a root.
`None` means a linear chain. Parents must precede children. After a
speculative forward, `accept` / `rollback` take a list of token indices.
The accepted path need not be a prefix of the list (a branch can be chosen)
but must be a connected path in the tree. That path is applied to:

- the paged KV cache (16-token blocks, attention layers only)
- conv-state snapshots (rolling state for conv layers)

`Scheduler` is a placeholder for continuous batching. `DraftModel::propose`
returns draft tokens and a tree for later tree-verification decode.

## State

- **Attention layers** keep a paged KV cache (`PagedKv`, blocks of 16
  tokens, dtype `KvDtype`) with prefix reuse across multi-turn turns.
- **Conv layers** keep a small rolling state of kernel-size minus one per
  layer (`ConvState`), snapshotted so a rejected draft branch can be rolled
  back.

## Crates

- `llm-core` — shapes, tensors, `Backend`, `Block`, `ModelConfig`, `LayerKind`, placeholder `Model`.
- `llm-cpu` — feature detection, `KernelSet`, CPU kernels.
- `llm-quant` — Q8_0 and Q4 block formats, `WeightFormat`, `KvDtype`, plus the future `.lfmb` container.
- `llm-engine` — sessions, paged KV / conv state, batch-first forward, sampling, tokenizer / chat template.
- `llm-capi` — C ABI (cbindgen header in a later milestone).
- `llm-server` — axum, `127.0.0.1` by default, OpenAI and Anthropic routes.
- `llm-gpu-wgpu` — default GPU path (Vulkan / Metal / DX12 via WGSL).
- `llm-cuda` / `llm-metal` — native fast paths, added later behind features.

Correctness tests live in `tests/` and the bench harness in `bench/`; both
are maintained by the maintainers. Official 40 tok/s numbers need a quiet
machine; the harness records load and pinned cores.

## Milestones

Each milestone has a test gate.

| | Goal | Gate |
| --- | --- | --- |
| **M1** | Run 700M on the CPU in fp32 | vs Hugging Face logits, 1e-3 overall, 1e-4 per layer |
| **M2** | Q8 and Q4 on the CPU with SIMD | the bench harness, decode tok/s |
| **M3** | Engine plus server | OpenAI and Anthropic routes with streaming, official SDK tests |
| **M4** | wgpu backend | logits match the CPU path |
| **M5** | LFM2-VL / LFM3-VL | vision encoder, projector, image inputs in both APIs |
| **M6** | 2B-class models, CUDA and Metal fast paths, C ABI and docs | correctness + benches on real GPUs |

The CPU throughput targets and the later GPU target (RTX 5090-class, 27B,
NVFP4, speculative tree verify, 8-stream totals) are in
[ROADMAP.md](ROADMAP.md).
