# Roadmap

## Phase: CPU first

The hard requirement is **40 tok/s** decode at batch size 1 with Q4 weights,
measured on the reference machine: 8 Xeon vCPUs with AVX-512/AMX.

That bar applies to:

- LFM2-700M
- LFM3-1.7B
- LFM3-VL-2B

LFM2-2.6B is **best effort** on the reference machine: the bandwidth ceiling
for Q4 at ~1.2+ GB/token is tight, and 40 tok/s may not hold.

Work order follows the milestones in [ARCHITECTURE.md](ARCHITECTURE.md):
fp32 correctness (M1), then Q4/Q8 SIMD and the bench harness (M2), then the
HTTP APIs (M3). Vision (M5) is required for LFM3-VL-2B to count. Official
TPS numbers are taken only in a quiet window, with load recorded, and are
compared to llama.cpp only in the same run.

A second CPU target is an x86-64-v3 Zen 2 host (AVX2 only). It is not the
40 tok/s gate. Binaries for that class of machine are built for x86-64-v3,
not `-C target-cpu=native`.

## Competitive targets

The CPU bar is to beat llama.cpp on the same machine, with the same model,
quantization, prompts, and thread count, measured back to back on a quiet
machine. The current reference point for LFM2-700M Q4_0 at 8 threads is
about **140 tok/s decode** and about **1290 tok/s prefill** on a 512-token
prompt.

Batch-1 decode is memory-bandwidth bound, so wins come from fewer bytes
per token (a lean 4-bit format and a quantized head), speculative decoding
(prompt lookup, an LFM2.5-230M draft model), AMX int8 prefill, and fused
LFM conv / SwiGLU kernels.

The multi-stream bar is to beat vLLM on total tok/s across 1, 4, and 8
streams, first on the CPU backend and later on GPU.

## Long-term: GPU

The later speed target is a single **RTX 5090-class** GPU running a
**27B** model at **>= 700 tok/s total** across **8 parallel streams**,
with a stretch goal of **1600 tok/s** total.

The path there is:

- continuous / micro-batching across streams
- speculative decoding with tree verification
- NVFP4 weights

This work sits after the CPU phase (M4 wgpu first for a portable GPU path,
then M6 CUDA / Metal). It is not benchmarked on the CPU-only reference
machine.

## Bench reports

When GPU (and the later CPU) numbers are reported, each result should
include:

- single-stream tok/s vs total tok/s across streams
- lane counts of 1, 4, and 8
- code vs prose prompts
- speculative decoding on vs off, with the accept rate when it is on
- prefill reported separately from decode
