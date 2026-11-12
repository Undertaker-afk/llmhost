# bench

SWE owns this directory.

The `llmhost-bench` harness (prefill vs decode tok/s, TTFT, p50/p95, load
and pinned-core recording, llama.cpp comparison in the same run) lives
here. Official 40 tok/s figures need a quiet window on the development box.

Report format for later GPU and multi-stream runs is in
[`../ROADMAP.md`](../ROADMAP.md): single-stream vs total, 1/4/8 lanes,
code vs prose, speculative decoding on/off with accept rate.
