# bench

`llmhost-bench` measures prefill and decode tok/s and time to first token
(p50/p95 over repeated runs), and records system load and pinned cores with
each result. A llama.cpp baseline can be run in the same session for
comparison. Numbers taken on a loaded machine are not reported as results.

Multi-stream and GPU runs use the report format in
[`../ROADMAP.md`](../ROADMAP.md): single-stream and total throughput, 1/4/8
lanes, code and prose prompts, and speculative decoding on and off with its
accept rate.
