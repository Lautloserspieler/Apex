# Phase 2 baseline measurement procedure

Apex currently uses upstream llama.cpp to compute all real model forward passes, so a fair performance comparison must be framed as **integration overhead**, not proof of an independently optimized engine.

1. Record model revision + GGUF SHA256, OS, compiler, CPU, RAM, GPU, CUDA/toolkit, clocks and upstream revision.
2. Hold quantization, prompt bytes, tokenization and maximum output length constant between runs.
3. Use `--temperature 0` for deterministic greedy decoding and log generated token IDs with the opt-in integration harness when necessary.
4. Separate **load time**, **prefill**, **time-to-first-token** and **decode** latency. Apex prints these diagnostics after one run.
5. Run repeated warm and cold samples, compute median and variability; do not compare a cold Apex run to a warmed llama.cpp process.
6. Capture peak RSS and VRAM using system tools; do not fabricate resource metrics.
7. Use a pinned llama.cpp CLI binary for the external baseline, with matching model/context/offload/sampling and explicit command line.
8. Preserve raw logs and a description of the measurement environment in `benchmarks/`.

No published speedups until reproducible numbers and correct token output exist.
