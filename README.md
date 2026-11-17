# Apex

**Apex** is an experimental C++20 LLM inference runtime that targets low latency, efficiency and eventually custom CUDA kernels / multi-GPU serving.

## Current status — Phase 2 (experimental)

- The **default** CPU-only build runs the explicit mock and unit tests; it does **not** perform real model inference.
- The **`cpu-inference` preset** integrates a pinned upstream llama.cpp reference backend (b12445, commit `ef3848f9d7d4cf7b6b2abfc3b4c960b55d6d9e91`). It loads compatible GGUF models, tokenizes prompts, performs autoregressive decoding, and returns true model-produced tokens.
- This is a correctness-first integration **using llama.cpp internals**, **not** a custom or faster Apex inference engine. Independent CUDA kernels and performance improvements are later phases.
- Single decoder-only model, one request/context at a time, raw text completion; chat templates, HTTP API and multi-GPU scheduling are **not** available yet.

## Prerequisites

- CMake >=3.21, Ninja and a C++20 compiler.
- Windows: x64 MSVC Developer PowerShell (run `cl` and confirm `für x64`). Linux: GCC or Clang.
- Internet access the first time you build to fetch pinned Catch2 and, for real inference, the upstream llama.cpp source.
- A **local GGUF file** for real inference. Model weights are **not included** in this repository.

## Clone, build and test

```bash
git clone https://github.com/Lautloserspieler/Apex.git
cd Apex
cmake --preset cpu-release
cmake --build --preset cpu-release --parallel
ctest --preset cpu-release --output-on-failure
./build/cpu-release/apex --mock --prompt "Hello, Apex!"
```

On Windows, run `./build/cpu-release/apex.exe` in PowerShell instead of `./build/cpu-release/apex`.

## Enable **real** GGUF inference (CPU)

```bash
cmake --preset cpu-inference
cmake --build --preset cpu-inference --parallel 2
ctest --preset cpu-inference --output-on-failure
./build/cpu-inference/apex --model /path/to/model.gguf --prompt "Once upon a time" --max-tokens 64 --ctx-size 2048 --temperature 0
```

**Windows Developer PowerShell:**

```powershell
cmake --preset cpu-inference
cmake --build --preset cpu-inference --parallel 2
ctest --preset cpu-inference --output-on-failure
.\build\cpu-inference\apex.exe --model "C:\Models\model.gguf" --prompt "Once upon a time" --max-tokens 64 --ctx-size 2048 --temperature 0
```

`--temperature 0` selects deterministic greedy decoding. For seeded stochastic sampling, use `--temperature 0.7 --top-k 40 --top-p 0.95 --seed 42`. `--gpu-layers 0` is CPU-only.

**Optional GPU offload through upstream llama.cpp:** `cmake --preset cuda-release` requires a compatible CUDA toolkit and 64-bit compiler; `--gpu-layers N` then requests offloaded layers. This is not an Apex custom CUDA kernel implementation.

## Suggested first model

[Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) provides a small official `qwen2.5-0.5b-instruct-q4_k_m.gguf` example (Apache-2.0 model license). See [Model setup and validation](docs/models.md). Download the individual GGUF file, **not** the entire repository. Start with a small prompt and context.

## Current CLI options

`--model`, `--prompt`, `--ctx-size`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--seed`, `--gpu-layers`, `--mock`, `--help`.

The result is printed to stdout. For real inference Apex reports prompt/output token counts, model-load time, time-to-first-token, prefill tokens/s and decode tokens/s to stderr. Each generation recreates the context to prevent KV state leaks between requests.

## Testing and limits

CPU-only CI, Windows-x64 CI, and a Linux backend compilation job run on pull requests. Ordinary CI does **not** download multi-hundred-MB model weights. Optionally set `APEX_TEST_MODEL` to the path of a suitable GGUF file before running CTest to enable a real-model integration test.

The GGUF loader and model implementation belong to pinned llama.cpp; Apex does not yet independently validate every model architecture/quantization. Unsupported models, malformed files or insufficient memory can cause load failure. Timing results are **unwarmed single-run diagnostics**, not apples-to-apples performance benchmarks. See [docs/support-matrix.md](docs/support-matrix.md) for exactly what has been built, tested and verified so far, versus what remains unverified.

Phase 2 reproducibility helpers: `benchmarks/phase2_baseline.py` runs repeated
single-process invocations and collects raw metrics; `benchmarks/compare_reference.py`
compares exact greedy token IDs with a **separate direct llama.cpp API caller**,
when real builds/tests are enabled. See [Benchmark methodology](docs/benchmark-methodology.md).

The roadmap is tracked in [Issues](https://github.com/Lautloserspieler/Apex/issues); [Phase 2](https://github.com/Lautloserspieler/Apex/issues/2) covers true generation and baseline validation, and [Phase 3](https://github.com/Lautloserspieler/Apex/issues/3) targets new CUDA optimizations.

## License

Original Apex code is MIT. llama.cpp is included as a separate third-party MIT-licensed dependency when enabled. Model weights have their own license; review upstream licenses before distributing binaries or models.
