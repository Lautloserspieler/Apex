# Apex

**Apex** is an experimental C++20/CUDA LLM inference engine focused on low latency, high throughput and efficient serving on consumer and multi-GPU hardware.

> **Status: Phase 1 foundation. Apex does not load or execute a real LLM yet.**
> The only supported generation today is a deliberately labeled mock, enabled with `--mock`. GGUF inference starts in [Phase 2](https://github.com/Lautloserspieler/Apex/issues/2). Do not use the placeholder for performance or model-quality comparisons.

## Goals

- Low time-to-first-token (TTFT) and efficient prefill/decode throughput
- Efficient memory usage, targeted CUDA kernels and reproducible benchmarks
- A clean C++20 core with an **optional** CUDA toolchain
- OpenAI-compatible serving when real inference is available

## Requirements

- CMake **3.21+**, a C++20 compiler, and Ninja
- Internet access on the first test configuration if Catch2 v3 is not already installed
- **No NVIDIA GPU or CUDA toolkit needed** for Phase 1 CPU builds or tests
- Optional CUDA configure: installed CUDA toolkit with `nvcc` (no CUDA kernels are included yet)

## Quick start: build and test (Linux)

```bash
git clone https://github.com/Lautloserspieler/Apex.git
cd Apex
cmake --preset cpu-release
cmake --build --preset cpu-release --parallel
ctest --preset cpu-release --output-on-failure
```

Run the **mock** smoke test:

```bash
./build/cpu-release/apex --mock --prompt "Hello, Apex!" --max-tokens 16
```

Expected output starts with `[MOCK OUTPUT - NOT LLM INFERENCE]`. It contains the prompt, not generated language-model tokens.

To test development builds:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
```

Optional CUDA-toolchain-only configure (this **does not** provide GPU inference):

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
```

When CUDA is enabled, CMake checks for `nvcc`; if unavailable, configuration fails with an actionable error. For explicit toolkit locations, pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.

## CLI behavior

- `apex --help` prints supported options.
- `apex --mock [--model mock://identifier] --prompt "Hello"` emits **only synthetic mock text**.
- `apex --model /path/to/model.gguf` is reserved for Phase 2 and rejects real inference for now.
- Invalid arguments, unsupported backends, and calls before loading produce errors rather than fake model output.

## Roadmap

1. [Phase 1 — minimal core, reliable CPU build and CI](https://github.com/Lautloserspieler/Apex/issues/1)
2. [Phase 2 — real GGUF models and text generation](https://github.com/Lautloserspieler/Apex/issues/2)
3. [Phase 3 — CUDA/KV-cache performance](https://github.com/Lautloserspieler/Apex/issues/3)
4. [Phase 4 — serving API](https://github.com/Lautloserspieler/Apex/issues/4)
5. [Phase 5 — multi-GPU and advanced techniques](https://github.com/Lautloserspieler/Apex/issues/5)
6. [Phase 6 — documentation and releases](https://github.com/Lautloserspieler/Apex/issues/6)

See [development guide](docs/development.md) and [architecture](docs/architecture.md).

## Testing

Catch2-based unit tests and CLI smoke tests run through CTest. GitHub Actions runs CPU-only builds and tests on every push to `main` and every pull request targeting `main`. CUDA is not required for the standard CI job. To use an already installed Catch2 v3, expose its CMake package location via `CMAKE_PREFIX_PATH`.

## License

MIT for original Apex code. Third-party dependencies and model weights retain their own licenses; review each before distributing.
