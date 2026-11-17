# Apex development guide — Phase 2

## Toolchain

Apex requires CMake 3.21+, Ninja, and a C++20 compiler. For Windows, launch an **x64 Visual Studio Developer PowerShell**; verify `cl` identifies x64. Ordinary PowerShell does not automatically load MSVC compiler environment.

## Fast, hardware-independent tests

```bash
cmake --preset cpu-release
cmake --build --preset cpu-release --parallel
ctest --preset cpu-release --output-on-failure
```

This builds only the mock engine and independent validation tests; **no real inference** happens.

## Compiling real GGUF inference

```bash
cmake --preset cpu-inference
cmake --build --preset cpu-inference --parallel 2
ctest --preset cpu-inference --output-on-failure
```

This downloads the **SHA-256-verified source archive** of the pinned llama.cpp commit **without cloning Git history**, compiles its CPU backend as a static dependency, and enables real model execution. Build time and disk usage are still larger than for Phase 1. To build offline, supply a populated `FETCHCONTENT_SOURCE_DIR_LLAMA_CPP` pointing to the **exact pinned source revision** and provide Catch2 v3 using `CMAKE_PREFIX_PATH`.

Optional CUDA: `cmake --preset cuda-release` requires `nvcc`, a compatible toolkit/driver and x64 C++ compiler. This uses the upstream llama.cpp CUDA backend, **not** a custom Apex implementation.

## Design & safe changes

- Engine public headers are backend-neutral; model/vocabulary objects live only in the backend.
- Mock mode accepts only a `mock://` identifier and is explicitly labeled.
- Real mode validates local file existence, attempts a GGUF model load, and throws a typed exception on failure. It never falls back to mock output.
- Generate is a raw text completion operation, not a chat template formatter.
- Per-request context construction prevents KV-cache state leaking between prompts.
- Decoding stops at EOS or max-tokens; context exhaustion fails before any generation.
- `APEX_TEST_MODEL` is an optional local integration test path and must not be used by ordinary CI without explicit model provisioning.
- Do not hardcode GPU performance claims or publish warm/cold benchmark comparisons without consistent methodology.

## CI and dependencies

Linux mock CI remains required as `build-and-test`; a separate Windows-x64 job validates MSVC; a Linux real-backend job compiles pinned llama.cpp without downloading model weights. Catch2 is pinned to v3.5.2 and llama.cpp pinned to b12445 / commit ef3848f9d7d4cf7b6b2abfc3b4c960b55d6d9e91.

Upstream dependencies have separate licenses. Record changes to model/loader/format support in `docs/models.md` and benchmarking changes in `docs/benchmark-methodology.md`.

## Fedora: switch an interrupted full Git clone to the archive download

If an older `cmake --preset cpu-inference` is still cloning `ggml-org/llama.cpp.git`, stop that **old** configure with Ctrl+C, then update Apex and remove only the incomplete generated CMake build directory:

```bash
cd ~/Entwicklung/Apex
git pull origin main
rm -rf build/cpu-inference
cmake --preset cpu-inference
cmake --build --preset cpu-inference --parallel 8
ctest --preset cpu-inference --output-on-failure
```

This removes **only generated files** under `build/cpu-inference`; your source tree and `models/` folder remain intact. For a completed, working build, cleanup is optional. The first archive download still requires Internet access and can be delayed by network issues.
