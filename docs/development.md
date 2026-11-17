# Apex development (Phase 1)

## Toolchains

Apex uses C++20 and CMake 3.21+. The recommended test toolchain is Ninja + GCC or Clang on Linux. Use the `cpu-debug` or `cpu-release` CMake preset. Do not install CUDA on ordinary CI runners just to compile the mock engine.

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset cpu-debug --output-on-failure
./build/cpu-debug/apex --mock --prompt "test"
```

For CUDA configuration, install a compatible NVIDIA CUDA toolkit and select `cuda-release`. This option enables the CUDA **toolchain**, not model GPU execution. `nvcc` must be discoverable; override with `-DCMAKE_CUDA_COMPILER=/absolute/path/to/nvcc` when necessary.

## Dependency policy

Catch2 v3 is used for unit tests. CMake first looks for an installed Catch2 package, then fetches the pinned v3.5.2 tag. In GitHub Actions the FetchContent directory is cached between CI runs. If working offline, install Catch2 v3 and set `CMAKE_PREFIX_PATH`; no network fetch is then needed.

Do not add large GGUF models, generated caches, credentials or CUDA binaries to the repository. Keep model-dependent integration tests opt-in when Phase 2 arrives.

## Current Engine semantics

`EngineMode::Real` is the default and will not simulate generation. Existing files produce an explicit "not implemented" error until Phase 2; missing paths produce clear errors.

`EngineMode::Mock` is explicitly selected via `--mock` or the C++ constructor. Only identifiers beginning with `mock://` are accepted. No file is read, no tokenizer runs, and no tensors are evaluated. `isLoaded()` means the mock session is initialized in this mode. Calls to `generate()` before loading throw `std::logic_error`; invalid arguments throw `std::invalid_argument`.

When replacing a mock model, `loadModel()` clears the old mock state before validating the new identifier. `unloadModel()` is idempotent. The Engine is not thread-safe; concurrency belongs to later phases.

## CI troubleshooting

If CMake says `Failed to find nvcc`, use `--preset cpu-release` instead of `cuda-release`. If Catch2 retrieval fails, inspect network restrictions or install Catch2 v3 locally. If tests are missing, run `cmake --build --preset cpu-release` before `ctest`.

## Contribution checklist

- Include tests for new state transitions and explicit error handling.
- Preserve CPU-only configure/build/CTest support.
- Never report mock throughput as LLM performance.
- Keep changes scoped and describe how they were tested.
- Prefer a pull request and a green CI run before merging.
