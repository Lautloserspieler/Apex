# Apex architecture and Phase 1 boundaries

Apex is being built incrementally. In Phase 1, the public C++ `apex::Engine` API, minimal CLI and tests exist; **real model execution does not**.

```text
CLI (src/main.cpp)
  |
  v
Engine (include/apex/engine.h, src/engine/engine.cpp)
  +-- EngineMode::Real: unsupported until Phase 2 (never returns fake inference)
  +-- EngineMode::Mock: deterministic, explicitly labeled smoke-test string

CTest / Catch2: tests/test_engine.cpp + tests/test_main.cpp
CMake: C++20 default; CUDA toolchain only when APEX_ENABLE_CUDA=ON
```

## Engineering boundaries for later phases

- **Phase 2:** add GGUF model loading, tokenizer, supported transformer forward-pass, token sampling and reproducible reference-output tests. Keep these behind the Engine contract.
- **Phase 3:** introduce a real CUDA backend, tensor/device ownership, KV-cache and isolated optimized kernels; compare numeric correctness and performance.
- **Phase 4:** add separate request parsing, admission/queue scheduling and OpenAI-compatible HTTP/SSE transport; never mix HTTP state with model state.
- **Phase 5:** add long-context memory optimization and multi-GPU only after single-GPU correctness and reproducible profiling.

## Ownership and error rules

The Engine owns its state and cannot be copied. A failed load leaves the engine unloaded; unloaded generation is an error. Invalid prompt/token-limit inputs must not be silently accepted. The mock identifier is **not a filesystem path** and cannot be used to assert that GGUF loading works.

Record changes to model/runtime architecture as design notes before implementing new backends. Real-inference tests must not be satisfied by the mock backend.
