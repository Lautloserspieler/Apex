# Apex architecture — Phase 2

```text
Apex CLI (src/main.cpp)
  |
  v
Engine API (include/apex/engine.h)
  |--- Explicit mock mode (no model, no real metrics)
  |--- Real GGUF backend (src/engine/llama_backend.cpp, opt-in compile)
          |
          +-- Pinned llama.cpp model loader / GGUF parser
          +-- llama.cpp tokenizer + decoder-only transformer execution
          +-- Greedy or seeded sampled tokens, EOS/length handling
          +-- Per-request llama_context / KV cache
          +-- Load / TTFT / prefill / decode diagnostics

CMake presets:
  cpu-debug / cpu-release: mock-only, fast tests
  cpu-inference: pinned llama.cpp CPU reference backend
  cuda-release: pinned llama.cpp CUDA offload (requires toolkit, not Apex kernels)
```

The backend is a correctness-first adapter. Original Apex optimized tensor kernels, scheduler, SSE serving and multi-GPU execution belong in later phases. Do not claim model compatibility beyond upstream llama.cpp's pinned support.

`Engine` rejects invalid configuration and file paths. A failed model replacement leaves the engine unloaded; it does not use a previous model silently. Per-call context creation avoids KV reuse across independent calls at the cost of additional overhead.

Third-party source is fetched through CMake with an exact Git commit. Catch2 tests are independent of any model weights. The `APEX_TEST_MODEL` integration test is opt-in.

See [Model testing](models.md) and [Benchmark procedure](benchmark-methodology.md).
