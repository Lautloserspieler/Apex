# Phase 2 support matrix

This is the authoritative, as-tested list of what Apex Phase 2 supports and
has actually verified, as of 2026-11-17. It consolidates evidence recorded
across [issue #2](https://github.com/Lautloserspieler/Apex/issues/2),
`docs/models.md`, `docs/benchmark-methodology.md` and `benchmarks/results-2026-11-17.md`.
Anything not listed here as tested should be treated as unverified, not
assumed to work.

## Build configurations

| Preset | Backend | CI coverage | Notes |
|---|---|---|---|
| `cpu-debug` | Mock only (`APEX_WITH_LLAMA` off) | not run in CI directly; used ad hoc for ASan/UBSan (see below) | Debug build, no real inference |
| `cpu-release` | Mock only | `build-and-test` (Ubuntu 24.04), `windows-x64` (Windows Server 2025, MSVC) | Fast mock/unit-test path; does **not** compile `llama_backend.cpp`'s real-backend branch |
| `cpu-inference` | Real GGUF via pinned llama.cpp, CPU | `inference-backend` (Ubuntu 24.04) | Pinned llama.cpp `b12445` / commit `ef3848f9d7d4cf7b6b2abfc3b4c960b55d6d9e91`, SHA256-verified source archive |
| `cuda-release` | Real GGUF via pinned llama.cpp, CUDA offload | **not run in GitHub-hosted CI** (no GPU runner); verified manually on Fedora 44 / RTX 5070 | Requires CUDA toolkit; `--gpu-layers N` controls offload |

There is no `windows-real-gguf` CPU-inference-on-Windows row above because that
coverage lives inside the `windows-x64` job's real-GGUF validation added in
[PR #12](https://github.com/Lautloserspieler/Apex/pull/12) — confirm the
workflow name in `.github/workflows/ci.yml` if this drifts.

## Operating systems

| OS | Mock build | Real CPU inference | Real CUDA inference |
|---|---|---|---|
| Linux (Ubuntu 24.04, CI) | CI-verified | CI-verified (TinyStories fixture) | not CI-verified (no GPU runner) |
| Linux (Fedora 44, manual) | not separately tested | manually verified (Qwen2.5-0.5B) | manually verified (RTX 5070) |
| Windows (Server 2025, MSVC x64, CI) | CI-verified | CI-verified (TinyStories fixture, PR #12) | not tested |
| macOS | not tested | not tested | not tested |

## Models tested

| Model | Quantization | Params | Purpose | Verified on |
|---|---|---|---|---|
| [tensorblock/TinyStories-656K-GGUF](https://huggingface.co/tensorblock/TinyStories-656K-GGUF) rev. `3099aa5` | Q4_K_M | 656K | Pinned CI smoke fixture (tiny, fast download); **not** a meaningful performance benchmark | Linux CI, Windows CI, Fedora manual |
| [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (SHA256 `74a4da8c...d7a9db`) | Q4_K_M | 630.17M | Primary correctness + performance baseline | Fedora manual (CPU + CUDA) |
| Qwen2.5-3B-Instruct | Q4_K_M | ~3B | Local smoke only (CPU + CUDA) | Fedora manual, not in CI |
| Qwen2.5-7B-Instruct | Q4_K_M | ~7B | Local smoke only (CPU + CUDA) | Fedora manual, not in CI |

Supported GGUF architectures/quantizations in general are bounded by the
pinned llama.cpp version — Apex does not maintain an independent allow-list
beyond rejecting encoder-decoder models (see below). No architecture outside
the `qwen2` family above has been exercised end-to-end.

## Correctness verification performed

| Check | Result | Evidence |
|---|---|---|
| Greedy token-ID parity vs. pinned llama.cpp C API (direct reference binary, not just "it generates text") | **Exact match**: 128/128, 172/172, 160/160 tokens across runs | [issue #2 checkpoint](https://github.com/Lautloserspieler/Apex/issues/2), PR #14 |
| Cross-platform parity (Linux + Windows CI, TinyStories fixture) | 16/16 tokens match | PR #14 |
| ASan/UBSan (Debug, CPU real backend, GCC 16.2.1, Fedora) | 589/589 build targets, 10/10 tests pass, no sanitizer diagnostics | manual, 2026-11-17 |
| Strict CPU-only device selection (`--gpu-layers 0` on a CUDA-enabled build must not use the GPU) | **Bug found and fixed** (PR #15) — see below | manual Fedora investigation, 2026-11-17 |
| Context-overflow boundary rejection (`prompt_tokens + max_tokens` landing exactly on `context_size`) | **Off-by-one bug found and fixed**, regression test added | PR #17, CI-verified (Linux + Windows, real model) |
| Mock-mode isolation (never silently simulates real inference) | Verified by unit tests | `tests/test_engine.cpp` |

### Known-fixed correctness bug: CUDA-build device selection (PR #15)

Before the fix, a CUDA-enabled binary run with `--gpu-layers 0` still allowed
llama.cpp to allocate CUDA compute buffers and produced a **different token
sequence** than the pure CPU build (172 vs. 164 vs. 160 tokens for the same
prompt/model/seed, depending on build and offload). Root cause: pinned
llama.cpp treats `llama_model_params.devices == nullptr` as "auto-discover
accelerators" — `n_gpu_layers=0` alone does not prevent that. Fix: Apex now
passes an explicit, empty, null-terminated accelerator device list whenever
`--gpu-layers 0` is requested. Verified with `CUDA_VISIBLE_DEVICES=-1`
producing the identical 172-token sequence as the pure CPU build, and
confirmed again post-fix on real GPU hardware (both `--gpu-layers 0` and
`--gpu-layers 99` match their respective references).

### Known-fixed correctness bug: context-overflow off-by-one (PR #17)

`required = tokens.size() + max_tokens - 1` (the 0-indexed position of the
last generated token) was compared with `>` instead of `>=` against
`context_size`, letting a request exactly one token over the limit slip past
the explicit pre-check and fail later with a generic decode error instead of
the intended `std::length_error`. Fixed and covered by a regression test
(`tests/test_inference.cpp`), verified on CPU and CUDA with a real model.

## Performance baselines recorded (not yet apples-to-apples)

| Model | Device | Decode tok/s | Source |
|---|---|---:|---|
| Qwen2.5-0.5B | CPU | 104.25 (median, Apex multi-process harness) | `benchmarks/phase2_baseline.py`, Fedora |
| Qwen2.5-0.5B | CUDA (RTX 5070) | 683.08 (median) | same |
| Qwen2.5-0.5B | CPU, official `llama-bench` (`tg128`) | 109.09 ± 0.18 | PR #16 |
| Qwen2.5-0.5B | CUDA (RTX 5070), official `llama-bench` (`tg128`) | 767.98 ± 1.12 | PR #16 |
| Qwen2.5-3B | CPU / GPU | 23.60 / 238.60 | `benchmarks/results-2026-11-17.md` |
| Qwen2.5-7B | CPU / GPU | 10.63 / 122.77 | `benchmarks/results-2026-11-17.md` |

**These numbers are not a validated Apex-vs-llama.cpp overhead claim.** The
Apex harness and the standalone `llama-bench` runs use different process
models, warm/cold protocols, and prompt/cache semantics — see
`docs/benchmark-methodology.md` for the explicit caveats. Treat all figures
above as Apex's own diagnostic baseline, not a competitive benchmark.

## Phase 3 instrumentation status

| Item | Status |
|---|---|
| Timing breakdown (tokenize, context setup, prefill, TTFT, decode, whole request) | Implemented; consistency asserted by the real-model integration test in CI |
| Cold vs warm runs in one process (`--repeat`) | Implemented; CI-verified with TinyStories |
| Pinned thread count (`--threads`) | Implemented; CI checks that greedy token IDs do not change with `--threads 1` |
| Peak RSS | Implemented (Linux/macOS `getrusage`, Windows `GetProcessMemoryInfo`); process high-water mark |
| Machine-readable records (`--metrics-json`) with build info and Git revision | Implemented |
| Benchmark matrix with correctness gate, JSON/CSV/Markdown (`phase3_bench.py`) | Implemented; CI smoke on Linux with reference parity |
| Energy per token, VRAM per context token | Derived in the harness; **not yet verified on real GPU hardware** |
| CUDA-event device timings, allocation counts, transfer volume | Not started |
| GPU CI job | Not started (needs a self-hosted GPU runner) |

## Explicit non-goals / out of scope for Phase 2

- Encoder-decoder models (rejected explicitly at load time)
- Chat templates / model-specific prompt formatting (raw completion only)
- Multiple concurrent requests or sequences (`n_seq_max = 1`)
- HTTP serving interface (Phase 4)
- Independent (non-llama.cpp) numerical validation of transformer math
- Any speedup claim relative to llama.cpp (that work starts in Phase 3)

## What remains open before Phase 2 can formally close

1. A controlled, like-for-like warm/cold benchmark comparing Apex's
   end-to-end timing against the standalone `llama-bench` baseline under
   identical prompt/context/sampler/thread/cache conditions, with raw
   artifacts uploaded (not just local JSON).
2. macOS build/test coverage is entirely untested — either test it or
   explicitly document it as unsupported.
