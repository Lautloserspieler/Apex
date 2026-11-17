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

## Repeated diagnostic harness (Phase 2)

Apex now includes `benchmarks/phase2_baseline.py` (Python 3 standard library only). It records the exact SHA-256 of the GGUF and Apex executable, Apex Git revision, context, output budget, seed-free greedy settings, driver/toolkit details if available, raw stdout/stderr per run, and median/min/max statistics in JSON.

On Fedora, after building `cpu-inference` and `cuda-release` and downloading a Qwen2.5 GGUF:

```bash
cd ~/Entwicklung/Apex

python3 benchmarks/phase2_baseline.py \
  --apex build/cpu-inference/apex \
  --model models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --label qwen7b-cpu --gpu-layers 0 --ctx-size 4096 \
  --max-tokens 256 --warmup 1 --runs 5

python3 benchmarks/phase2_baseline.py \
  --apex build/cuda-release/apex \
  --model models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --label qwen7b-gpu --gpu-layers 99 --ctx-size 4096 \
  --max-tokens 256 --warmup 1 --runs 5
```

The harness supplies **the same fixed Qwen ChatML prompt** to every invocation. For other models or model-specific prompts, pass `--prompt-file path/to/prompt.txt` consistently to both commands. Results are saved below `benchmarks/results/<UTC timestamp>-<label>-<id>/report.json` with per-run logs and generated text. These outputs are git-ignored because prompts and generations may be private.

**Measurement caveat:** Every run is a fresh **process** that reloads the model and creates a context. The first excluded run only warms system filesystem caches / general runtime state; it does **not** measure a long-lived in-process hot model or KV cache. For strict cold-run comparisons use controlled OS/GPU state and record them separately. Reported `prefill_tok_s`, `decode_tok_s` and TTFT are Apex's diagnostic measurements, not whole-request wall-time metrics; compare like with like, never mix with the external tool's metrics without aligning definitions.

This script does **not** establish an independent upstream llama.cpp baseline or validate logits/token IDs. Those Phase 2 acceptance requirements remain open and must be completed separately. Do not claim Apex speedups from CPU/GPU deltas: both paths use the llama.cpp backend.

## Exact greedy token-ID parity against pinned llama.cpp C API

Phase 2 also builds a separate `apex_llama_reference` binary when real
inference and tests are enabled. It directly calls the **exact pinned** upstream
llama.cpp public API, not any Apex Engine class. The separate program and the
`benchmarks/compare_reference.py` runner verify full generated token-ID parity
in greedy decoding, including correct stop behavior and prompt token count.
Linux and Windows real-GGUF CI run this check with the SHA-256-verified
TinyStories-656K GGUF fixture. Unlike a performance benchmark, the parity check
must match exactly (token IDs can be compared without normalizing text).

Example on Fedora, using an already-built `cpu-inference` tree and a model
previously downloaded to `models/`:

```bash
cd ~/Entwicklung/Apex
git pull origin main
cmake --build --preset cpu-inference --parallel 8
python3 benchmarks/compare_reference.py \
  --apex build/cpu-inference/apex \
  --reference build/cpu-inference/apex_llama_reference \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --ctx-size 2048 --max-tokens 128 --gpu-layers 0 \
  --prompt-file benchmarks/phase2_qwen_prompt.txt
```

For CUDA use the `cuda-release` binaries, set `--gpu-layers 99`, and verify
through the raw reference and Apex logs that offloading occurred. Reference and
Apex share the same pinned upstream numerical compute implementation: this
checks Apex's adapter semantics but is **not** evidence of independent
transformer numerical validation or a speedup over stock llama.cpp CLI.

Each comparison writes `comparison.json` and separate binary logs under
`benchmarks/results/`, which is git-ignored. A failed/mismatching comparison
returns nonzero and retains the evidence.

## Explicit CPU-only device selection (Phase 2)

Pinned llama.cpp b12445 interprets `llama_model_params.devices == nullptr` as automatic accelerator discovery. Consequently `n_gpu_layers=0` by itself does **not** necessarily prevent all CUDA compute allocations in a CUDA build. Apex now provides an explicit null-terminated empty accelerator device list (`devices[0] == nullptr`) whenever `--gpu-layers 0` is requested. The separate direct-API comparison executable applies the same device constraint to ensure a matched reference configuration. CUDA-enabled GPU-offload runs are not changed. Verify actual CUDA buffer absence and Qwen exact-token CPU parity on a real GPU system before marking this edge case closed.

## Phase 3: CUDA graph decode verification (ggml-cuda GGML_CUDA_GRAPHS)

Apex does not implement its own CUDA graph capture. The pinned llama.cpp
backend (ggml-cuda) already has one, controlled by the `GGML_CUDA_GRAPHS`
CMake option, whose upstream default has varied by version. Apex's
`CMakeLists.txt` now forces `GGML_CUDA_GRAPHS=ON` whenever `APEX_ENABLE_CUDA`
is on, because Apex's autoregressive decode loop (`LlamaBackend::generate`)
reuses one `llama_context` with a stable batch=1 shape per decode step — the
exact case CUDA graph capture targets.

**To verify it actually took effect and measure the delta on real GPU
hardware:**

```bash
# 1. Confirm the option is actually ON for this configure.
cmake --preset cuda-release 2>&1 | grep "Apex CUDA graphs"
# Expect: -- Apex CUDA graphs (ggml-cuda GGML_CUDA_GRAPHS): ON

cmake --build --preset cuda-release --parallel 2

# 2. Baseline run with graphs explicitly disabled at runtime (no rebuild
#    needed - this is the upstream opt-out env var).
GGML_CUDA_DISABLE_GRAPHS=1 python3 benchmarks/phase2_baseline.py \
  --apex build/cuda-release/apex \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --label cuda-graphs-off --gpu-layers 99 --ctx-size 2048 \
  --max-tokens 256 --warmup 1 --runs 5

# 3. Same run with graphs enabled (the default now).
python3 benchmarks/phase2_baseline.py \
  --apex build/cuda-release/apex \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --label cuda-graphs-on --gpu-layers 99 --ctx-size 2048 \
  --max-tokens 256 --warmup 1 --runs 5
```

Compare the `decode_tok_s` medians between the two labeled result
directories. This isolates exactly one variable (graph capture on/off) with
everything else — model, prompt, context, offload — held constant, so any
delta is directly attributable to CUDA graphs, unlike the cross-tool
comparisons above. Record the result (including a null result, i.e. no
measurable difference) in `benchmarks/results-2026-11-17.md` or a new dated
file before concluding anything.

## External synthetic baseline recorded (2026-11-17)

The official pinned `llama-bench` executable was built separately for CPU-only and CUDA on the Fedora RTX 5070 system. CPU `pp37/tg128`: **372.10 ± 4.07 / 109.09 ± 0.18 tok/s**; CUDA `pp37/tg128`: **11643.48 ± 3151.74 / 767.98 ± 1.12 tok/s**. These are standalone synthetic figures, **not** a like-for-like end-to-end performance comparison with the earlier multi-process Apex benchmark (CPU median prefill/decode **372.85 / 104.25**, CUDA **583.05 / 683.08 tok/s**). For full commands and constraints see [the dated evidence record](../benchmarks/results-2026-11-17.md#repeated-official-llamacpp-standalone-benchmark--fedora-44-2026-11-17). The standalone measurements establish a first external baseline only. Do not quote their differences as independent Apex kernel overhead; the same pinned llama.cpp kernels implement Apex Phase 2 real inference.
