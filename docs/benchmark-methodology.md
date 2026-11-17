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
