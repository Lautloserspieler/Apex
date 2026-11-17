# Phase 2 model testing

## Recommended correctness baseline

- Official repository: [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
- Example filename: `qwen2.5-0.5b-instruct-q4_k_m.gguf`
- Model license: Apache 2.0 (consult the official model card before redistribution)
- Backend: pinned llama.cpp `b12445` / commit `ef3848f9d7d4cf7b6b2abfc3b4c960b55d6d9e91`
- Reference CPU configuration: greedy sampling (`--temperature 0`), `--gpu-layers 0`, context 2048 and max 64 new tokens.
- **Strict CPU selection:** `--gpu-layers 0` explicitly passes an empty accelerator-device list to the pinned llama.cpp model loader. This avoids CUDA compute buffers even when the binary was built with CUDA and a GPU is visible. `--gpu-layers N` with `N > 0` keeps automatic accelerator selection. This distinction was required after reproducible Fedora tests showed that `n_gpu_layers=0` alone still allowed CUDA buffer allocation and changed greedy token output.


### Download one model file

On Linux/macOS:

```bash
mkdir -p models
curl -L --fail -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"
sha256sum models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

On Windows Developer PowerShell:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
Invoke-WebRequest "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true" -OutFile "models\qwen2.5-0.5b-instruct-q4_k_m.gguf"
Get-FileHash "models\qwen2.5-0.5b-instruct-q4_k_m.gguf" -Algorithm SHA256
```

Downloading models is manual and independent of CI. Store the hash, model repository revision and file size with each benchmark report. Do not check models into Git. Model repositories can change: pin a Hugging Face revision for formal experiments.

### Run the model

```bash
cmake --preset cpu-inference
cmake --build --preset cpu-inference --parallel 2
./build/cpu-inference/apex --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf --prompt "Write a one-sentence greeting:" --ctx-size 2048 --max-tokens 64 --temperature 0
```

Windows uses `./build/cpu-inference/apex.exe`.

**Important:** This is raw prompt **completion**, not model-specific chat formatting. For instruct models, chat quality may suffer without the correct special-token chat template. Model-specific chat templates are not implemented in this phase.

### Optional integration test

```bash
export APEX_TEST_MODEL="$PWD/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
ctest --preset cpu-inference --output-on-failure
```

PowerShell:

```powershell
$env:APEX_TEST_MODEL = (Resolve-Path .\models\qwen2.5-0.5b-instruct-q4_k_m.gguf).Path
ctest --preset cpu-inference --output-on-failure
```

Only enable `APEX_TEST_MODEL` for builds with `APEX_ENABLE_LLAMA=ON`. When no model path is provided, the integration test explicitly skips real generation.

## Pinned cross-platform CI smoke fixture

- Model: [tensorblock/TinyStories-656K-GGUF](https://huggingface.co/tensorblock/TinyStories-656K-GGUF)
- Revision: `3099aa586494d47fcc8a64f71528539c9384915f`
- File: `TinyStories-656K-Q4_K_M.gguf`
- SHA256: `bc31753cc99ac43a6b611b1e390c6054e1e6a32b3b0c3090fe36ac018f8389b4`
- Purpose: verify actual token generation and repeated deterministic outputs under Linux and Windows x64 CPU reference builds.
- This tiny 656K-parameter model is **not** a meaningful CPU/GPU performance benchmark. Model weights are downloaded only in real-inference test jobs, never committed to the repository.

## Phase 2 limitations

- Decoder-only text models, raw completions, one active request/context
- Supported GGUF architectures and quantizations determined by the pinned llama.cpp version
- Context must accommodate prompt plus max generated tokens
- No reliable quality claims or speedup claims are made
- A real model test and fair external baseline comparison are required before closing Phase 2

## Optional deterministic Qwen2.5 reference parity

After building the real `cpu-inference` target and downloading the documented
Qwen2.5-0.5B-Instruct Q4_K_M GGUF, invoke:

```bash
python3 benchmarks/compare_reference.py \
  --apex build/cpu-inference/apex \
  --reference build/cpu-inference/apex_llama_reference \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt-file benchmarks/phase2_qwen_prompt.txt \
  --ctx-size 2048 --max-tokens 128 --gpu-layers 0
```

The runner writes exact token IDs and records the local GGUF hash, executable
hashes and raw logs. The same approach works with the CUDA binaries. The
Windows/Linux CI uses the smaller, pinned TinyStories fixture to avoid a
large external model dependency in ordinary pull requests. Keep the formal
Qwen reference checkpoint open until these runs and hashes are actually saved.

## Phase 2 backend isolation: Fedora RTX 5070

A Qwen2.5-0.5B-Instruct Q4_K_M validation (GGUF SHA256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`) found that, **before the strict CPU fix**, the pure CPU build produced 172 tokens, the CUDA-enabled build with `--gpu-layers 0` produced 164 tokens, and full CUDA offload produced 160 tokens. All runs reproduced identically within each backend and matched the pinned direct llama.cpp reference. With `CUDA_VISIBLE_DEVICES=-1`, the CUDA build produced the identical 172-token CPU sequence.

The strict `--gpu-layers 0` device-selection fix is intended to make that CPU behavior independent of whether a GPU is visible. Actual CUDA-enabled builds should be validated on GPU hardware after updating Apex:

```bash
cmake --build --preset cpu-inference --parallel 8
cmake --build --preset cuda-release --parallel 6
python3 benchmarks/compare_reference.py \
  --apex build/cuda-release/apex \
  --reference build/cuda-release/apex_llama_reference \
  --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --prompt-file benchmarks/phase2_qwen_prompt.txt \
  --ctx-size 2048 --max-tokens 256 --gpu-layers 0
```

Compare its full `apex_token_ids` to the previously recorded 172-token pure CPU baseline. Inspect `reference.stderr.log` for actual CUDA device/compute buffers; merely seeing `FLASH_ATTN_EXT` in the graph is not evidence of CUDA execution. The GPU-offload mode (`--gpu-layers 99`) should continue passing independent per-backend exact-token reference checks.
