# Apex

**High-performance C++/CUDA LLM inference engine** focused on low latency, high throughput and efficient serving.

Apex is a minimal, optimized inference core designed for maximum performance on consumer and multi-GPU hardware. The goal is to deliver measurable improvements over existing solutions (llama.cpp, vLLM, etc.) in targeted scenarios such as low-latency serving, long-context efficiency, and reduced overhead.

## Goals
- Extremely low time-to-first-token (TTFT) and high decode throughput
- Efficient memory usage and kernel optimizations
- Clean C++/CUDA codebase with minimal overhead
- Reproducible benchmarks against established baselines
- Practical serving interface (OpenAI-compatible)

## Current Status
Early stage. Folder structure, license and basic CMake setup are in place. No inference code yet.

## Roadmap / Milestones

### Milestone 1 – Minimal Inference Core (current focus)
- Load a small model (e.g. via GGUF or simple format)
- Run basic generation (greedy or simple sampling)
- Simple benchmark harness to measure tokens/s and latency
- Clear comparison baseline against llama.cpp

### Milestone 2
- First custom CUDA kernels / optimizations
- Measurable performance improvement in at least one scenario

### Milestone 3
- Basic serving interface
- Documentation and examples

## Building
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## License
MIT
