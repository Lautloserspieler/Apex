# Apex

**High-performance C++/CUDA LLM inference engine** focused on low latency, high throughput and efficient serving.

Apex is a minimal, optimized inference core designed for maximum performance on consumer and multi-GPU hardware. The goal is to deliver measurable improvements over existing solutions (llama.cpp, vLLM, etc.) in targeted scenarios such as low-latency serving, long-context efficiency, and reduced overhead.

## Goals
- Extremely low time-to-first-token (TTFT) and high decode throughput
- Efficient memory usage and kernel optimizations
- Clean C++/CUDA codebase with minimal overhead
- Reproducible benchmarks against established baselines
- Practical serving interface (OpenAI-compatible)

## Status
Early stage – foundation and first optimizations in progress.

## License
MIT
