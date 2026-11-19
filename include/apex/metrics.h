#pragma once

#include "apex/engine.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace apex {

// Build configuration embedded at CMake configure time, so every benchmark
// record states exactly which code and backend produced it.
struct BuildInfo {
    std::string apex_version;
    std::string apex_git_revision; // `git describe --always --dirty`, or "unknown"
    std::string llama_cpp_revision; // pinned upstream tag/commit, or "disabled"
    std::string build_type;
    std::string compiler;
    std::string system;
    bool llama_backend = false;
    bool cuda_backend = false;
    bool cuda_graphs = false;
};

BuildInfo build_info();

// Peak resident set size of this process in MiB, or 0.0 if unavailable.
double process_peak_rss_mib() noexcept;

struct MetricsRun {
    int index = 0;            // 0-based position inside this process
    GenerationResult result;
};

// One process invocation: the model is loaded once, then `runs` generations
// are executed. Run 0 is the first generation after load ("cold" in-process:
// first context allocation, first touch of memory-mapped weights); later
// runs are "warm". The prompt and generated text are deliberately omitted
// because they may be private; the token IDs are kept for correctness checks.
struct MetricsReport {
    BuildInfo build;
    std::string model_path;
    EngineOptions options;
    int max_tokens = 0;
    std::uint64_t prompt_bytes = 0;
    int requested_runs = 0;
    std::vector<MetricsRun> runs;
    std::string error; // empty unless a run failed
};

std::string json_escape(std::string_view text);
void write_metrics_json(std::ostream& out, const MetricsReport& report);

} // namespace apex
