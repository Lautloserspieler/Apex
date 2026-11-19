#include "apex/metrics.h"

#include "apex_build_info.h"

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace apex {

BuildInfo build_info() {
    BuildInfo info;
    info.apex_version = APEX_BUILD_VERSION;
    info.apex_git_revision = APEX_BUILD_GIT_REVISION;
    info.llama_cpp_revision = APEX_BUILD_LLAMA_CPP_REVISION;
    info.build_type = APEX_BUILD_TYPE;
    info.compiler = APEX_BUILD_COMPILER;
    info.system = APEX_BUILD_SYSTEM;
#if defined(APEX_WITH_LLAMA)
    info.llama_backend = true;
#endif
#if defined(APEX_WITH_CUDA)
    info.cuda_backend = true;
#endif
    info.cuda_graphs = APEX_BUILD_CUDA_GRAPHS != 0;
    return info;
}

double process_peak_rss_mib() noexcept {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
        return static_cast<double>(counters.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#elif defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss <= 0) {
        return 0.0;
    }
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0); // bytes
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0; // KiB on Linux/BSD
#endif
#else
    return 0.0;
#endif
}

std::string json_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size() + 2);
    for (const char raw : text) {
        const auto c = static_cast<unsigned char>(raw);
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                out += buffer;
            } else {
                out += raw; // UTF-8 bytes pass through unchanged
            }
        }
    }
    return out;
}

namespace {

class JsonWriter {
public:
    explicit JsonWriter(std::ostream& out) : out_(out) {}

    void string(std::string_view value) { out_ << '"' << json_escape(value) << '"'; }
    void boolean(bool value) { out_ << (value ? "true" : "false"); }
    template <class T>
    void integer(T value) { out_ << value; }
    void number(double value) {
        if (!std::isfinite(value)) {
            out_ << "null"; // JSON has no NaN/Infinity
            return;
        }
        std::ostringstream text;
        text.imbue(std::locale::classic());
        text << std::setprecision(10) << value;
        out_ << text.str();
    }
    void key(std::string_view name) {
        string(name);
        out_ << ':';
    }

private:
    std::ostream& out_;
};

void write_stats(std::ostream& out, JsonWriter& json, const GenerationStats& s) {
    out << '{';
    json.key("prompt_tokens"); json.integer(s.prompt_tokens); out << ',';
    json.key("generated_tokens"); json.integer(s.generated_tokens); out << ',';
    json.key("stop_reason"); json.string(s.stop_reason); out << ',';
    json.key("model_load_seconds"); json.number(s.model_load_seconds); out << ',';
    json.key("tokenize_seconds"); json.number(s.tokenize_seconds); out << ',';
    json.key("context_setup_seconds"); json.number(s.context_setup_seconds); out << ',';
    json.key("prefill_seconds"); json.number(s.prefill_seconds); out << ',';
    json.key("ttft_seconds"); json.number(s.ttft_seconds); out << ',';
    json.key("decode_seconds"); json.number(s.decode_seconds); out << ',';
    json.key("request_seconds"); json.number(s.request_seconds); out << ',';
    json.key("peak_rss_mib"); json.number(s.peak_rss_mib); out << ',';
    json.key("peak_vram_mib"); json.number(s.peak_vram_mib); out << ',';
    json.key("avg_power_watts"); json.number(s.avg_power_watts);
    out << '}';
}

} // namespace

void write_metrics_json(std::ostream& out, const MetricsReport& report) {
    JsonWriter json(out);
    const BuildInfo& b = report.build;
    const EngineOptions& o = report.options;

    out << "{\n";
    json.key("schema"); json.string("apex.metrics/1"); out << ",\n";
    json.key("build"); out << '{';
    json.key("apex_version"); json.string(b.apex_version); out << ',';
    json.key("apex_git_revision"); json.string(b.apex_git_revision); out << ',';
    json.key("llama_cpp_revision"); json.string(b.llama_cpp_revision); out << ',';
    json.key("build_type"); json.string(b.build_type); out << ',';
    json.key("compiler"); json.string(b.compiler); out << ',';
    json.key("system"); json.string(b.system); out << ',';
    json.key("llama_backend"); json.boolean(b.llama_backend); out << ',';
    json.key("cuda_backend"); json.boolean(b.cuda_backend); out << ',';
    json.key("cuda_graphs"); json.boolean(b.cuda_graphs);
    out << "},\n";

    json.key("config"); out << '{';
    json.key("model_path"); json.string(report.model_path); out << ',';
    json.key("prompt_bytes"); json.integer(report.prompt_bytes); out << ',';
    json.key("max_tokens"); json.integer(report.max_tokens); out << ',';
    json.key("context_size"); json.integer(o.context_size); out << ',';
    json.key("gpu_layers"); json.integer(o.gpu_layers); out << ',';
    json.key("threads"); json.integer(o.threads); out << ',';
    json.key("temperature"); json.number(o.temperature); out << ',';
    json.key("top_k"); json.integer(o.top_k); out << ',';
    json.key("top_p"); json.number(o.top_p); out << ',';
    json.key("seed"); json.integer(o.seed); out << ',';
    json.key("requested_runs"); json.integer(report.requested_runs);
    out << "},\n";

    json.key("runs"); out << '[';
    for (std::size_t i = 0; i < report.runs.size(); ++i) {
        const MetricsRun& run = report.runs[i];
        out << (i == 0 ? "\n" : ",\n") << '{';
        json.key("index"); json.integer(run.index); out << ',';
        json.key("phase"); json.string(run.index == 0 ? "cold" : "warm"); out << ',';
        json.key("stats"); write_stats(out, json, run.result.stats); out << ',';
        json.key("token_ids"); out << '[';
        const auto& ids = run.result.generated_token_ids;
        for (std::size_t t = 0; t < ids.size(); ++t) {
            if (t != 0) out << ',';
            out << ids[t];
        }
        out << "]}";
    }
    out << "\n]";
    if (!report.error.empty()) {
        out << ",\n";
        json.key("error"); json.string(report.error);
    }
    out << "\n}\n";
}

} // namespace apex
