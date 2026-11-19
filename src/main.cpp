#include "apex/engine.h"
#include "apex/metrics.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {
void usage(std::ostream& out) {
    out << "Apex Phase 3 — GGUF inference (llama.cpp reference backend)\n"
        << "Usage: apex --model FILE.gguf --prompt TEXT [options]\n"
        << "       apex --mock --prompt TEXT [options]\n"
        << "Options:\n"
        << "  --model FILE       GGUF path (or mock://ID with --mock)\n"
        << "  --prompt TEXT      Raw completion prompt (no chat template yet)\n"
        << "  --max-tokens N     Output cap (default 64)\n"
        << "  --ctx-size N       Context length (default 2048)\n"
        << "  --temperature F    Zero for greedy decoding (default 0)\n"
        << "  --top-k N          Sampling top-k (default 40)\n"
        << "  --top-p F          Sampling top-p (default 0.95)\n"
        << "  --seed N           Sampling seed (default 42)\n"
        << "  --gpu-layers N     Offloaded layers; 0 excludes GPU devices (default 0)\n"
        << "  --threads N        CPU threads for prefill/decode; 0 = llama.cpp default (default 0)\n"
        << "  --repeat N         Generate N times in this process after one model load;\n"
        << "                     run 1 is cold, later runs are warm (default 1)\n"
        << "  --metrics-json F   Write build info, config and per-run metrics as JSON to F\n"
        << "  --print-token-ids  Print generated token IDs to stderr (diagnostic)\n"
        << "  --mock             Explicit synthetic response; no model loaded\n"
        << "  --help             Show this message\n";
}

template <class T>
bool parse_integer(const std::string& value, T& destination) {
    const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), destination);
    return ec == std::errc{} && ptr == value.data() + value.size();
}

bool parse_float(const std::string& value, float& destination) {
    try {
        std::size_t used = 0;
        destination = std::stof(value, &used);
        return used == value.size() && std::isfinite(destination);
    } catch (const std::exception&) { return false; }
}

void stats(const apex::GenerationStats& s, int run, int runs) {
    const double prefill_tps = s.prefill_seconds > 0 ? s.prompt_tokens / s.prefill_seconds : 0;
    const double decode_tps = s.decode_seconds > 0 && s.generated_tokens > 1
                                  ? (s.generated_tokens - 1) / s.decode_seconds : 0;
    std::cerr << "[Apex] prompt_tokens=" << s.prompt_tokens
              << " output_tokens=" << s.generated_tokens
              << " stop=" << s.stop_reason
              << " load_s=" << s.model_load_seconds
              << " ttft_s=" << s.ttft_seconds
              << " prefill_tok_s=" << prefill_tps
              << " decode_tok_s=" << decode_tps
              << " tokenize_s=" << s.tokenize_seconds
              << " ctx_setup_s=" << s.context_setup_seconds
              << " request_s=" << s.request_seconds;
    if (s.peak_rss_mib > 0.0) {
        std::cerr << " peak_rss_mib=" << s.peak_rss_mib;
    }
    // 0.0 means "not measured" (CPU-only build/run, mock, or nvidia-smi
    // unavailable) - see the GenerationStats doc comment. Only print when
    // actually sampled so CPU runs don't show a misleading "0 MiB / 0 W".
    if (s.peak_vram_mib > 0.0 || s.avg_power_watts > 0.0) {
        std::cerr << " peak_vram_mib=" << s.peak_vram_mib
                  << " avg_power_watts=" << s.avg_power_watts;
    }
    if (runs > 1) {
        std::cerr << " run=" << (run + 1) << "/" << runs
                  << " phase=" << (run == 0 ? "cold" : "warm");
    }
    std::cerr << "\n";
}

bool write_report(const std::string& path, const apex::MetricsReport& report) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        std::cerr << "[Apex error] Cannot open metrics file: " << path << '\n';
        return false;
    }
    apex::write_metrics_json(file, report);
    file.flush();
    if (!file) {
        std::cerr << "[Apex error] Failed writing metrics file: " << path << '\n';
        return false;
    }
    return true;
}
} // namespace

int main(int argc, char** argv) {
    apex::EngineOptions cfg;
    bool mock = false;
    bool print_token_ids = false;
    int max_tokens = 64;
    int repeat = 1;
    std::string metrics_json;
    std::string model;
    std::string prompt = "Hello, Apex!";

    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--help" || option == "-h") {
            usage(std::cout);
            return 0;
        }
        if (option == "--mock") { mock = true; continue; }
        if (option == "--print-token-ids") { print_token_ids = true; continue; }
        if (option != "--model" && option != "--prompt" && option != "--max-tokens" &&
            option != "--ctx-size" && option != "--seed" && option != "--temperature" &&
            option != "--top-k" && option != "--top-p" && option != "--gpu-layers" &&
            option != "--threads" && option != "--repeat" && option != "--metrics-json") {
            std::cerr << "Unknown option: " << option << "\n";
            return 2;
        }
        if (i + 1 == argc) {
            std::cerr << "Missing value for " << option << "\n";
            return 2;
        }
        const std::string value(argv[++i]);
        if (option == "--model") model = value;
        else if (option == "--prompt") prompt = value;
        else if (option == "--metrics-json") {
            if (value.empty()) {
                std::cerr << "Invalid value for --metrics-json: empty path\n";
                return 2;
            }
            metrics_json = value;
        }
        else {
            bool valid = false;
            if (option == "--max-tokens") valid = parse_integer(value, max_tokens) && max_tokens > 0;
            if (option == "--ctx-size") valid = parse_integer(value, cfg.context_size) && cfg.context_size >= 16;
            if (option == "--seed") valid = parse_integer(value, cfg.seed);
            if (option == "--threads") valid = parse_integer(value, cfg.threads) && cfg.threads >= 0;
            if (option == "--repeat") valid = parse_integer(value, repeat) && repeat >= 1 && repeat <= 10000;
            if (option == "--gpu-layers") valid = parse_integer(value, cfg.gpu_layers) && cfg.gpu_layers >= 0;
            if (option == "--top-k") valid = parse_integer(value, cfg.top_k) && cfg.top_k >= 0;
            if (option == "--temperature") valid = parse_float(value, cfg.temperature) && cfg.temperature >= 0;
            if (option == "--top-p") valid = parse_float(value, cfg.top_p) && cfg.top_p > 0 && cfg.top_p <= 1;
            if (!valid) {
                std::cerr << "Invalid value for " << option << ": " << value << "\n";
                return 2;
            }
        }
    }

    if (mock && print_token_ids) {
        std::cerr << "--print-token-ids requires a real model backend\n";
        return 2;
    }
    // Mock output has no real timings; never let it masquerade as a
    // benchmark record.
    if (mock && !metrics_json.empty()) {
        std::cerr << "--metrics-json requires a real model backend\n";
        return 2;
    }
    if (mock && model.empty()) model = "mock://cli";
    if (!mock && model.empty()) {
        usage(std::cerr);
        return 2;
    }

    apex::MetricsReport report;
    report.build = apex::build_info();
    report.model_path = model;
    report.options = cfg;
    report.max_tokens = max_tokens;
    report.prompt_bytes = prompt.size();
    report.requested_runs = repeat;

    int exit_code = 0;
    try {
        apex::Engine engine(mock ? apex::EngineMode::Mock : apex::EngineMode::Real, cfg);
        engine.loadModel(model);
        for (int run = 0; run < repeat; ++run) {
            auto result = engine.generateDetailed(prompt, max_tokens);
            // Print the completion once; later runs are for measurement.
            if (run == 0) {
                std::cout << result.text << '\n';
            }
            if (!mock) {
                stats(result.stats, run, repeat);
                if (print_token_ids && run == 0) { // one record; JSON has all runs
                    std::cerr << "APEX_TOKEN_IDS=";
                    for (std::size_t index = 0; index < result.generated_token_ids.size(); ++index) {
                        if (index != 0) std::cerr << ',';
                        std::cerr << result.generated_token_ids[index];
                    }
                    std::cerr << '\n';
                }
            }
            report.runs.push_back(apex::MetricsRun{run, std::move(result)});
        }
    } catch (const std::exception& e) {
        std::cerr << "[Apex error] " << e.what() << '\n';
        report.error = e.what();
        exit_code = 1;
    }
    // Written even after a failure so partial evidence is not lost.
    if (!metrics_json.empty() && !write_report(metrics_json, report)) {
        return 1;
    }
    return exit_code;
}
