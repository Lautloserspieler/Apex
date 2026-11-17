#include "apex/engine.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {
void usage(std::ostream& out) {
    out << "Apex Phase 2 — GGUF inference (llama.cpp reference backend)\n"
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
        << "  --gpu-layers N     GPU offload layers (default 0, CPU only)\n"
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

void stats(const apex::GenerationStats& s) {
    const double prefill_tps = s.prefill_seconds > 0 ? s.prompt_tokens / s.prefill_seconds : 0;
    const double decode_tps = s.decode_seconds > 0 && s.generated_tokens > 1
                                  ? (s.generated_tokens - 1) / s.decode_seconds : 0;
    std::cerr << "[Apex] prompt_tokens=" << s.prompt_tokens
              << " output_tokens=" << s.generated_tokens
              << " stop=" << s.stop_reason
              << " load_s=" << s.model_load_seconds
              << " ttft_s=" << s.ttft_seconds
              << " prefill_tok_s=" << prefill_tps
              << " decode_tok_s=" << decode_tps << "\n";
}
} // namespace

int main(int argc, char** argv) {
    apex::EngineOptions cfg;
    bool mock = false;
    int max_tokens = 64;
    std::string model;
    std::string prompt = "Hello, Apex!";

    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--help" || option == "-h") {
            usage(std::cout);
            return 0;
        }
        if (option == "--mock") { mock = true; continue; }
        if (option != "--model" && option != "--prompt" && option != "--max-tokens" &&
            option != "--ctx-size" && option != "--seed" && option != "--temperature" &&
            option != "--top-k" && option != "--top-p" && option != "--gpu-layers") {
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
        else {
            bool valid = false;
            if (option == "--max-tokens") valid = parse_integer(value, max_tokens) && max_tokens > 0;
            if (option == "--ctx-size") valid = parse_integer(value, cfg.context_size) && cfg.context_size >= 16;
            if (option == "--seed") valid = parse_integer(value, cfg.seed);
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

    if (mock && model.empty()) model = "mock://cli";
    if (!mock && model.empty()) {
        usage(std::cerr);
        return 2;
    }

    try {
        apex::Engine engine(mock ? apex::EngineMode::Mock : apex::EngineMode::Real, cfg);
        engine.loadModel(model);
        auto result = engine.generateDetailed(prompt, max_tokens);
        std::cout << result.text << '\n';
        if (!mock) stats(result.stats);
    } catch (const std::exception& e) {
        std::cerr << "[Apex error] " << e.what() << '\n';
        return 1;
    }
    return 0;
}
