#include "apex/engine.h"

#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

void printUsage(std::ostream& out) {
    out << "Apex Phase 1 - no real LLM inference is implemented yet\n"
        << "Usage:\n"
        << "  apex --mock [--model mock://name] [--prompt TEXT] [--max-tokens N]\n"
        << "  apex --model FILE.gguf [--prompt TEXT] [--max-tokens N]  (Phase 2, unavailable)\n"
        << "  apex --help\n"
        << "The --mock flag is REQUIRED to produce placeholder output.\n";
}

} // namespace

int main(int argc, char** argv) {
    bool mock = false;
    std::string model_reference;
    std::string prompt = "Hello, Apex!";
    int max_tokens = 64;

    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--help" || option == "-h") {
            printUsage(std::cout);
            return 0;
        }
        if (option == "--mock") {
            mock = true;
            continue;
        }
        if (option != "--model" && option != "--prompt" && option != "--max-tokens") {
            std::cerr << "Unknown argument: " << option << "\n";
            printUsage(std::cerr);
            return 2;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << option << "\n";
            return 2;
        }

        const std::string value(argv[++i]);
        if (option == "--model") {
            model_reference = value;
        } else if (option == "--prompt") {
            prompt = value;
        } else {
            const auto* begin = value.data();
            const auto* end = begin + value.size();
            auto [ptr, ec] = std::from_chars(begin, end, max_tokens);
            if (ec != std::errc{} || ptr != end || max_tokens <= 0) {
                std::cerr << "--max-tokens must be a positive integer\n";
                return 2;
            }
        }
    }

    if (mock && model_reference.empty()) {
        model_reference = "mock://cli";
    }
    if (!mock && model_reference.empty()) {
        std::cerr << "A real model requires --model, but real inference is not available "
                     "until Phase 2. Use --mock for a placeholder test.\n";
        return 2;
    }

    try {
        apex::Engine engine(mock ? apex::EngineMode::Mock : apex::EngineMode::Real);
        engine.loadModel(model_reference);
        std::cout << engine.generate(prompt, max_tokens) << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Apex error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
