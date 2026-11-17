#include "apex/engine.h"
#include "llama_backend.h"

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace apex {

Engine::Engine(EngineMode mode, EngineOptions options)
    : mode_(mode), options_(options) {
    if (options_.context_size < 16) {
        throw std::invalid_argument("context_size must be at least 16");
    }
    if (options_.gpu_layers < 0) {
        throw std::invalid_argument("gpu_layers must be non-negative");
    }
    if (!std::isfinite(options_.temperature) || options_.temperature < 0.0f) {
        throw std::invalid_argument("temperature must be finite and non-negative");
    }
    if (!std::isfinite(options_.top_p) || options_.top_p <= 0.0f ||
        options_.top_p > 1.0f) {
        throw std::invalid_argument("top_p must be in (0, 1]");
    }
    if (options_.top_k < 0) {
        throw std::invalid_argument("top_k must be non-negative");
    }
}

Engine::~Engine() = default;

bool Engine::loadModel(const std::string& model_reference) {
    unloadModel();
    if (model_reference.empty()) {
        throw std::invalid_argument("Model reference must not be empty");
    }

    if (mode_ == EngineMode::Mock) {
        constexpr std::string_view prefix = "mock://";
        if (model_reference.rfind(prefix, 0) != 0 ||
            model_reference.size() <= prefix.size()) {
            throw std::invalid_argument("Mock mode only accepts mock:// identifiers");
        }
        model_loaded_ = true;
        return true;
    }

    if (!std::filesystem::is_regular_file(model_reference)) {
        throw std::runtime_error("Model file does not exist or is not a regular file: " + model_reference);
    }

#if defined(APEX_WITH_LLAMA)
    backend_ = std::make_unique<LlamaBackend>(model_reference, options_);
    model_loaded_ = true;
    return true;
#else
    throw std::logic_error(
        "Real GGUF backend was not compiled. Reconfigure with cmake --preset cpu-inference");
#endif
}

GenerationResult Engine::generateDetailed(const std::string& prompt, int max_tokens) const {
    if (!model_loaded_) {
        throw std::logic_error("No model is loaded");
    }
    if (prompt.empty()) {
        throw std::invalid_argument("Prompt must not be empty");
    }
    if (max_tokens <= 0) {
        throw std::invalid_argument("max_tokens must be greater than zero");
    }
    if (mode_ == EngineMode::Mock) {
        GenerationResult result;
        result.text = "[MOCK OUTPUT - NOT LLM INFERENCE] prompt=\"" + prompt +
                      "\" (max_tokens=" + std::to_string(max_tokens) + ")";
        result.stats.stop_reason = "mock";
        return result;
    }
    if (!backend_) {
        throw std::logic_error("Real inference backend is unavailable");
    }
    return backend_->generate(prompt, max_tokens);
}

std::string Engine::generate(const std::string& prompt, int max_tokens) const {
    return generateDetailed(prompt, max_tokens).text;
}

void Engine::unloadModel() noexcept {
    backend_.reset();
    model_loaded_ = false;
}

bool Engine::isLoaded() const noexcept {
    return model_loaded_;
}

} // namespace apex
