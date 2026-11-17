#include "apex/engine.h"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace apex {

Engine::Engine(EngineMode mode) noexcept : mode_(mode) {}

bool Engine::loadModel(const std::string& model_reference) {
    // A failed replacement must never leave an old model marked as loaded.
    unloadModel();

    if (model_reference.empty()) {
        throw std::invalid_argument("Model reference must not be empty");
    }

    if (mode_ == EngineMode::Mock) {
        constexpr const char* prefix = "mock://";
        if (model_reference.rfind(prefix, 0) != 0 ||
            model_reference.size() <= std::char_traits<char>::length(prefix)) {
            throw std::invalid_argument(
                "Mock mode requires an explicit mock:// identifier; no GGUF file is loaded");
        }
        model_loaded_ = true;
        return true;
    }

    if (!std::filesystem::is_regular_file(model_reference)) {
        throw std::runtime_error("Model file does not exist or is not a regular file: " + model_reference);
    }

    throw std::logic_error(
        "Real model loading and inference are not implemented yet (Phase 2). "
        "Use --mock only for a clearly labeled smoke test.");
}

std::string Engine::generate(const std::string& prompt, int max_tokens) const {
    if (!model_loaded_) {
        throw std::logic_error("No model is loaded");
    }
    if (prompt.empty()) {
        throw std::invalid_argument("Prompt must not be empty");
    }
    if (max_tokens <= 0) {
        throw std::invalid_argument("max_tokens must be greater than zero");
    }

    // No fake tokens/s, no simulated sampling and no misleading "model" response.
    return "[MOCK OUTPUT - NOT LLM INFERENCE] prompt=\"" + prompt +
           "\" (max_tokens=" + std::to_string(max_tokens) + ")";
}

void Engine::unloadModel() noexcept {
    model_loaded_ = false;
}

bool Engine::isLoaded() const noexcept {
    return model_loaded_;
}

} // namespace apex
