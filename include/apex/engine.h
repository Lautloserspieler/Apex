#pragma once

#include <string>

namespace apex {

// Real inference is deliberately unavailable until Phase 2.
// Mock is an explicitly opted-in, deterministic test implementation.
enum class EngineMode {
    Real,
    Mock
};

class Engine final {
public:
    explicit Engine(EngineMode mode = EngineMode::Real) noexcept;

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Returns true on success; throws on invalid input or unsupported real inference.
    // In mock mode, model_reference must be a mock:// identifier, never a GGUF path.
    bool loadModel(const std::string& model_reference);

    // Returns mock text only in explicit mock mode. Throws on invalid state/input.
    std::string generate(const std::string& prompt, int max_tokens = 128) const;

    void unloadModel() noexcept;
    bool isLoaded() const noexcept;

private:
    EngineMode mode_;
    bool model_loaded_ = false;
};

} // namespace apex
