#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace apex {

enum class EngineMode {
    Real,
    Mock
};

struct EngineOptions {
    std::uint32_t context_size = 2048;
    int gpu_layers = 0;
    float temperature = 0.0f;
    float top_p = 0.95f;
    int top_k = 40;
    std::uint32_t seed = 42;
};

struct GenerationStats {
    int prompt_tokens = 0;
    int generated_tokens = 0;
    double model_load_seconds = 0.0;
    double prefill_seconds = 0.0;
    double ttft_seconds = 0.0;
    double decode_seconds = 0.0;
    std::string stop_reason; // eos, length or mock
};

struct GenerationResult {
    std::string text;
    std::vector<std::int32_t> generated_token_ids;
    GenerationStats stats;
};

class LlamaBackend;

// Phase 2: real GGUF execution delegates to a pinned llama.cpp reference backend.
class Engine final {
public:
    explicit Engine(EngineMode mode = EngineMode::Real,
                    EngineOptions options = {});
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool loadModel(const std::string& model_reference);
    std::string generate(const std::string& prompt, int max_tokens = 128) const;
    GenerationResult generateDetailed(const std::string& prompt,
                                      int max_tokens = 128) const;
    void unloadModel() noexcept;
    bool isLoaded() const noexcept;

private:
    EngineMode mode_;
    EngineOptions options_;
    bool model_loaded_ = false;
    std::unique_ptr<LlamaBackend> backend_;
};

} // namespace apex
