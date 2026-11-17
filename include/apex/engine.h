#pragma once

#include <string>
#include <vector>

namespace apex {

class Engine {
public:
    Engine() = default;
    ~Engine() = default;

    // Load a model from the given path
    bool loadModel(const std::string& path);

    // Generate text from a prompt (basic version)
    std::string generate(const std::string& prompt, int max_tokens = 128);

    // Simple status check
    bool isLoaded() const;

private:
    bool model_loaded_ = false;
};

} // namespace apex
