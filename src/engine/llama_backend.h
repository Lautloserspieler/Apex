#pragma once

#include "apex/engine.h"
#include <memory>
#include <string>

namespace apex {

// Opaque reference runtime keeps upstream C API out of Apex's public headers.
class LlamaBackend final {
public:
    LlamaBackend(const std::string& path, const EngineOptions& options);
    ~LlamaBackend();

    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;

    GenerationResult generate(const std::string& prompt, int max_tokens) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace apex
