#include <catch2/catch_test_macros.hpp>
#include "apex/engine.h"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

TEST_CASE("Optional real GGUF integration (APEX_TEST_MODEL)", "[inference]") {
    const char* path = std::getenv("APEX_TEST_MODEL");
    if (path == nullptr || std::string(path).empty()) {
        SUCCEED("No APEX_TEST_MODEL configured; integration test is opt-in");
        return;
    }
    REQUIRE(std::filesystem::is_regular_file(path));

    apex::EngineOptions options;
    options.context_size = 256;
    options.gpu_layers = 0;
    options.temperature = 0;
    apex::Engine engine(apex::EngineMode::Real, options);
    REQUIRE(engine.loadModel(path));
    auto first = engine.generateDetailed("A short greeting:", 8);
    REQUIRE(first.stats.prompt_tokens > 0);
    REQUIRE(first.stats.generated_tokens >= 0);
    REQUIRE(first.stats.generated_tokens <= 8);
    REQUIRE(first.generated_token_ids.size() ==
            static_cast<std::size_t>(first.stats.generated_tokens));

    // The KV/context must reset between generation requests.
    auto second = engine.generateDetailed("A short greeting:", 8);
    REQUIRE(second.generated_token_ids == first.generated_token_ids);
    engine.unloadModel();
    REQUIRE_FALSE(engine.isLoaded());
}

TEST_CASE("Context overflow is rejected even at the exact boundary (APEX_TEST_MODEL)", "[inference]") {
    const char* path = std::getenv("APEX_TEST_MODEL");
    if (path == nullptr || std::string(path).empty()) {
        SUCCEED("No APEX_TEST_MODEL configured; integration test is opt-in");
        return;
    }

    apex::EngineOptions options;
    options.context_size = 16;
    options.gpu_layers = 0;
    options.temperature = 0;
    apex::Engine engine(apex::EngineMode::Real, options);
    REQUIRE(engine.loadModel(path));

    // "A short greeting:" tokenizes to a handful of tokens; max_tokens is
    // chosen so prompt_tokens + max_tokens lands exactly on context_size,
    // the boundary that must be rejected (last position == context_size,
    // out of the valid 0..context_size-1 range).
    const auto probe = engine.generateDetailed("A short greeting:", 1);
    const int prompt_tokens = probe.stats.prompt_tokens;
    const int max_tokens = static_cast<int>(options.context_size) - prompt_tokens;
    REQUIRE(max_tokens > 0);
    REQUIRE_THROWS_AS(engine.generateDetailed("A short greeting:", max_tokens), std::length_error);
}
