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

    // "A short greeting:" tokenizes to a handful of tokens. The last
    // generated token sits at position prompt_tokens + max_tokens - 1, so
    // +1 beyond context_size - prompt_tokens makes that position land
    // exactly on context_size, the boundary that must be rejected (valid
    // positions are 0..context_size-1).
    const auto probe = engine.generateDetailed("A short greeting:", 1);
    const int prompt_tokens = probe.stats.prompt_tokens;
    const int max_tokens = static_cast<int>(options.context_size) - prompt_tokens + 1;
    REQUIRE(max_tokens > 0);
    REQUIRE_THROWS_AS(engine.generateDetailed("A short greeting:", max_tokens), std::length_error);
}

TEST_CASE("Phase 3 timing breakdown is consistent (APEX_TEST_MODEL)", "[inference]") {
    const char* path = std::getenv("APEX_TEST_MODEL");
    if (path == nullptr || std::string(path).empty()) {
        SUCCEED("No APEX_TEST_MODEL configured; integration test is opt-in");
        return;
    }

    apex::EngineOptions options;
    options.context_size = 256;
    options.gpu_layers = 0;
    options.temperature = 0;
    apex::Engine reference_engine(apex::EngineMode::Real, options);
    REQUIRE(reference_engine.loadModel(path));
    const auto reference = reference_engine.generateDetailed("Once upon a time", 8);

    // A pinned thread count must not change greedy token output.
    options.threads = 1;
    apex::Engine engine(apex::EngineMode::Real, options);
    REQUIRE(engine.loadModel(path));
    const auto result = engine.generateDetailed("Once upon a time", 8);
    REQUIRE(result.generated_token_ids == reference.generated_token_ids);

    const auto& s = result.stats;
    REQUIRE(s.tokenize_seconds >= 0.0);
    REQUIRE(s.context_setup_seconds > 0.0);
    REQUIRE(s.prefill_seconds > 0.0);
    REQUIRE(s.ttft_seconds >= s.prefill_seconds);
    REQUIRE(s.request_seconds > 0.0);
    // The parts are measured on disjoint, consecutive intervals of the same
    // generate() call, so together they cannot exceed the whole request.
    REQUIRE(s.tokenize_seconds + s.context_setup_seconds + s.ttft_seconds +
                s.decode_seconds <= s.request_seconds + 1e-6);
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
    REQUIRE(s.peak_rss_mib > 0.0);
#endif
    // CPU-only run: GPU diagnostics stay "not measured".
    REQUIRE(s.peak_vram_mib == 0.0);
    REQUIRE(s.avg_power_watts == 0.0);
}
