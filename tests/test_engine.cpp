#include <catch2/catch_test_macros.hpp>

#include "apex/engine.h"

#include <stdexcept>
#include <string>

TEST_CASE("Real inference is not silently simulated", "[engine]") {
    apex::Engine engine;
    REQUIRE_FALSE(engine.isLoaded());
    REQUIRE_THROWS_AS(engine.generate("Hello"), std::logic_error);
    REQUIRE_THROWS_AS(engine.loadModel(""), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.loadModel("missing-phase-1-model.gguf"), std::runtime_error);
    // This source file exists at test time, but is not an LLM model.
    REQUIRE_THROWS(engine.loadModel(__FILE__));
    REQUIRE_FALSE(engine.isLoaded());
}

TEST_CASE("Mock mode must be selected and identified explicitly", "[engine]") {
    apex::Engine engine(apex::EngineMode::Mock);
    REQUIRE_THROWS_AS(engine.loadModel("test_model.gguf"), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.loadModel("mock://"), std::invalid_argument);
    REQUIRE(engine.loadModel("mock://unit-test"));
    REQUIRE(engine.isLoaded());

    const std::string output = engine.generate("Hello", 10);
    REQUIRE(output.find("MOCK OUTPUT - NOT LLM INFERENCE") != std::string::npos);
    REQUIRE(output.find("Hello") != std::string::npos);
    REQUIRE(output.find("max_tokens=10") != std::string::npos);
}

TEST_CASE("Generation validates its input and engine state", "[engine]") {
    apex::Engine engine(apex::EngineMode::Mock);
    REQUIRE_THROWS_AS(engine.generate("Hello"), std::logic_error);
    REQUIRE(engine.loadModel("mock://valid"));
    REQUIRE_THROWS_AS(engine.generate(""), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.generate("Hello", 0), std::invalid_argument);
    REQUIRE_THROWS_AS(engine.generate("Hello", -10), std::invalid_argument);
    REQUIRE_NOTHROW(engine.generate("Hello", 1));
}

TEST_CASE("Failed replacement unloads the prior mock", "[engine]") {
    apex::Engine engine(apex::EngineMode::Mock);
    REQUIRE(engine.loadModel("mock://first"));
    REQUIRE(engine.loadModel("mock://second"));
    REQUIRE_THROWS_AS(engine.loadModel("invalid.gguf"), std::invalid_argument);
    REQUIRE_FALSE(engine.isLoaded());
    REQUIRE_THROWS_AS(engine.generate("Hello"), std::logic_error);

    REQUIRE(engine.loadModel("mock://third"));
    engine.unloadModel();
    REQUIRE_FALSE(engine.isLoaded());
    engine.unloadModel();
    REQUIRE_FALSE(engine.isLoaded());
}

TEST_CASE("Engine options are validated before loading", "[engine]") {
    apex::EngineOptions options;
    options.context_size = 8;
    REQUIRE_THROWS_AS(apex::Engine(apex::EngineMode::Mock, options), std::invalid_argument);

    options = {};
    options.gpu_layers = -1;
    REQUIRE_THROWS_AS(apex::Engine(apex::EngineMode::Real, options), std::invalid_argument);

    options = {};
    options.temperature = -0.5f;
    REQUIRE_THROWS_AS(apex::Engine(apex::EngineMode::Mock, options), std::invalid_argument);

    options = {};
    options.top_p = 0.0f;
    REQUIRE_THROWS_AS(apex::Engine(apex::EngineMode::Mock, options), std::invalid_argument);

    options = {};
    options.top_k = -1;
    REQUIRE_THROWS_AS(apex::Engine(apex::EngineMode::Mock, options), std::invalid_argument);
}
