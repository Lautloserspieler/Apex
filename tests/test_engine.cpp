#include <catch2/catch_test_macros.hpp>
#include "apex/engine.h"

TEST_CASE("Engine loads model", "[engine]") {
    apex::Engine engine;
    REQUIRE(engine.loadModel("test_model.gguf") == true);
    REQUIRE(engine.isLoaded() == true);
}

TEST_CASE("Engine generates text", "[engine]") {
    apex::Engine engine;
    engine.loadModel("test_model.gguf");

    std::string result = engine.generate("Hello");
    REQUIRE(result.find("Hello") != std::string::npos);
}

TEST_CASE("Generate without model fails gracefully", "[engine]") {
    apex::Engine engine;
    std::string result = engine.generate("test");
    REQUIRE(result.find("Error") != std::string::npos);
}
