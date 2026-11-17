#include <catch2/catch_test_macros.hpp>

TEST_CASE("Sanity check", "[core]") {
    REQUIRE(1 + 1 == 2);
}

TEST_CASE("Apex placeholder", "[core]") {
    // Placeholder – will be replaced with real tests as the engine grows
    REQUIRE(true);
}
