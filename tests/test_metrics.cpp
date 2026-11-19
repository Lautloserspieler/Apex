#include <catch2/catch_test_macros.hpp>
#include "apex/engine.h"
#include "apex/metrics.h"

#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}
} // namespace

TEST_CASE("JSON escaping covers quotes, backslashes and control bytes", "[metrics]") {
    REQUIRE(apex::json_escape("plain") == "plain");
    REQUIRE(apex::json_escape("a\"b\\c") == "a\\\"b\\\\c");
    REQUIRE(apex::json_escape("line\nnext\ttab\r") == "line\\nnext\\ttab\\r");
    REQUIRE(apex::json_escape(std::string("\x01", 1)) == "\\u0001");
    // UTF-8 (e.g. a Windows path with an umlaut) passes through unchanged.
    REQUIRE(apex::json_escape("C:\\Modelle\\gr\xc3\xb6\xc3\x9f" "e.gguf") ==
            "C:\\\\Modelle\\\\gr\xc3\xb6\xc3\x9f" "e.gguf");
}

TEST_CASE("Build info states which backend produced a measurement", "[metrics]") {
    const auto info = apex::build_info();
    REQUIRE_FALSE(info.apex_version.empty());
    REQUIRE_FALSE(info.apex_git_revision.empty());
    REQUIRE_FALSE(info.compiler.empty());
#if defined(APEX_TEST_EXPECT_LLAMA)
    REQUIRE(info.llama_backend);
#endif
    if (!info.llama_backend) {
        REQUIRE(info.llama_cpp_revision == "disabled");
        REQUIRE_FALSE(info.cuda_backend);
    }
    if (!info.cuda_backend) {
        REQUIRE_FALSE(info.cuda_graphs);
    }
}

TEST_CASE("Peak RSS is reported on supported platforms", "[metrics]") {
#if defined(_WIN32) || defined(__linux__) || defined(__APPLE__)
    const double rss = apex::process_peak_rss_mib();
    REQUIRE(rss > 0.0);
    REQUIRE(rss < 1024.0 * 1024.0); // sanity: below 1 TiB
#else
    SUCCEED("Peak RSS not implemented on this platform");
#endif
}

TEST_CASE("Metrics report serializes runs, phases and token IDs", "[metrics]") {
    apex::MetricsReport report;
    report.build = apex::build_info();
    report.model_path = "models/a\"b.gguf";
    report.options.context_size = 256;
    report.options.threads = 4;
    report.max_tokens = 3;
    report.prompt_bytes = 16;
    report.requested_runs = 2;

    for (int index = 0; index < 2; ++index) {
        apex::MetricsRun run;
        run.index = index;
        run.result.generated_token_ids = {11, 22, 33};
        run.result.stats.prompt_tokens = 5;
        run.result.stats.generated_tokens = 3;
        run.result.stats.stop_reason = "length";
        run.result.stats.decode_seconds = 0.5;
        run.result.stats.request_seconds = 0.75;
        run.result.stats.peak_vram_mib = std::numeric_limits<double>::quiet_NaN();
        report.runs.push_back(run);
    }

    std::ostringstream out;
    apex::write_metrics_json(out, report);
    const std::string json = out.str();

    REQUIRE(contains(json, "\"schema\":\"apex.metrics/1\""));
    REQUIRE(contains(json, "\"model_path\":\"models/a\\\"b.gguf\""));
    REQUIRE(contains(json, "\"context_size\":256"));
    REQUIRE(contains(json, "\"threads\":4"));
    REQUIRE(contains(json, "\"phase\":\"cold\""));
    REQUIRE(contains(json, "\"phase\":\"warm\""));
    REQUIRE(contains(json, "\"token_ids\":[11,22,33]"));
    REQUIRE(contains(json, "\"request_seconds\":0.75"));
    // Non-finite values must not produce invalid JSON.
    REQUIRE(contains(json, "\"peak_vram_mib\":null"));
    REQUIRE_FALSE(contains(json, "nan"));
    REQUIRE_FALSE(contains(json, "\"error\""));

    report.error = "boom";
    std::ostringstream failed;
    apex::write_metrics_json(failed, report);
    REQUIRE(contains(failed.str(), "\"error\":\"boom\""));
}

TEST_CASE("Negative thread counts are rejected", "[metrics]") {
    apex::EngineOptions options;
    options.threads = -1;
    REQUIRE_THROWS_AS(apex::Engine(apex::EngineMode::Mock, options), std::invalid_argument);
    options.threads = 0;
    REQUIRE_NOTHROW(apex::Engine(apex::EngineMode::Mock, options));
}
