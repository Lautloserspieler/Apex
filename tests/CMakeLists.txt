# Prefer an already installed Catch2 package; only fetch it if necessary.
find_package(Catch2 3 QUIET CONFIG)

if(Catch2_FOUND)
    # Catch2's CMake test-discovery helper is shipped next to Catch2Config.cmake.
    list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
else()
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    # FetchContent does not automatically install Catch.cmake on CMAKE_MODULE_PATH.
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()

add_executable(apex_tests test_main.cpp test_engine.cpp test_inference.cpp test_metrics.cpp)
target_link_libraries(apex_tests PRIVATE Catch2::Catch2WithMain apex_engine)
if(APEX_ENABLE_LLAMA)
    target_compile_definitions(apex_tests PRIVATE APEX_TEST_EXPECT_LLAMA=1)
endif()

include(Catch)
catch_discover_tests(apex_tests)

add_test(NAME apex_cli_mock_smoke
    COMMAND $<TARGET_FILE:apex> --mock --prompt "Hello, Apex!" --max-tokens 8)
set_tests_properties(apex_cli_mock_smoke
    PROPERTIES PASS_REGULAR_EXPRESSION "MOCK OUTPUT - NOT LLM INFERENCE")

add_test(NAME apex_cli_help COMMAND $<TARGET_FILE:apex> --help)
set_tests_properties(apex_cli_help
    PROPERTIES PASS_REGULAR_EXPRESSION "--model FILE")

add_test(NAME apex_missing_model
    COMMAND $<TARGET_FILE:apex> --model apex-nonexistent-model.gguf --prompt "hello")
set_tests_properties(apex_missing_model PROPERTIES WILL_FAIL TRUE)

# Mock output must never be recorded as a benchmark measurement.
add_test(NAME apex_mock_rejects_metrics_json
    COMMAND $<TARGET_FILE:apex> --mock --prompt "hello" --metrics-json mock-metrics.json)
set_tests_properties(apex_mock_rejects_metrics_json PROPERTIES WILL_FAIL TRUE)

add_test(NAME apex_rejects_zero_repeat
    COMMAND $<TARGET_FILE:apex> --mock --prompt "hello" --repeat 0)
set_tests_properties(apex_rejects_zero_repeat PROPERTIES WILL_FAIL TRUE)

add_test(NAME apex_rejects_negative_threads
    COMMAND $<TARGET_FILE:apex> --mock --prompt "hello" --threads -2)
set_tests_properties(apex_rejects_negative_threads PROPERTIES WILL_FAIL TRUE)

add_test(NAME apex_mock_repeat_threads
    COMMAND $<TARGET_FILE:apex> --mock --prompt "hello" --repeat 3 --threads 2)
set_tests_properties(apex_mock_repeat_threads
    PROPERTIES PASS_REGULAR_EXPRESSION "MOCK OUTPUT - NOT LLM INFERENCE")
