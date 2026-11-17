# Prefer an already installed Catch2 package; only fetch it if necessary.
find_package(Catch2 3 QUIET CONFIG)

if(NOT TARGET Catch2::Catch2WithMain)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.5.2
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()

add_executable(apex_tests test_main.cpp test_engine.cpp)
target_link_libraries(apex_tests PRIVATE Catch2::Catch2WithMain apex_engine)

include(Catch)
catch_discover_tests(apex_tests)

add_test(NAME apex_cli_mock_smoke
    COMMAND $<TARGET_FILE:apex> --mock --prompt "Hello, Apex!" --max-tokens 8)
set_tests_properties(apex_cli_mock_smoke
    PROPERTIES PASS_REGULAR_EXPRESSION "MOCK OUTPUT - NOT LLM INFERENCE")

add_test(NAME apex_cli_help COMMAND $<TARGET_FILE:apex> --help)
set_tests_properties(apex_cli_help
    PROPERTIES PASS_REGULAR_EXPRESSION "no real LLM inference")
