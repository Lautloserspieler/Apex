include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2
)

FetchContent_MakeAvailable(Catch2)

add_executable(apex_tests
  test_main.cpp
)

target_link_libraries(apex_tests PRIVATE Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(apex_tests)
