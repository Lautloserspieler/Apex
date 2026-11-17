cmake_minimum_required(VERSION 3.21)
project(Apex VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(APEX_ENABLE_LLAMA "Build real GGUF inference via pinned llama.cpp" OFF)
option(APEX_ENABLE_CUDA "Enable CUDA for the llama.cpp backend (requires nvcc)" OFF)

if(APEX_ENABLE_CUDA AND NOT APEX_ENABLE_LLAMA)
    message(FATAL_ERROR "APEX_ENABLE_CUDA requires APEX_ENABLE_LLAMA=ON in Phase 2")
endif()

if(APEX_ENABLE_LLAMA)
    include(FetchContent)
    # Immutable upstream release commit: ggml-org/llama.cpp b12445 (2026-11-16).
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA ${APEX_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    # A commit-pinned source archive avoids a full Git history download.
    # SHA256 verified on GitHub Actions against upstream commit source archive.
    FetchContent_Declare(llama_cpp
        URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/ef3848f9d7d4cf7b6b2abfc3b4c960b55d6d9e91
        URL_HASH SHA256=5dc44685ba4bf265eb9101724b90c06d2c6999ffef6adee7903c39ea337a7f3f
    )
    FetchContent_MakeAvailable(llama_cpp)
endif()

include(CTest)

add_library(apex_engine
    src/engine/engine.cpp
    src/engine/llama_backend.cpp
)
target_include_directories(apex_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(apex_engine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/engine")

if(APEX_ENABLE_LLAMA)
    target_compile_definitions(apex_engine PRIVATE APEX_WITH_LLAMA=1)
    target_link_libraries(apex_engine PRIVATE llama)
endif()

if(MSVC)
    target_compile_options(apex_engine PRIVATE /W4 /utf-8)
else()
    target_compile_options(apex_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(apex src/main.cpp)
target_link_libraries(apex PRIVATE apex_engine)
set_target_properties(apex PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

message(STATUS "Apex real GGUF backend: ${APEX_ENABLE_LLAMA}")
message(STATUS "Apex CUDA backend: ${APEX_ENABLE_CUDA}")
