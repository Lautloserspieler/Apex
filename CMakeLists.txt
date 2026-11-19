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
    if(APEX_ENABLE_CUDA)
        # Phase 3: explicitly request ggml-cuda's own CUDA graph capture
        # rather than relying on upstream's current default (it has varied
        # by version). This targets decode, where Apex's autoregressive loop
        # reuses one llama_context with a stable batch=1 shape per step - the
        # exact case upstream's graph capture is built for. Apex does not
        # implement its own CUDA graph logic; it only requests the pinned
        # backend's existing one. Runtime opt-out: GGML_CUDA_DISABLE_GRAPHS=1.
        set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
    endif()
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    # A commit-pinned source archive avoids a full Git history download.
    # SHA256 verified on GitHub Actions against upstream commit source archive.
    FetchContent_Declare(llama_cpp
        URL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/ef3848f9d7d4cf7b6b2abfc3b4c960b55d6d9e91
        URL_HASH SHA256=5dc44685ba4bf265eb9101724b90c06d2c6999ffef6adee7903c39ea337a7f3f
    )
    FetchContent_MakeAvailable(llama_cpp)
    # llama.cpp b12445 uses u8 literals as const char* in its chat template
    # implementation. MSVC C++20 enables char8_t by default, which makes
    # these upstream calls ill-formed. Scope the compatibility switch to the
    # pinned third-party llama target; Apex itself remains strict C++20.
    if(MSVC)
        target_compile_options(llama PRIVATE /Zc:char8_t-)
    endif()
endif()

include(CTest)

# Build metadata embedded into every --metrics-json record (Phase 3 step 1:
# log commit SHA and backend configuration next to each measurement).
set(APEX_GIT_REVISION "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" describe --always --dirty --abbrev=40
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _apex_git_revision
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _apex_git_result)
    if(_apex_git_result EQUAL 0 AND _apex_git_revision)
        set(APEX_GIT_REVISION "${_apex_git_revision}")
    endif()
endif()
if(APEX_ENABLE_LLAMA)
    set(APEX_LLAMA_CPP_REVISION "b12445/ef3848f9d7d4cf7b6b2abfc3b4c960b55d6d9e91")
    if(FETCHCONTENT_SOURCE_DIR_LLAMA_CPP)
        # A local source tree bypasses the SHA256-verified archive; do not
        # let benchmark records claim the pinned revision unverified.
        string(APPEND APEX_LLAMA_CPP_REVISION " (local source override, unverified)")
    endif()
else()
    set(APEX_LLAMA_CPP_REVISION "disabled")
endif()
if(CMAKE_BUILD_TYPE)
    set(APEX_BUILD_TYPE_NAME "${CMAKE_BUILD_TYPE}")
else()
    set(APEX_BUILD_TYPE_NAME "unspecified")
endif()
if(APEX_ENABLE_CUDA AND GGML_CUDA_GRAPHS)
    set(APEX_CUDA_GRAPHS_FLAG 1)
else()
    set(APEX_CUDA_GRAPHS_FLAG 0)
endif()
configure_file(src/utils/build_info.h.in
    "${CMAKE_CURRENT_BINARY_DIR}/generated/apex_build_info.h" @ONLY)
message(STATUS "Apex Git revision: ${APEX_GIT_REVISION}")

# Direct upstream API reference, kept entirely separate from apex_engine.
# Used only by real-model CI and opt-in Phase 2 correctness comparisons.
if(APEX_ENABLE_LLAMA AND BUILD_TESTING)
    add_executable(apex_llama_reference benchmarks/llama_reference.cpp)
    target_link_libraries(apex_llama_reference PRIVATE llama)
    set_target_properties(apex_llama_reference PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    if(MSVC)
        target_compile_options(apex_llama_reference PRIVATE /W4 /utf-8)
    else()
        target_compile_options(apex_llama_reference PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

add_library(apex_engine
    src/engine/engine.cpp
    src/engine/llama_backend.cpp
    src/utils/metrics.cpp
)
target_include_directories(apex_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(apex_engine PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
if(WIN32)
    # GetProcessMemoryInfo for peak working-set reporting.
    target_link_libraries(apex_engine PRIVATE psapi)
endif()

if(APEX_ENABLE_LLAMA)
    target_compile_definitions(apex_engine PRIVATE APEX_WITH_LLAMA=1)
    target_link_libraries(apex_engine PRIVATE llama)
endif()

if(APEX_ENABLE_CUDA)
    # Gates the nvidia-smi-based VRAM/power sampler in llama_backend.cpp.
    # Separate from APEX_WITH_LLAMA because cpu-inference has LLAMA=ON but
    # CUDA=OFF, and the sampler is meaningless without a GPU.
    target_compile_definitions(apex_engine PRIVATE APEX_WITH_CUDA=1)
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
if(APEX_ENABLE_CUDA)
    message(STATUS "Apex CUDA graphs (ggml-cuda GGML_CUDA_GRAPHS): ${GGML_CUDA_GRAPHS}")
endif()
