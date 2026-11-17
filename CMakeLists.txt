cmake_minimum_required(VERSION 3.21)
project(Apex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(APEX_ENABLE_CUDA "Enable the optional CUDA toolchain (kernels arrive in Phase 3)" OFF)

if(APEX_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "APEX_ENABLE_CUDA=ON requires nvcc and a CUDA toolkit. Set CUDAToolkit_ROOT or CMAKE_CUDA_COMPILER.")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include(CTest)

add_library(apex_engine src/engine/engine.cpp)
target_include_directories(apex_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
    target_compile_options(apex_engine PRIVATE /W4)
else()
    target_compile_options(apex_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(apex src/main.cpp)
target_link_libraries(apex PRIVATE apex_engine)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

message(STATUS "Apex Phase 1: C++ mock foundation; real inference is not implemented")
message(STATUS "Apex CUDA toolchain enabled: ${APEX_ENABLE_CUDA}")
