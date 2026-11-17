cmake_minimum_required(VERSION 3.18)
project(Apex LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

enable_testing()

message(STATUS "Apex: High-performance inference engine")

# Engine library
add_library(apex_engine
    src/engine/engine.cpp
)

target_include_directories(apex_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Main executable
add_executable(apex src/main.cpp)
target_link_libraries(apex PRIVATE apex_engine)

# Tests
add_subdirectory(tests)
