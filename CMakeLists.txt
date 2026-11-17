cmake_minimum_required(VERSION 3.18)
project(Apex LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

enable_testing()

# Placeholder – will be expanded as the project grows
message(STATUS "Apex: High-performance inference engine")

add_subdirectory(tests)
