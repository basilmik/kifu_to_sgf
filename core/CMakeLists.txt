cmake_minimum_required(VERSION 3.16)
project(kifu_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # нужно для .so и pybind11

add_library(kifu STATIC
    src/pipeline.cpp
)

target_include_directories(kifu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

