cmake_minimum_required(VERSION 3.16)
project(kifu_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

add_library(kifu STATIC
    src/pipeline.cpp
)

target_include_directories(kifu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(kifu PUBLIC ${OpenCV_LIBS})