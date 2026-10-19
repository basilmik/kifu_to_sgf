cmake_minimum_required(VERSION 3.16)
project(kifu_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Подключаем core
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../core
                 ${CMAKE_CURRENT_BINARY_DIR}/core_build)

# pybind11
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_kifu_core
    bindings.cpp
)
target_link_libraries(_kifu_core PRIVATE kifu)

# Кладём .so рядом с main.py — чтобы import _kifu_core работал без установки
set_target_properties(_kifu_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)