# SPDX-License-Identifier: GPL-2.0-or-later
# examples/cpp_minimal/CMakeLists.txt
#
# Standalone smoke test for the public umbrella header. Built as part of the
# main project when SINGLET_GPU_BUILD_EXAMPLES=ON, OR independently via
# FetchContent / find_package by downstream users (this file uses both
# patterns so it can serve as a reference).

cmake_minimum_required(VERSION 3.24)
project(singlet_gpu_cpp_minimal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# In-tree build: the parent CMakeLists.txt has already declared the
# `singlet-gpu::singlet-gpu` INTERFACE target. Standalone build: locate it.
if(NOT TARGET singlet-gpu::singlet-gpu)
    find_package(singlet-gpu REQUIRED)
endif()

add_executable(cpp_minimal main.cpp)
target_link_libraries(cpp_minimal PRIVATE singlet-gpu::singlet-gpu)
set_target_properties(cpp_minimal PROPERTIES CUDA_ARCHITECTURES "70;80;90")
