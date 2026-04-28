# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/python/CMakeLists.txt
#
# Delegates to the parent singlet-gpu CMakeLists.txt, then adds the pybind11
# extension module target. Intended to be the CMake root when scikit-build-core
# drives the wheel build (cmake.source-dir = "..").
#
# Activated only when SINGLET_GPU_BUILD_PYTHON=ON (set by pyproject.toml's
# cmake.args).  All C++ kernel / test / bench targets from the parent are
# suppressed (SINGLET_GPU_BUILD_TESTS=OFF, SINGLET_GPU_BUILD_BENCH=OFF).

cmake_minimum_required(VERSION 3.24)

# scikit-build-core sets the project version from pyproject.toml.
project(singlet_gpu_python VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80 90)

# ---------------------------------------------------------------------------
# Pull in the parent singlet-gpu INTERFACE library.
# When invoked by scikit-build-core, the source root is singlet-gpu/ (the
# parent of python/).  If invoked standalone (e.g. manual cmake -S . -B build
# from python/), use CMAKE_CURRENT_SOURCE_DIR/../.
# ---------------------------------------------------------------------------
option(SINGLET_GPU_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(SINGLET_GPU_BUILD_TESTS  "Build C++ correctness tests"          OFF)
option(SINGLET_GPU_BUILD_BENCH  "Build C++ benchmark drivers"          OFF)

# Resolve parent source dir — works whether called from python/ directly or
# from the repo root with -DSINGLET_GPU_BUILD_PYTHON=ON.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" AND
   NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    # Called from python/ directly (standalone build).
    set(_PARENT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/..")
else()
    # scikit-build-core sets CMAKE_SOURCE_DIR to singlet-gpu/; python/ is a subdir.
    set(_PARENT_SRC "${CMAKE_SOURCE_DIR}")
endif()

add_subdirectory("${_PARENT_SRC}" singlet_gpu_parent EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# pybind11
# ---------------------------------------------------------------------------
find_package(pybind11 2.11 REQUIRED)

# ---------------------------------------------------------------------------
# _core extension module
# ---------------------------------------------------------------------------
pybind11_add_module(_core MODULE
    src/_singlet_gpu_core.cpp
)

# Inherit singlet-gpu headers, factornet, CUDA, zstd from the INTERFACE target.
target_link_libraries(_core PRIVATE singlet-gpu::singlet-gpu)

# pybind11 best practice: hide all symbols except the module init.
target_compile_options(_core PRIVATE -fvisibility=hidden)

# The module init function name must match the pybind11_add_module target name.
# Output file: _core.cpython-3xx-…so → installed into singlet_gpu/.
set_target_properties(_core PROPERTIES
    OUTPUT_NAME "_core"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/singlet_gpu"
)

# Install the .so into the singlet_gpu package directory inside the wheel.
install(TARGETS _core
    COMPONENT python
    LIBRARY DESTINATION singlet_gpu
)

# Install the pure-Python package alongside it.
install(DIRECTORY singlet_gpu/
    COMPONENT python
    DESTINATION singlet_gpu
    FILES_MATCHING PATTERN "*.py"
)
