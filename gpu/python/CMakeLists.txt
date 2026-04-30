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

# Pull in the parent singlet-gpu INTERFACE library if it isn't already present.
# Three invocation paths must all work:
#   (1) Standalone: cmake -S python -B build  →  target doesn't exist, pull parent in.
#   (2) Parent-driven: cmake -S . -B build -DSINGLET_GPU_BUILD_PYTHON=ON
#       and parent CMakeLists has add_subdirectory(python)  →  target already
#       exists by the time the wrapper runs; skip the redundant add.
#   (3) scikit-build-core (default `cmake.source-dir = "python"`):
#       cmake source root is python/; target doesn't exist; pull parent in via
#       a relative path. cmake supports add_subdirectory pointing outside the
#       cmake source tree; we name the binary subdir explicitly to avoid the
#       "directory must be a subdir of CMAKE_SOURCE_DIR" warning.
#
# Earlier versions of this file derived _PARENT_SRC from CMAKE_SOURCE_DIR. In
# scikit-build-core's default config CMAKE_SOURCE_DIR == CMAKE_CURRENT_SOURCE_DIR
# (both point at python/), so add_subdirectory(${CMAKE_SOURCE_DIR}) recursed
# infinitely. Fixed by using CMAKE_CURRENT_SOURCE_DIR/.. unconditionally and
# guarding with the target check.
if(NOT TARGET singlet-gpu::singlet-gpu)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/.."
        "${CMAKE_BINARY_DIR}/singlet_gpu_parent"
        EXCLUDE_FROM_ALL
    )
endif()

# ---------------------------------------------------------------------------
# pybind11
# ---------------------------------------------------------------------------
find_package(pybind11 2.11 REQUIRED)

# CYCLE-110: anno/reference_map.h has an inline NPZ parser that calls
# zlib's inflate() to decompress the DEFLATE-compressed npy entries inside
# .npz archives. The parent INTERFACE target doesn't link zlib because
# the foundational kernels don't need it; the binding TU does.
find_package(ZLIB REQUIRED)

# ---------------------------------------------------------------------------
# _core extension module
# ---------------------------------------------------------------------------
pybind11_add_module(_core MODULE
    src/_singlet_gpu_core.cpp
)

# CYCLE-107: the umbrella + module headers contain `__global__` <<< >>> kernel
# launch syntax inline. Compile the pybind11 entry TU as CUDA so nvcc parses it.
# Without this, host g++ chokes on the launch syntax (same Cycle-92 lesson at
# the binding layer).
set_source_files_properties(src/_singlet_gpu_core.cpp PROPERTIES LANGUAGE CUDA)
set_target_properties(_core PROPERTIES CUDA_ARCHITECTURES "70;80;90")

# Inherit singlet-gpu headers + CUDA + zstd from the INTERFACE target.
target_link_libraries(_core PRIVATE singlet-gpu::singlet-gpu ZLIB::ZLIB)

# pybind11 best practice: hide all symbols except the module init.
# Use device-aware visibility flag so nvcc threads it through to host pass.
target_compile_options(_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
)

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
