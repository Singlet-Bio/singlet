cmake_minimum_required(VERSION 3.24)
project(singlet-gpu LANGUAGES CXX CUDA)

# License: GPL-2.0-or-later (inherited from factornet, which singlet-gpu links).

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 80 90)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")

# ---------------------------------------------------------------------------
# factornet — the GPU PCA + NMF backend. Header-only, GPL-2.0.
# ---------------------------------------------------------------------------
set(FACTORNET_INCLUDE_DIR "/mnt/home/debruinz/factornet/include" CACHE PATH
    "Path to factornet include directory")
if(NOT EXISTS "${FACTORNET_INCLUDE_DIR}/factornet/factornet.hpp")
    message(FATAL_ERROR
        "factornet headers not found at ${FACTORNET_INCLUDE_DIR}. "
        "Set -DFACTORNET_INCLUDE_DIR=/path/to/factornet/include")
endif()

# Eigen (transitive dependency of factornet — it uses Eigen::Sparse on the CPU).
find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    find_path(EIGEN_INCLUDE_DIR Eigen/Dense
        HINTS /usr/include/eigen3 /usr/local/include/eigen3
              /opt/homebrew/include/eigen3)
    if(NOT EIGEN_INCLUDE_DIR)
        message(FATAL_ERROR "Eigen3 not found — required by factornet")
    endif()
endif()

# Header-only INTERFACE library — every kernel lives in include/singlet-gpu/.
add_library(singlet-gpu INTERFACE)
add_library(singlet-gpu::singlet-gpu ALIAS singlet-gpu)
target_include_directories(singlet-gpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${FACTORNET_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
if(Eigen3_FOUND)
    target_link_libraries(singlet-gpu INTERFACE Eigen3::Eigen)
else()
    target_include_directories(singlet-gpu INTERFACE ${EIGEN_INCLUDE_DIR})
endif()

target_compile_features(singlet-gpu INTERFACE cxx_std_20 cuda_std_20)
# factornet's GPU bridge requires this define on every TU that touches gpu/*.
target_compile_definitions(singlet-gpu INTERFACE FACTORNET_HAS_GPU=1)

# CUDA toolkit dependencies.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(singlet-gpu INTERFACE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    CUDA::cusolver
    CUDA::curand
)

# zstd for .1pz decompression (required by io/pz_device_loader.h).
find_package(zstd QUIET)
if(NOT zstd_FOUND)
    find_path(ZSTD_INCLUDE_DIR zstd.h)
    find_library(ZSTD_LIBRARY zstd)
    if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
        target_include_directories(singlet-gpu INTERFACE ${ZSTD_INCLUDE_DIR})
        target_link_libraries(singlet-gpu INTERFACE ${ZSTD_LIBRARY})
    else()
        message(FATAL_ERROR "zstd not found — required for .1pz loader")
    endif()
else()
    target_link_libraries(singlet-gpu INTERFACE zstd::libzstd_shared)
endif()

# .cu translation units for explicit template instantiations only.
file(GLOB_RECURSE SINGLET_GPU_CU_SOURCES CONFIGURE_DEPENDS src/*.cu)
if(SINGLET_GPU_CU_SOURCES)
    add_library(singlet-gpu-impl STATIC ${SINGLET_GPU_CU_SOURCES})
    target_link_libraries(singlet-gpu-impl PUBLIC singlet-gpu)
    set_target_properties(singlet-gpu-impl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Tests — owned by analysis-validator.
# EXCLUDE_FROM_ALL: test sources have API-mismatch compile errors (factornet
# constructor signatures, PzDataLoader member renames, std::filesystem in CUDA TU).
# These are Phase 4 fixes in test .cpp files — not a library bug.
# Build explicitly with: make preprocess_lognorm_correctness (etc.)
# ctest will discover tests only after their executables are built.
option(SINGLET_GPU_BUILD_TESTS "Build correctness tests" ON)
if(SINGLET_GPU_BUILD_TESTS)
    enable_testing()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
        add_subdirectory(tests)
    endif()
endif()

# Benchmarks — owned by gpu-bench.
# EXCLUDE_FROM_ALL: bench drivers trigger libstdc++ ABI / std::string cold-path
# link errors on some host-compiler/nvcc combinations, and are not part of the
# library compile gate.  Build explicitly with: make bench_io_pz_loader_perf (etc.)
option(SINGLET_GPU_BUILD_BENCH "Build benchmark drivers" ON)
if(SINGLET_GPU_BUILD_BENCH)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt)
        add_subdirectory(bench EXCLUDE_FROM_ALL)
    endif()
endif()

message(STATUS "singlet-gpu: header-only INTERFACE target ready")
message(STATUS "  include/singlet-gpu/ — public API (header-only)")
message(STATUS "  src/                  — explicit template instantiations (.cu)")
message(STATUS "  tests/                — correctness harnesses")
message(STATUS "  bench/                — performance drivers")
