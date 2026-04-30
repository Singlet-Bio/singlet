cmake_minimum_required(VERSION 3.24)
project(singlet-gpu LANGUAGES CXX CUDA)

# License: GPL-2.0-or-later

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
# factornet — OPTIONAL legacy backend (CYCLE-105: no longer required).
# Pass -DFACTORNET_INCLUDE_DIR=/path/to/factornet/include to opt back in.
# Old builds that set this variable will continue to work; new builds simply
# skip it. The factornet INTERFACE include is only added if the directory exists.
# ---------------------------------------------------------------------------
set(FACTORNET_INCLUDE_DIR "" CACHE PATH
    "Optional: path to factornet include directory (not required as of CYCLE-105)")
if(FACTORNET_INCLUDE_DIR AND EXISTS "${FACTORNET_INCLUDE_DIR}")
    message(STATUS "singlet-gpu: factornet include dir found (optional, not required): ${FACTORNET_INCLUDE_DIR}")
else()
    if(FACTORNET_INCLUDE_DIR)
        message(WARNING "singlet-gpu: FACTORNET_INCLUDE_DIR set but not found: ${FACTORNET_INCLUDE_DIR} (ignored)")
    endif()
    set(FACTORNET_INCLUDE_DIR "")
endif()

# Eigen — required only if factornet was also requested (transitive dep of factornet).
# For pure singlet-gpu, Eigen is not needed.
if(FACTORNET_INCLUDE_DIR)
    find_package(Eigen3 QUIET)
    if(NOT Eigen3_FOUND)
        find_path(EIGEN_INCLUDE_DIR Eigen/Dense
            HINTS
                "${FACTORNET_INCLUDE_DIR}/../subprojects/eigen-3.4.0"
                "/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3"
                /usr/include/eigen3 /usr/local/include/eigen3
                /opt/homebrew/include/eigen3)
        if(NOT EIGEN_INCLUDE_DIR)
            message(WARNING "Eigen3 not found — factornet compat mode skipped")
            set(FACTORNET_INCLUDE_DIR "")
        endif()
    endif()
endif()

# Header-only INTERFACE library — every kernel lives in include/singlet-gpu/.
add_library(singlet-gpu INTERFACE)
add_library(singlet-gpu::singlet-gpu ALIAS singlet-gpu)
target_include_directories(singlet-gpu INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Only add factornet include if present (migration safety valve).
if(FACTORNET_INCLUDE_DIR)
    target_include_directories(singlet-gpu INTERFACE
        $<BUILD_INTERFACE:${FACTORNET_INCLUDE_DIR}>)
    target_compile_definitions(singlet-gpu INTERFACE FACTORNET_HAS_GPU=1)
    if(Eigen3_FOUND)
        target_link_libraries(singlet-gpu INTERFACE Eigen3::Eigen)
    elseif(EIGEN_INCLUDE_DIR)
        target_include_directories(singlet-gpu INTERFACE ${EIGEN_INCLUDE_DIR})
    endif()
endif()

target_compile_features(singlet-gpu INTERFACE cxx_std_20 cuda_std_20)

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
option(SINGLET_GPU_BUILD_TESTS "Build correctness tests" ON)
if(SINGLET_GPU_BUILD_TESTS)
    enable_testing()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
        add_subdirectory(tests)
    endif()
endif()

# Benchmarks — owned by gpu-bench.
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
if(FACTORNET_INCLUDE_DIR)
    message(STATUS "  factornet include:    ${FACTORNET_INCLUDE_DIR} (migration compat)")
else()
    message(STATUS "  factornet:            NOT USED (native kernels active, CYCLE-105)")
endif()
