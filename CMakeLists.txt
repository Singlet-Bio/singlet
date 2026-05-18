cmake_minimum_required(VERSION 3.14)
project(singlet VERSION 2.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────────────────────────
option(SINGLET_BUILD_GPU      "Build GPU kernels (requires CUDA 12+)" OFF)
option(SINGLET_BUILD_PIPELINE "Build singlet binary"                 OFF)
option(SINGLET_BUILD_PYTHON   "Build Python bindings"                 OFF)
option(SINGLET_BUILD_TESTS    "Build test suite"                      OFF)

# ─── Dependencies ─────────────────────────────────────────────────────────────
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

# zstd (required by pz and fq codecs)
find_library(ZSTD_LIBRARY NAMES zstd)
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h)
if(NOT ZSTD_LIBRARY)
    message(FATAL_ERROR "zstd not found. Install libzstd-dev or set ZSTD_LIBRARY.")
endif()

# ─── Header-only libraries (always available) ─────────────────────────────────

# singlet::pz — .1pz sparse matrix codec
add_library(singlet_pz INTERFACE)
add_library(Singlet::pz ALIAS singlet_pz)
target_include_directories(singlet_pz INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(singlet_pz INTERFACE ${ZSTD_LIBRARY})

# singlet::fq — .1fq encoded FASTQ codec
add_library(singlet_fq INTERFACE)
add_library(Singlet::fq ALIAS singlet_fq)
target_include_directories(singlet_fq INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(singlet_fq INTERFACE ${ZSTD_LIBRARY} ZLIB::ZLIB)

# singlet::pileup — streaming BAM pileup engine
add_library(singlet_pileup INTERFACE)
add_library(Singlet::pileup ALIAS singlet_pileup)
target_include_directories(singlet_pileup INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(singlet_pileup INTERFACE singlet_pz singlet_fq Threads::Threads)

# ─── Pipeline binary (optional) ──────────────────────────────────────────────
if(SINGLET_BUILD_PIPELINE)
    find_package(OpenMP REQUIRED)
    include(FindPkgConfig)

    # NCBI VDB library for native SRA reading
    find_library(NCBI_VDB_LIB ncbi-vdb
        HINTS $ENV{CONDA_PREFIX}/lib ENV LD_LIBRARY_PATH)
    find_path(NCBI_VDB_INCLUDE vdb/manager.h
        HINTS $ENV{CONDA_PREFIX}/include)
    if(NOT NCBI_VDB_LIB OR NOT NCBI_VDB_INCLUDE)
        message(FATAL_ERROR
            "libncbi-vdb not found. Install sra-tools or set CONDA_PREFIX.\n"
            "  conda install -c bioconda sra-tools")
    endif()
    message(STATUS "Found ncbi-vdb: ${NCBI_VDB_LIB}")
    message(STATUS "Found ncbi-vdb headers: ${NCBI_VDB_INCLUDE}")

    # htslib (required for BAM/CRAM I/O)
    pkg_check_modules(HTSLIB REQUIRED htslib>=1.10)

    # HDF5 C library (for h5ad/loom export)
    find_library(HDF5_C_LIB hdf5
        HINTS $ENV{CONDA_PREFIX}/lib /usr/lib64)
    find_path(HDF5_C_INCLUDE hdf5.h
        HINTS $ENV{CONDA_PREFIX}/include /usr/include)
    if(HDF5_C_LIB)
        set(HDF5_AVAILABLE TRUE)
        message(STATUS "Found HDF5: ${HDF5_C_LIB}")
    else()
        set(HDF5_AVAILABLE FALSE)
        message(WARNING "HDF5 not found — h5ad/loom export disabled")
    endif()

    add_subdirectory(src/star)
    add_subdirectory(src/pipeline)
endif()

# ─── GPU library (optional, requires CUDA) ───────────────────────────────────
# Integrated from former standalone singlet-gpu repo (2026-05-13).
# Header-only INTERFACE library — every kernel lives in include/singlet/gpu/.
if(SINGLET_BUILD_GPU)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 80 90)
    endif()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")

    find_package(CUDAToolkit REQUIRED)

    # singlet::gpu — header-only INTERFACE target
    add_library(singlet_gpu INTERFACE)
    add_library(Singlet::gpu ALIAS singlet_gpu)
    target_include_directories(singlet_gpu INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_compile_features(singlet_gpu INTERFACE cxx_std_20 cuda_std_20)
    target_link_libraries(singlet_gpu INTERFACE
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::cusolver
        CUDA::curand
        ${ZSTD_LIBRARY}
    )
    target_include_directories(singlet_gpu INTERFACE ${ZSTD_INCLUDE_DIR})

    # Optional .cu translation units (template instantiations, kernel helpers)
    file(GLOB_RECURSE SINGLET_GPU_CU_SOURCES CONFIGURE_DEPENDS src/gpu/*.cu)
    if(SINGLET_GPU_CU_SOURCES)
        add_library(singlet_gpu_impl STATIC ${SINGLET_GPU_CU_SOURCES})
        target_link_libraries(singlet_gpu_impl PUBLIC singlet_gpu)
        set_target_properties(singlet_gpu_impl PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
        )
    endif()

    # Benchmarks — owned by gpu-bench. Built only when SINGLET_BUILD_GPU=ON
    # (drivers stay .cpp; kernel-launching helpers live in bench/common/*.cu).
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench/CMakeLists.txt)
        add_subdirectory(bench EXCLUDE_FROM_ALL)
    endif()
endif()

# ─── Python bindings (optional) ──────────────────────────────────────────────
if(SINGLET_BUILD_PYTHON)
    find_package(pybind11 REQUIRED)
    add_subdirectory(src/bindings/python)
endif()

# ─── Tests ───────────────────────────────────────────────────────────────────
if(SINGLET_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()

# ─── Install targets ─────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/singlet DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS singlet_pz singlet_fq singlet_pileup
    EXPORT SingletTargets
)
install(EXPORT SingletTargets
    NAMESPACE Singlet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Singlet
)

configure_package_config_file(
    cmake/SingletConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/SingletConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Singlet
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SingletConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SingletConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SingletConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Singlet
)
