cmake_minimum_required(VERSION 3.14)
project(singlet VERSION 1.0.0 LANGUAGES CXX C)

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

# singlet::pz2 — store sparse-matrix codec (header-only C++17; SIMD paths are runtime-dispatched, so NO -march flags are needed)
add_library(singlet_pz2 INTERFACE)
add_library(Singlet::pz2 ALIAS singlet_pz2)
target_include_directories(singlet_pz2 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(singlet_pz2 INTERFACE ${ZSTD_LIBRARY} Threads::Threads)

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

# ─── singlet-stage: fetch-tier encoder .1fq -> .1fg2 (src/stage; docs/research-beta/download-and-staging.md). Engine headers + libzstd only. ───
add_executable(singlet-stage src/stage/singlet_stage.cpp)
target_include_directories(singlet-stage PRIVATE src/stage)
target_link_libraries(singlet-stage PRIVATE singlet_fq Threads::Threads)
target_compile_options(singlet-stage PRIVATE -msse4.2)
add_executable(test_fg2_runset src/stage/test_fg2_runset.cpp)   # the gate for include/singlet/fq/fg2_runset.h against the .1fq RunSet
target_link_libraries(test_fg2_runset PRIVATE singlet_fq Threads::Threads)
target_compile_options(test_fg2_runset PRIVATE -msse4.2)

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

    # GPU engine (`singlet --engine gpu`, docs/engine-gpu.md): STAR's mapping of each distinct cDNA sequence on the GPU, into
    # STAR's sequence cache; every other stage is the CPU engine's code. AUTO builds it when a CUDA compiler is found.
    set(SINGLET_ENGINE_GPU "AUTO" CACHE STRING "GPU engine for singlet --engine gpu: AUTO, ON or OFF")
    set(SINGLET_ENGINE_GPU_ENABLED OFF)
    if(NOT SINGLET_ENGINE_GPU STREQUAL "OFF")
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            find_package(CUDAToolkit REQUIRED)
            set(SINGLET_ENGINE_GPU_ENABLED ON)
            message(STATUS "GPU engine: ON (${CMAKE_CUDA_COMPILER})")
        elseif(SINGLET_ENGINE_GPU STREQUAL "ON")
            message(FATAL_ERROR "SINGLET_ENGINE_GPU=ON but no CUDA compiler was found (set CMAKE_CUDA_COMPILER)")
        else()
            message(STATUS "GPU engine: OFF (no CUDA compiler; singlet --engine gpu falls back to the CPU engine)")
        endif()
    endif()

    add_subdirectory(src/star)
    if(SINGLET_ENGINE_GPU_ENABLED)
        add_library(singlet_engine_gpu STATIC src/engine_gpu/engine_gpu.cu src/engine_gpu/nonhost_gpu.cu src/engine_gpu/pileup_gpu.cu src/engine_gpu/records_gpu.cu src/engine_gpu/hostloop_gpu.cu
                    src/engine_gpu/solo_gpu.cu)
        # E11 (docs/gpu-engine/fable/E11-solowire.md): E9's Solo library runs its rules through __host__ __device__ lambdas, on the host with OpenMP
        set_source_files_properties(src/engine_gpu/solo_gpu.cu PROPERTIES COMPILE_OPTIONS "--extended-lambda;-Xcompiler=-fopenmp")
        set_target_properties(singlet_engine_gpu PROPERTIES
            CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
            set_target_properties(singlet_engine_gpu PROPERTIES CUDA_ARCHITECTURES "80;90")
        endif()
        # no fused multiply-add: the port's double arithmetic must round as STAR's does on the host
        target_compile_options(singlet_engine_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -fmad=false>)
        target_include_directories(singlet_engine_gpu PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/engine_gpu)
        target_link_libraries(singlet_engine_gpu PUBLIC CUDA::cudart_static Threads::Threads PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(singlet_star PRIVATE SINGLET_HAVE_ENGINE_GPU)
        target_include_directories(singlet_star PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/engine_gpu)
    endif()
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
install(TARGETS singlet_pz singlet_pz2 singlet_fq singlet_pileup
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
