cmake_minimum_required(VERSION 3.14)
project(singlet VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options ──────────────────────────────────────────────────────────────────
option(SINGLET_BUILD_GPU      "Build GPU kernels (requires CUDA 12+)" OFF)
option(SINGLET_BUILD_PIPELINE "Build singlify binary"                 OFF)
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
    add_subdirectory(src/star)
    add_subdirectory(src/pipeline)
endif()

# ─── GPU library (optional, requires CUDA) ───────────────────────────────────
if(SINGLET_BUILD_GPU)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    add_subdirectory(src/gpu)
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
install(DIRECTORY include/singlet DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS singlet_pz singlet_fq singlet_pileup
    EXPORT SingletTargets
)
install(EXPORT SingletTargets
    NAMESPACE Singlet::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Singlet
)
