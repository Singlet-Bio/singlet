cmake_minimum_required(VERSION 3.15)
project(singlify_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Find htslib via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(HTSLIB REQUIRED IMPORTED_TARGET htslib)
pkg_check_modules(ZSTD REQUIRED IMPORTED_TARGET libzstd)

# singlet-pileup headers (relative to this CMakeLists.txt)
set(PILEUP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../singlet-pileup")

pybind11_add_module(_core singlify/_core.cpp)

target_include_directories(_core PRIVATE
    ${PILEUP_DIR}/include
)

target_link_libraries(_core PRIVATE
    PkgConfig::HTSLIB
    PkgConfig::ZSTD
    pthread
)

target_compile_options(_core PRIVATE -O3 -march=native)

install(TARGETS _core LIBRARY DESTINATION singlify)
