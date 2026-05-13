# STAR aligner — compiled as object library, linked into singlet
# Vendored from STAR 2.7.11b (MIT license, see LICENSE.STAR).
# Optimizations: PGO, LTO, SA boundary prefetch, lazy winBin reset,
# hash barcode lookup, 14-mer chunk sort, NUMA interleave.

file(GLOB STAR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB STAR_OPAL "${CMAKE_CURRENT_SOURCE_DIR}/opal/*.cpp")
file(GLOB STAR_SGT "${CMAKE_CURRENT_SOURCE_DIR}/SimpleGoodTuring/*.cpp")

# Several .cpp files in STAR are code fragments #include-d by other
# translation units; compiling them standalone produces errors.
list(REMOVE_ITEM STAR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ReadAlign_CIGAR.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/stitchGapIndel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Transcript_variationOutput.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ReadAlign_outputVariation.cpp"
)

# bam_cat.c is compiled as C++ (uses C++ linkage in STAR)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/bam_cat.c PROPERTIES LANGUAGE CXX)

add_library(singlet_star OBJECT
    ${STAR_SOURCES}
    ${STAR_OPAL}
    ${STAR_SGT}
    ${CMAKE_CURRENT_SOURCE_DIR}/bam_cat.c
)

target_include_directories(singlet_star PUBLIC
    ${CMAKE_SOURCE_DIR}/include/singlet/star
    ${CMAKE_CURRENT_SOURCE_DIR}/htslib
)

# STAR source files #include "VERSION" and "parametersDefault.xxd" from their
# own directory. Ensure the source dir is on the include path.
target_include_directories(singlet_star PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/opal
    ${HTSLIB_INCLUDE_DIRS}
)

target_compile_options(singlet_star PRIVATE
    -O3 -march=native
    $<$<BOOL:${SINGLET_ENABLE_PGO_USE}>:-fprofile-use=${CMAKE_SOURCE_DIR}/data/pgo_profile>
)

# STAR's Makefile injects these via -D flags; reproduce them here.
string(TIMESTAMP _STAR_BUILD_DATE "%Y-%m-%dT%H:%M:%S")
target_compile_definitions(singlet_star PRIVATE
    "COMPILATION_TIME_PLACE=\"${_STAR_BUILD_DATE} singlet-bundled\""
    "GIT_BRANCH_COMMIT_DIFF=\"singlet-bundled\""
)

target_link_libraries(singlet_star PUBLIC
    ${HTSLIB_LIBRARIES}
    ZLIB::ZLIB
    Threads::Threads
)
