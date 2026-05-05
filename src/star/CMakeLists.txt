# STAR aligner — compiled as object library, linked into singlify
# singlet-lite fork: 48% faster than stock STAR via PGO, LTO, and
# algorithmic optimizations (SA boundary prefetch, lazy winBin reset,
# hash barcode lookup, 14-mer chunk sort, NUMA interleave)

file(GLOB STAR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB STAR_OPAL "${CMAKE_CURRENT_SOURCE_DIR}/opal/*.cpp")
file(GLOB STAR_SGT "${CMAKE_CURRENT_SOURCE_DIR}/SimpleGoodTuring/*.cpp")

add_library(singlet_star OBJECT
    ${STAR_SOURCES}
    ${STAR_OPAL}
    ${STAR_SGT}
)

target_include_directories(singlet_star PUBLIC
    ${CMAKE_SOURCE_DIR}/include/singlet/star
    ${CMAKE_SOURCE_DIR}/include/singlet/star/htslib
)

target_compile_options(singlet_star PRIVATE
    -O3 -march=native
    $<$<BOOL:${SINGLET_ENABLE_PGO_USE}>:-fprofile-use=${CMAKE_SOURCE_DIR}/data/pgo_profile>
)

target_link_libraries(singlet_star PUBLIC
    ZLIB::ZLIB
    Threads::Threads
)
