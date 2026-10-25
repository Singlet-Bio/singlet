# singlet binary
add_executable(singlet singlet.cpp)
target_link_libraries(singlet PRIVATE
    singlet_pileup
    singlet_fq
    singlet_pz
    singlet_star
    ${HTSLIB_LIBRARIES}
    ${ZSTD_LIBRARY}
    ${NCBI_VDB_LIB}
    ZLIB::ZLIB
    OpenMP::OpenMP_CXX
    Threads::Threads
)
target_link_directories(singlet PRIVATE ${HTSLIB_LIBRARY_DIRS})
target_include_directories(singlet PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/singlet/pileup
    ${CMAKE_SOURCE_DIR}/include/singlet/star
    ${HTSLIB_INCLUDE_DIRS}
    ${NCBI_VDB_INCLUDE}
)
target_compile_options(singlet PRIVATE -O3 -march=native)

# Engine git commit compiled into the binary (pileup_stats.json "provenance"). Pass -DSINGLET_GIT_COMMIT=<sha> when
# building from a tree without .git (e.g. an rsync copy); otherwise it is read from the source tree.
set(SINGLET_GIT_COMMIT "" CACHE STRING "engine git commit recorded in the binary")
set(_singlet_commit "${SINGLET_GIT_COMMIT}")
if(NOT _singlet_commit)
    execute_process(COMMAND git -C ${CMAKE_SOURCE_DIR} rev-parse HEAD
                    OUTPUT_VARIABLE _singlet_commit OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT _singlet_commit)
    set(_singlet_commit "unknown")
endif()
target_compile_definitions(singlet PRIVATE SINGLET_GIT_COMMIT="${_singlet_commit}")

if(HDF5_AVAILABLE)
    target_link_libraries(singlet PRIVATE ${HDF5_C_LIB})
    target_include_directories(singlet PRIVATE ${HDF5_C_INCLUDE})
endif()
