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

if(HDF5_AVAILABLE)
    target_link_libraries(singlet PRIVATE ${HDF5_C_LIB})
    target_include_directories(singlet PRIVATE ${HDF5_C_INCLUDE})
endif()
