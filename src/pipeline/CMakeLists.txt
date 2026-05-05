# singlify binary
find_package(OpenMP REQUIRED)

add_executable(singlify singlify.cpp)
target_link_libraries(singlify PRIVATE
    singlet_pileup
    singlet_fq
    singlet_pz
    singlet_star
    ${ZSTD_LIBRARY}
    ZLIB::ZLIB
    OpenMP::OpenMP_CXX
)
target_include_directories(singlify PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/singlet/star
)
target_compile_options(singlify PRIVATE -O3 -march=native)
