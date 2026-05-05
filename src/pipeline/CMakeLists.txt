# singlify binary
add_executable(singlify singlify.cpp)
target_link_libraries(singlify PRIVATE
    singlet_pileup
    singlet_fq
    singlet_pz
    star_lib
    htslib
    OpenMP::OpenMP_CXX
)
target_compile_options(singlify PRIVATE -O3 -march=native)
