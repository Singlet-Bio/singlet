# singlet GPU kernels
add_library(singlet_gpu SHARED)
add_library(Singlet::gpu ALIAS singlet_gpu)
target_include_directories(singlet_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(singlet_gpu PUBLIC
    CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::curand
)
set_target_properties(singlet_gpu PROPERTIES
    CUDA_ARCHITECTURES "70;80;90"
)
