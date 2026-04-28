# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/CMakeLists.txt
# Owned by analysis-validator. Add a new gtest entry per correctness harness.

cmake_minimum_required(VERSION 3.24)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
# Avoid overriding the parent project's compiler/linker settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# ---------------------------------------------------------------------------
# g001 ABI fix (Cycle 54 Phase 6 / Cycle 55b):
# g001 uses gcc-toolset-13 as the compiler toolchain via scl enable, but the
# system libstdc++.so.6 (GCC 11) does not export _M_replace_cold which is a
# newer libstdc++ symbol.  gtest compiled by nvcc/g++-13 emits calls to this
# symbol, so we must link the gcc-toolset-13 static libstdc++.a to resolve it.
# We apply this fix to the gtest and gtest_main imported targets so that every
# test executable that links against them automatically inherits the fix.
# ---------------------------------------------------------------------------
set(GCC_TOOLSET13_LIBSTDCXX /opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13/libstdc++.a)
if(EXISTS "${GCC_TOOLSET13_LIBSTDCXX}")
    foreach(_gtest_tgt gtest gtest_main gmock gmock_main)
        if(TARGET ${_gtest_tgt})
            target_link_libraries(${_gtest_tgt} PUBLIC "${GCC_TOOLSET13_LIBSTDCXX}")
        endif()
    endforeach()
    message(STATUS "singlet-gpu: Applied gcc-toolset-13 libstdc++.a fix to gtest targets")
else()
    message(STATUS "singlet-gpu: gcc-toolset-13 libstdc++.a not found at ${GCC_TOOLSET13_LIBSTDCXX} — skipping ABI fix (non-g001 build)")
endif()

# ---------------------------------------------------------------------------
# NOTE (Phase 4 fix): LANGUAGE CUDA policy for test .cpp files.
# Test files that include singlet-gpu feature headers with __global__ or
# __device__ functions must be compiled by nvcc (LANGUAGE CUDA).  CUDA 12.8
# with -std=c++20 handles std::filesystem correctly so this is safe.
# Test files that only use CUDA runtime API (cudaMalloc, cudaMemcpy, etc.)
# but do NOT include headers with __global__/__device__ are compiled by g++.
# ---------------------------------------------------------------------------
# io/pz_device_loader correctness harness  (Cycle 56 Feature 0)
# ---------------------------------------------------------------------------
add_executable(io_pz_device_loader_correctness
    io_pz_device_loader_correctness.cpp
)
# pz_device_loader.h includes detail/uint_to_float_kernel.h which has
# __global__ CUDA kernels — must compile with nvcc (LANGUAGE CUDA).
set_source_files_properties(io_pz_device_loader_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(io_pz_device_loader_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
target_link_libraries(io_pz_device_loader_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables the GPU DeviceMemory code-path in factornet headers.
target_compile_definitions(io_pz_device_loader_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(io_pz_device_loader_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(io_pz_device_loader_correctness
    PROPERTIES LABELS "correctness;io;pz_device_loader")

# ---------------------------------------------------------------------------
# preprocess/lognorm correctness harness
# ---------------------------------------------------------------------------
add_executable(preprocess_lognorm_correctness
    preprocess_lognorm_correctness.cpp
)
# lognorm.h contains __global__ kernel definitions — must compile with nvcc.
set_source_files_properties(preprocess_lognorm_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(preprocess_lognorm_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Tests use CUDA runtime for device-to-host copies; link explicitly.
target_link_libraries(preprocess_lognorm_correctness PRIVATE CUDA::cudart)
set_target_properties(preprocess_lognorm_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(preprocess_lognorm_correctness
    PROPERTIES LABELS "correctness;preprocess")

# ---------------------------------------------------------------------------
# preprocess/hvg correctness harness
# ---------------------------------------------------------------------------
add_executable(preprocess_hvg_correctness
    preprocess_hvg_correctness.cpp
)
# hvg.h contains __global__ kernel definitions — must compile with nvcc.
set_source_files_properties(preprocess_hvg_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(preprocess_hvg_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
target_link_libraries(preprocess_hvg_correctness PRIVATE CUDA::cudart)
set_target_properties(preprocess_hvg_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(preprocess_hvg_correctness
    PROPERTIES LABELS "correctness;preprocess")

# ---------------------------------------------------------------------------
# reduce/svd adapter correctness harness
# ---------------------------------------------------------------------------
add_executable(reduce_svd_correctness
    reduce_svd_correctness.cpp
)
# reduce_svd_correctness.cpp transitively includes factornet SVD .cuh headers
# that contain CUDA device code (__shfl_sync, __syncthreads, etc.).  Those
# device intrinsics require nvcc — set LANGUAGE CUDA on this source file only.
# CUDA 12.8 nvcc handles std::filesystem with -std=c++20 so this is safe.
set_source_files_properties(reduce_svd_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(reduce_svd_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# SVD adapters link against factornet GPU headers (CUDA device code in .cuh).
# The factornet INTERFACE target should propagate include dirs; add CUDA::cudart
# for stream synchronization used indirectly by factornet's GPUContext.
target_link_libraries(reduce_svd_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU must be defined so factornet GPU code-paths are compiled in.
target_compile_definitions(reduce_svd_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(reduce_svd_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(reduce_svd_correctness
    PROPERTIES LABELS "correctness;reduce;svd")

# ---------------------------------------------------------------------------
# reduce/nmf adapter correctness harness  (cycle 6)
# ---------------------------------------------------------------------------
add_executable(reduce_nmf_correctness
    reduce_nmf_correctness.cpp
)
# reduce_nmf_correctness.cpp transitively includes factornet NMF .cuh headers
# (batch_nnls.cuh, loss.cuh, mu_update.cuh etc.) that contain CUDA device code.
# LANGUAGE CUDA on this file forces nvcc compilation; CUDA 12.8 handles
# std::filesystem with -std=c++20 correctly.
set_source_files_properties(reduce_nmf_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(reduce_nmf_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# NMF adapters link against factornet GPU headers (.cuh) and the chunked-NMF
# DataLoader path.  CUDA::cudart is needed for cudaMemcpy / stream sync inside
# factornet's GPUContext.
target_link_libraries(reduce_nmf_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables the GPU code-paths in factornet headers.
target_compile_definitions(reduce_nmf_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(reduce_nmf_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(reduce_nmf_correctness
    PROPERTIES LABELS "correctness;reduce;nmf")

# ---------------------------------------------------------------------------
# streaming/pipeline correctness harness  (cycle 7)
# ---------------------------------------------------------------------------
add_executable(streaming_pipeline_correctness
    streaming_pipeline_correctness.cpp
)
# streaming_pipeline_correctness.cpp includes lognorm.h, hvg.h which contain
# __global__ kernels, and transitively pulls in factornet NMF/SVD .cuh files
# with device intrinsics.  Must be compiled by nvcc.  CUDA 12.8 with
# -std=c++20 handles std::filesystem correctly.
set_source_files_properties(streaming_pipeline_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(streaming_pipeline_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# The streaming pipeline composes cycles 3–6 headers (lognorm, hvg, svd, nmf)
# all of which touch factornet GPU code-paths.  CUDA::cudart is required for
# cudaStream_t and cudaMemcpy* used in the test fixture and the in-memory
# reference path.
target_link_libraries(streaming_pipeline_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths inside factornet headers (svd, nmf).
target_compile_definitions(streaming_pipeline_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(streaming_pipeline_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(streaming_pipeline_correctness
    PROPERTIES LABELS "correctness;streaming")

# ---------------------------------------------------------------------------
# graph/knn + graph/snn correctness harness  (cycle 8 / cycle 62 CAGRA+SNN)
# ---------------------------------------------------------------------------
add_executable(graph_knn_correctness
    graph_knn_correctness.cpp
)
# knn.h and snn.h contain __global__/__device__ kernels — must compile with nvcc.
set_source_files_properties(graph_knn_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(graph_knn_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# kNN/SNN kernels use CUDA runtime for device allocations, cudaStreamCreate,
# cudaMemcpyAsync, and radix-select / prefix-scan device kernels.
target_link_libraries(graph_knn_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths inside factornet headers consumed
# by DenseMatrixGPU and DeviceMemory used in the kNN/SNN kernels.
target_compile_definitions(graph_knn_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(graph_knn_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
# Optional: link cuVS if present on the build node (enables the CAGRA recall test).
# Probe for cuvs only once (it may have been found already by another target above).
if(NOT TARGET cuvs::cuvs)
    find_package(cuvs QUIET)
endif()
if(cuvs_FOUND OR TARGET cuvs::cuvs)
    target_link_libraries(graph_knn_correctness PRIVATE cuvs::cuvs)
    target_compile_definitions(graph_knn_correctness PRIVATE SINGLET_GPU_HAS_CUVS)
    message(STATUS "singlet-gpu/tests: cuVS found — Knn_CagraBackend_RecallAtK will run.")
else()
    message(STATUS "singlet-gpu/tests: cuVS not found — Knn_CagraBackend_RecallAtK will be SKIPPED.")
endif()
gtest_discover_tests(graph_knn_correctness
    PROPERTIES LABELS "correctness;graph;knn;snn")

# ---------------------------------------------------------------------------
# graph/leiden correctness harness  (cycle 9)
# ---------------------------------------------------------------------------
add_executable(graph_leiden_correctness
    graph_leiden_correctness.cpp
)
# leiden.h contains __global__/__device__ kernels — must compile with nvcc.
set_source_files_properties(graph_leiden_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(graph_leiden_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Leiden wraps cuGraph which links against CUDA runtime and RAFT.
# CUDA::cudart is required for cudaMemcpyAsync, cudaStreamCreate, and the
# DeviceMemory<T> RAII helpers (DeviceMemory<int> for labels).
target_link_libraries(graph_leiden_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths inside factornet/core headers that
# DeviceMemory and DeviceDense pull in.
target_compile_definitions(graph_leiden_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(graph_leiden_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
# Optional: link cugraph if present on the build node.
# Without cugraph the leiden.h header compilation will fail at link time, so
# this find_package is a build-time gate rather than a skip guard.  On a node
# without cugraph the target still compiles if leiden.h provides a stub that
# throws std::runtime_error("cugraph not available") — consistent with how
# the CAGRA backend is gated in graph_knn_correctness.
find_package(cugraph QUIET)
if(cugraph_FOUND)
    target_link_libraries(graph_leiden_correctness PRIVATE cugraph::cugraph)
    target_compile_definitions(graph_leiden_correctness PRIVATE SINGLET_GPU_HAS_CUGRAPH)
    message(STATUS "singlet-gpu/tests: cuGraph found — Leiden GPU tests will run.")
else()
    message(STATUS "singlet-gpu/tests: cuGraph not found — all Leiden GPU tests will GTEST_SKIP().")
endif()
gtest_discover_tests(graph_leiden_correctness
    PROPERTIES LABELS "correctness;graph;leiden")

# ---------------------------------------------------------------------------
# embed/umap correctness harness  (cycle 10)
# ---------------------------------------------------------------------------
add_executable(embed_umap_correctness
    embed_umap_correctness.cpp
)
# umap.h contains __global__/__device__ kernels — must compile with nvcc.
set_source_files_properties(embed_umap_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(embed_umap_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# UMAP wraps cuML which uses CUDA runtime, RAFT, and cuBLAS internally.
# CUDA::cudart is required for cudaMemcpy, cudaGetDeviceCount, and
# DeviceMemory<T> RAII helpers.
target_link_libraries(embed_umap_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths inside factornet/core headers
# (DeviceMemory, GPUContext) consumed transitively by the kNN kernel.
target_compile_definitions(embed_umap_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/umap_reference.py.
target_compile_definitions(embed_umap_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(embed_umap_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
# Optional: link cuml if present.  Without cuml the umap.h header must provide
# a stub that throws std::runtime_error("cuml not available"), consistent with
# the cugraph pattern in graph_leiden_correctness.
find_package(cuml QUIET)
if(cuml_FOUND)
    target_link_libraries(embed_umap_correctness PRIVATE cuml::cuml++)
    target_compile_definitions(embed_umap_correctness PRIVATE SINGLET_GPU_HAS_CUML)
    message(STATUS "singlet-gpu/tests: cuML found — UMAP GPU tests will run.")
else()
    message(STATUS "singlet-gpu/tests: cuML not found — all UMAP GPU tests will GTEST_SKIP().")
endif()
gtest_discover_tests(embed_umap_correctness
    PROPERTIES LABELS "correctness;embed;umap")

# ---------------------------------------------------------------------------
# de/wilcoxon correctness harness  (cycle 11)
# ---------------------------------------------------------------------------
add_executable(de_wilcoxon_correctness
    de_wilcoxon_correctness.cpp
)
# wilcoxon.h contains __global__ kernel definitions — must compile with nvcc.
set_source_files_properties(de_wilcoxon_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(de_wilcoxon_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Wilcoxon DE uses CUDA runtime for device allocations, cudaMemcpyAsync, and
# the histogram-based radix-sort pass (cub) internally.  CUDA::cudart is
# required for stream/device management in the test fixture.
target_link_libraries(de_wilcoxon_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory, GPUContext) consumed by the DE kernel and the io/loader.
target_compile_definitions(de_wilcoxon_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(de_wilcoxon_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(de_wilcoxon_correctness
    PROPERTIES LABELS "correctness;de;wilcoxon")

# ---------------------------------------------------------------------------
# de/ttest correctness harness  (cycle 11)
# ---------------------------------------------------------------------------
add_executable(de_ttest_correctness
    de_ttest_correctness.cpp
)
# ttest.h contains __global__ kernel definitions — must compile with nvcc.
set_source_files_properties(de_ttest_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(de_ttest_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# t-test DE uses CUDA runtime for device allocations and the two-pass Welford
# per-gene per-cluster reduction.  CUDA::cudart is required.
target_link_libraries(de_ttest_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers.
target_compile_definitions(de_ttest_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(de_ttest_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(de_ttest_correctness
    PROPERTIES LABELS "correctness;de;ttest")

# ---------------------------------------------------------------------------
# qc/metrics correctness harness  (feature 6)
# ---------------------------------------------------------------------------
add_executable(qc_metrics_correctness
    qc_metrics_correctness.cpp
)
# metrics.h contains __global__ CUDA kernel definitions — must compile with nvcc.
set_source_files_properties(qc_metrics_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(qc_metrics_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# QC kernels use cub::BlockReduce and atomicAdd; CUDA::cudart required for
# stream/device management and cudaDeviceSynchronize in the test fixture.
target_link_libraries(qc_metrics_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory, SparseMatrixGPU) used by both the kernel and the test.
target_compile_definitions(qc_metrics_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(qc_metrics_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(qc_metrics_correctness
    PROPERTIES LABELS "correctness;qc;metrics")

# ---------------------------------------------------------------------------
# anno/marker_score correctness harness  (cycle 12)
# ---------------------------------------------------------------------------
add_executable(anno_marker_score_correctness
    anno_marker_score_correctness.cpp
)
# marker_score.h contains __global__ kernel definitions — must compile with nvcc.
set_source_files_properties(anno_marker_score_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(anno_marker_score_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Marker scoring uses CUDA runtime for device allocations, cudaMemcpyAsync,
# and cuBLAS GEMM (Mlm kernel) + cub reductions (UCell ranking).
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(anno_marker_score_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory, GPUContext) consumed by the annotation kernels.
target_compile_definitions(anno_marker_score_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory so the C++ test can locate the Python script.
target_compile_definitions(anno_marker_score_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(anno_marker_score_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(anno_marker_score_correctness
    PROPERTIES LABELS "correctness;anno;marker_score")

# ---------------------------------------------------------------------------
# anno/reference_map correctness harness  (cycle 12)
# ---------------------------------------------------------------------------
add_executable(anno_reference_map_correctness
    anno_reference_map_correctness.cpp
)
# reference_map.h contains __global__/__device__ kernels — must compile with nvcc.
set_source_files_properties(anno_reference_map_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(anno_reference_map_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Reference map uses cuBLAS GEMM (embedding × weights^T), a per-row softmax
# kernel, and cub::DeviceReduce::ArgMax.  CUDA::cudart is required.
target_link_libraries(anno_reference_map_correctness PRIVATE CUDA::cudart)
# load_celltypist_model reads DEFLATE-compressed .npz files via zlib.
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    target_link_libraries(anno_reference_map_correctness PRIVATE ZLIB::ZLIB)
    target_compile_definitions(anno_reference_map_correctness PRIVATE SINGLET_GPU_HAS_ZLIB)
    message(STATUS "singlet-gpu/tests: zlib found — anno_reference_map NPZ DEFLATE reading enabled.")
else()
    message(STATUS "singlet-gpu/tests: zlib not found — anno_reference_map NPZ tests may skip.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers.
target_compile_definitions(anno_reference_map_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path for Python subprocess calls.
target_compile_definitions(anno_reference_map_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(anno_reference_map_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(anno_reference_map_correctness
    PROPERTIES LABELS "correctness;anno;reference_map")

# ---------------------------------------------------------------------------
# gsea/fgsea correctness harness  (cycle 13)
# ---------------------------------------------------------------------------
add_executable(gsea_fgsea_correctness
    gsea_fgsea_correctness.cpp
)
# fgsea.h contains __global__ kernel definitions — must compile with nvcc.
set_source_files_properties(gsea_fgsea_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(gsea_fgsea_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# GSEA fgsea uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device for stats vector), and cuRAND permutation
# generation inside fgsea.h.  CUDA::cudart is required.
target_link_libraries(gsea_fgsea_correctness PRIVATE CUDA::cudart)
# cuRAND for adaptive multilevel permutation kernel inside fgsea.h.
target_link_libraries(gsea_fgsea_correctness PRIVATE CUDA::curand)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext) consumed by the GSEA kernel.
target_compile_definitions(gsea_fgsea_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the R
# reference script at tests/refs/fgsea_r_reference.R.
target_compile_definitions(gsea_fgsea_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(gsea_fgsea_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(gsea_fgsea_correctness
    PROPERTIES LABELS "correctness;gsea;fgsea")

# ---------------------------------------------------------------------------
# gsea/aucell correctness harness  (cycle 13)
# ---------------------------------------------------------------------------
add_executable(gsea_aucell_correctness
    gsea_aucell_correctness.cpp
)
# aucell.h contains __global__ kernel definitions — must compile with nvcc.
set_source_files_properties(gsea_aucell_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(gsea_aucell_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# AUCell uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device CSC upload, device→host score download),
# and the histogram-binned per-cell ranking kernel.  CUDA::cudart is required.
target_link_libraries(gsea_aucell_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by AUCell.
target_compile_definitions(gsea_aucell_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the R
# reference script at tests/refs/aucell_r_reference.R.
target_compile_definitions(gsea_aucell_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(gsea_aucell_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(gsea_aucell_correctness
    PROPERTIES LABELS "correctness;gsea;aucell")

# ---------------------------------------------------------------------------
# integrate/harmony correctness harness  (cycle 14)
# ---------------------------------------------------------------------------
add_executable(integrate_harmony_correctness
    integrate_harmony_correctness.cpp
)
# harmony.h contains __global__/__device__ kernels — must compile with nvcc.
set_source_files_properties(integrate_harmony_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(integrate_harmony_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Harmony uses CUDA runtime for DeviceDense / DeviceMemory allocations,
# cudaMallocAsync, cudaMemcpyAsync, and the soft k-means correction kernel.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(integrate_harmony_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, DeviceDense) consumed by the Harmony kernel.
target_compile_definitions(integrate_harmony_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory so the C++ test can locate Python reference scripts.
target_compile_definitions(integrate_harmony_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(integrate_harmony_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(integrate_harmony_correctness
    PROPERTIES LABELS "correctness;integrate;harmony")

# ---------------------------------------------------------------------------
# integrate/bbknn correctness harness  (cycle 14)
# ---------------------------------------------------------------------------
add_executable(integrate_bbknn_correctness
    integrate_bbknn_correctness.cpp
)
# integrate/bbknn.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(integrate_bbknn_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(integrate_bbknn_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# BBKNN builds on the cycle-8 kNN kernel (same cuBLAS GEMM tile structure,
# per-batch masked sub-calls).  CUDA::cudart is required for DeviceMemory<int>
# and cudaMemcpy used in the BBKNN kernel and the test fixture.
target_link_libraries(integrate_bbknn_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers consumed
# by DeviceMemory<T>, DeviceDense, and the kNN sub-kernel reused by BBKNN.
target_compile_definitions(integrate_bbknn_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path for Python subprocess calls.
target_compile_definitions(integrate_bbknn_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(integrate_bbknn_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
# Optional: link cugraph for the Leiden downstream test.
# Same pattern as graph_leiden_correctness — without cugraph the Leiden test
# calls GTEST_SKIP() rather than failing to compile.
find_package(cugraph QUIET)
if(cugraph_FOUND)
    target_link_libraries(integrate_bbknn_correctness PRIVATE cugraph::cugraph)
    target_compile_definitions(integrate_bbknn_correctness PRIVATE SINGLET_GPU_HAS_CUGRAPH)
    message(STATUS "singlet-gpu/tests: cuGraph found — BBKNN_LeidenDownstream will run.")
else()
    message(STATUS "singlet-gpu/tests: cuGraph not found — BBKNN_LeidenDownstream will GTEST_SKIP().")
endif()
gtest_discover_tests(integrate_bbknn_correctness
    PROPERTIES LABELS "correctness;integrate;bbknn")

# ---------------------------------------------------------------------------
# preprocess/velocity_prep correctness harness  (cycle 15)
# ---------------------------------------------------------------------------
add_executable(preprocess_velocity_prep_correctness
    preprocess_velocity_prep_correctness.cpp
)
# velocity_prep.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(preprocess_velocity_prep_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(preprocess_velocity_prep_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# velocity_prep uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device CSC upload, device→host result download),
# and the tiled kNN-moments smoothing + gamma regression kernels.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(preprocess_velocity_prep_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by the velocity
# prep kernel and the pz_device_loader used in the GSM4037629 real-data test.
target_compile_definitions(preprocess_velocity_prep_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/velocity_prep_scvelo_reference.py.
target_compile_definitions(preprocess_velocity_prep_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(preprocess_velocity_prep_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(preprocess_velocity_prep_correctness
    PROPERTIES LABELS "correctness;preprocess;velocity_prep")

# ---------------------------------------------------------------------------
# preprocess/scale + regress_out correctness harness  (Feature 7)
# ---------------------------------------------------------------------------
add_executable(preprocess_scale_correctness
    preprocess_scale_correctness.cpp
)
# scale.h and regress_out contain __global__ CUDA kernels — must compile with nvcc.
set_source_files_properties(preprocess_scale_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(preprocess_scale_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
target_link_libraries(preprocess_scale_correctness PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
target_compile_definitions(preprocess_scale_correctness PRIVATE FACTORNET_HAS_GPU)
set_target_properties(preprocess_scale_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(preprocess_scale_correctness
    PROPERTIES LABELS "correctness;preprocess;scale")

# ---------------------------------------------------------------------------
# anno/mt_lineage correctness harness  (cycle 16)
# ---------------------------------------------------------------------------
add_executable(anno_mt_lineage_correctness
    anno_mt_lineage_correctness.cpp
)
# mt_lineage.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(anno_mt_lineage_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(anno_mt_lineage_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# MT lineage uses CUDA runtime for DeviceMemory<int/float> allocations,
# cudaMemcpyAsync (host→device CSC upload, device→host labels download),
# and the per-site binomial EM + cell-level Gaussian mixture kernels.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(anno_mt_lineage_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by the MT lineage
# kernel and the pz_device_loader used in the GSM4037629 real-data test.
target_compile_definitions(anno_mt_lineage_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/mt_lineage_mquad_reference.py.
target_compile_definitions(anno_mt_lineage_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# zlib is needed by the minimal NPZ reader inside the test (DEFLATE members).
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    target_link_libraries(anno_mt_lineage_correctness PRIVATE ZLIB::ZLIB)
    target_compile_definitions(anno_mt_lineage_correctness PRIVATE SINGLET_GPU_HAS_ZLIB)
    message(STATUS "singlet-gpu/tests: zlib found — NPZ DEFLATE reading enabled.")
else()
    message(STATUS "singlet-gpu/tests: zlib not found — NPZ DEFLATE members will be skipped (stored-only NPZ still works).")
endif()
set_target_properties(anno_mt_lineage_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(anno_mt_lineage_correctness
    PROPERTIES LABELS "correctness;anno;mt_lineage")

# ---------------------------------------------------------------------------
# de/donor_pseudobulk correctness harness  (cycle 17)
# ---------------------------------------------------------------------------
add_executable(de_donor_pseudobulk_correctness
    de_donor_pseudobulk_correctness.cpp
)
# donor_pseudobulk.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(de_donor_pseudobulk_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(de_donor_pseudobulk_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# donor_pseudobulk uses CUDA runtime for DeviceMemory<float/int> allocations,
# cudaMemcpyAsync (host→device CSC + label upload, device→host result download),
# cub::DeviceRadixSort + cub::DeviceSegmentedReduce for pseudobulk aggregation,
# and the IRLS NB-GLM kernel (cuSOLVER for the small per-group Hessian solve).
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(de_donor_pseudobulk_correctness PRIVATE CUDA::cudart)
# cuSOLVER for the block-level weighted least-squares solve inside the NB-GLM.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(de_donor_pseudobulk_correctness PRIVATE CUDA::cusolver)
    message(STATUS "singlet-gpu/tests: cuSOLVER found — donor_pseudobulk NB-GLM path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuSOLVER link skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by the
# donor_pseudobulk kernel and by the io/pz_device_loader used in the
# GSM4037629 real-data test.
target_compile_definitions(de_donor_pseudobulk_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the R
# reference script at tests/refs/donor_pseudobulk_deseq2_reference.R.
target_compile_definitions(de_donor_pseudobulk_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(de_donor_pseudobulk_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(de_donor_pseudobulk_correctness
    PROPERTIES LABELS "correctness;de;donor_pseudobulk")

# ---------------------------------------------------------------------------
# embed/cell2fate correctness harness  (cycle 27)
# ---------------------------------------------------------------------------
add_executable(embed_cell2fate_correctness
    embed_cell2fate_correctness.cpp
)
# cell2fate.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(embed_cell2fate_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(embed_cell2fate_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# cell2fate uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device CSC upload, device→host result download),
# cuRAND Philox for minibatch sampling and posterior draws, and
# cub::DeviceRadixSort for deterministic minibatch ordering.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(embed_cell2fate_correctness PRIVATE CUDA::cudart)
# cuRAND for Philox-seeded minibatch order and posterior sampling in cell2fate.h.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(embed_cell2fate_correctness PRIVATE CUDA::curand)
    message(STATUS "singlet-gpu/tests: cuRAND found — Cell2fate stochastic path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuRAND link skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by the cell2fate
# kernel (NMF module init, Adam optimizer) and by the io/pz_device_loader
# used in the GSM4037629 real-data test.
target_compile_definitions(embed_cell2fate_correctness PRIVATE FACTORNET_HAS_GPU)
# CELL2FATE_EXPOSE_GRAD_FOR_TEST exposes cell2fate_init() + cell2fate_analytical_grad()
# + cell2fate_eval_elbo() for the gradient-check test (Cell2fate_GradientCheck_FiniteDiff).
# These symbols are gated by this define so they don't bloat the production build.
target_compile_definitions(embed_cell2fate_correctness PRIVATE
    CELL2FATE_EXPOSE_GRAD_FOR_TEST)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/cell2fate_python_reference.py.
target_compile_definitions(embed_cell2fate_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(embed_cell2fate_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(embed_cell2fate_correctness
    PROPERTIES LABELS "correctness;embed;cell2fate")

# ---------------------------------------------------------------------------
# reduce/nmf/csi_gep correctness harness  (cycle 28)
# ---------------------------------------------------------------------------
add_executable(reduce_nmf_csi_gep_correctness
    reduce_nmf_csi_gep_correctness.cpp
)
# csi_gep.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(reduce_nmf_csi_gep_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(reduce_nmf_csi_gep_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# CSI-GEP uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device CSC upload, device→host program/usage download),
# cuRAND Philox seeded per run (run_seed = base_seed + run_idx), and
# cub::DeviceRadixSort for top-30 gene Jaccard on device.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(reduce_nmf_csi_gep_correctness PRIVATE CUDA::cudart)
# cuRAND for per-run Philox seed inside csi_gep.h.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(reduce_nmf_csi_gep_correctness PRIVATE CUDA::curand)
    message(STATUS "singlet-gpu/tests: cuRAND found — CSI-GEP stochastic path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuRAND link skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by the CSI-GEP
# kernel (inner NMF loop via factornet::nmf::fit_gpu) and by the
# io/pz_device_loader used in the GSM4037629 real-data test.
target_compile_definitions(reduce_nmf_csi_gep_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/csi_gep_python_reference.py.
target_compile_definitions(reduce_nmf_csi_gep_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(reduce_nmf_csi_gep_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(reduce_nmf_csi_gep_correctness
    PROPERTIES LABELS "correctness;reduce;nmf;csi_gep")

# ---------------------------------------------------------------------------
# spatial/stagate correctness harness  (cycle 29)
# ---------------------------------------------------------------------------
add_executable(spatial_stagate_correctness
    spatial_stagate_correctness.cpp
)
# stagate.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(spatial_stagate_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(spatial_stagate_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# STAGATE uses CUDA runtime for DeviceMemory<float/int32_t> allocations,
# cudaMemcpyAsync (host→device CSC + coords upload, device→host embedding +
# labels download), cuRAND Philox for layer weight init, cuBLAS GEMM for MLP
# decoder, cuSPARSE SpMM for the first encoder projection, and
# cub::DeviceSegmentedReduce for per-neighbor attention aggregation.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(spatial_stagate_correctness PRIVATE CUDA::cudart)
# cuRAND for Philox-seeded weight initialisation inside stagate.h.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(spatial_stagate_correctness PRIVATE CUDA::curand)
    target_link_libraries(spatial_stagate_correctness PRIVATE CUDA::cusparse)
    message(STATUS "singlet-gpu/tests: cuRAND + cuSPARSE found — STAGATE full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuRAND/cuSPARSE links skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext) consumed by the STAGATE kernel and by the
# io/pz_device_loader path used in shape/determinism tests.
target_compile_definitions(spatial_stagate_correctness PRIVATE FACTORNET_HAS_GPU)
# STAGATE_EXPOSE_GRAD_FOR_TEST exposes stagate_init(), stagate_analytical_grad(),
# and stagate_eval_loss() for the finite-difference gradient check
# (Stagate_GradientCheck_FiniteDiff). Gated so they don't bloat production builds.
target_compile_definitions(spatial_stagate_correctness PRIVATE
    STAGATE_EXPOSE_GRAD_FOR_TEST)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/stagate_pyg_reference.py.
target_compile_definitions(spatial_stagate_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# Optional: link cugraph for the post-training Leiden test.
# Without cugraph the leiden.h header must provide a stub that throws
# std::runtime_error("cugraph not available") — consistent with prior cycles.
find_package(cugraph QUIET)
if(cugraph_FOUND)
    target_link_libraries(spatial_stagate_correctness PRIVATE cugraph::cugraph)
    target_compile_definitions(spatial_stagate_correctness PRIVATE SINGLET_GPU_HAS_CUGRAPH)
    message(STATUS "singlet-gpu/tests: cuGraph found — Stagate_PostTrainingLeiden will run full Leiden.")
else()
    message(STATUS "singlet-gpu/tests: cuGraph not found — Stagate_PostTrainingLeiden uses fallback k-means clustering.")
endif()
set_target_properties(spatial_stagate_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(spatial_stagate_correctness
    PROPERTIES LABELS "correctness;spatial;stagate")

# ---------------------------------------------------------------------------
# generative/discrete_diffusion correctness harness  (cycle 30)
# ---------------------------------------------------------------------------
add_executable(generative_discrete_diffusion_correctness
    generative_discrete_diffusion_correctness.cpp
)
# discrete_diffusion.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(generative_discrete_diffusion_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(generative_discrete_diffusion_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Discrete diffusion uses CUDA runtime for DeviceMemory<float/int32_t>
# allocations, cudaMemcpyAsync (host→device CSC upload, device→host sample
# download), cuRAND Philox for masking + sampling, and cuBLAS GEMM for QKV
# projections and feed-forward layers.  CUDA::cudart is required.
target_link_libraries(generative_discrete_diffusion_correctness PRIVATE CUDA::cudart)
# cuRAND for per-cell Philox masking and sampling steps inside discrete_diffusion.h.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(generative_discrete_diffusion_correctness PRIVATE CUDA::curand)
    target_link_libraries(generative_discrete_diffusion_correctness PRIVATE CUDA::cublas)
    message(STATUS "singlet-gpu/tests: cuRAND + cuBLAS found — discrete_diffusion full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuRAND/cuBLAS links skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by discrete_diffusion.h
# and by the io/pz_device_loader path used in the GSM4037629 real-data test.
target_compile_definitions(generative_discrete_diffusion_correctness PRIVATE FACTORNET_HAS_GPU)
# DISCRETE_DIFFUSION_EXPOSE_GRAD_FOR_TEST exposes discrete_diffusion_init(),
# discrete_diffusion_analytical_grad(), and discrete_diffusion_eval_loss() for the
# finite-difference gradient check (Diffusion_GradientCheck_FiniteDiff).
# Gated so these symbols do not bloat production builds.
target_compile_definitions(generative_discrete_diffusion_correctness PRIVATE
    DISCRETE_DIFFUSION_EXPOSE_GRAD_FOR_TEST)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/discrete_diffusion_python_reference.py.
target_compile_definitions(generative_discrete_diffusion_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(generative_discrete_diffusion_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(generative_discrete_diffusion_correctness
    PROPERTIES LABELS "correctness;generative;discrete_diffusion")

# ---------------------------------------------------------------------------
# qc/doublet_score correctness harness  (cycle 31)
# ---------------------------------------------------------------------------
add_executable(qc_doublet_correctness
    qc_doublet_correctness.cpp
)
# doublet_score.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(qc_doublet_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(qc_doublet_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# doublet_score uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device PCA embedding upload, device→host score
# download), cuRAND Philox for synthetic doublet pair selection, and
# reuses the cycle-8 compute_knn kernel for the combined kNN step.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(qc_doublet_correctness PRIVATE CUDA::cudart)
# cuRAND for Philox-seeded synthetic doublet generation inside doublet_score.h.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(qc_doublet_correctness PRIVATE CUDA::curand)
    message(STATUS "singlet-gpu/tests: cuRAND found — doublet_score stochastic path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuRAND link skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by doublet_score.h
# (which reuses the cycle-8 kNN kernel) and by the io/pz_device_loader
# used in the GSM4037629 real-data test.
target_compile_definitions(qc_doublet_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/doublet_scrublet_reference.py.
target_compile_definitions(qc_doublet_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(qc_doublet_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(qc_doublet_correctness
    PROPERTIES LABELS "correctness;qc;doublet_score")

# ---------------------------------------------------------------------------
# perturbation/perturb_graph correctness harness  (cycle 32)
# ---------------------------------------------------------------------------
add_executable(perturbation_perturb_graph_correctness
    perturbation_perturb_graph_correctness.cpp
)
# perturb_graph.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(perturbation_perturb_graph_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(perturbation_perturb_graph_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# perturb_graph uses CUDA runtime for DeviceMemory<float/int32_t> allocations,
# cudaMemcpyAsync (host→device CSC + label + dose upload, device→host prediction
# download), cuRAND Philox for weight init and minibatch ordering, cuBLAS GEMM
# for encoder/decoder MLP layers, and cuSPARSE SpMM for the GNN perturbation
# graph propagation (PerturbGraph variant).
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(perturbation_perturb_graph_correctness PRIVATE CUDA::cudart)
# cuRAND for Philox-seeded weight initialisation and minibatch sampling inside
# perturb_graph.h (CPA training loop and GNN init).
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(perturbation_perturb_graph_correctness PRIVATE CUDA::curand)
    target_link_libraries(perturbation_perturb_graph_correctness PRIVATE CUDA::cublas)
    target_link_libraries(perturbation_perturb_graph_correctness PRIVATE CUDA::cusparse)
    message(STATUS "singlet-gpu/tests: cuRAND + cuBLAS + cuSPARSE found — perturb_graph full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuRAND/cuBLAS/cuSPARSE links skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by perturb_graph.h
# (NB log-likelihood accumulator via cycle-17 pattern, Adam optimizer via
# cycle-27 pattern) and by the io/pz_device_loader path used in the
# determinism and smoke tests.
target_compile_definitions(perturbation_perturb_graph_correctness PRIVATE FACTORNET_HAS_GPU)
# PERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST exposes perturb_graph_init(),
# perturb_graph_analytical_grad(), and perturb_graph_eval_loss() for the
# finite-difference gradient check (PerturbGraph_GradientCheck_FiniteDiff).
# Gated so these symbols do not bloat production builds.
target_compile_definitions(perturbation_perturb_graph_correctness PRIVATE
    PERTURB_GRAPH_EXPOSE_GRAD_FOR_TEST)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/cpa_python_reference.py.
target_compile_definitions(perturbation_perturb_graph_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(perturbation_perturb_graph_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(perturbation_perturb_graph_correctness
    PROPERTIES LABELS "correctness;perturbation;perturb_graph")

# ---------------------------------------------------------------------------
# spatial/flash_deconv correctness harness  (cycle 33)
# ---------------------------------------------------------------------------
add_executable(spatial_flash_deconv_correctness
    spatial_flash_deconv_correctness.cpp
)
# flash_deconv.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(spatial_flash_deconv_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(spatial_flash_deconv_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# FlashDeconv uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device CSC + reference B + spatial coords upload,
# device→host W + uncertainty + top_gene_indices download), cuSOLVER for the
# B^T B SVD (leverage-score computation, n_types × n_types — small),
# cuRAND Philox for weighted gene subsampling and bootstrap seed sequence,
# cuBLAS GEMM for the ADMM primal update (Y_sub ≈ W B_sub),
# cuSPARSE SpMM for the spatial Laplacian regularization step (kNN graph),
# and cub::DeviceSegmentedReduce for reference aggregation.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(spatial_flash_deconv_correctness PRIVATE CUDA::cudart)
# cuSOLVER for leverage-score SVD; cuRAND for Philox-seeded sketching + bootstrap;
# cuBLAS for ADMM GEMM; cuSPARSE for spatial Laplacian SpMM.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(spatial_flash_deconv_correctness PRIVATE
        CUDA::cusolver
        CUDA::curand
        CUDA::cublas
        CUDA::cusparse
    )
    message(STATUS "singlet-gpu/tests: cuSOLVER + cuRAND + cuBLAS + cuSPARSE found "
                   "— spatial_flash_deconv full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSOLVER/cuRAND/cuBLAS/cuSPARSE links skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by flash_deconv.h
# (cycle-5 NMF NNLS inner step, cycle-8 kNN for spatial Laplacian) and by
# the io/pz_device_loader path used in the GSM4037629 real-data test.
target_compile_definitions(spatial_flash_deconv_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/flash_deconv_cell2location_reference.py.
target_compile_definitions(spatial_flash_deconv_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(spatial_flash_deconv_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(spatial_flash_deconv_correctness
    PROPERTIES LABELS "correctness;spatial;flash_deconv")

# ---------------------------------------------------------------------------
# atac/chromvar correctness harness  (cycle 34)
# ---------------------------------------------------------------------------
add_executable(atac_chromvar_correctness
    atac_chromvar_correctness.cpp
)
# chromvar.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(atac_chromvar_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(atac_chromvar_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# chromVAR uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device peak×cell CSC + motif×peak annotation upload,
# device→host deviation/variability/pvalue download), cuSPARSE SpMM for the
# core motif × peak * peak × cell matrix multiply, cuRAND Philox for
# background peak sampling (N=50 per peak) and label-permutation p-values
# (K=1000 permutations via batched SpMMs), and cub::DeviceReduce for
# per-peak expected accessibility and per-motif variability summation.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(atac_chromvar_correctness PRIVATE CUDA::cudart)
# cuSPARSE for the core motif × peak SpMM (observed + background scoring).
# cuRAND for Philox-seeded background peak sampling and permutation labels.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(atac_chromvar_correctness PRIVATE CUDA::cusparse)
    target_link_libraries(atac_chromvar_correctness PRIVATE CUDA::curand)
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuRAND found — atac_chromvar full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuSPARSE/cuRAND links skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by chromvar.h
# (cuSPARSE SpMM batched over K permutations) and by the io/pz_device_loader
# used in the GSM4037629 real-data test.
target_compile_definitions(atac_chromvar_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the R
# reference script at tests/refs/chromvar_r_reference.R.
target_compile_definitions(atac_chromvar_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(atac_chromvar_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(atac_chromvar_correctness
    PROPERTIES LABELS "correctness;atac;chromvar")

# ---------------------------------------------------------------------------
# cna/numbat correctness harness  (cycle 35)
# ---------------------------------------------------------------------------
add_executable(cna_numbat_correctness
    cna_numbat_correctness.cpp
)
# numbat.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(cna_numbat_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(cna_numbat_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Numbat CNA uses CUDA runtime for DeviceMemory<float/int> allocations,
# cudaMemcpyAsync (host→device dense expr matrix + gene annotation upload,
# device→host CNA states + posteriors + clone labels download),
# cuFFT for optional FFT-based chromosome smoothing (short windows fall back
# to a direct convolution kernel), cuRAND Philox for clone clustering init,
# and reuses the cycle-9 Leiden kernel for clone assignment.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(cna_numbat_correctness PRIVATE CUDA::cudart)
# cuFFT for FFT-based chromosome smoothing (optional — direct kernel is
# used when window size < 32; cuFFT accelerates large windows).
# cuRAND for Philox-seeded clone clustering initialisation inside numbat.h.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(cna_numbat_correctness PRIVATE CUDA::cufft)
    target_link_libraries(cna_numbat_correctness PRIVATE CUDA::curand)
    message(STATUS "singlet-gpu/tests: cuFFT + cuRAND found — Numbat CNA full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found — cuFFT/cuRAND links skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by numbat.h
# (Leiden clone clustering via cycle-9 adapter) and by the io/pz_device_loader
# used in the Numbat_GSM_RealTumor test.
target_compile_definitions(cna_numbat_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the R
# reference script at tests/refs/numbat_r_reference.R.
target_compile_definitions(cna_numbat_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
# Optional: link cugraph for the Leiden-based clone assignment path in numbat.h.
# Without cugraph, numbat.h falls back to GPU k-means for clone calling.
find_package(cugraph QUIET)
if(cugraph_FOUND)
    target_link_libraries(cna_numbat_correctness PRIVATE cugraph::cugraph)
    target_compile_definitions(cna_numbat_correctness PRIVATE SINGLET_GPU_HAS_CUGRAPH)
    message(STATUS "singlet-gpu/tests: cuGraph found — Numbat Leiden clone assignment enabled.")
else()
    message(STATUS "singlet-gpu/tests: cuGraph not found — Numbat uses k-means clone assignment fallback.")
endif()
set_target_properties(cna_numbat_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(cna_numbat_correctness
    PROPERTIES LABELS "correctness;cna;numbat")

# ---------------------------------------------------------------------------
# grn/granie correctness harness  (cycle 36)
# ---------------------------------------------------------------------------
add_executable(grn_granie_correctness
    grn_granie_correctness.cpp
)
# granie.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(grn_granie_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(grn_granie_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# GRaNIE uses CUDA runtime for DeviceMemory<float> allocations,
# cudaMemcpyAsync (host→device GEX CSC + peaks CSC + tf_motif upload,
# device→host tf_activity + peak_gene_corr + edge list download),
# cuSPARSE SpMM for TF activity computation (tf_motif × peaks SpMM),
# cuBLAS GEMM for TF-target scoring (TF activity × GEX),
# cuRAND Philox for FDR permutation null distribution,
# and cub::DeviceSegmentedReduce for per-TF peak-accessibility means.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(grn_granie_correctness PRIVATE CUDA::cudart)
# cuSPARSE for TF activity SpMM and peak-gene correlation SpMM batch.
# cuBLAS for TF-target GEMM (n_tfs × n_cells × n_genes).
# cuRAND for Philox-seeded FDR permutation null distribution.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(grn_granie_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
        CUDA::curand
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS + cuRAND found "
                   "— grn_granie full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS/cuRAND links skipped.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU) consumed by granie.h
# (cycle-7 Leiden for community detection, cycle-2 loader for the
# real multiome test) and by the io/pz_device_loader path used in
# GRaNIE_GSM_RealMultiome.
target_compile_definitions(grn_granie_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the R
# reference script at tests/refs/granie_r_reference.R.
target_compile_definitions(grn_granie_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(grn_granie_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(grn_granie_correctness
    PROPERTIES LABELS "correctness;grn;granie")

# ---------------------------------------------------------------------------
# comm/cellchat correctness harness  (cycle 37)
# ---------------------------------------------------------------------------
add_executable(comm_cellchat_correctness
    comm_cellchat_correctness.cpp
)
# cellchat.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(comm_cellchat_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(comm_cellchat_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# CellChat GPU uses:
#   CUDA::cudart  — DeviceMemory<float> alloc/copy, stream management
#   CUDA::cusparse — pseudobulk SpMM (one-hot × expression)
#   CUDA::cublas   — batched GEMM for L-R outer products
#   CUDA::curand   — Philox4x32_10 for permutation label shuffles
# cub::DeviceRadixSort (header-only via CUDA toolkit) for on-device sort.
target_link_libraries(comm_cellchat_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(comm_cellchat_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
        CUDA::curand
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS + cuRAND found "
                   "— comm_cellchat full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS/cuRAND links skipped for comm_cellchat.")
endif()
# FACTORNET_HAS_GPU enables factornet GPU types (DeviceMemory<T>, GPUContext,
# SparseMatrixGPU<float>) consumed by comm/cellchat.h and by the real-data
# io/pz_device_loader path in CellChat_GSM_RealData.
target_compile_definitions(comm_cellchat_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ source directory so the C++ test can locate R scripts at
# tests/refs/cellchat_r_reference.R and tests/refs/cellchatdb_mini.tsv.
target_compile_definitions(comm_cellchat_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(comm_cellchat_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(comm_cellchat_correctness
    PROPERTIES LABELS "correctness;comm;cellchat")

# ---------------------------------------------------------------------------
# eqtl/nebula correctness harness  (cycle 38)
# ---------------------------------------------------------------------------
add_executable(eqtl_nebula_correctness
    eqtl_nebula_correctness.cpp
)
# nebula.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(eqtl_nebula_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(eqtl_nebula_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# NEBULA eQTL uses:
#   CUDA::cudart   — DeviceMemory<float/int32_t> alloc/copy, stream management
#   CUDA::cusparse — per-SNP segmented SpMM for donor aggregation
#   CUDA::cublas   — batched 2×2 Hessian solves inside Fisher scoring (small,
#                    fp64 accumulator; cuBLAS DGEMM at block size 2)
# cub::DeviceSegmentedReduce (header-only via CUDA toolkit) for donor sums.
target_link_libraries(eqtl_nebula_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(eqtl_nebula_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS found "
                   "— eqtl_nebula full path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS links skipped for eqtl_nebula.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by
# eqtl/nebula.h (donor aggregation reuses cycle-15 pseudobulk pattern;
# Fisher scoring block kernel reuses cycle-17 NB-GLM fp32/fp64 strategy)
# and by io/pz_device_loader used in Nebula_GSM_RealMultisample.
target_compile_definitions(eqtl_nebula_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the R
# reference script at tests/refs/nebula_r_reference.R.
target_compile_definitions(eqtl_nebula_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(eqtl_nebula_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(eqtl_nebula_correctness
    PROPERTIES LABELS "correctness;eqtl;nebula")

# ---------------------------------------------------------------------------
# qc/omnidoublet correctness harness  (cycle 39)
# ---------------------------------------------------------------------------
add_executable(qc_omnidoublet_correctness
    qc_omnidoublet_correctness.cpp
)
# omnidoublet.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(qc_omnidoublet_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(qc_omnidoublet_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# OmniDoublet uses:
#   CUDA::cudart   — DeviceMemory<float/double/int32_t> alloc/copy, stream mgmt
#   CUDA::curand   — Philox4x32_10 for doublet-pair sampling (sim_doublets_kernel)
#   CUDA::cusparse — SpMM gather for the sparse doublet simulation tile loop
#                    and for the joint PCA input (RNA HVG + ADT concatenation)
#   CUDA::cublas   — batched IRLS for the logistic classifier (4-feature, 20 iters;
#                    fp64 Hessian inverse reuses cycle-38 Fisher-scoring pattern)
# cub::DeviceRadixSort (header-only, via CUDAToolkit) for FDR threshold sort.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(qc_omnidoublet_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(qc_omnidoublet_correctness PRIVATE
        CUDA::curand
        CUDA::cusparse
        CUDA::cublas
    )
    message(STATUS "singlet-gpu/tests: cuRAND + cuSPARSE + cuBLAS found "
                   "— qc_omnidoublet full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuRAND/cuSPARSE/cuBLAS links skipped for qc_omnidoublet.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by
# omnidoublet.h (joint PCA via factornet::svd::randomized_gpu reuse from
# cycle 4; kNN via cycle 8 compute_exact; logistic classifier via cycle 38
# Fisher-scoring pattern) and by io/pz_device_loader used in the
# Omnidoublet_GSM_CITEseq_RealData test.
target_compile_definitions(qc_omnidoublet_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/omnidoublet_py_reference.py.
target_compile_definitions(qc_omnidoublet_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(qc_omnidoublet_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(qc_omnidoublet_correctness
    PROPERTIES LABELS "correctness;qc;omnidoublet")

# ---------------------------------------------------------------------------
# ase/daesc correctness harness  (cycle 40)
# ---------------------------------------------------------------------------
add_executable(ase_daesc_correctness
    ase_daesc_correctness.cpp
)
# daesc.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(ase_daesc_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(ase_daesc_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# DAESC ASE uses:
#   CUDA::cudart   — DeviceMemory<float/int32_t> alloc/copy, stream management
#   CUDA::cusparse — per-SNP cell gather from CSC (snp_cell_gather_kernel)
#   CUDA::cublas   — fp64 2×2/3×3 Fisher information matrix inversion inside
#                    batched beta-binomial MLE (small analytic inverse per SNP)
# cub::DeviceSegmentedSort (header-only via CUDAToolkit) for BH FDR correction.
target_link_libraries(ase_daesc_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(ase_daesc_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS found "
                   "— ase_daesc full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS links skipped for ase_daesc.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext) consumed by ase/daesc.h (Fisher scoring
# pattern reuses cycle-38 NEBULA fp64 Hessian inverse) and by
# io/pz_device_loader used in Daesc_GSM_RealData.
target_compile_definitions(ase_daesc_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the R
# reference script at tests/refs/daesc_r_reference.R.
target_compile_definitions(ase_daesc_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(ase_daesc_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(ase_daesc_correctness
    PROPERTIES LABELS "correctness;ase;daesc")

# ---------------------------------------------------------------------------
# fate/cospar correctness harness  (cycle 41)
# ---------------------------------------------------------------------------
add_executable(fate_cospar_correctness
    fate_cospar_correctness.cpp
)
# cospar.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(fate_cospar_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(fate_cospar_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Cospar GPU uses:
#   CUDA::cudart   — DeviceMemory<float/int32_t> alloc/copy, stream management
#   CUDA::cusparse — SpMM for K_state @ T (state coherence regularization) and
#                    lineage coupling sparse gather (lineage_coupling_kernel)
#   CUDA::cublas   — GEMM for fate_bias forward iteration (T^k @ terminal_ind)
# cub::DeviceReduce (header-only, via CUDAToolkit) for Frobenius residual check.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(fate_cospar_correctness PRIVATE CUDA::cudart)
# cuSPARSE for K_state SpMM (state similarity × transition map gradient).
# cuBLAS for fate_bias forward iteration GEMM (n_t0 × n_t1 × n_fates).
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(fate_cospar_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS found "
                   "— fate_cospar full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS links skipped for fate_cospar.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by
# fate/cospar.h (cycle-8 compute_exact kNN for K_state, cycle-33 simplex
# projection pattern for per-row prox step) and by io/pz_device_loader
# used in Cospar_GSM_RealLineage.
target_compile_definitions(fate_cospar_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate the Python
# reference script at tests/refs/cospar_py_reference.py.
target_compile_definitions(fate_cospar_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(fate_cospar_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(fate_cospar_correctness
    PROPERTIES LABELS "correctness;fate;cospar")

# ---------------------------------------------------------------------------
# variants/monopogen correctness harness  (cycle 42)
# ---------------------------------------------------------------------------
add_executable(variants_monopogen_correctness
    variants_monopogen_correctness.cpp
)
# monopogen.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(variants_monopogen_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(variants_monopogen_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# Monopogen uses:
#   CUDA::cudart   — DeviceMemory<float/int32_t> alloc/copy, stream management
#   CUDA::cusparse — per-SNP pileup aggregation (row-sum of snp_ad / snp_dp CSC)
#                    and LD-refined scoring (sparse SpMV against LD panel CSR)
# cub::DeviceSegmentedSort (header-only, via CUDAToolkit) for BH FDR correction.
# CUDA::cudart is required for stream/device management in the test fixture.
target_link_libraries(variants_monopogen_correctness PRIVATE CUDA::cudart)
# cuSPARSE for the pileup SpMM (n_snps × n_cells → n_snps) and LD SpMV.
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(variants_monopogen_correctness PRIVATE CUDA::cusparse)
    message(STATUS "singlet-gpu/tests: cuSPARSE found "
                   "— variants_monopogen full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE link skipped for variants_monopogen.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext) consumed by variants/monopogen.h
# (germline Bayesian kernel reuses cycle-38 NEBULA fp32 pattern) and by
# io/pz_device_loader used in Monopogen_GSM_RealData.
target_compile_definitions(variants_monopogen_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate:
#   tests/refs/monopogen_py_reference.py
#   tests/refs/ld_panel_mini.tsv
target_compile_definitions(variants_monopogen_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(variants_monopogen_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(variants_monopogen_correctness
    PROPERTIES LABELS "correctness;variants;monopogen")

# ---------------------------------------------------------------------------
# fate/cellrank2 correctness harness  (cycle 43)
# ---------------------------------------------------------------------------
add_executable(fate_cellrank2_correctness
    fate_cellrank2_correctness.cpp
)
# cellrank2.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(fate_cellrank2_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(fate_cellrank2_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# CellRank2 GPU uses:
#   CUDA::cudart   — DeviceMemory<float/int32_t> alloc/copy, stream management
#   CUDA::cusparse — SpMV inside batched GMRES for (I - Q) solves
#   CUDA::cublas   — Arnoldi orthogonalization (Hessenberg GEMV/GEMM)
#   CUDA::cusolver — optional Schur decomposition for auto terminal-state ID
# cub::DeviceReduce (header-only, via CUDAToolkit) for residual norm checks.
target_link_libraries(fate_cellrank2_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(fate_cellrank2_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
        CUDA::cusolver
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS + cuSOLVER found "
                   "— fate_cellrank2 full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS/cuSOLVER links skipped for fate_cellrank2.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by
# fate/cellrank2.h (cycle-8 compute_exact kNN for transition matrix,
# cycle-4 factornet SVD Lanczos for terminal state Schur decomp) and by
# io/pz_device_loader used in Cellrank2_GSM_RealData.
target_compile_definitions(fate_cellrank2_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate:
#   tests/refs/cellrank2_py_reference.py
target_compile_definitions(fate_cellrank2_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(fate_cellrank2_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(fate_cellrank2_correctness
    PROPERTIES LABELS "correctness;fate;cellrank2")

# ---------------------------------------------------------------------------
# enrich/ssgsea + enrich/progeny correctness harness  (cycle 44)
# ---------------------------------------------------------------------------
add_executable(enrich_ssgsea_progeny_correctness
    enrich_ssgsea_progeny_correctness.cpp
)
# ssgsea.h / progeny.h contain __global__ kernels — must compile with nvcc.
set_source_files_properties(enrich_ssgsea_progeny_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(enrich_ssgsea_progeny_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# ssGSEA uses:
#   CUDA::cudart   — DeviceMemory<float/int32_t> alloc/copy, stream management
#   CUDA::cusparse — SpMM for per-cell sparse × dense KS accumulation
#   cub::BlockRadixSort (header-only, via CUDAToolkit) for per-cell gene ranking
# PROGENy uses:
#   CUDA::cudart   — DeviceMemory<float> alloc/copy
#   CUDA::cublas   — sgemm: expression (n_cells × n_genes) @ W^T (n_genes × n_pathways)
#   CUDA::cusparse — SpMM when expression is sparse CSC (cycle 0 loader path)
# Pathway normalization (Welford z-score) uses cub::DeviceReduce (header-only).
target_link_libraries(enrich_ssgsea_progeny_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(enrich_ssgsea_progeny_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS found "
                   "— enrich_ssgsea_progeny full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS links skipped for enrich_ssgsea_progeny.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by both
# enrich/ssgsea.h (per-cell rank kernel) and enrich/progeny.h (SGEMM path),
# and by io/pz_device_loader used in the real-data tests.
target_compile_definitions(enrich_ssgsea_progeny_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate:
#   tests/refs/ssgsea_r_reference.R
#   tests/refs/progeny_py_reference.py
#   tests/refs/ssgsea_mini_sets.tsv
#   tests/refs/progeny_human_top100.tsv
target_compile_definitions(enrich_ssgsea_progeny_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(enrich_ssgsea_progeny_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(enrich_ssgsea_progeny_correctness
    PROPERTIES LABELS "correctness;enrich;ssgsea;progeny")

# ---------------------------------------------------------------------------
# fate/palantir correctness harness  (cycle 45)
# ---------------------------------------------------------------------------
add_executable(fate_palantir_correctness
    fate_palantir_correctness.cpp
)
# palantir.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(fate_palantir_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(fate_palantir_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# fate/palantir.h uses:
#   CUDA::cudart   — DeviceMemory<float> alloc/copy, stream management
#   CUDA::cusparse — kNN diffusion operator (sparse P) construction +
#                    anisotropic correction (D^{-1/2} P D^{-1/2})
#   CUDA::cublas   — PCA embedding (dense matrix ops), eigenvector scaling
#   CUDA::cusolver — symmetric eigendecomposition of P' (Lanczos / cuSOLVER SYEVD)
#                    via factornet lanczos_gpu
# Branch probability absorption reuses cycle 43 CellRank2 GMRES (cuSPARSE + cuBLAS).
target_link_libraries(fate_palantir_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(fate_palantir_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
        CUDA::cusolver
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS + cuSOLVER found "
                   "— fate_palantir full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS/cuSOLVER links skipped for fate_palantir.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by:
#   fate/palantir.h (spectral decomp via factornet lanczos_gpu,
#                    kNN from cycle 8 compute_exact,
#                    branch probs via cycle 43 GMRES)
#   io/pz_device_loader (Palantir_GSM_RealData test)
target_compile_definitions(fate_palantir_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate:
#   tests/refs/palantir_py_reference.py
target_compile_definitions(fate_palantir_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(fate_palantir_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(fate_palantir_correctness
    PROPERTIES LABELS "correctness;fate;palantir")

# ---------------------------------------------------------------------------
# network/hdwgcna correctness harness  (cycle 46)
# ---------------------------------------------------------------------------
add_executable(network_hdwgcna_correctness
    network_hdwgcna_correctness.cpp
)
# hdwgcna.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(network_hdwgcna_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(network_hdwgcna_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# network/hdwgcna.h uses:
#   CUDA::cudart   — DeviceMemory<float> alloc/copy, stream management
#   CUDA::cublas   — cuBLAS SYRK for Pearson correlation matrix,
#                    cuBLAS GEMM for TOM numerator computation
#   CUDA::cusparse — cuSPARSE SpMM for large-HVG chunked correlation paths
# Eigengene PCA reuses cycle 4 factornet randomized_gpu (CUDA::cusolver via factornet).
target_link_libraries(network_hdwgcna_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(network_hdwgcna_correctness PRIVATE
        CUDA::cublas
        CUDA::cusparse
    )
    message(STATUS "singlet-gpu/tests: cuBLAS + cuSPARSE found "
                   "— network_hdwgcna full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuBLAS/cuSPARSE links skipped for network_hdwgcna.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by:
#   network/hdwgcna.h (correlation kernel, TOM kernel, eigengene Randomized SVD)
#   io/pz_device_loader (Hdwgcna_GSM_RealData test)
target_compile_definitions(network_hdwgcna_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate:
#   tests/refs/hdwgcna_r_reference.R
#   tests/refs/wgcna_py_reference.py
target_compile_definitions(network_hdwgcna_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(network_hdwgcna_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(network_hdwgcna_correctness
    PROPERTIES LABELS "correctness;network;hdwgcna")

# ---------------------------------------------------------------------------
# abundance/milo correctness harness  (cycle 47)
# ---------------------------------------------------------------------------
add_executable(abundance_milo_correctness
    abundance_milo_correctness.cpp
)
# milo.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(abundance_milo_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(abundance_milo_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# abundance/milo.h uses:
#   CUDA::cudart   — DeviceMemory<float/int32_t> alloc/copy, stream management
#   CUDA::cusparse — SpMV for graph-weighted FDR (NH-NH adjacency BH correction)
#   CUDA::cublas   — optional cuBLAS GEMM in kNN distance computations
# Batched NB-GLM IRLS reuses cycle-15 donor pseudobulk + cycle-38 Fisher-scoring
# fp32/fp64 Hessian-inverse pattern.
target_link_libraries(abundance_milo_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(abundance_milo_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS found "
                   "— abundance_milo full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS links skipped for abundance_milo.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by:
#   abundance/milo.h (kNN graph reuses cycle-8 compute_exact via factornet)
#   io/pz_device_loader (Milo_GSM_RealData test)
target_compile_definitions(abundance_milo_correctness PRIVATE FACTORNET_HAS_GPU)
# gcc-toolset-13 libstdc++ required for _M_replace_cold (same ABI fix as bench/ and disease_scdrs).
# GTest::gtest_main was compiled with gcc-toolset-13; link its stdc++ explicitly.
if(EXISTS "/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13/libstdc++.a")
    target_link_libraries(abundance_milo_correctness PRIVATE
        /opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13/libstdc++.a)
endif()
# Expose the tests/ directory path so the C++ test can locate:
#   tests/refs/milo_r_reference.R
target_compile_definitions(abundance_milo_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(abundance_milo_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(abundance_milo_correctness
    PROPERTIES LABELS "correctness;abundance;milo")

# ---------------------------------------------------------------------------
# integration/e2e correctness harness  (cycle 49a)
# DEFERRED: compile-gate cycle 54/phase-6 deferred this target pending full
# 9-stage API revalidation.  Build it only when explicitly requested.
# DAG: CYCLE-55-E2E-REVALIDATE — realign test against current feature headers.
# To re-enable: cmake -DSINGLET_GPU_BUILD_E2E_TEST=ON ..
# ---------------------------------------------------------------------------
option(SINGLET_GPU_BUILD_E2E_TEST "Build integration_e2e_correctness (deferred)" OFF)
if(SINGLET_GPU_BUILD_E2E_TEST)
add_executable(integration_e2e_correctness
    integration_e2e_correctness.cpp
)
# End-to-end pipeline includes headers with __global__ kernels — must compile with nvcc.
set_source_files_properties(integration_e2e_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(integration_e2e_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# integration_e2e exercises all 9 pipeline stages; link all CUDA libraries
# that the constituent kernels need:
#   CUDA::cudart    — DeviceMemory<T> alloc/copy, stream management
#   CUDA::cusparse  — io/pz_device_loader (SpMM), lognorm (SpMV), kNN (SpGEMM)
#   CUDA::cublas    — SVD (SGEMM via factornet), UMAP (dense ops)
#   CUDA::cusolver  — SVD (gesvdj / syevd via factornet)
target_link_libraries(integration_e2e_correctness PRIVATE CUDA::cudart)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(integration_e2e_correctness PRIVATE
        CUDA::cusparse
        CUDA::cublas
        CUDA::cusolver
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuBLAS + cuSOLVER found "
                   "— integration_e2e full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuBLAS/cuSOLVER links skipped for integration_e2e.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths consumed by every stage:
#   io/pz_device_loader, preprocess/lognorm, preprocess/hvg,
#   reduce/svd/randomized (factornet adapter), graph/knn,
#   graph/leiden, embed/umap, de/wilcoxon, anno/marker_score.
target_compile_definitions(integration_e2e_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate:
#   tests/refs/e2e_scanpy_reference.py
target_compile_definitions(integration_e2e_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(integration_e2e_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(integration_e2e_correctness
    PROPERTIES LABELS "correctness;integration;e2e")
endif() # SINGLET_GPU_BUILD_E2E_TEST

# ---------------------------------------------------------------------------
# disease/scdrs correctness harness  (cycle 50)
# ---------------------------------------------------------------------------
add_executable(disease_scdrs_correctness
    disease_scdrs_correctness.cpp
)
# scdrs.h contains __global__ kernels — must compile with nvcc.
set_source_files_properties(disease_scdrs_correctness.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(disease_scdrs_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# disease/scdrs.h uses:
#   CUDA::cudart   — DeviceMemory<float> alloc/copy, stream management
#   CUDA::cusparse — batched SpMM: expression (n_genes × n_cells) @
#                    weight_matrix (n_genes × (n_diseases + n_controls))
#                    → score_matrix (n_cells × (n_diseases + n_controls));
#                    single SpMM call handles disease + control sets together
#   CUDA::curand   — Philox4x32_10 seeded RNG for control gene set sampling
#                    (host-side Philox; device-side only if GPU binning is used)
# cub::DeviceSegmentedReduce (header-only via CUDA toolkit) for per-cell null
# mean/std computation over n_controls score columns.
# cub::DeviceRadixSort for BH FDR correction sort step.
target_link_libraries(disease_scdrs_correctness PRIVATE CUDA::cudart)
# gcc-toolset-13 libstdc++ required for _M_replace_cold (same ABI fix as bench/).
# GTest::gtest_main was compiled with gcc-toolset-13; link its stdc++ explicitly
# so the cold-path std::string symbols resolve.
if(EXISTS "/opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13/libstdc++.a")
    target_link_libraries(disease_scdrs_correctness PRIVATE
        /opt/rh/gcc-toolset-13/root/usr/lib/gcc/x86_64-redhat-linux/13/libstdc++.a)
endif()
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(disease_scdrs_correctness PRIVATE
        CUDA::cusparse
        CUDA::curand
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuRAND found "
                   "— disease_scdrs full GPU path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuRAND links skipped for disease_scdrs.")
endif()
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, GPUContext, SparseMatrixGPU<float>) consumed by:
#   disease/scdrs.h (batched SpMM score computation, per-cell null normalization)
#   io/pz_device_loader (Scdrs_GSM_RealData test — GSM4037629 exon_counts.1pz)
target_compile_definitions(disease_scdrs_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory path so the C++ test can locate:
#   tests/refs/scdrs_py_reference.py
#   tests/refs/scdrs_diseases_mini.tsv
target_compile_definitions(disease_scdrs_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(disease_scdrs_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(disease_scdrs_correctness
    PROPERTIES LABELS "correctness;disease;scdrs")

# ---------------------------------------------------------------------------
# preprocess/deconv_size_factors correctness harness  (Cycle 87 Phase D)
# ---------------------------------------------------------------------------
add_executable(preprocess_deconv_size_factors_correctness
    preprocess_deconv_size_factors_correctness.cpp
)
# deconv_size_factors.h contains __global__/__device__ kernel definitions —
# must compile with nvcc.  CUDA 12.8 with -std=c++20 handles std::filesystem.
set_source_files_properties(
    preprocess_deconv_size_factors_correctness.cpp
    PROPERTIES LANGUAGE CUDA
)
target_link_libraries(preprocess_deconv_size_factors_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
# deconv_size_factors.h uses CUDA runtime for device allocations,
# cudaMemcpyAsync, cudaStreamSynchronize, and the pool-based linear-system
# solve (cuSOLVER).  CUDA::cudart is required.
target_link_libraries(preprocess_deconv_size_factors_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU code-paths in factornet/core headers
# (DeviceMemory<T>, SparseMatrixGPU<float>, GPUContext) used by the kernel
# and by the io/pz_device_loader (Test 3 real-data path).
target_compile_definitions(preprocess_deconv_size_factors_correctness
    PRIVATE FACTORNET_HAS_GPU)
# Expose the tests/ directory so the C++ test can locate:
#   tests/refs/scran_ref.R  (R scran reference for Test 2/3)
target_compile_definitions(preprocess_deconv_size_factors_correctness PRIVATE
    SINGLET_GPU_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
set_target_properties(preprocess_deconv_size_factors_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
# Optional: link cuSPARSE + cuSOLVER if present (pool solve uses cuSOLVER).
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    target_link_libraries(preprocess_deconv_size_factors_correctness PRIVATE
        CUDA::cusparse
        CUDA::cusolver
    )
    message(STATUS "singlet-gpu/tests: cuSPARSE + cuSOLVER found "
                   "— deconv_size_factors full solve path enabled.")
else()
    message(STATUS "singlet-gpu/tests: CUDAToolkit not found "
                   "— cuSPARSE/cuSOLVER links skipped for deconv_size_factors.")
endif()
gtest_discover_tests(preprocess_deconv_size_factors_correctness
    PROPERTIES LABELS "correctness;preprocess;deconv_size_factors")

# ---------------------------------------------------------------------------
# preprocess/hvg deviance (scry binomial + Poisson) correctness harness  (Cycle 88 Phase D)
# ---------------------------------------------------------------------------
add_executable(preprocess_hvg_deviance_correctness
    preprocess_hvg_deviance_correctness.cpp
)
# hvg.h contains __global__/__device__ kernel definitions — must compile with nvcc.
# CUDA 12.8 with -std=c++20 handles std::filesystem correctly.
set_source_files_properties(
    preprocess_hvg_deviance_correctness.cpp
    PROPERTIES LANGUAGE CUDA
)
target_link_libraries(preprocess_hvg_deviance_correctness PRIVATE
    singlet-gpu::singlet-gpu
    GTest::gtest_main
)
target_link_libraries(preprocess_hvg_deviance_correctness PRIVATE CUDA::cudart)
# FACTORNET_HAS_GPU enables GPU DeviceMemory / SparseMatrixGPU code paths.
target_compile_definitions(preprocess_hvg_deviance_correctness PRIVATE FACTORNET_HAS_GPU)
# Expose tests/ directory so the harness can locate refs/ scripts.
target_compile_definitions(preprocess_hvg_deviance_correctness PRIVATE
    SINGLET_GPU_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
set_target_properties(preprocess_hvg_deviance_correctness PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)
gtest_discover_tests(preprocess_hvg_deviance_correctness
    PROPERTIES LABELS "correctness;preprocess;hvg_deviance")
