# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/CMakeLists.txt
# Owned by gpu-bench. Add a new cuda_add_executable entry per perf driver.
#
# Depends on:
#   singlet-gpu::singlet-gpu   — header-only INTERFACE (includes, compile flags)
#   CUDA::cudart               — cuEvents for timing
#
# Bench drivers are NOT linked against GoogleBenchmark — we use a minimal
# header-only cuEvent wrapper (bench/include/singlet_gpu/bench/harness.h)
# to avoid an extra external dependency. This makes bench drivers build in
# the same environment as the library itself.

cmake_minimum_required(VERSION 3.24)

# The bench include directory (harness.h).
set(BENCH_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------
# Helper macro — adds one benchmark driver as a CUDA executable.
# Usage: add_bench_driver(<name> <source.cpp>)
# ---------------------------------------------------------------------------
macro(add_bench_driver _target _source)
    # Force .cpp bench drivers to be compiled as CUDA TUs: they include CUDA kernel
    # headers (cub, factornet/gpu/) which require nvcc, not g++.
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/${_source}" PROPERTIES LANGUAGE CUDA)
    add_executable(${_target} "${CMAKE_CURRENT_SOURCE_DIR}/${_source}")
    target_include_directories(${_target} PRIVATE ${BENCH_INCLUDE_DIR})
    target_link_libraries(${_target} PRIVATE
        singlet-gpu::singlet-gpu
        CUDA::cudart
    )
    set_target_properties(${_target} PROPERTIES
        CXX_STANDARD  20
        CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        # Allow compilation as CUDA translation unit (contains CUDA headers).
        CUDA_SEPARABLE_COMPILATION ON
    )
    # Register as CTest test so set_tests_properties can tag it with "bench" label.
    add_test(NAME ${_target} COMMAND ${_target})
    set_tests_properties(${_target} PROPERTIES LABELS "bench" PASS_REGULAR_EXPRESSION "BENCH")
endmacro()

# ---------------------------------------------------------------------------
# Cycle 53a drivers: io + preprocess + reduce
# ---------------------------------------------------------------------------

add_bench_driver(bench_io_pz_loader_perf        bench_io_pz_loader_perf.cpp)
add_bench_driver(bench_preprocess_lognorm_perf   bench_preprocess_lognorm_perf.cpp)
add_bench_driver(bench_preprocess_hvg_perf       bench_preprocess_hvg_perf.cpp)
add_bench_driver(bench_reduce_svd_perf           bench_reduce_svd_perf.cpp)
add_bench_driver(bench_reduce_nmf_perf           bench_reduce_nmf_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 53b drivers (graph/embed/de) — source files already present.
# ---------------------------------------------------------------------------
add_bench_driver(bench_graph_knn_perf          bench_graph_knn_perf.cpp)
add_bench_driver(bench_graph_leiden_perf       bench_graph_leiden_perf.cpp)
add_bench_driver(bench_embed_umap_perf         bench_embed_umap_perf.cpp)
add_bench_driver(bench_de_wilcoxon_perf        bench_de_wilcoxon_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 57 drivers: feature 0 Phase E comprehensive benchmark
# ---------------------------------------------------------------------------
add_bench_driver(io_pz_device_loader_bench    io_pz_device_loader_bench.cpp)

# ---------------------------------------------------------------------------
# Cycle 58 drivers: feature 2 lognorm Phase E benchmark
# ---------------------------------------------------------------------------
add_bench_driver(bench_preprocess_lognorm_phaseE  bench_preprocess_lognorm_phaseE.cpp)

# ---------------------------------------------------------------------------
# Cycle 59 drivers: feature 3 HVG Phase E benchmark
# ---------------------------------------------------------------------------
add_bench_driver(bench_preprocess_hvg_phaseE  bench_preprocess_hvg_phaseE.cpp)

# ---------------------------------------------------------------------------
# Cycle 60 drivers: feature 4 SVD adapters Phase E benchmark
# ---------------------------------------------------------------------------
add_bench_driver(bench_reduce_svd_phaseE  bench_reduce_svd_phaseE.cpp)

# ---------------------------------------------------------------------------
# Cycle 61 drivers: feature 5 NMF adapters Phase E benchmark
# ---------------------------------------------------------------------------
add_bench_driver(bench_reduce_nmf_phaseE  bench_reduce_nmf_phaseE.cpp)

# ---------------------------------------------------------------------------
# Cycle 85 drivers: DE wilcoxon + ttest multi-scale scanpy CPU baseline
# ---------------------------------------------------------------------------
add_bench_driver(bench_de_ttest_perf          bench_de_ttest_perf.cpp)
add_bench_driver(bench_de_wilcoxon_perf_c85   bench_de_wilcoxon_perf_c85.cpp)

# ---------------------------------------------------------------------------
# Cycle 86 drivers: OPTIM-NMF-K50 MU-default verification bench + Phase B profiling
# ---------------------------------------------------------------------------
add_bench_driver(bench_reduce_nmf_c86          bench_reduce_nmf_c86.cpp)
add_bench_driver(bench_nmf_profile_c86         bench_nmf_profile_c86.cpp)

# ---------------------------------------------------------------------------
# Cycle 86 Phase E drivers: NMF post-fix adapter bench + QC metrics bench
# ---------------------------------------------------------------------------
add_bench_driver(bench_nmf_phaseE_c86          bench_nmf_phaseE_c86.cpp)
add_bench_driver(bench_qc_perf                 bench_qc_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 157 drivers: preprocess/pearson_residuals standalone Phase E benchmark
# ---------------------------------------------------------------------------
add_bench_driver(bench_preprocess_pearson_residuals_perf  bench_preprocess_pearson_residuals_perf.cpp)

# ---------------------------------------------------------------------------
# Status message
# ---------------------------------------------------------------------------
message(STATUS "singlet-gpu bench: 21 drivers registered (cycles 53a+53b+57+58+59+60+61+85+86+86E+157)")
message(STATUS "  53a: bench_io_pz_loader_perf")
message(STATUS "  53a: bench_preprocess_lognorm_perf")
message(STATUS "  53a: bench_preprocess_hvg_perf")
message(STATUS "  53a: bench_reduce_svd_perf")
message(STATUS "  53a: bench_reduce_nmf_perf")
message(STATUS "  53b: bench_graph_knn_perf")
message(STATUS "  53b: bench_graph_leiden_perf")
message(STATUS "  53b: bench_embed_umap_perf")
message(STATUS "  53b: bench_de_wilcoxon_perf")
message(STATUS "  59: bench_preprocess_hvg_phaseE")
message(STATUS "  61: bench_reduce_nmf_phaseE")
message(STATUS "  85: bench_de_ttest_perf")
message(STATUS "  85: bench_de_wilcoxon_perf_c85")
message(STATUS "  86: bench_reduce_nmf_c86")
message(STATUS "  86E: bench_nmf_phaseE_c86")
message(STATUS "  86E: bench_qc_perf")
