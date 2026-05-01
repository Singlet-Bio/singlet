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
# Cycle 158 drivers: enrich/score_genes standalone Phase E benchmark
# ---------------------------------------------------------------------------
add_bench_driver(bench_enrich_score_genes_perf  bench_enrich_score_genes_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 159 drivers: embed/diffmap standalone Phase E benchmark
# ---------------------------------------------------------------------------
add_bench_driver(bench_embed_diffmap_perf  bench_embed_diffmap_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 161 drivers: embed/dpt standalone Phase E benchmark (§J.6 AT-RISK test)
# ---------------------------------------------------------------------------
add_bench_driver(bench_embed_dpt_perf  bench_embed_dpt_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 162 drivers: preprocess/model_gene_var standalone Phase E benchmark
# GPU port of scran::modelGeneVarByPoisson (Lun-McCarthy-Marioni 2016).
# O(nnz) scatter + cub::DeviceRadixSort top-N; NOT at risk of §J.6 dense-n×n bug.
# ---------------------------------------------------------------------------
add_bench_driver(bench_preprocess_model_gene_var_perf  bench_preprocess_model_gene_var_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 163 drivers: enrich/decoupler_wsum + decoupler_wmean Phase E benchmark
# cuSPARSE SpMM (X^T · W) + per-pathway normalisation kernels.
# WSUM (L1-norm) and WMEAN (nnz-count) — NOT at risk of §J.6 dense-n×n bug.
# ---------------------------------------------------------------------------
add_bench_driver(bench_enrich_decoupler_wsum_perf  bench_enrich_decoupler_wsum_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 164 drivers: enrich/decoupler_ulm Phase E benchmark
# ULM (Univariate Linear Model) OLS slope β_1[c,p] = cov(X_c,W_p)/var(W_p).
# 5 passes: mean_X scatter, W stats fused warp-shuffle, var_W, SpMM, score.
# NOT at risk of §J.6 dense-n×n bug (SpMM O(nnz × p)).
# ---------------------------------------------------------------------------
add_bench_driver(bench_enrich_decoupler_ulm_perf  bench_enrich_decoupler_ulm_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 165 drivers: enrich/decoupler_mlm Phase E benchmark
# MLM (Multivariate Linear Model) Cholesky OLS B = (W^T W + ridge*I)^{-1} W^T X.
# 5 passes: Sgemm W^T W, ridge kernel, SpMM X^T W, Spotrf/Spotrs, Sgeam.
# NOT at risk of §J.6 dense-n×n bug (SpMM O(nnz × p); solve O(p^2 × n), p=50).
# ---------------------------------------------------------------------------
add_bench_driver(bench_enrich_decoupler_mlm_perf  bench_enrich_decoupler_mlm_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 166 drivers: enrich/decoupler_ora Phase E benchmark
# ORA (Over-Representation Analysis): top-K (5%) genes per cell via 64-bucket
# shared-mem histogram; hypergeometric -log10(p) via lgamma log-sum-exp kernel.
# 4 passes: topk_smem, build_T, Sgemm T^T·M, ora_hypergeo_kernel.
# NOT at risk of §J.6 dense-n×n bug (T is m×n with m≤5k genes; lgamma O(n*p*K)).
# ---------------------------------------------------------------------------
add_bench_driver(bench_enrich_decoupler_ora_perf  bench_enrich_decoupler_ora_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 167 drivers: enrich/decoupler_viper Phase E benchmark
# VIPER/aREA (Alvarez et al. 2016 Nat Genet): per-cell rank transform via CUB
# DeviceSegmentedRadixSort, normcdfinvf qnorm → T1, cuBLAS Sgemm T1^T·W,
# per-regulon L1-norm warp-shuffle scale.
# V0 simplification: NES = ES1 = (T1^T · W) / L1(W).
# NOT at risk of §J.6 dense-n×n bug (T1 is m×n, m=5k genes; Sgemm O(m*n*n_reg)).
# ---------------------------------------------------------------------------
add_bench_driver(bench_enrich_decoupler_viper_perf  bench_enrich_decoupler_viper_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 169 drivers: integrate/lisi Phase E benchmark
# LISI — Local Inverse Simpson's Index (Korsunsky et al. 2019 Nat Methods).
# One block per cell; shared-mem histogram of n_labels ints; per-cell D =
# sum(p_l^2), LISI = 1/D. k=10, n_batches=4 synthetic round-robin labels.
# kNN (Exact, L2, k=10) is built once as an untimed warmup; only lisi() is
# timed. Scales: 10k / 30k cells × 50 PCs.
# ---------------------------------------------------------------------------
add_bench_driver(bench_integrate_lisi_perf  bench_integrate_lisi_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 170 drivers: integrate/asw Phase E benchmark
# ASW — Average Silhouette Width (Rousseeuw 1987; scIB default, Luecken 2022).
# One block per cell; shared-mem float smem_sum[n_labels] + int smem_cnt[n_labels];
# thread 0 sweeps kNN row, computes a (intra-cluster mean dist), b (min other-
# cluster mean dist), silhouette = (b-a)/max(a,b). cub::DeviceReduce::Sum for
# scalar ASW mean. k=10/15, n_clusters=4 synthetic round-robin labels.
# kNN (Exact, L2) is built once as an untimed warmup; only asw() is timed.
# Scales: 10k cells × 50 PCs (k=10), 30k cells × 50 PCs (k=15).
# ---------------------------------------------------------------------------
add_bench_driver(bench_integrate_asw_perf  bench_integrate_asw_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 171 drivers: integrate/kbet Phase E benchmark
# kBET — k-nearest neighbor Batch Effect Test (Büttner et al. 2019 Nat Methods).
# One block per cell; shared-mem int local_count[n_batches]; thread 0 serial
# scan builds batch histogram, computes chi2 = Σ_b (obs-exp)²/max(exp,1),
# p-value via Wilson-Hilferty cube-root transform (df==1: erfc exact).
# kbet_reject_kernel marks p<0.05; cub::DeviceReduce::Sum for mean_chi2 and
# reject_rate. k=10/15, n_batches=4 synthetic round-robin labels.
# kNN (Exact, L2) is built once as an untimed warmup; only kbet() is timed.
# Scales: 10k cells × 50 PCs (k=10), 30k cells × 50 PCs (k=15).
# ---------------------------------------------------------------------------
add_bench_driver(bench_integrate_kbet_perf  bench_integrate_kbet_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 172 drivers: graph/kmeans Phase E benchmark
# Lloyd k-means clustering on PCA embedding (CYCLE-149 kernel).
# Forgy init; cublasSgemm D=X^T·C per iter; km_assign_kernel (dist+argmin,
# one block/cell); atomic-scatter centroid update; inertia kernel.
# Scales: 10k / 30k cells × 50 PCs, k=10, max_iter=20.
# ---------------------------------------------------------------------------
add_bench_driver(bench_graph_kmeans_perf  bench_graph_kmeans_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 173 drivers: embed/dendrogram Phase E benchmark
# UPGMA hierarchical clustering on cluster centroids (CYCLE-146 kernel).
# Atomic-scatter centroid aggregation (one block/cell); center + L2-normalize;
# cublasSgemm corr = μ_norm^T · μ_norm (k × k); d = 1 - corr; host UPGMA O(k³).
# Scales: 10k / 30k cells × 50 PCs, k=20 clusters, round-robin labels, dense CSC.
# ---------------------------------------------------------------------------
add_bench_driver(bench_embed_dendrogram_perf  bench_embed_dendrogram_perf.cpp)

# ---------------------------------------------------------------------------
# Cycle 174 drivers: preprocess/magic Phase E benchmark
# MAGIC graph-diffusion imputation (van Dijk et al. 2018 Cell 174:716-729).
# cuSPARSE SpMM-heavy: row_sum + normalize_weights + csc_transpose_to_dense
# + cusparseSpMM × t=3 (ping-pong dense buffers, CUSPARSE_SPMM_ALG_DEFAULT).
# Scales: 10k / 30k cells × 5000 genes, density=5%, k=10 SNN, t=3.
# ---------------------------------------------------------------------------
add_bench_driver(bench_preprocess_magic_perf  bench_preprocess_magic_perf.cpp)

# ---------------------------------------------------------------------------
# Status message
# ---------------------------------------------------------------------------
message(STATUS "singlet-gpu bench: 36 drivers registered (cycles 53a+53b+57+58+59+60+61+85+86+86E+157+158+159+161+162+163+164+165+166+167+169+170+171+172+173+174)")
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
message(STATUS "  57:  io_pz_device_loader_bench")
message(STATUS "  58:  bench_preprocess_lognorm_phaseE")
message(STATUS "  60:  bench_reduce_svd_phaseE")
message(STATUS "  157: bench_preprocess_pearson_residuals_perf")
message(STATUS "  158: bench_enrich_score_genes_perf")
message(STATUS "  159: bench_embed_diffmap_perf")
message(STATUS "  161: bench_embed_dpt_perf")
message(STATUS "  162: bench_preprocess_model_gene_var_perf")
message(STATUS "  163: bench_enrich_decoupler_wsum_perf")
message(STATUS "  164: bench_enrich_decoupler_ulm_perf")
message(STATUS "  165: bench_enrich_decoupler_mlm_perf")
message(STATUS "  166: bench_enrich_decoupler_ora_perf")
message(STATUS "  167: bench_enrich_decoupler_viper_perf")
message(STATUS "  169: bench_integrate_lisi_perf")
message(STATUS "  170: bench_integrate_asw_perf")
message(STATUS "  171: bench_integrate_kbet_perf")
message(STATUS "  172: bench_graph_kmeans_perf")
message(STATUS "  173: bench_embed_dendrogram_perf")
message(STATUS "  174: bench_preprocess_magic_perf")
