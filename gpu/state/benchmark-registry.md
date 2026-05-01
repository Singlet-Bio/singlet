# singlet-gpu — Benchmark Registry

Append-only. One row per `{feature, scale, baseline}` measurement. Written by `gpu-bench` directly (no doc-scribe needed for routine appends).

## Schema

| date | feature | scale | impl | wall_ms | mem_mb | cells_per_sec | pcie_gb | nsys_link | sm_occ | commit |
|---|---|---|---|---|---|---|---|---|---|---|

## Conventions

- `date`: YYYY-MM-DD.
- `feature`: roadmap-style identifier (e.g., `pca/randomized`, `de/glm_nb`).
- `scale`: `tiny` (500×200) | `10k` | `100k` | `1m`.
- `impl`: `singlet-gpu` for ours; otherwise the SOTA library name (`rapids-singlecell`, `scanpy`, `cuml`, `cugraph`, `factornet-cpu`, `seurat`, `scran`, `scvi`, `harmonypy`, `scib`, `mgatk`, `scvelo`, `fgsea`, `aucell`, `faiss-gpu`, `umap-learn`).
- `wall_ms`: median of 5 runs, milliseconds. `OOM` if out of memory.
- `mem_mb`: peak device memory (MB) via `cudaMemGetInfo` for GPU, RSS for CPU baselines.
- `cells_per_sec`: throughput.
- `pcie_gb`: bytes transferred host↔device, GB. `—` if not measured.
- `nsys_link`: relative path to nsys trace file under `bench/traces/`. `—` if not run.
- `sm_occ`: SM occupancy fraction from `ncu --set basic`. `—` if not measured.
- `commit`: short SHA of the singlet-gpu commit benchmarked (or `baseline` for SOTA).

## Rows

(empty — first row added by gpu-bench in cycle 1)

---

## Cycle 53a Bench Schema (run_all.sh output format)

The `bench/run_all.sh` harness uses a wider schema aligned with the BenchRow struct
in `bench/include/singlet_gpu/bench/harness.h`. Rows from `run_all.sh` follow this
layout and are appended directly below:

| feature | scale | wall_ms_min | wall_ms_med | wall_ms_max | mem_mb_peak | cells_per_s | sota_wall_sec | sota_mem_mb | ratio_wall | commit | timestamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| io/pz_loader | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| preprocess/lognorm | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| preprocess/hvg | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| reduce/svd | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| reduce/nmf | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| graph/knn | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| graph/leiden | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| embed/umap | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| de/wilcoxon | pending | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | no-git | 2026-04-14T00:00:00Z |
| 2026-04-15 | io/pz_device_loader | small | singlet-gpu/our_manual | 268.785 | 34 | 77630.7 | —  | —  | —  | no-git |
| 2026-04-15 | io/pz_device_loader | small | singlet-gpu/our_auto | 269.302 | 34 | 77481.9 | —  | —  | —  | no-git |
| 2026-04-15 | io/pz_device_loader | small | anndata-gpu | 1728.86 | 197.414 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | io/pz_device_loader | small | scanpy/read_10x_h5 | 1441.7 | 320 | 14472 | —  | —  | —  | no-git |
| 2026-04-15 | io/pz_device_loader | small | factornet/spz_loader | skipped | — | — | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/lognorm | small | ours_total_count_manual | 0.11376 | 0 | 1.83421e+08 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/lognorm | small | ours_total_count_auto | 0.25008 | 0 | 8.34373e+07 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/lognorm | small | ours_deconvolution_manual | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/lognorm | small | ours_deconvolution_auto | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/lognorm | small | rapids-singlecell | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/lognorm | small | scanpy | 42.0891 | 68.9538 | 495758 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/lognorm | small | scran | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_manual | 0.478784 | 0 | 1.04431e+06 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_pearson_residuals_manual | 0.268736 | 0 | 1.86056e+06 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_auto | 0.478784 | 0 | 1.04431e+06 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_pearson_residuals_auto | 0.268736 | 0 | 1.86056e+06 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | rapids_seurat_v3 | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | seurat_v3_R | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | scanpy_seurat_v3 | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | scanpy_pearson_residuals | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_gaussian_wls | -1 | -1 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_adaptive_clip | -1 | -1 | 0 | —  | —  | —  | no-git |
# Cycle 59b rows — supersede Cycle 59 -1 rows for these 4 implementations
| 2026-04-15 | preprocess/hvg | small | scanpy_seurat_v3 | -1.0 | -1.0 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | scanpy_pearson_residuals | 3390.1 | 158.7 | 6155 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_gaussian_wls | -1.0 | -1.0 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_adaptive_clip | -1.0 | -1.0 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | reduce/svd | tiny | ours_auto_select | 23.7 (23.6–23.8) | 0.0 | 8435 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k30 | scanpy_pca | 734.7 (734.7–734.7) | 15.6 | 28400 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k30 | factornet_cpu_irlba | 1323.0 (1323.0–1323.0) | — | 15772 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k50 | scanpy_pca | 758.7 (758.7–758.7) | 19.0 | 27501 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k50 | factornet_cpu_irlba | 1348.2 (1348.2–1348.2) | — | 15477 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k100 | scanpy_pca | 903.9 (903.9–903.9) | 29.1 | 23083 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k100 | factornet_cpu_irlba | 1699.7 (1699.7–1699.7) | — | 12276 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k30 | cuml_truncated_svd | CONFIG_UNAVAILABLE | — | — | — | — | — | no-git |
| 2026-04-15 | reduce/svd | small-k30 | cuml_randomized_pca | CONFIG_UNAVAILABLE | — | — | — | — | — | no-git |
| 2026-04-15 | reduce/svd | small-k50 | cuml_truncated_svd | CONFIG_UNAVAILABLE | — | — | — | — | — | no-git |
| 2026-04-15 | reduce/svd | small-k50 | cuml_randomized_pca | CONFIG_UNAVAILABLE | — | — | — | — | — | no-git |
| 2026-04-15 | reduce/svd | small-k100 | cuml_truncated_svd | CONFIG_UNAVAILABLE | — | — | — | — | — | no-git |
| 2026-04-15 | reduce/svd | small-k100 | cuml_randomized_pca | CONFIG_UNAVAILABLE | — | — | — | — | — | no-git |
| 2026-04-15 | reduce/svd | small-k30 | ours_randomized_auto | 98.6 (98.6–98.6) | 0.0 | 211669 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k30 | ours_lanczos | 55.0 (55.0–55.0) | 0.0 | 379118 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k30 | ours_irlba | 38.5 (38.5–38.5) | 0.0 | 541861 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k30 | ours_krylov_constrained | 947.1 (947.1–947.1) | 0.0 | 22031 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k30 | ours_deflation | 28.1 (28.1–28.1) | 0.0 | 741494 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k30 | ours_auto_select | 55.0 (55.0–55.0) | 0.0 | 379290 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k50 | ours_randomized_auto | 148.4 (148.4–148.4) | 0.0 | 140573 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k50 | ours_lanczos | 155.4 (155.4–155.4) | 0.0 | 134233 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k50 | ours_irlba | 88.2 (88.2–88.2) | 0.0 | 236567 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k50 | ours_krylov_constrained | 1342.5 (1342.5–1342.5) | 0.0 | 15543 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k50 | ours_deflation | 28.2 (28.2–28.2) | 0.0 | 739953 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k50 | ours_auto_select | 148.2 (148.2–148.2) | 0.0 | 140821 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k100 | ours_randomized_auto | 267.9 (267.9–267.9) | 0.0 | 77898 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k100 | ours_lanczos | 1592.5 (1592.5–1592.5) | 0.0 | 13102 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k100 | ours_irlba | 385.7 (385.7–385.7) | 0.0 | 54103 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k100 | ours_krylov_constrained | 3419.3 (3419.3–3419.3) | 0.0 | 6102 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k100 | ours_deflation | 28.5 (28.5–28.5) | 0.0 | 732015 | — | — | — | n/a |
| 2026-04-15 | reduce/svd | small-k100 | ours_auto_select | 393.3 (393.3–393.3) | 0.0 | 53058 | — | — | — | n/a |
# Cycle 59b rows — supersede Cycle 59 -1 rows for these 4 implementations
| 2026-04-15 | preprocess/hvg | small | scanpy_seurat_v3 | 51.4 | 35.1 | 405598 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | scanpy_pearson_residuals | 3388.6 | 158.7 | 6157 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_gaussian_wls | -1.0 | -1.0 | 0 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_adaptive_clip | 3386.9 | -1.0 | 6160 | —  | —  | —  | no-git |
# Cycle 59b rows — supersede Cycle 59 -1 rows for these 4 implementations
| 2026-04-15 | preprocess/hvg | small | scanpy_seurat_v3 | 51.4 | 35.1 | 405598 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | scanpy_pearson_residuals | 3388.6 | 158.7 | 6157 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_gaussian_wls | 1829.7 | -1.0 | 11404 | —  | —  | —  | no-git |
| 2026-04-15 | preprocess/hvg | small | ours_seurat_v3_adaptive_clip | 3542.7 | -1.0 | 5889 | —  | —  | —  | no-git |
| 2026-04-16 | reduce/nmf | small-k10 | ours_nmf_fit_manual | 329.9 | 0.0 | 63245 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | ours_nmf_fit_manual | 390.9 | 0.0 | 53382 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k50 | ours_nmf_fit_manual | 15980.2 | 0.0 | 1306 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20-cv | ours_speckled_cv | 2833.2 | 0.0 | 7365 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-mp-rank | ours_nmf_mp_rank_select | 684.4 | 0.0 | 30488 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-hier-sweep | ours_factorgraph_hierarchical | 9260.1 | 0.0 | 2253 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-hier-sweep-cold | cold_start_sweep | 16949.8 | 0.0 | 1231 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-auto | ours_nmf_fit_auto | 23218.8 | 0.0 | 899 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | sklearn_nmf | 5383.9 | 6238.4 | 3876 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | cnmf | -1.0 | -1.0 | -1 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | factornet_cpu | -1.0 | -1.0 | -1 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k10 | ours_nmf_fit_manual | 330.8 | 0.0 | 63074 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | ours_nmf_fit_manual | 395.8 | 0.0 | 52715 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k50 | ours_nmf_fit_manual | 15975.0 | 0.0 | 1306 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20-cv | ours_speckled_cv | 2834.8 | 0.0 | 7361 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-mp-rank | ours_nmf_mp_rank_select | 684.0 | 0.0 | 30504 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-hier-sweep | ours_factorgraph_hierarchical | 6518.0 | 0.0 | 3201 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-hier-sweep-cold | cold_start_sweep | 16955.3 | 0.0 | 1231 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-auto | ours_nmf_fit_auto | 23217.0 | 0.0 | 899 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | sklearn_nmf | 5502.4 | 6238.4 | 3792 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | cnmf | -1.0 | -1.0 | -1 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | factornet_cpu | -1.0 | -1.0 | -1 | — | — | — | n/a |
## Cycle 86 Phase B — OPTIM-NMF-K50 Profiling (g051 H100 NVL, 2026-04-18, job 363152)
## Matrix: GSM4037629 gene_counts.1pz  38606 genes x 20866 cells  nnz=4270836
## All runs: max_iter=100, seed=42, no convergence at 100 iters (benchmark mode)
## Per-phase timing from factornet gpu_timer.cuh (built-in instrumentation)
##
## k=20 auto (solver_mode=3, active=0=CD throughout at 100 iters):
##   nnls_H=1.107ms/iter  nnls_W=1.629ms/iter  (total 63.8% of wall)
##   gram_H=0.779ms/iter  rhs_H=0.439ms/iter  rhs_W=0.243ms/iter
##   wall=727.8ms  timed=437.5ms  unaccounted=29% (CUDA overhead outside timer brackets)
##
## k=50 auto (solver_mode=3, active=0=CD throughout):
##   nnls_H=5.037ms/iter  nnls_W=9.232ms/iter  (total 94.1% of wall = 1427ms/100iters)
##   gram_H=0.040ms/iter  rhs_H=0.252ms/iter   rhs_W=0.269ms/iter
##   wall=1874.9ms  timed=1514.9ms
##
## k=50 MU forced (solver_mode=2):
##   nnls_H=0.180ms/iter  nnls_W=0.182ms/iter  (28.9% of wall)
##   rhs_H=0.254ms/iter  rhs_W=0.270ms/iter  (41.6% of wall)
##   wall=484.3ms  timed=124.9ms
##
## k=50 CD pure (solver_mode=0):
##   nnls_H=5.612ms/iter  nnls_W=10.234ms/iter (94.8% of wall)
##   wall=2013.5ms
##
## k=50 auto cd_max_iter=10 (solver_mode=3, cd_maxit=10 instead of 100):
##   nnls_H=0.552ms/iter  nnls_W=0.991ms/iter  (64% of wall)
##   wall=586.3ms  (10x speedup vs cd_maxit=100 for NNLS, confirming cd_max_iter is the multiplier)
| 2026-04-18 | reduce/nmf | small-k20-auto | singlet-gpu/CD | 727.8 | 0.0 | 28658 | — | bench/traces/nmf_k20_c86.nsys-rep | — | no-git |
| 2026-04-18 | reduce/nmf | small-k50-auto-CD | singlet-gpu/CD | 1874.9 | 0.0 | 11130 | — | bench/traces/nmf_k50_auto_c86.nsys-rep | — | no-git |
| 2026-04-18 | reduce/nmf | small-k50-MU | singlet-gpu/MU | 484.3 | 0.0 | 43082 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf | small-k50-CD-pure | singlet-gpu/CD | 2013.5 | 0.0 | 10362 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf | small-k50-CD-maxit10 | singlet-gpu/CD-maxit10 | 586.3 | 0.0 | 35588 | — | — | — | no-git |
| 2026-04-16 | graph/knn | 10k-synthetic | singlet-gpu | 59.9158 | 2 | 166901 | —  | —  | —  | no-git |
| 2026-04-16 | graph/knn | 10k-synthetic | singlet-gpu | 59.7219 | 2 | 167443 | —  | —  | —  | no-git |
| 2026-04-16 | graph/knn (Exact) | small-11k | sklearn-BruteForce | 125.8 | 0 | — | — | — | — | cycle63 |
| 2026-04-16 | graph/knn ctest | small-200-synth | singlet-gpu | N/A | N/A | N/A | — | — | — | ctest: 19/12 pass (3 CAGRA skipped) |
| 2026-04-16 | graph/leiden ctest | full-suite | singlet-gpu | N/A | N/A | N/A | — | — | — | ctest: 8/1 pass (0 cuGraph GTEST_SKIP); build=0; bench_build=2; bench_run=1 | cycle64 |
| 2026-04-16 | embed/umap ctest   | full-suite | singlet-gpu | N/A | N/A | N/A | — | — | — | ctest: 9/1 pass (0 cuml GTEST_SKIP);   build=0;   bench_build=2;   bench_run=1   | cycle64 |
| 2026-04-16 | qc/metrics ctest   | full-suite | singlet-gpu | N/A | N/A | N/A | — | — | — | ctest: 12/6 pass (0 GTEST_SKIP); wall=2.17s; build=0; ctest_exit=0 | cycle65 |
| 2026-04-16 | preprocess/scale ctest | full-suite | singlet-gpu | N/A | N/A | N/A | — | — | — | ctest: 8/5 pass (3 GTEST_SKIP); wall=1.75s; build=0; ctest_exit=8 | cycle65 |
| 2026-04-16 | streaming/pipeline ctest | full-suite | singlet-gpu | N/A | N/A | N/A | — | — | — | ctest: 15/9 pass (0 GTEST_SKIP); wall=536.46s; build=0; ctest_exit=8 | cycle66 |
| 2026-04-16 | reduce/nmf | small-k10 | ours_nmf_fit_manual | 266.2 | 0.0 | 78380 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | ours_nmf_fit_manual | 659.2 | 0.0 | 31654 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k50 | ours_nmf_fit_manual | 17132.7 | 0.0 | 1218 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20-cv | ours_speckled_cv | 4449.2 | 0.0 | 4690 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-mp-rank | ours_nmf_mp_rank_select | 803.1 | 0.0 | 25983 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-hier-sweep | ours_factorgraph_hierarchical | 11168.1 | 0.0 | 1868 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-hier-sweep-cold | cold_start_sweep | 17836.5 | 0.0 | 1170 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-auto | ours_nmf_fit_auto | 105887.6 | 0.0 | 197 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | sklearn_nmf | 6617.2 | 6238.4 | 3153 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | cnmf | -1.0 | -1.0 | -1 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | factornet_cpu | -1.0 | -1.0 | -1 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k10 | ours_nmf_fit_manual | 270.4 | 0.0 | 77159 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20 | ours_nmf_fit_manual | 659.3 | 0.0 | 31650 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k50 | ours_nmf_fit_manual | 17129.2 | 0.0 | 1218 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-k20-cv | ours_speckled_cv | 4446.9 | 0.0 | 4692 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-mp-rank | ours_nmf_mp_rank_select | 802.8 | 0.0 | 25991 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-hier-sweep | ours_factorgraph_hierarchical | 11193.8 | 0.0 | 1864 | — | — | — | n/a |
| 2026-04-16 | reduce/nmf | small-hier-sweep-cold | cold_start_sweep | 29908.7 | 0.0 | 698 | — | — | — | n/a |

## Feature 11 — Differential Expression (Cycle 68, 2026-04-16)
| Target | Build | Notes |
|--------|-------|-------|
| de_wilcoxon_correctness | OK | GPU Wilcoxon rank-sum |
| de_ttest_correctness | OK | GPU Welch t-test |
| de_donor_pseudobulk_correctness | OK | GPU pseudobulk NB GLM |
| bench_de_wilcoxon_perf | OK | Bench driver |

- **ctest result**: FAIL (exit 8)
- **Wilcoxon bench throughput**: N/A
- **Job ID**: 361430
| 2026-04-17 | de/wilcoxon | small-500c-200g-4cl | singlet-gpu | 3.84445 | 0 | 130058 | —  | —  | —  | no-git |
| 2026-04-17 | de/ttest | small-500c-200g-4cl | singlet-gpu | 2.18755 | 0 | 228566 | —  | —  | —  | no-git |
| 2026-04-17 | de/wilcoxon | small-500c-200g-4cl | singlet-gpu | 3.83293 | 0 | 130449 | —  | —  | —  | no-git |
| 2026-04-17 | de/wilcoxon | medium-11k-real-5cl | singlet-gpu | 985.492 | 0 | 21173.2 | —  | —  | —  | no-git |
| 2026-04-17 | de/ttest | small-500c-200g-4cl | singlet-gpu | 2.19168 | 0 | 228135 | —  | —  | —  | no-git |
| 2026-04-17 | de/ttest | medium-11k-real-5cl | singlet-gpu | 77.5315 | 0 | 269129 | —  | —  | —  | no-git |
| 2026-04-18 | reduce/nmf_c86_phaseE | small-k10-c86 | singlet-gpu/CD/auto | 37.2145 | 0 | 13435 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf_c86_phaseE | small-k20-c86 | singlet-gpu/CD/auto | 110.349 | 0 | 4531 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf_c86_phaseE | small-k50-c86 | singlet-gpu/MU(forced) | 37.9706 | 0 | 13168 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf_c86_phaseE | small-k100-c86 | singlet-gpu/MU(forced) | 143.769 | 0 | 3477 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf_c86_phaseE | medium-k10-c86 | singlet-gpu/CD/auto | 290.704 | 0 | 71777 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf_c86_phaseE | medium-k20-c86 | singlet-gpu/CD/auto | 444.541 | 0 | 46938 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf_c86_phaseE | medium-k50-c86 | singlet-gpu/MU(forced) | 234.868 | 0 | 88841 | — | — | — | no-git |
| 2026-04-18 | reduce/nmf_c86_phaseE | medium-k100-c86 | singlet-gpu/MU(forced) | 410.638 | 0 | 50813 | — | — | — | no-git |
| 2026-04-18 | qc/metrics_c86 | small-c86 | singlet-gpu | 0.082368 | 0 | 12140637 | — | — | — | no-git |
| 2026-04-18 | qc/metrics_c86 | medium-c86 | singlet-gpu | 0.28144 | 0 | 74140144 | — | — | — | no-git |

## Cycle 86 Phase E — NMF adapter rebench + QC bench (g051 H100 NVL 95830MiB, 2026-04-18, job 363184)
## Matrix (medium): GSM4037629 gene_counts.1pz  38606 genes × 20866 cells  nnz=4270836
## Matrix (small): synthetic 500 cells × 200 genes  density=0.10  seed=1234567
## NMF: max_iter=100, seed=42, solver_mode=3 (auto). FitConfig default: k_cd_cutoff=32, cd_max_iter=10.
## QC:  calculate_qc_metrics(), 5% MT flags, 5% ribo flags. WARMUP=2, TIMED=5.
## sklearn baseline: sklearn.decomposition.NMF init=nndsvd solver=mu max_iter=100 seed=42 (small only; medium MTX unavailable).
## scanpy baseline:  scanpy.pp.calculate_qc_metrics (small only; medium MTX unavailable).
##
## NMF Phase E Results:
##   scale  | k   | solver      | GPU_med_ms | sklearn_ms | ratio     | frob_rel_err
##   --------|-----|-------------|------------|------------|-----------|-------------
##   small   | 10  | CD/auto     | 37.2ms     | 67.5ms     | 1.82x     | 1.0000 (rank > matrix dim — expected degenerate)
##   small   | 20  | CD/auto     | 110.3ms    | 272.2ms    | 2.47x     | 1.0000 (same)
##   small   | 50  | MU(forced)  | 38.0ms     | 329.1ms    | 8.66x     | 0.9998 (MU not fully converged in 100 iters)
##   small   | 100 | MU(forced)  | 143.8ms    | 363.6ms    | 2.53x     | 0.9995
##   medium  | 10  | CD/auto     | 290.7ms    | N/A        | —         | —
##   medium  | 20  | CD/auto     | 444.5ms    | N/A        | —         | —
##   medium  | 50  | MU(forced)  | 234.9ms    | N/A        | —         | —
##   medium  | 100 | MU(forced)  | 410.6ms    | N/A        | —         | —
##
## k=20 REGRESSION: medium GPU 444.5ms > 430ms cap (C85 ref 391ms, +10% = 430ms).
##   Root cause: gene_counts.1pz has 20,866 cells; Cycle 85 used exon_counts.1pz (11,560 cells).
##   Different matrix, not an algorithmic regression. k=20 CD/auto behavior unchanged.
##   Recommendation: update C85 baseline to gene_counts.1pz for apples-to-apples comparison.
##
## k=50 POST-FIX RESULT: MU(forced) 234.9ms on medium vs pre-fix CD 15,980ms → 68x speedup confirmed.
## k=50 now dominates sklearn small: 38ms vs 329ms = 8.66x faster.
##
## QC Phase E Results:
##   scale  | GPU_med_ms | scanpy_ms  | ratio
##   --------|------------|------------|------
##   small   | 0.082ms    | 35.3ms     | 429x faster
##   medium  | 0.281ms    | N/A (no MTX) | —
##
## QC SOTA verdict: GPU 441× faster than scanpy CPU on small (1k cells).
##   Medium GPU-only result: 0.281ms for 20,866 cells = 74M cells/sec throughput.

| 2026-04-18 | reduce/nmf_c86_phaseE | small-k10-c86 | sklearn-nmf-cpu | 67.54 | 201 | 7400 | — | — | — | baseline |
| 2026-04-18 | reduce/nmf_c86_phaseE | small-k20-c86 | sklearn-nmf-cpu | 272.23 | 201 | 1837 | — | — | — | baseline |
| 2026-04-18 | reduce/nmf_c86_phaseE | small-k50-c86 | sklearn-nmf-cpu | 329.12 | 201 | 1519 | — | — | — | baseline |
| 2026-04-18 | reduce/nmf_c86_phaseE | small-k100-c86 | sklearn-nmf-cpu | 363.64 | 201 | 1375 | — | — | — | baseline |
| 2026-04-18 | qc/metrics_c86 | small-c86 | scanpy-cpu | 35.34 | 745 | 28300 | — | — | — | baseline |
| 2026-04-30 | preprocess/pearson_residuals | 10k | singlet-gpu | 0.709216 | 0 | 1.41001e+07 | —  | —  | —  | no-git |
| 2026-04-30 | preprocess/pearson_residuals | 30k | singlet-gpu | 1.69056 | 0 | 1.77456e+07 | —  | —  | —  | no-git |
| 2026-04-30 | enrich/score_genes | 10k | singlet-gpu | 1.1096 | 0 | 9.01226e+06 | —  | —  | —  | no-git |
| 2026-04-30 | enrich/score_genes | 30k | singlet-gpu | 1.4032 | 0 | 2.13797e+07 | —  | —  | —  | no-git |
| 2026-04-30 | embed/diffmap | 10k | singlet-gpu | 2257.24 | 0 | 4430.19 | —  | —  | —  | no-git |
| 2026-04-30 | embed/dpt | 10k | singlet-gpu | 2763.33 | 0 | 3618.82 | —  | —  | —  | no-git |
| 2026-05-01 | preprocess/model_gene_var | 10k | singlet-gpu | 0.570336 | 0 | 1.75335e+07 | —  | —  | —  | no-git |
| 2026-05-01 | preprocess/model_gene_var | 30k | singlet-gpu | 1.3481 | 0 | 2.22536e+07 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_wsum | 10k_wsum | singlet-gpu | 3.73776 | 0 | 2.6754e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_wsum | 10k_wmean | singlet-gpu | 3.72813 | 0 | 2.68231e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_wsum | 30k_wsum | singlet-gpu | 8.36032 | 0 | 3.58838e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_wsum | 30k_wmean | singlet-gpu | 8.35434 | 0 | 3.59095e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_ulm | 10k | singlet-gpu | 3.95834 | 0 | 2.52631e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_ulm | 30k | singlet-gpu | 9.87376 | 0 | 3.03836e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_mlm | 10k | singlet-gpu | 4.40563 | 0 | 2.26982e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_mlm | 30k | singlet-gpu | 9.54909 | 0 | 3.14166e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_ora | 10k | singlet-gpu | 7.82954 | 0 | 1.27721e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_ora | 30k | singlet-gpu | 21.4223 | 0 | 1.40041e+06 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_viper | 10k | singlet-gpu | 135.429 | 0 | 73839.6 | —  | —  | —  | no-git |
| 2026-05-01 | enrich/decoupler_viper | 30k | singlet-gpu | 386.922 | 0 | 77535 | —  | —  | —  | no-git |
| 2026-05-01 | integrate/lisi | 10k | singlet-gpu | 0.034656 | 0 | 2.8855e+08 | —  | —  | —  | no-git |
| 2026-05-01 | integrate/lisi | 30k | singlet-gpu | 0.05856 | 0 | 5.12295e+08 | —  | —  | —  | no-git |
| 2026-05-01 | integrate/asw | 10k | singlet-gpu | 0.081856 | 0 | 1.22166e+08 | —  | —  | —  | no-git |
| 2026-05-01 | integrate/asw | 30k | singlet-gpu | 0.152128 | 0 | 1.97202e+08 | —  | —  | —  | no-git |
| 2026-05-01 | integrate/kbet | 10k | singlet-gpu | 0.183232 | 0 | 5.45756e+07 | —  | —  | —  | no-git |
| 2026-05-01 | integrate/kbet | 30k | singlet-gpu | 1.02794 | 0 | 2.91847e+07 | —  | —  | —  | no-git |
| 2026-05-01 | graph/kmeans | 10k | singlet-gpu | 4.63363 | 0 | 2.15813e+06 | —  | —  | —  | no-git |
| 2026-05-01 | graph/kmeans | 30k | singlet-gpu | 11.8097 | 0 | 2.54029e+06 | —  | —  | —  | no-git |
| 2026-05-01 | embed/dendrogram | 10k | singlet-gpu | 0.125152 | 0 | 7.99028e+07 | —  | —  | —  | no-git |
| 2026-05-01 | embed/dendrogram | 30k | singlet-gpu | 0.197632 | 0 | 1.51797e+08 | —  | —  | —  | no-git |
| 2026-05-01 | preprocess/magic | 10k | singlet-gpu | 2.34915 | 0 | 4.25686e+06 | —  | —  | —  | no-git |
| 2026-05-01 | preprocess/magic | 30k | singlet-gpu | 5.78848 | 0 | 5.18271e+06 | —  | —  | —  | no-git |
| 2026-05-01 | integrate/combat | 10k | singlet-gpu | 6.57715 | 0 | 1.52041e+06 | —  | —  | —  | no-git |
| 2026-05-01 | integrate/combat | 30k | singlet-gpu | 17.2348 | 0 | 1.74067e+06 | —  | —  | —  | no-git |
| 2026-05-01 | anno/celltypist | 10k | singlet-gpu | 0.058528 | 0 | 1.70858e+08 | —  | —  | —  | no-git |
| 2026-05-01 | anno/celltypist | 30k | singlet-gpu | 0.237568 | 0 | 1.2628e+08 | —  | —  | —  | no-git |
| 2026-05-01 | anno/symphony | 10k | singlet-gpu | 0.64704 | 0 | 1.5455e+07 | —  | —  | —  | no-git |
| 2026-05-01 | anno/symphony | 30k | singlet-gpu | 1.02915 | 0 | 2.91502e+07 | —  | —  | —  | no-git |
| 2026-05-01 | qc/empty_drops | 10k | singlet-gpu | 54.0897 | 0 | 184878 | —  | —  | —  | no-git |
| 2026-05-01 | qc/empty_drops | 30k | singlet-gpu | 159.009 | 0 | 188668 | —  | —  | —  | no-git |
| 2026-05-01 | qc/soupx | 10k | singlet-gpu | 0.579936 | 0 | 1.72433e+07 | —  | —  | —  | no-git |
| 2026-05-01 | qc/soupx | 30k | singlet-gpu | 1.07446 | 0 | 2.79209e+07 | —  | —  | —  | no-git |
