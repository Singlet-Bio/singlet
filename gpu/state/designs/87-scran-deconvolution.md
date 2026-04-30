# Cycle 87 — Feature #2 Sub-Variant: scran Deconvolution Size Factors (Lun 2016)

## Problem Statement

Current frontier lognorm uses total-count + log1p normalization (~370× scanpy speedup). However, the Seurat/scran/Bioconductor ecosystem shows that `scran::computeSumFactors` (Lun, Bach, Marioni 2016, *Genome Biology*) is the foundational deconvolution-based normalization for sparse UMI data, reducing total-count bias at low library sizes. No GPU-native port exists; rapids-singlecell omits it. User directive (2026-04-18) explicitly targets ecosystem compatibility.

## Phase B Findings

### Literature Review
- **scran algorithm**: Pools cells into sliding windows (sizes: 21, 41, 61, 81, 101). Builds ~5·n_cells-equation linear system A·s = p. Solves via constrained weighted QR (limSolve package).
- **Clustering**: Splits clusters >3000 cells; performs inter-cluster scaling via reference cluster.
- **Benchmarking**: Ahlmann-Eltze & Huber (2023) rank scran deconvolution top-tier for sparse UMI vs 21 alternatives.
- **GPU gap**: No existing GPU implementation. rapids-singlecell does NOT include it.

### Code Reference Analysis
- **Solver**: Weighted least-squares with non-negativity constraints; modern libscran uses sparse QR (R Matrix package backend).
- **Reference implementation**: libscran C++ (github.com/LTLA/libscran) is GPU-portable reference.
- **Prerequisite**: quickCluster (rank-correlation PCA or igraph SNN). Skippable for datasets ≤3000 cells or single-cluster experiments.
- **GPU mapping**: Per-cluster batched QR → cuSOLVER `cusolverDn*geqrf` + `cusolverDn*ormqr`.

## Phase C Decision

### Architecture

**Header**: `include/singlet-gpu/preprocess/deconv_size_factors.h` (~400–600 LOC, header-only)

### Algorithm Steps

1. **Library size computation**: Per-cell sum via `cub::DeviceSegmentedReduce` on DeviceCsc columns.
2. **Deterministic sort**: Cells by library size via `cub::DeviceRadixSort`.
3. **Auto-clustering**: Split into clusters when n_cells > max_cluster_size (3000 default); maintain sorted-order contiguity.
4. **Pool-membership matrix**: Per cluster, 5 window sizes; compute pool sums via atomicAdd kernel over overlapping windows.
5. **LLS solve**: cuSOLVER QR on dense pool matrix (fp32 default; fp64 fallback if fp32 fails per Rule 8).
6. **NNLS projection**: If positive=true, clip negatives; re-solve up to max_nnls_iters (3 default).
7. **Inter-cluster scaling**: Via reference cluster (median-library-size).
8. **Normalization**: Divide by median size factor → median(output) = 1.

### Configuration

| Parameter | Value | Rationale |
|---|---|---|
| pool_sizes | [21, 41, 61, 81, 101] | scran defaults |
| max_cluster_size | 3000 | scran default; auto-split above |
| positive | true | scran default (non-negativity) |
| max_nnls_iters | 3 | convergence safety |

## Correctness Testing (Phase D Validator Scope)

### Reference Implementations
- **Primary**: R `scran::computeSumFactors` via Rscript helper
- **Secondary**: scranPY Python port
- **Fallback**: Analytic synthetic (equal totals → SF=1.0)

### Test Suite & Tolerances

| Test | Input | Expected | Tolerance |
|---|---|---|---|
| EqualTotals smoke | All cells, library=1.0 | SF = 1.0 | max\|SF-1.0\| < 1e-4 |
| GradedLibrarySize analytic | Uniform distribution (1-100×) | SF ∝ lib_size | Spearman ≥0.9999, max rel err <2% |
| RealData vs scran R | GSM4037629 (11.5k cells) | Spearman agreement | Spearman ≥0.999, max rel err <5% |
| Determinism | Repeated calls, same input | Bitwise identical | 0 variance |
| TwoClusterScaling | Synthetic two-cluster data | Balanced inter-cluster ratios | Ratio rel err <1% |

## Streaming & Performance

### Streaming Capability
- **Pool-sum accumulation**: Shardable (associative summation). Partial results per shard → combine.
- **QR solve**: Requires full system per cluster; mitigated by max_cluster_size cap (~3000 cells → dense system ~200×200, acceptable).
- **Gram-matrix path** (A^T A): Deferred to feature 17 integration cycle.

### Determinism & Reproducibility
- **Fully deterministic**: Sorted inputs + no stochasticity. atomicAdd on pool_sum is associative. No non-deterministic operations on size_factors output.

### Target Performance
- **scran R baseline**: 5–30s on 11.5k cells (single thread, CPU).
- **GPU target**: <500ms on H100 (10–60× dominance).

## Adoption Strategy

Single algorithm, single backend. No consolidation needed in v1. Promote to frontier post-Phase-F validation if ≥10× speedup and correctness within tolerance.

## Parallel Dispatch (Phase D)

- **gpu-kernel-dev** (Sonnet): Implement the deconv_size_factors header.
- **analysis-validator** (Sonnet): Write and execute 5 test suites (EqualTotals, GradedLibrarySize, RealData_vs_scran, Determinism, TwoClusterScaling).

## Success Criteria

✓ Header compiles without warnings (fp32 + fp64 paths tested).
✓ All 5 Phase D tests pass within tolerance.
✓ Determinism validation: bitwise identical on repeated runs.
✓ Performance: <500ms H100 on GSM4037629 (11.5k cells).
✓ Interface: `compute_deconv_size_factors(counts: DeviceCsc, clusters: DeviceVector, cfg: Config) → DeviceVector size_factors`.

---

**Design authored**: 2026-04-18  
**Status**: Phase C recorded; Phase D dispatched  
**Reference**: scran R package (Lun et al. 2016); libscran (LTLA, github.com/LTLA/libscran)
