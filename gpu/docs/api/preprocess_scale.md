# `preprocess::scale` + `preprocess::regress_out`

Feature #7. Z-score scaling (zero-center + unit-variance) of a sparse `.1pz` into a dense matrix, followed by optional covariate regression. Both functions on the **frontier** as of Cycle 7; bench gate currently unfilled in `state/pareto-frontier.md` (TF32-tolerance correctness fix landed in Cycle 7 retest).

## C++ signature

```cpp
namespace singlet_gpu::preprocess {

struct ScaleConfig {
    float max_value     = 10.0f;   // clip scaled values to [-max_value, max_value]
    bool  zero_center   = true;    // subtract per-gene mean
    bool  unit_variance = true;    // divide by per-gene std
};

// Sparse → dense: builds a row-major DenseMatrix(n_genes × n_cells) with
// (x_ij - mean_i) / std_i applied gene-wise. Mean/std are caller-provided —
// typically from a prior `qc::calculate_qc_metrics` call.
core::DeviceDense scale(const core::DeviceCSC& mat,
                        const float* d_mean,
                        const float* d_std,
                        const ScaleConfig& cfg = {},
                        cudaStream_t stream = nullptr);

// Convenience overload: takes DeviceMemory<float> wrappers for mean/std.
core::DeviceDense scale(const core::DeviceCSC& mat,
                        const core::DeviceMemory<float>& d_mean,
                        const core::DeviceMemory<float>& d_std,
                        const ScaleConfig& cfg = {},
                        cudaStream_t stream = nullptr);

// Lower-level overload: raw device pointers.
core::DeviceDense scale(const int* d_indptr,
                        const int* d_indices,
                        const float* d_data,
                        int n_genes, int n_cells,
                        const float* d_mean,
                        const float* d_std,
                        const ScaleConfig& cfg = {},
                        cudaStream_t stream = nullptr);

// regress_out — remove linear effects of p covariates from a dense matrix.
// X is row-major n_genes × n_cells, modified in place.
// C is column-major n_cells × p, p ≤ 32 (cuSOLVER QR ceiling for the small batch).
void regress_out(float* X, int n_genes, int n_cells,
                 const float* C, int p,
                 cudaStream_t stream = nullptr);

}  // namespace singlet_gpu::preprocess
```

## Python signature

> **Status (2026-04-29)**: Python wrappers for `scale` + `regress_out` are **not yet present** as of the CYCLE-101 audit. The C++ functions are on the frontier; the wrappers are tracked in `state/wrapper-gaps.md` → CYCLE-103. Once shipped, signatures will follow scanpy convention:
>
> ```python
> # planned — not yet in pip install singlet-gpu
> sg.preprocess.scale(adata, max_value=10.0, zero_center=True, layer=None)
> sg.preprocess.regress_out(adata, keys=['n_counts', 'pct_mt'], layer=None)
> ```
>
> Until then, call the C++ functions directly via the pybind11 `_core` module.

## R signature

Same status — R wrappers tracked in CYCLE-103.

## Inputs

### `scale`

- **mat** — `core::DeviceCSC` of (typically log-normalized) values. Genes × cells.
- **d_mean / d_std** — `[n_genes]` device pointers. Pass directly from `qc::QcResult.gene_mean` and `sqrt(QcResult.gene_var)` to skip recomputation.
- **cfg.max_value** — clip scaled values to `[-10, 10]` by default. Matches scanpy's `pp.scale(max_value=10)`.
- **cfg.zero_center** — subtract per-gene mean. `false` produces unit-variance scaling without centering — useful when downstream PCA does its own centering.
- **cfg.unit_variance** — divide by per-gene std. `false` produces only mean-centering.

### `regress_out`

- **X** — row-major `n_genes × n_cells` dense matrix on device. **Modified in place.** Typically the output of `scale`.
- **C** — column-major `n_cells × p` design matrix on device. Each column is one covariate (e.g. n_umis, pct_mt). Add an intercept column as needed; the function does NOT add one automatically.
- **p** — number of covariates. Capped at 32 — the cuSOLVER batched-QR path is tuned for small p; larger covariate sets should be done in batches or via a separate Sgels call.

## Outputs

`scale` returns a freshly allocated `core::DeviceDense` (row-major n_genes × n_cells). The input `core::DeviceCSC` is not modified.

`regress_out` modifies `X` in place. No return value.

## Complexity

| Function | Scale | Wall (V100S) | SOTA wall | Speedup |
|---|---|---|---|---|
| `scale` | small (10k cells × 30k genes) | TBD (bench unfilled) | TBD (scanpy `pp.scale` ≈ ~100 ms) | TBD |
| `regress_out` (p=2) | small | TBD | TBD | TBD |

Memory: `scale` allocates `n_genes × n_cells × 4 bytes` device dense — 30k × 10k = 1.2 GB. At 100k cells × 30k genes you're at 12 GB; streaming becomes mandatory. The benchmark row was deferred while the TF32-tolerance correctness fix landed; a backfill bench cycle is in the queue.

## Streaming behavior

- **`scale`**: cell-shardable. Per-shard pass requires only `d_mean / d_std` (already global, computed once by `qc::calculate_qc_metrics`). Output is a per-shard DenseMatrix that downstream consumers (PCA, regress_out) consume directly.
- **`regress_out`**: requires a global QR of the design matrix `C`, then a Sgemv-style residual update per cell. The QR is one-shot (p ≤ 32, cheap); the residual update is fully cell-shardable.

For ≥ 100k cells, both scale and regress_out should run in the streaming driver; the in-memory dense allocation otherwise OOMs.

## Determinism

Both functions are deterministic given fp32 reduction-order tolerance. No atomicAdd in either; the sparse-to-dense scatter is one block per gene, threads stride over the gene's nonzeros (no inter-warp races for the gene's destination row). The cuSOLVER QR in `regress_out` is deterministic per the cuSOLVER documentation.

## Correctness contract

| Function | Reference | Tolerance | Sample |
|---|---|---|---|
| `scale` | scanpy `pp.scale(max_value=10)` | rel err ≤ 1e-5 element-wise | small synthetic + GSM4037629; ctest PASS post TF32 fix (Cycle 7) |
| `regress_out` (p=2) | scanpy `pp.regress_out(['n_counts', 'pct_mt'])` | residual rel err ≤ 1e-4 | small synthetic; ctest PASS |

The TF32 tolerance fix (Cycle 7) was the unlock: cuBLAS GEMM uses TF32 on Ampere+ by default, which gives slightly different residuals than scanpy's fp32 numpy. Tolerance widened from 1e-6 to 1e-4 with documented justification in the test header.

## Citations

- **z-score scaling** is standard; no method paper.
- **regress_out** linear residualization is the Seurat / Scanpy default. Implementation here uses cuSOLVER's batched QR factorization (`Sgeqrf` + `Sormqr`) followed by a cuBLAS Sgemv residual update. The factornet "DeviceMemory + RAII pool allocator" pattern is used throughout — no raw cudaMalloc in the kernel.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/preprocess/lognorm.h>      // until released
#include <singlet-gpu/preprocess/hvg.h>
#include <singlet-gpu/preprocess/scale.h>
#include <singlet-gpu/qc/metrics.h>

int main() {
    namespace sg = singlet_gpu;
    auto pz = sg::load_pz("/path/to/exon_counts.1pz");
    cudaStreamSynchronize(pz.producer_stream);

    /* ... lognorm + HVG-subset mat to mat_hvg ... */

    sg::core::DeviceMemory<uint8_t> is_mt(pz.mat.rows), is_ribo(pz.mat.rows);
    /* fill from rownames */

    auto qc = sg::qc::calculate_qc_metrics(pz.mat, is_mt, is_ribo, pz.producer_stream);

    // scale: sparse → dense row-major (n_genes × n_cells)
    sg::preprocess::ScaleConfig scfg{};
    scfg.max_value = 10.0f;
    auto X = sg::preprocess::scale(pz.mat, qc.gene_mean.data(), /*std=*/nullptr,
                                   scfg, pz.producer_stream);
    // (compute std from gene_var on the device side first; example simplified)

    // regress_out: remove n_umis effect — pass it as a single column
    sg::preprocess::regress_out(X.data.data(), pz.mat.rows, pz.mat.cols,
                                qc.n_umis.data(), /*p=*/1,
                                pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // X is now ready for PCA via cuBLAS GEMM or factornet SVD on dense input.
}
```

## Pareto-frontier row

| scale | wall_ms | sota_wall_ms | sota_lib | dominates_on |
|---|---|---|---|---|
| small | TBD (bench backfill pending) | TBD (scanpy `pp.scale` baseline) | scanpy 1.10.x | TBD |

Frontier (correctness-signed) but bench row is `TBD`. CYCLE-99-FOLLOWUP: backfill scale bench in the next available bench-venv cycle.

## Links

- Design doc: [`state/designs/07-scale.md`](../../state/designs/07-scale.md)
- Frontier entry: pending wall-bench backfill in `state/pareto-frontier.md` (CLAUDE.md roadmap currently lists feature #7 as "frontier (all tests pass after TF32 tolerance fix)")
- Equivalence notebook: `docs/notebooks/scaling.ipynb` (pending)
- Related: [`qc_metrics.md`](qc_metrics.md) (provides mean/std), [`reduce_svd.md`](reduce_svd.md) (consumes the dense scaled matrix)
