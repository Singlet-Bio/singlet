---
feature: 7
roadmap_id: 7
module: include/singlet-gpu/preprocess/scale.h
status: design (Phase C)
tolerance: <=1e-5 relative error vs scanpy sc.pp.scale and sc.pp.regress_out
target_perf: scale 100k cells <= 5ms; regress_out 100k cells x 3 covariates <= 50ms
ooc_plan: scale is per-gene, streams per-shard independently (with pre-computed global mean/var). regress_out needs global OLS fit but can apply per-shard.
---

# Feature 7 — Scaling + regress_out

## Z-score scaling (sc.pp.scale equivalent)

Per-gene z-score: `X_scaled[g,c] = (X[g,c] - mean[g]) / std[g]`

With optional clipping: `X_scaled = clip(X_scaled, -max_value, max_value)` (scanpy default: max_value=10).

### Kernel design

**Input**: `DeviceCSC` (genes x cells), pre-computed `gene_mean[n_genes]` and `gene_std[n_genes]` from QC metrics (Feature 6) or HVG (Feature 3).

**Output**: Modified in-place OR new dense matrix. Note: scaling a sparse matrix produces a DENSE matrix (subtracting the mean fills in zeros). So the output is `DenseMatrixGPU<float>` of shape (n_genes_hvg x n_cells) — typically 2000 x n_cells after HVG selection.

Two approaches:
1. **Sparse → dense + scale fused**: One kernel that reads CSC and writes dense, applying (x - mean) / std per element, filling non-sparse entries with (-mean / std).
2. **cuSPARSE csc2dense + scale**: Two-step, simpler.

Prefer (1) for fusion. The kernel is one thread per output element:
```
__global__ void sparse_to_scaled_dense(
    const int* indptr, const int* indices, const float* data,
    int n_genes, int n_cells,
    const float* mean, const float* std,
    float max_value, float* out)  // out[n_genes * n_cells], row-major
{
    int gene = blockIdx.y * blockDim.y + threadIdx.y;
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (gene >= n_genes || cell >= n_cells) return;
    
    // Binary search for (gene, cell) in CSC column `cell`
    int col_start = indptr[cell], col_end = indptr[cell+1];
    float val = 0.0f;  // default: gene not expressed in this cell
    // Binary search for `gene` in indices[col_start..col_end]
    int pos = binary_search(indices + col_start, col_end - col_start, gene);
    if (pos >= 0) val = data[col_start + pos];
    
    float scaled = (val - mean[gene]) / max(std[gene], 1e-12f);
    scaled = fminf(fmaxf(scaled, -max_value), max_value);
    out[gene * n_cells + cell] = scaled;
}
```

Memory: for 2000 HVG genes x 100k cells = 800 MB fp32. Fits in device memory for reasonable scales.

### Alternative: zero-centered sparse

For downstream PCA, we can avoid densification entirely using implicit centering:
- PCA on `X - mean` can be computed as `(X - mean)^T (X - mean) = X^T X - n * mean * mean^T`
- factornet's SVD adapters already support implicit centering

So scaling may be OPTIONAL if PCA handles it. But for other downstream uses (DE, heatmaps), the dense scaled matrix is needed.

## regress_out (sc.pp.regress_out equivalent)

Remove confounding covariates from the expression matrix by linear regression.

**Input**: Dense scaled matrix `X[n_genes x n_cells]`, design matrix `C[n_cells x p]` where p = number of covariates (typically 1-5: n_counts, pct_mt, batch indicators, cell cycle scores).

**Algorithm**: Per-gene OLS regression, then subtract the predicted values:
```
For each gene g:
    beta_g = (C^T C)^{-1} C^T X[g,:]   # p x 1
    X_residual[g,:] = X[g,:] - C @ beta_g
```

### Kernel design

**Batch OLS via cuBLAS**: Since all genes share the same design matrix C:
1. Pre-compute `(C^T C)^{-1} C^T` once = pseudoinverse `P[p x n_cells]` via cuSOLVER QR or explicit inverse (p is tiny, 1-5).
2. `Beta[p x n_genes] = P @ X^T` — single cuBLAS GEMM.
3. `Predicted[n_genes x n_cells] = Beta^T @ C^T` — single cuBLAS GEMM.
4. `Residual = X - Predicted` — fused element-wise subtract.

Total: 2 GEMMs + 1 small QR factorization + 1 subtract. Highly efficient.

**Memory**: X is already dense (from scaling). P is tiny (p x n_cells). Beta is tiny (p x n_genes). Predicted reuses X's buffer (in-place subtraction).

### Covariates

Common covariates for single-cell:
- `total_counts`: from QC metrics (Feature 6)
- `pct_mt`: from QC metrics
- `n_genes`: from QC metrics
- Batch indicators: one-hot encoded from metadata
- Cell cycle scores: from singlify's `cell_cycle_scores.tsv` or Feature 6's cell cycle scoring

The design matrix is constructed on host and uploaded once. Tiny memory (<1MB for 100k cells x 5 covariates).

## Streaming (billion-cell)

**Scaling**: Pre-compute global mean/std across all shards (Feature 6 streaming Welford). Then apply per-shard: load shard → scale → write dense output chunk. Each chunk is `n_hvg_genes x shard_cells`.

**regress_out**: Pre-compute global pseudoinverse P (requires global C^T C, which is a p x p matrix — tiny). Then apply per-shard: `Beta_shard = P @ X_shard^T`, subtract `Beta_shard^T @ C_shard^T`.

Both are one-pass after the global statistics are computed.

## API

```cpp
namespace singlet_gpu::preprocess {
    struct ScaleConfig {
        float max_value = 10.0f;     // clip scaled values to [-max, max]
        bool zero_center = true;      // subtract mean
        bool unit_variance = true;    // divide by std
    };
    
    // Sparse HVG matrix → dense scaled matrix
    core::DenseMatrixGPU<float> scale(
        const core::DeviceCSC& mat,       // HVG-subsetted sparse
        const DeviceMemory<float>& mean,  // from QC or HVG
        const DeviceMemory<float>& std,   // from QC or HVG
        const ScaleConfig& cfg = {},
        cudaStream_t stream = nullptr);
    
    struct RegressConfig {
        // covariates passed as dense matrix C[n_cells x p]
    };
    
    // In-place residualization
    void regress_out(
        core::DenseMatrixGPU<float>& X,     // [n_genes x n_cells], modified in-place
        const core::DenseMatrixGPU<float>& C, // [n_cells x p] design matrix
        cudaStream_t stream = nullptr);
}
```

## Correctness test spec

1. Tiny synthetic (500x200): scale output matches `scanpy.pp.scale(X, max_value=10)` within 1e-5
2. GSM4037629: scale → PCA → compare PCs vs scanpy pipeline within tolerance
3. regress_out: synthetic with known covariate effect → verify residual has zero correlation with covariates
4. Edge cases: gene with zero variance → output should be 0 (not NaN/Inf)

## Target performance

| Operation | Scale | Target wall | Notes |
|---|---|---|---|
| scale | 2k genes x 11.5k cells | <1ms | tiny: just sparse→dense + arithmetic |
| scale | 2k genes x 100k cells | <5ms | 800 MB dense output |
| scale | 2k genes x 1M cells | <50ms | 8 GB dense — may need streaming |
| regress_out | 2k x 100k, p=3 | <50ms | 2 GEMMs dominate |
| regress_out | 2k x 1M, p=3 | <500ms | large GEMM |
