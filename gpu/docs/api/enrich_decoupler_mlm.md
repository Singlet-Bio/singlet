# `enrich::mlm`

Feature #32. Multivariate Linear Model (MLM) pathway scoring from decoupleR. **All 5 correctness tests PASS (CYCLE-136, job 369267).** Jointly regresses all pathways on each cell via closed-form OLS: β = (W^T W)^{-1} W^T X, accounting for inter-pathway correlations.

Extends ULM (univariate regression) to the multivariate setting with Cholesky-based linear solve via cuSOLVER.

## C++ signature

```cpp
namespace singlet_gpu::enrich {

struct MlmConfig {
    float ridge       = 1e-6f;  // ridge regularization for W^T W
    bool  deterministic = true; // no-op: MLM is deterministic by design
};

struct MlmResult {
    core::DeviceMemory<float> scores;  // n_cells × n_pathways, col-major
    int n_cells;
    int n_pathways;
};

MlmResult mlm(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    const MlmConfig&          cfg    = {},
    cudaStream_t              stream = nullptr);

}  // namespace singlet_gpu::enrich
```

## Python signature

```python
import singlet_gpu as sg

sg.enrich.mlm(
    adata,
    pathway_weights=W,        # m_genes × n_pathways, col-major
    ridge=1e-6,
    layer=None, inplace=True, copy=False,
)
# → adata.obsm['mlm_scores']  (n_cells × n_pathways)
```

## R signature

```r
singletGpu::mlm(adata, W, ridge = 1e-6)
```

## Inputs

- **X** — `io::PzDeviceMatrix` sparse CSC expression (m genes × n cells).
- **d_W** — device pointer to dense pathway weights (m × p col-major fp32). Must match X row order.
- **n_pathways** — number of pathways p. Constraint: p ≤ m (else W^T W is rank-deficient).
- **cfg.ridge** — Tikhonov regularization (L2 penalty); added to diagonal of W^T W for stability. Default 1e-6.

## Outputs

`MlmResult`:
- **scores** `[n_cells × n_pathways]` col-major device matrix. Entry [c, p] = β[c, p], the OLS coefficient of pathway p in the multivariate regression of X[:, c] on W. Accounts for inter-pathway correlations. Caller must sync stream before reading.
- **n_cells**, **n_pathways** — host-side dimensions.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Notes |
|---|---|---|---|
| small (1k cells, 10 pathways) | ~1.8 ms | ~16 MB (A=W^T W, Y=W^T X) | cuBLAS Sgemm + cuSOLVER Cholesky + solve |
| medium (20k cells, 10 pathways) | ~8.5 ms | ~32 MB | SpMM dominates O(nnz × p) |
| large (100k+ cells) | O(nnz × p) streaming | per-slab chunks | CYCLE-151 target |

Critical path: Pass 3 (SpMM X^T W, O(nnz × p)) and Pass 4a (Cholesky factorization, O(p³)). For p ≤ 100, Cholesky is ≤ 1M floating-point ops, negligible.

## Streaming behavior

Streams per cell slab naturally: Pass 1 (Sgemm W^T W) is global (one-time, m × p × p ops). Pass 3 (SpMM) chunks per slab; each slab's RHS contributes columns to Y. Cholesky and solve are per-column (trivial cost per cell slab).

## Determinism

Fully deterministic. cuBLAS Sgemm (Pass 1) is deterministic. Ridge addition is deterministic. cuSPARSE SpMM (Pass 3) is deterministic. cuSOLVER Spotrf (Cholesky, Pass 4a) is deterministic. Spotrs (triangular solve, Pass 4b) is deterministic. Identical inputs → bit-identical outputs.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| TinyClosedForm | analytical OLS vs GPU | max_rel_err ≤ 1e-4 | m=20, n=10, p=3 | PASS |
| VsNumPy_Random | vs numpy.linalg.lstsq | MSE ratio ≤ 1.01 | synthetic random | PASS |
| RidgeStabilizes | adds ridge to diag | A[i,i] increases | inspect d_A after Pass 2 | PASS |
| CholeskySuccess | positive definite | info = 0 after Spotrf | random W | PASS |
| Determinism_BitIdentical | same input twice | rel_err = 0 | 10k cell matrix | PASS |

All tests in `tests/enrich_decoupler_mlm_correctness.cpp` (CYCLE-136, ctest 5/5 PASS).

## Citation

> Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. (2022). decoupleR: ensemble of computational methods to infer biological activities from omics data. _Bioinformatics Advances_ 2:vbac016. https://doi.org/10.1093/bioadv/vbac016

MLM is decoupleR's multivariate regression approach: jointly fit all pathways on each cell's expression profile, yielding coefficients that automatically account for pathway-pathway covariance. Singlet-gpu's GPU port uses cuBLAS for the Gram matrix and cross-product, cuSOLVER Potrf/Potrs for the Cholesky-based linear solve.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/enrich/decoupler_mlm.h>

int main() {
    namespace sg = singlet_gpu;
    
    auto pz = sg::load_pz("/path/to/counts.1pz");
    int m = pz.mat.rows, n = pz.mat.cols, p = 8;
    
    // Check constraint: p <= m
    if (p > m) {
        throw std::runtime_error("n_pathways > n_genes; W^T W is rank-deficient");
    }
    
    // Load pathway weight matrix
    std::vector<float> h_W(m * p, 0.1f);
    sg::core::DeviceMemory<float> d_W(h_W.size());
    cudaMemcpy(d_W.get(), h_W.data(), h_W.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    
    sg::enrich::MlmConfig cfg{};
    cfg.ridge = 1e-6f;  // regularization for numerical stability
    
    auto res = sg::enrich::mlm(pz.mat, d_W.get(), p, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);
    
    // res.scores[c + p_idx * n_cells] = β[c, p_idx]
}
```

## Pitfalls and notes

1. **Pathway count must not exceed gene count.** If n_pathways > n_genes, W^T W is rank-deficient and Cholesky factorization fails. Kernel throws std::runtime_error before attempting Spotrf.

2. **Ridge parameter controls numerical stability.** Default 1e-6 is suitable for log-normalized data. For highly correlated pathways (e.g., many TF regulons), increase ridge to 1e-5 or 1e-4. Too large a ridge flattens coefficients toward zero.

3. **Cholesky failure is informative.** If Spotrf returns info ≠ 0, it means W^T W + ridge·I is not positive definite — usually indicating insufficient ridge regularization. The kernel checks and throws with a helpful message.

4. **Inter-pathway correlations are resolved.** Unlike ULM (which regresses each pathway independently), MLM jointly solves for all pathways, so coefficients reflect the unique contribution of each pathway after removing variance explained by others.

5. **Transposition overhead.** The kernel transposes Y from (n × p) to (p × n) and back for cuSOLVER compatibility (Spotrs expects (p × n_rhs) layout). This adds two cuBLAS Sgeam passes but is negligible (O(n × p) memory move).

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-1k | 1.8 | 16 | correctness (all tests PASS), statistical rigor (multivariate OLS accounts for covariance) |
| medium-20k | 8.5 | 32 | wall efficiency (cuSOLVER Cholesky fast for p ≤ 100) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369267, CYCLE-136). Pending Phase E benchmark vs decoupleR CPU. Constraint: p ≤ m (enforced at entry).

## Links

- Design docs: [`state/designs/32-decoupler-mlm.md`](../../state/designs/32-decoupler-mlm.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § enrich/decoupler_mlm
- Tests: `tests/enrich_decoupler_mlm_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`enrich_decoupler_ulm.md`](enrich_decoupler_ulm.md) (univariate regression), [`enrich_decoupler_wsum.md`](enrich_decoupler_wsum.md) (linear non-parametric alternative)
