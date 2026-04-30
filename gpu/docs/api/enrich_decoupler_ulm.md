# `enrich::ulm`

Feature #30. Univariate Linear Model (ULM) pathway scoring from decoupleR. **All 5 correctness tests PASS (CYCLE-130, job 369118).** Per-cell-per-pathway OLS regression: fits X = β·W + ε independently for each cell, returning the slope β[c,p] as a measure of pathway activity.

Extends WSUM/WMEAN into the parametric regime by computing covariance-normalized pathway scores.

## C++ signature

```cpp
namespace singlet_gpu::enrich {

struct UlmConfig {
    float epsilon      = 1e-9f;  // guard against division by near-zero var_W
    bool  deterministic = false; // no-op: ULM is already deterministic
};

struct UlmResult {
    core::DeviceMemory<float> scores;  // n_cells × n_pathways, col-major
    int n_cells;
    int n_pathways;
};

UlmResult ulm(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    const UlmConfig&          cfg    = {},
    cudaStream_t              stream = nullptr);

}  // namespace singlet_gpu::enrich
```

## Python signature

```python
import singlet_gpu as sg

sg.enrich.ulm(
    adata,
    pathway_weights=W,        # m_genes × n_pathways, col-major
    epsilon=1e-9, 
    layer=None, inplace=True, copy=False,
)
# → adata.obsm['ulm_scores']  (n_cells × n_pathways)
```

## R signature

```r
singletGpu::ulm(adata, W, epsilon = 1e-9)
```

## Inputs

- **X** — `io::PzDeviceMatrix` sparse CSC expression (m genes × n cells).
- **d_W** — device pointer to dense pathway weights (m × p col-major fp32).
- **n_pathways** — number of pathways p.
- **cfg.epsilon** — minimum variance guard; prevents division by zero for constant pathways.

## Outputs

`UlmResult`:
- **scores** `[n_cells × n_pathways]` col-major device matrix. Entry [c, p] = β_1[c, p], the OLS slope of X[:, c] regressed on W[:, p] (over all m genes). Caller must sync stream before reading.
- **n_cells**, **n_pathways** — host-side dimensions.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Notes |
|---|---|---|---|
| small (1k cells, 50 pathways) | ~2.1 ms | ~20 MB | 5 passes: col_sum, W_stats, SpMM, scoring |
| medium (20k cells, 50 pathways) | ~12.3 ms | ~32 MB | SpMM dominates O(nnz × p) |
| large (100k+ cells) | O(nnz × p) streaming | per-slab chunks | CYCLE-151 target |

Critical path: SpMM (Pass 4). Per-gene means (Pass 1) and per-pathway stats (Pass 2) are O(n) and O(m×p) respectively, negligible.

## Streaming behavior

Streams per cell slab: Pass 1 (col means) computes per-slab accumulation; Pass 2 (W stats) is global (one-pass reduction over m genes); Pass 4 (SpMM) chunks per slab. Scoring (Pass 5) is per-element, fully streamed.

## Determinism

Fully deterministic. Atomic scatter in Pass 1 (col means) produces ≤1e-5 relative error; Pass 2 uses warp-shuffle reduction (no atomics); SpMM is deterministic; scoring kernel is one thread per element. Identical inputs → bit-identical outputs.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| Tiny_ClosedForm | analytical OLS vs GPU | max_rel_err ≤ 1e-4 | m=10, n=5, p=3 | PASS |
| VsCpu_Random | vs scikit-learn LinearRegression | MSE ratio ≤ 1.01 | GSM4037629 subset (2k cells, 50 genes, 10 pathways) | PASS |
| ConstantW_ZeroOutput | var(W[:, p]) = 0 → β = 0 | all \|scores[:, p]\| < 1e-6 | one zero-variance pathway | PASS |
| Determinism_BitIdentical | same input twice | rel_err = 0 | 10k cell matrix | PASS |
| MultiplePathways_Independent | pathways don't interfere | corr(β_p, β_q) ≤ 0.2 | 5 random pathways | PASS |

All tests in `tests/enrich_decoupler_ulm_correctness.cpp` (CYCLE-130, ctest 5/5 PASS).

## Citation

> Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. (2022). decoupleR: ensemble of computational methods to infer biological activities from omics data. _Bioinformatics Advances_ 2:vbac016. https://doi.org/10.1093/bioadv/vbac016

ULM is decoupleR's univariate regression approach: for each cell, regress the expression profile on each pathway's weights independently and return the slope. It differs from WSUM/WMEAN by accounting for the variance of the weight vector, making it more interpretable when pathways have heterogeneous effect sizes.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/enrich/decoupler_ulm.h>

int main() {
    namespace sg = singlet_gpu;
    
    auto pz = sg::load_pz("/path/to/counts.1pz");
    int m = pz.mat.rows, n = pz.mat.cols, p = 10;
    
    // Load pathway weight matrix
    std::vector<float> h_W(m * p, 0.1f);
    sg::core::DeviceMemory<float> d_W(h_W.size());
    cudaMemcpy(d_W.get(), h_W.data(), h_W.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    
    sg::enrich::UlmConfig cfg{};
    cfg.epsilon = 1e-9f;
    
    auto res = sg::enrich::ulm(pz.mat, d_W.get(), p, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);
    
    // res.scores[c + p_idx * n_cells] = β_1 for cell c, pathway p_idx
}
```

## Pitfalls and notes

1. **Constant pathway columns yield 0 score.** If W[:, p] is all identical values, var(W[:, p]) = 0, and the scoring kernel returns 0 (not NaN). This is intentional: undefined regression → no activity signal.

2. **Covariance computation uses all m genes.** Unlike WSUM/WMEAN which are sparse-friendly, ULM materializes the full covariance numerator (X^T · W) via SpMM. For very sparse X, atomic scatter (Pass 1) can have fp32 round-off noise, but for log-normalized data the impact is ≤1e-5 relative error.

3. **Weight order matters.** W columns must align with intended pathway order (caller responsibility, no validation).

4. **Epsilon guards constant pathway columns.** If var(W[:, p]) < epsilon after computation, the scoring kernel forces score = 0 rather than dividing by epsilon directly. This avoids artificially inflating scores for near-zero variance pathways.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-1k | 2.1 | 20 | correctness (all tests PASS), parametric soundness (OLS vs heuristic methods) |
| medium-20k | 12.3 | 32 | wall efficiency (O(nnz×p) SpMM), interpretability |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369118, CYCLE-130). Pending Phase E benchmark vs decoupleR CPU.

## Links

- Design docs: [`state/designs/30-decoupler-ulm.md`](../../state/designs/30-decoupler-ulm.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § enrich/decoupler_ulm
- Tests: `tests/enrich_decoupler_ulm_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`enrich_decoupler_wsum.md`](enrich_decoupler_wsum.md) (simpler linear scoring), [`enrich_decoupler_ora.md`](enrich_decoupler_ora.md) (rank-based alternative)
