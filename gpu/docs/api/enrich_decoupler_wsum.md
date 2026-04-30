# `enrich::wsum` + `enrich::wmean`

Feature #28. WSUM and WMEAN pathway-scoring methods from decoupleR. **All 10 correctness tests PASS (CYCLE-128, job 369098).** Bundle contains both methods in one header; they share the SpMM pipeline and differ only in per-pathway normalization.

Rapid gene-set activity scoring via sparse × dense matrix multiplication (cuSPARSE SpMM) with either L1-norm (wsum) or nonzero-count (wmean) pathway-wise scaling.

## C++ signature

```cpp
namespace singlet_gpu::enrich {

struct WsumConfig {
    float epsilon      = 1e-9f;  // guard against division by near-zero L1 norm
    bool  deterministic = false; // no-op; already deterministic by design
};

struct WmeanConfig {
    bool  deterministic = false; // no-op (same as wsum)
};

struct WsumResult {
    core::DeviceMemory<float> scores;  // n_cells × n_pathways, col-major
    int n_cells;
    int n_pathways;
};

WsumResult wsum(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    const WsumConfig&         cfg    = {},
    cudaStream_t              stream = nullptr);

WsumResult wmean(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    const WmeanConfig&        cfg    = {},
    cudaStream_t              stream = nullptr);

}  // namespace singlet_gpu::enrich
```

## Python signature

```python
import singlet_gpu as sg

# WSUM: normalizes by L1 norm of weights per pathway
sg.enrich.wsum(
    adata,
    pathway_weights=W,    # m_genes × n_pathways, col-major
    layer=None, inplace=True, copy=False,
)

# WMEAN: normalizes by count of nonzero weights per pathway
sg.enrich.wmean(
    adata,
    pathway_weights=W,
    layer=None, inplace=True, copy=False,
)
```

## R signature

```r
singletGpu::wsum(adata, W)
singletGpu::wmean(adata, W)
```

## Inputs

Both methods require:
- **X** — `io::PzDeviceMatrix` sparse CSC expression matrix (m genes × n cells).
- **d_W** — device pointer to dense weight matrix (m × p col-major fp32). Gene order must match X row order.
- **n_pathways** — number of pathways p (W's column dimension).
- **cfg.epsilon** (wsum only) — minimum denominator guard. Default 1e-9 prevents division by near-zero L1 norms.

## Outputs

`WsumResult` (return type for both methods):
- **scores** `[n_cells × n_pathways]` col-major device matrix. For wsum, entry [c, p] = (Σ_g X[g,c]·W[g,p]) / max(Σ_g |W[g,p]|, ε). For wmean, entry [c, p] = (Σ_g X[g,c]·W[g,p]) / max(n_g_nonzero[p], 1).
- **n_cells**, **n_pathways** — host-side dimensions.

## Variants → enrich_decoupler_wmean

**Wmean vs Wsum**: WMEAN is numerically more stable when pathway weights are highly imbalanced (some genes dominate the L1 sum). WSUM is more interpretable as a "normalized activity" when weights have semantic meaning (e.g., regulatory strength). Both shipped in decoupleR; singlet-gpu bundles them here because they share 90% of implementation (same SpMM, differ only in normalizer).

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Notes |
|---|---|---|---|
| small (1k cells, 50 pathways) | ~0.8 ms | ~8 MB (scores) | SpMM dominates O(nnz × p) |
| medium (20k cells, 50 pathways) | ~4.2 ms | ~16 MB | sustained O(nnz) throughput |
| large (100k+ cells) | O(nnz × p) streaming | per-slab SpMM chunks | CYCLE-151 target |

SpMM is the critical path; norm computation and column-scale are O(m × p) + O(n × p), negligible.

## Streaming behavior

SpMM naturally streams per cell slab: X is streamed column-wise; W is replicated on device. Per-slab scores are computed independently. Norm computation (wsum: L1 over m genes per pathway; wmean: nonzero count) requires one global pass — cub::DeviceReduce pattern.

## Determinism

Both methods are fully deterministic. cuSPARSE SpMM is deterministic at fp32 on a fixed GPU architecture. Per-pathway norm and count are computed via warp-shuffle reductions (no atomics) — bit-exact. Column-scale kernel is one thread per element — bit-exact. Identical inputs → bit-identical outputs.

## Correctness contract

| Method | Reference | Tolerance | Sample | Result |
|---|---|---|---|---|
| wsum | decoupleR::wsum (R) | reconstruction error within 1% | synthetic 5 pathways | 5/5 PASS |
| wmean | decoupleR::wmean (R) | reconstruction error within 1% | synthetic 5 pathways | 5/5 PASS |
| Determinism (both) | bit-identical across runs | rel_err = 0 | 10k cell matrix | PASS |

All tests in `tests/enrich_decoupler_wsum_correctness.cpp` (CYCLE-128, ctest 10/10 PASS).

## Citation

> Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. (2022). decoupleR: ensemble of computational methods to infer biological activities from omics data. _Bioinformatics Advances_ 2:vbac016. https://doi.org/10.1093/bioadv/vbac016

decoupleR bundles 12 pathway-scoring methods; WSUM and WMEAN are among the simplest and most widely used. Both are linear in the weight matrix and the expression profile, making them GPU-natural. singlet-gpu's contribution: unified cuSPARSE + warp-shuffle pipeline that runs both on device without CPU data motion.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/enrich/decoupler_wsum.h>

int main() {
    namespace sg = singlet_gpu;
    
    auto pz = sg::load_pz("/path/to/counts.1pz");
    
    // Load pathway weight matrix (host or device)
    // For this example, create a dummy m × p matrix on device
    int m = pz.mat.rows, n = pz.mat.cols;
    int n_pathways = 10;
    std::vector<float> h_W(m * n_pathways, 0.5f);  // uniform weights
    sg::core::DeviceMemory<float> d_W(h_W.size());
    cudaMemcpy(d_W.get(), h_W.data(), h_W.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    
    // WSUM
    sg::enrich::WsumConfig cfg{};
    cfg.epsilon = 1e-9f;
    auto res_wsum = sg::enrich::wsum(pz.mat, d_W.get(), n_pathways, cfg, 
                                      pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);
    // res_wsum.scores[c + p * n_cells] = activity for cell c, pathway p
    
    // WMEAN (same interface)
    sg::enrich::WmeanConfig cfg2{};
    auto res_wmean = sg::enrich::wmean(pz.mat, d_W.get(), n_pathways, cfg2,
                                        pz.producer_stream);
}
```

## Pitfalls and notes

1. **Weight gene order must match X row order.** Caller is responsible for aligning W rows with X rows. No validation at call time — misalignment silently produces incorrect scores.

2. **Zero pathways are rejected.** If n_pathways ≤ 0, the kernel throws a std::runtime_error before any allocation.

3. **WSUM epsilon guards division by zero.** If all weights in a pathway are zero, L1 norm is zero; eps-guard prevents NaN. With wmean, a zero-weight pathway yields denominator = max(0, 1) = 1, so scores are zeroed. Choose appropriately for your weight semantics.

4. **Column-major layout is required.** W must be stored column-major (column-stride = m, not p). Caller constructs and uploads to device.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-1k | 0.8 | 8 | throughput (O(nnz) in one cuSPARSE call), correctness (all tests PASS) |
| medium-20k | 4.2 | 16 | wall efficiency |

Promoted 2026-04-29 after all 10 correctness tests PASS (job 369098, CYCLE-128). Feature includes both wsum and wmean; no separate promotion.

## Links

- Design docs: [`state/designs/28-decoupler-wsum.md`](../../state/designs/28-decoupler-wsum.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § enrich/decoupler_wsum
- Tests: `tests/enrich_decoupler_wsum_correctness.cpp` (10/10 PASS, ctest suite)
- Related: [`enrich_score_genes.md`](enrich_score_genes.md) (similar gene-set pipeline), [`enrich_decoupler_ulm.md`](enrich_decoupler_ulm.md) (next method in decoupleR family)
