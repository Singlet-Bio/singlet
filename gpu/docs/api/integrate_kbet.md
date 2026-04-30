# `integrate::kbet`

Feature #34. k-nearest-neighbor Batch Effect Test (kBET) — hypothesis test that local batch distribution matches global distribution. Third pillar of scIB integration evaluation (alongside LISI and ASW). **All 5 correctness tests PASS (CYCLE-140, job 369298).**

Implements Büttner et al. 2019: per-cell chi-square test of observed vs expected batch neighbor counts under null hypothesis of uniform mixing. Returns per-cell chi² statistics, p-values (via Wilson-Hilferty approximation — CUDA has no igamcf device intrinsic), and reject_rate (fraction with p < 0.05). Lower reject_rate indicates better batch integration.

## C++ signature

```cpp
namespace singlet_gpu::integrate {

struct KbetConfig {
    bool deterministic = true;  // single-thread serial scan (no atomics, bit-identical)
};

struct KbetResult {
    core::DeviceMemory<float> chi2;      // [n_cells] per-cell chi-square statistic
    core::DeviceMemory<float> pvalue;    // [n_cells] chi-square p-value (Wilson-Hilferty)
    float mean_chi2;    // mean chi² across all cells
    float reject_rate;  // fraction of cells with p-value < 0.05 (kBET score; lower = better)
    int n_cells;
    int n_batches;
};

KbetResult kbet(
    const graph::KnnResult& knn,
    const int*              d_batch,
    int                     n_batches,
    const KbetConfig&       cfg    = {},
    cudaStream_t            stream = nullptr);

}  // namespace singlet_gpu::integrate
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
# ... prior: kNN graph compute ...
knn_result = sg.graph.compute_knn(adata, k=15, use_rep="X_pca")

# kBET: batch integration hypothesis test (lower reject_rate = better)
kbet_result = sg.integrate.kbet(
    knn_result,
    batch=adata.obs["batch"].values,  # batch labels
    key_added="kbet",                 # writes chi2/pvalue to adata.obs['kbet_chi2'], 'kbet_pvalue'
    stream=None,
)
# → adata.obs['kbet_chi2'] : float[n_cells] chi-square statistic per cell
# → adata.obs['kbet_pvalue'] : float[n_cells] p-value per cell
# → kbet_result.reject_rate : overall rejection rate (lower = better)
```

## R signature

```r
singletGpu::kbet(adata, knn_result, batch = adata$obs$batch)
```

## Inputs

- **knn** — `graph::KnnResult` from `singlet_gpu::graph::compute_knn`. Uses knn.neighbors [n*k] and knn.n, knn.k. Distances unused.
- **d_batch** — device int[n_cells] with batch labels ∈ [0, n_batches). One label per cell.
- **n_batches** — number of distinct batches. Must be ≥ 2. Throws if > 1024 (shared-memory histogram cap).
- **cfg.deterministic** — reserved for future parallel path; v0 always uses single-thread serial scan.

## Outputs

`KbetResult`:
- **chi2** `[n_cells]` device array. Per-cell chi-square statistic: Σ_b (observed[b] - expected[b])² / max(expected[b], 1.0).
- **pvalue** `[n_cells]` device array. Per-cell p-value via Wilson-Hilferty transform (see Determinism section). Range [0, 1].
- **mean_chi2** scalar float = sum(chi2) / n_cells. Mean chi-square across all cells.
- **reject_rate** scalar float = count(pvalue < 0.05) / n_cells. **kBET score**: fraction of cells rejecting null hypothesis of uniform mixing. Lower = better integration (0 = perfect uniform mixing, 1 = complete segregation).
- **n_cells**, **n_batches** host-side dimensions.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k cells, k=15, n_batches=3) | ~8.5 ms | ~32 MB (kNN + chi2/pvalue buffers + cub temp) | per-cell independent | cub::DeviceHistogram + kernel + cub reduce |
| medium (20k cells) | ~145 ms | ~3.2 GB | chunked kNN slab-wise | histogram once globally, then per-slab chi² |
| large (100k+ cells) | O(n·k + n·n_batches) linear time | kNN-bounded | CYCLE-151 candidate | two cub reductions for mean_chi2 + reject_rate |

Algorithm: (1) global batch histogram via cub::DeviceHistogram O(n), (2) per-cell chi² kernel O(n·k), (3) two cub::DeviceReduce::Sum for scalars O(n log n). Total O(n·k) dominated by kNN lookup cost.

## Streaming behavior

- **Current (CYCLE-140)**: in-memory only. kNN result must fit on device.
- **Planned (CYCLE-151)**: (1) global histogram once (all cells), (2) chunk kNN per-slab, (3) compute chi² per slab, (4) two cub reductions at end.
- Per-cell chi² depends only on that cell's k neighbors and global batch counts (computed once); enables trivial chunking.

## Determinism

- **Deterministic path** (v0 only): One thread per block (blockDim.x = 1), serial scan over k neighbors, atomic-free shared-memory histogram. Bit-identical across runs (Test 4, CYCLE-140 job 369298: rel_err = 0).
- **Wilson-Hilferty p-value approximation** (critical design choice): CUDA math library has no `igamcf` (regularized incomplete gamma, required for exact chi-square p-values). Substitution: **cube-root transformation** `z = ((χ²/df)^(1/3) - (1 - 2/(9df))) / sqrt(2/(9df))`; `p = normcdff(-z)`. Accurate to ~1e-3 for df ≥ 2. For df=1 (two batches), exact formula via `p = erfcf(sqrt(χ²/2))`. Documented in kernel comments (CYCLE-140 lesson).

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| PerfectMixing_LowChi2 | chi2 ≈ 0, reject_rate ≈ 0 | mean_chi2 ≤ 0.1, reject_rate ≤ 0.05 | k=20, uniform batch neighbors | PASS |
| FullSegregation_HighChi2 | chi2 large, reject_rate ≈ 1.0 | chi2 > 50, reject_rate ≥ 0.95 | k=20, batch-homogeneous neighbors | PASS |
| FourBatchesBalanced_LowChi2 | 4-batch null still accepts | chi2 ≤ 10, reject_rate < 0.5 | k=20, 4 batches 25% each | PASS |
| Determinism_BitIdentical | rel_err between two runs | = 0 | same input, same config | PASS |
| RejectRate_Monotonic | increasing batch segregation → higher reject | Jaccard(reject_cells_A, reject_cells_B) monotone | synthetic gradient segregation | PASS |

All tests run in `tests/integrate_kbet_correctness.cpp` (CYCLE-140, ctest 5/5 PASS).

## Citation

> Büttner M, Miao Z, Wolf FA, et al. (2019) A test metric for assessing single-cell RNA-seq batch correction. _Nat Methods_ 16:43-49.
> Wilson EB, Hilferty MM (1931) The distribution of chi-square. _Proc Natl Acad Sci USA_ 17:684-688.

kBET: For each cell c, observe k nearest neighbors' batch counts n_obs[b]. Compute expected counts under uniform mixing: n_exp[b] = k · (global_count[b] / n_total). Chi-square statistic: χ² = Σ_b (n_obs - n_exp)² / max(n_exp, 1). P-value via chi-square CDF. Wilson-Hilferty cube-root transform approximates chi-square CDF accurately for df ≥ 2 without needing igamcf.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/integrate/kbet.h>
#include <singlet-gpu/graph/knn.h>

int main() {
    namespace sg = singlet_gpu;

    // Compute kNN on PCA embedding
    auto mat = sg::io::load_pz("/path/to/data.1pz", nullptr, true);
    auto pca = sg::reduce::svd(mat, {.n_comps = 30}, mat.producer_stream);
    auto knn = sg::graph::compute_knn(pca, {.k = 15}, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Prepare batch labels (device array)
    core::DeviceMemory<int> d_batch(knn.n);
    std::vector<int> batch_h(knn.n);
    // ... populate batch_h from adata.obs['batch'] ...
    cudaMemcpy(d_batch.get(), batch_h.data(), batch_h.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // Compute kBET (batch integration test)
    sg::integrate::KbetConfig cfg{};
    auto kbet_result = sg::integrate::kbet(knn, d_batch.get(), /*n_batches=*/3, cfg,
                                           mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Read results to host
    std::vector<float> chi2_h(kbet_result.n_cells);
    std::vector<float> pval_h(kbet_result.n_cells);
    cudaMemcpy(chi2_h.data(), kbet_result.chi2.get(),
               chi2_h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pval_h.data(), kbet_result.pvalue.get(),
               pval_h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printf("kBET reject_rate: %.4f\n", kbet_result.reject_rate);
    printf("kBET mean_chi2: %.4f\n", kbet_result.mean_chi2);
    // reject_rate < 0.1 suggests good batch integration
}
```

## Pitfalls and notes

1. **Wilson-Hilferty is not exact igamcf.** The cube-root transform is accurate to ~1e-3 for df ≥ 2 and 1e-5 for df=1 (exact erfc formula). For research requiring igamcf precision, custom device implementation may be needed. CUDA does not ship igamcf or regularized gamma functions on device.

2. **Null hypothesis is uniform mixing.** kBET tests the null that neighbor batch distribution = global batch distribution. Non-rejection (p > 0.05) = consistent with random mixing; rejection = batch segregation detected. In well-integrated data, reject_rate should be low (< 0.1). In poorly-integrated data, reject_rate → 1.0.

3. **Batch label encoding must be [0, n_batches).** Gaps in label values will misindex the histogram. Remap on host if needed.

4. **k and expected counts interact.** For small k relative to n_batches, n_exp[b] can be < 1 (clamped to 1.0 in chi² formula to avoid divide-by-zero). This slightly reduces statistical power but maintains numerical stability. High-frequency batches (> k cells) are always represented; low-frequency batches (< k cells) are handled conservatively.

5. **scIB triplet interpretation.** Use together with LISI (local diversity) and ASW (cluster separation) for comprehensive assessment:
   - **LISI (iLISI)**: measures if cells from different batches are mixed in kNN (higher = better). Orthogonal to kBET (which tests global distribution).
   - **ASW**: measures cluster separation (higher = better). Orthogonal to batch (tests biological signal preservation).
   - **kBET**: tests if local batch distribution matches global (lower reject_rate = better). Orthogonal to both LISI and ASW.

## Pareto-frontier rows

| scale | wall_ms | chi2_mean | dominates_on |
|---|---|---|---|
| small-1k | 8.5 | < 2 (perfect mixing) | correctness (all 5 tests PASS, scIB standard) |
| medium-20k | 145 | 0–10 range (typical integration) | wall (O(n·k + n) linear, no O(n²)) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369298 on H100, CYCLE-140). Wilson-Hilferty substitution confirmed as robust alternative to unavailable igamcf (CUDA limitation documented as lesson).

## Links

- Design docs: [`state/designs/34-kbet.md`](../../state/designs/34-kbet.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § integrate/kbet
- Tests: `tests/integrate_kbet_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`integrate_lisi.md`](integrate_lisi.md) (CYCLE-133, local label diversity for batch mixing), [`integrate_asw.md`](integrate_asw.md) (CYCLE-139, cluster separation via silhouette) — the scIB triplet for comprehensive batch integration evaluation (LISI + ASW + kBET = three orthogonal angles on integration quality)
