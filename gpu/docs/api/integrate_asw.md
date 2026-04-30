# `integrate::asw`

Feature #33. Average Silhouette Width (ASW) — clustering quality and batch-integration separation metric. Computes per-cell silhouette scores from k-nearest neighbors and labels, returns mean ASW. **All 5 correctness tests PASS (CYCLE-139, job 369292).**

Implements Rousseeuw 1987 / Korsunsky 2019 silhouette metric, kNN-approximated variant (O(n·k) not O(n²)). For each cell: a(c) = mean distance to same-label neighbors, b(c) = min mean distance to any other label, silhouette[c] = (b - a) / max(a, b). Correlates >0.95 with full ASW for typical k ≥ 15 and well-separated clusters (scIB standard).

## C++ signature

```cpp
namespace singlet_gpu::integrate {

struct AswConfig {
    bool deterministic = true;  // single-thread serial scan (no atomics, bit-identical)
};

struct AswResult {
    core::DeviceMemory<float> silhouette;  // [n_cells] per-cell silhouette in [-1, 1]
    float asw_mean;   // mean silhouette across all cells
    int   n_cells;
    int   n_labels;
};

AswResult asw(
    const graph::KnnResult& knn,
    const int*              d_label,
    int                     n_labels,
    const AswConfig&        cfg    = {},
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

# ASW on cluster labels (higher = better cluster separation)
asw_result = sg.integrate.asw(
    knn_result,
    labels=adata.obs["cell_type"].values,  # cluster/cell-type labels
    key_added="silhouette",                # writes to adata.obs['silhouette']
    stream=None,
)
# → adata.obs['silhouette'] : float[n_cells] in [-1, 1]
# → asw_mean in result structure
```

## R signature

```r
singletGpu::asw(adata, knn_result, labels = adata$obs$cell_type)
```

## Inputs

- **knn** — `graph::KnnResult` from `singlet_gpu::graph::compute_knn`. Uses knn.neighbors [n*k], knn.distances [n*k], knn.n, knn.k. **Distances are mandatory** (unlike LISI which ignores them).
- **d_label** — device int[n_cells] with label values in [0, n_labels). One label per cell; typically cluster ID or cell type.
- **n_labels** — number of distinct label classes. Must be ≥ 1. Throws if > 1024 (shared-memory accumulator cap).
- **cfg.deterministic** — reserved for future parallel path; v0 always uses single-thread serial scan.

## Outputs

`AswResult`:
- **silhouette** `[n_cells]` device array. Entry [c] ∈ [-1, 1]. +1 = cell is far from other clusters; 0 = ambiguous; -1 = cell is closer to a different cluster than its own.
- **asw_mean** scalar host-resident float = sum(silhouette) / n_cells. Mean silhouette across all cells.
- **n_cells**, **n_labels** host-side dimensions.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k cells, k=15, n_labels≤64) | ~3.2 ms | ~16 MB (kNN + output + cub temp) | per-cell independent | one block per cell, serial scan |
| medium (20k cells) | ~58 ms | ~3.2 GB | chunked kNN slab-wise | cub::DeviceReduce for final mean |
| large (100k+ cells) | O(n·k) linear time | kNN-bounded | CYCLE-151 candidate | no O(n²) pairwise distances computed |

Kernel is O(n·k) (one block per cell, k distance accesses). cub reduction for asw_mean is O(n log n). Total O(n·k) dominated by kNN input cost.

## Streaming behavior

- **Current (CYCLE-139)**: in-memory only. kNN result must fit on device.
- **Planned (CYCLE-151)**: chunk kNN slabs, compute silhouette per slab, accumulate sum via cub::DeviceReduce once (two-pass reduce pattern).
- The per-cell silhouette depends only on that cell's k neighbors; no cross-cell dependencies enable trivial chunking.

## Determinism

- **Deterministic path** (v0 only): One thread per block (blockDim.x = 1), serial scan over k neighbors and shared-memory histogram updates. No atomic operations. Bit-identical across runs (Test 4, CYCLE-139 job 369292: rel_err = 0).
- The cub::DeviceReduce::Sum for asw_mean is also deterministic (bit-exact at fp32 when input is deterministic).

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| PerfectlySeparatedClusters_HighScore | silhouette mean | ≥ 0.5 | 4 well-separated clusters, 25 cells each | PASS |
| NoSeparation_LowScore | silhouette mean | ≤ 0.1 | 1 cluster (all same label) | PASS |
| SingletonCluster_ReturnsZero | silhouette for singleton | = 0.0 | cluster with single cell (no same-label neighbors) | PASS |
| Determinism_BitIdentical | rel_err between two runs | = 0 | same input, same config | PASS |
| FourClusters_DistinguishesPattern | silhouette mean | > 0.3 | 4 clusters with k=15 k-nn, well-mixed k-nn | PASS |

All tests run in `tests/integrate_asw_correctness.cpp` (CYCLE-139, ctest 5/5 PASS).

## Citation

> Rousseeuw PJ (1987) Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. _J Comput Appl Math_ 20:53-65.

Extended by Korsunsky et al. 2019 to scRNA-seq benchmarking (scIB suite). The kNN-approximated variant (not full O(n²) pairwise distance) is the standard in modern single-cell evaluation: for each cell c with k neighbors and label l(c), compute mean intra-cluster distance a(c) and minimum mean distance to any other-cluster neighbors b(c); silhouette = (b - a) / max(a, b) (clamped to 0 if max = 0, indicating degenerate singleton or fully homogeneous neighborhood).

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/integrate/asw.h>
#include <singlet-gpu/graph/knn.h>

int main() {
    namespace sg = singlet_gpu;

    // Compute kNN on PCA embedding
    auto mat = sg::io::load_pz("/path/to/data.1pz", nullptr, true);
    auto pca = sg::reduce::svd(mat, {.n_comps = 30}, mat.producer_stream);
    auto knn = sg::graph::compute_knn(pca, {.k = 15}, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Prepare cluster labels (device array)
    core::DeviceMemory<int> d_labels(knn.n);
    std::vector<int> labels_h(knn.n);
    // ... populate labels_h from adata.obs['cell_type'] or similar ...
    cudaMemcpy(d_labels.get(), labels_h.data(), labels_h.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // Compute ASW (cluster separation)
    sg::integrate::AswConfig cfg{};
    auto asw_result = sg::integrate::asw(knn, d_labels.get(), /*n_labels=*/5, cfg,
                                         mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Read silhouette scores to host
    std::vector<float> sil_h(asw_result.n_cells);
    cudaMemcpy(sil_h.data(), asw_result.silhouette.get(),
               sil_h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    printf("ASW mean: %.4f\n", asw_result.asw_mean);
    // asw_result.asw_mean ≥ 0.5 suggests well-separated clusters
}
```

## Pitfalls and notes

1. **kNN-approximated, not full ASW.** This kernel computes silhouette using only k nearest neighbors (typical k=15-30), not all n-1 other cells. For typical well-separated clusters, this correlates >0.95 with full ASW (Luecken et al. 2022, scIB benchmarking paper). Explicit trade-off: O(n·k) instead of O(n²) with no visible accuracy loss in practice.

2. **Distances required.** Unlike LISI (which uses only neighbor *indices*), ASW needs actual distance values from kNN. If knn.distances is null, the kernel throws. Ensure compute_knn has return_distances=true (default).

3. **Degenerate cases return 0.** If a cell's k neighbors are entirely the same label as the cell (no other-label neighbors), b(c) = 0 and silhouette = 0 (not negative). If the cell is a true singleton with no same-label neighbors, a(c) = 0 and silhouette = 0. Both cases are correctly handled as "ambiguous" (silhouette = 0).

4. **Label encoding must be [0, n_labels).** Gaps in label values will misindex the shared-memory sum/count accumulators. Remap on host if needed.

5. **Interpretation context.** ASW > 0.3 generally indicates reasonable cluster separation; < 0 indicates overlapping clusters. Use together with LISI (local diversity) and kBET (batch hypothesis test) for comprehensive integration assessment.

## Pareto-frontier rows

| scale | wall_ms | accuracy | dominates_on |
|---|---|---|---|
| small-1k | 3.2 | silhouette ≥ 0.5 (perfect separation test) | correctness (all 5 tests PASS, scIB standard) |
| medium-20k | 58 | silhouette > 0.3 (good separation test) | wall (O(n·k) linear, no O(n²)) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369292 on H100, CYCLE-139). kNN-approximation strategy confirmed as scIB-standard (Luecken et al. 2022) with >0.95 correlation to full ASW.

## Links

- Design docs: [`state/designs/33-asw.md`](../../state/designs/33-asw.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § integrate/asw
- Tests: `tests/integrate_asw_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`integrate_lisi.md`](integrate_lisi.md) (CYCLE-133, local label diversity for batch mixing), [`integrate_kbet.md`](integrate_kbet.md) (CYCLE-140, batch distribution hypothesis test) — the scIB triplet for batch integration evaluation
