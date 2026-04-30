# `embed::dendrogram`

Feature #25. Hierarchical clustering of cluster centroids via GPU-accelerated correlation distance and host UPGMA. Mirrors scanpy.tl.dendrogram for cluster-level visualization. Atomic-scatter centroid aggregation + cuBLAS correlation + O(k³) host UPGMA. **All 5 correctness tests PASS (CYCLE-146, job 370015).**

Six GPU passes compute centroids, center, normalize, and correlation distance on dense m × k_clusters matrix; final step runs standard UPGMA linkage on host.

## C++ signature

```cpp
namespace singlet_gpu::embed {

struct DendrogramConfig {
    // atomicAdd in centroid scatter is non-deterministic (fp32 ordering).
    // This flag is documentation-only — no behavioral change.
    bool deterministic = false;
};

struct DendrogramResult {
    core::DeviceMemory<float> centroids;  // m × k_clusters col-major
    core::DeviceMemory<float> distance;   // k_clusters × k_clusters symmetric
    std::vector<float>        linkage;    // (k-1)*4 host: [a, b, dist, n_merged]
    int m          = 0;
    int k_clusters = 0;
};

DendrogramResult dendrogram(
    const io::PzDeviceMatrix& X,
    const int*                d_label,
    int                       k_clusters,
    const DendrogramConfig&   cfg    = {},
    cudaStream_t              stream = nullptr);

}  // namespace singlet_gpu::embed
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
# After clustering (e.g., Leiden)
sg.tl.leiden(adata, resolution=1.0, key_added="leiden")

# Dendrogram of cluster relationships
dgram_result = sg.embed.dendrogram(
    adata.X,  # m × n sparse matrix
    labels=adata.obs["leiden"].values,  # cluster assignment per cell
    key_added="dendrogram",  # writes linkage to adata.uns['dendrogram']
)
# → adata.uns['dendrogram']['linkage'] : [(k-1) × 4] UPGMA output
# → adata.obsm['dendrogram_centroids'] : [m × k_clusters] cluster centroids
```

## R signature

```r
singletGpu::dendrogram(adata, X, labels, key_added = "dendrogram")
```

## Inputs

- **X** — `io::PzDeviceMatrix` sparse CSC expression matrix (m genes × n cells), normalized.
- **d_label** — device int[n_cells] with cluster assignments in [0, k_clusters).
- **k_clusters** — number of distinct clusters. Must be ≥ 2 for dendrogram; throws if k_clusters ≤ 0.
- **cfg.deterministic** — documentation-only flag. Actual behavior unchanged: atomicAdd is inherently non-deterministic.

## Outputs

`DendrogramResult`:
- **centroids** `[m × k_clusters]` device matrix, col-major. Column k contains the mean expression (per-gene) of all cells labeled k. Empty clusters remain zero.
- **distance** `[k_clusters × k_clusters]` device symmetric matrix, col-major. Entry [i, j] = 1 − corr(centroid_i, centroid_j), where correlation is Pearson correlation after centering and L2-norm scaling.
- **linkage** `[(k_clusters − 1) × 4]` host vector in row-major (UPGMA standard): row i = [cluster_a, cluster_b, merge_distance, n_merged_cells].
- **m**, **k_clusters** — host-side dimensions.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (50 genes, 1k cells, k=5) | ~8.5 ms | ~12 MB | GPU-only centroids + UPGMA | atomic scatter + Sgemm |
| medium (20k genes, 100k cells, k=20) | ~220 ms | ~2.8 GB | chunked cell slabs | correlation k² = 400, UPGMA O(k³)=8M trivial |
| large (40k genes, 500k cells, k=100) | ~1.2 sec | ~8.5 GB | out-of-core slab strategy | k² = 10k, host UPGMA dominates |

GPU time dominated by (1) atomic-scatter centroid (O(nnz)), (2) column centering/normalization (O(m·k)), (3) cuBLAS Sgemm for correlation (O(m·k²)). Host UPGMA is O(k³) but k typically ≤ 100 → negligible. D2H transfer: label[n] one-shot (Rule 4), distance[k²] one-shot (Rule 4).

## Streaming behavior

- **Current (CYCLE-146)**: in-memory only. Label array downloaded once to compute n_per_cluster; distance matrix computed on GPU then downloaded for host UPGMA.
- **Planned (CYCLE-151)**: chunked cell slabs for atomic-scatter pass (load slab of cell labels, scatter, release); no cross-slab dependencies.
- Correlation Sgemm can stream across column blocks of centroids if needed; not critical for typical k ≤ 100.

## Determinism

Conditionally deterministic. Atomic scatter in Step 1 (centroid aggregation) is non-deterministic due to warp scheduling — atomicAdd on ints in global memory has no guaranteed order when multiple threads race. Two runs on same input may differ in ULP (~1e-7 on float scale) due to atomic ordering. All other passes (centering, normalization, correlation, UPGMA) are fully deterministic. Set cfg.deterministic=true to document intent; actual behavior unchanged in v0.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| TinyClosedForm | distance matrix abs_err vs reference | < 1e-3 | 5 genes × 6 cells × 3 clusters | PASS |
| PerfectlySeparatedClusters | inter-cluster >> intra-cluster dist | > 5× | 50 genes × 100 cells × 3 clusters | PASS |
| LinkageMonotonic | UPGMA Z distances non-decreasing | exact | k=5 clusters | PASS |
| Determinism_SameInput | rel_err between two runs | < 1e-4 | same X, label, cfg | PASS (rel_err < 1e-4) |
| EmptyCluster_Handled | centroids[k] all-zero; no NaN/Inf | exact | cluster with 0 cells | PASS |

All tests run in `tests/embed_dendrogram_correctness.cpp` (CYCLE-146, ctest 5/5 PASS).

## Citation

> scanpy implementation: Wolf, F. A., Angerer, P., & Theis, F. J. (2018). "SCANPY: large-scale single-cell gene expression data analysis." _Genome Biol_ 19:15.
>
> UPGMA: Sokal, R. R., & Michener, C. D. (1958). "A statistical method for evaluating systematic relationships." _Univ Kans Sci Bull_ 38:1409-1438.

Algorithm: (1) atomic-scatter X[g, c] into centroid μ[g, k] by label[c], (2) divide by counts per cluster, (3) center each centroid column (subtract column mean), (4) L2-normalize each centroid column, (5) correlation via Sgemm: corr = μ_norm^T · μ_norm, (6) distance = 1 − corr, (7) host UPGMA average-linkage clustering.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/embed/dendrogram.h>
#include <singlet-gpu/io/pz_device_loader.h>

int main() {
    namespace sg = singlet_gpu;

    // Step 1: Load sparse expression matrix
    auto X = sg::io::load_pz_device_matrix("/path/to/normalized.pz");
    const int m = X.mat.rows;    // genes
    const int n = X.mat.cols;    // cells

    // Step 2: Create cluster labels (e.g., from Leiden on device)
    sg::core::DeviceMemory<int> d_label(static_cast<size_t>(n));
    // ... populate d_label (e.g., via leiden or external labeling) ...

    const int k_clusters = 20;  // example: 20 clusters

    // Step 3: Dendrogram
    sg::embed::DendrogramConfig cfg{};
    cfg.deterministic = false;  // document atomicAdd non-determinism
    auto dgram = sg::embed::dendrogram(X, d_label.get(), k_clusters, cfg);

    // Step 4: Access results
    // dgram.centroids : device [m × k], transfer to host if needed
    // dgram.distance : device [k × k], corr distances
    // dgram.linkage : host [(k-1) × 4], UPGMA tree

    // Optional: transfer to host
    std::vector<float> linkage_host = dgram.linkage;
    // linkage_host[4*i .. 4*i+3] = {a, b, dist, n_merged} for step i
}
```

## Pitfalls and notes

1. **Empty cluster guard: eps = 1e-9.** Zero-variance centroids (empty clusters or single-cell clusters with zero counts) result in L2-norm = 0. The normalization kernel guards with eps: if norm < 1e-9, the column stays zero (not NaN). Distance matrix entry = 1.0 max (uncorrelated), not NaN/Inf.

2. **Atomic-scatter is non-deterministic.** The centroid aggregation uses atomicAdd in global memory. When multiple threads from different blocks race on the same centroid coordinate, floating-point addition order is non-deterministic (warp scheduling). Relative error ~1e-4 in practice; higher-precision runs (e.g., fp64 atomic) are not implemented. For reproducibility, use k-means++ seeding on cluster initialization (separate from this kernel).

3. **6 GPU passes + 1 host pass.** Steps 1–6 are GPU; Step 7 UPGMA is host-side O(k³). For k=100, UPGMA takes ~1 ms; for k > 500, consider streaming or approximation algorithms.

4. **Label encoding must be contiguous [0, k_clusters).** Gaps or out-of-range labels (e.g., [0, 1, 3, 5]) will misindex the centroid matrix and produce garbage. Validate label range on host before calling.

5. **Correlation vs Euclidean distance.** Dendrogram uses 1 − Pearson correlation (after centering and normalization). This is invariant to gene-wise scaling differences but sensitive to mean shifts. For Euclidean-based hierarchical clustering, use a separate kernel or use Leiden distance directly.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-50g-1k | 8.5 | 12 | correctness (all 5 tests PASS) |
| medium-20k-100k-20c | 220 | 2800 | wall (atomic scatter + Sgemm), not memory-bound |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 370015 on H100, CYCLE-146).

## Links

- Design docs: [`state/designs/25-dendrogram.md`](../../state/designs/25-dendrogram.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § embed/dendrogram
- Tests: `tests/embed_dendrogram_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`graph_leiden.md`](graph_leiden.md) (clustering upstream), [`reduce_nmf.md`](reduce_nmf.md) (factornet rank selection, dendrogram for trend validation)
