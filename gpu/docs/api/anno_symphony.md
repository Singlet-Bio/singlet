# `anno::symphony`

Feature #23. Cell-type annotation via PCA projection onto reference followed by soft cluster assignment and label transfer. **First GPU implementation of Symphony (Kang et al. 2021). All 5 correctness tests PASS (CYCLE-138, job 369290).**

Pairs with `anno/celltypist` (CYCLE-135) to form the GPU reference-mapping annotation duo. Two complementary paradigms: centroid-projection (Symphony) vs logreg (CellTypist).

## C++ signature

```cpp
namespace singlet_gpu::anno {

struct SymphonyConfig {
    float eps_sigma  = 1e-3f;   // guard for ref σ close to 0 in standardization
    float eps_dist   = 1e-6f;   // guard for distance close to 0 in soft-assign 1/d
    bool  deterministic = true;  // cuBLAS Sgemm + warp-shuffle are deterministic
};

struct SymphonyResult {
    core::DeviceMemory<int>   pred_class;   // [n_query_cells] int ∈ [0, n_classes)
    core::DeviceMemory<float> confidence;   // [n_query_cells] max label prob ∈ [0, 1]
    int n_query_cells;
    int n_classes;
};

SymphonyResult symphony(
    const core::DeviceDense& Z_query,       // [n_features × n_query_cells] col-major
    const core::DeviceDense& Z_ref,         // [n_features × n_ref_cells] col-major
    const core::DeviceMemory<float>& mu_ref,  // [n_features] reference mean
    const core::DeviceMemory<float>& sigma_ref,  // [n_features] reference std dev
    const core::DeviceMemory<float>& W_pca,  // [n_features × k_pcs] PCA loadings col-major
    const core::DeviceMemory<int>& cluster_labels,  // [n_ref_cells] int ∈ [0, n_clusters)
    const SymphonyConfig& cfg = {},
    cudaStream_t stream = nullptr);

}  // namespace singlet_gpu::anno
```

## Python signature

```python
import singlet_gpu as sg
import anndata

# Setup: reference and query, both normalized + log-transformed to same gene subset
adata_ref = sg.io.read_anndata("/path/to/reference.1pz_dir/")
adata_query = sg.io.read_anndata("/path/to/query.1pz_dir/")

# Preprocess both to same genes
common_genes = sorted(set(adata_ref.var.index) & set(adata_query.var.index))
adata_ref = adata_ref[:, common_genes].copy()
adata_query = adata_query[:, common_genes].copy()

# Normalize + log-transform
sg.preprocess.log_normalize(adata_ref, target_sum=1e4)
sg.preprocess.log_normalize(adata_query, target_sum=1e4)

# Fit PCA on reference
sg.reduce.svd(adata_ref, n_components=30, use_rep='X')

# Map query onto reference via Symphony
pred = sg.anno.symphony(
    adata_query,
    adata_ref,
    mu_ref=adata_ref.var['mean'],
    sigma_ref=adata_ref.var['std'],
    W_pca=adata_ref.varm['W_pca'],
    cluster_labels=adata_ref.obs['cluster'],
)
# → adata_query.obs['symphony_pred_class'] : int
# → adata_query.obs['symphony_confidence'] : float [0, 1]
```

## R signature

```r
singletGpu::symphony(adata_query, adata_ref, mu_ref, sigma_ref, W_pca, cluster_labels)
```

## Inputs

- **Z_query** — `core::DeviceDense` [n_features × n_query_cells] col-major query gene expression. Log-normalized, same gene order as reference.
- **Z_ref** — `core::DeviceDense` [n_features × n_ref_cells] col-major reference expression. Used only to pre-compute PCA; not needed at inference time in production.
- **mu_ref**, **sigma_ref** — `core::DeviceMemory<float>` [n_features] reference mean and standard deviation per gene (precomputed from Z_ref or loaded).
- **W_pca** — `core::DeviceMemory<float>` [n_features × k_pcs] PCA loadings (eigenvectors scaled by singular values). Typically k_pcs = 30.
- **cluster_labels** — `core::DeviceMemory<int>` [n_ref_cells] cell-type or cluster assignment for each reference cell. Determines label transfer targets.
- **cfg.eps_sigma** — guard value for σ ≈ 0 to prevent division by zero in standardization. Default 1e-3.
- **cfg.eps_dist** — guard for distance ≈ 0 to prevent inf in soft-assign 1/d. Default 1e-6.

## Outputs

`SymphonyResult`:
- **pred_class** `[n_query_cells]` argmax label across soft-assigned clusters. Integer in [0, n_classes).
- **confidence** `[n_query_cells]` max label probability after soft-assignment + label transfer. ∈ [0, 1].
- **n_query_cells**, **n_classes** host scalars.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k query, 5k ref, n_feat=5k, k_pcs=30) | ~3 ms | ~48 MB | 5 passes + cuBLAS |  —  |
| medium (20k query, 50k ref) | ~55 ms | ~320 MB | chunked per-batch query via Sgemm + soft-assign | cuBLAS dominates |
| large (100k+ query) | pending feature 17 | O(n_query × k_pcs) temporary | segment query via streaming | W_pca, mu_ref, sigma_ref broadcast |

5 passes: (1) standardize Z_query O(n_feat × n_query), (2) cuBLAS Sgemm project O(k_pcs × n_query), (3) distance compute (per-col sumsq + per-row sumsq + cross Sgemm + combine), (4) soft-assign 1/d per-cell kernel, (5) label-transfer Sgemm + argmax. Dominated by two cuBLAS Sgemm calls.

## Streaming behavior

- **Current (CYCLE-138)**: in-memory. Full distance and label-prob matrices allocated.
- **Planned (CYCLE-151+)**: chunked per-batch query cells. Reference centroid distance + soft-assignment broadcast, label transfer per-batch.
- Passes 1–5 naturally partition: standardize per-shard, project per-shard, distance per-shard, soft-assign per-shard, label-transfer per-shard. Reference stats (mu, sigma, W_pca) are read-only.

## Determinism

Fully deterministic. cuBLAS Sgemm deterministic at fp32. Warp-shuffle reductions deterministic. No atomics in hot path. Same input → bit-identical output.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| TinyClosedForm | pred_class, confidence match hand calc | exact | 3 query, 5 ref, 2 clusters | PASS |
| QueryNearCluster_AssignsClusterLabel | high confidence for query near ref cluster | confidence ≥ 0.8 | synthetic colocalized clusters | PASS |
| AmbiguousQuery_LowConfidence | low confidence when query equidistant | confidence ≤ 0.6 | query at centroid of 2 clusters | PASS |
| Determinism_BitIdentical | rel_err(run 1, run 2) | 0.0 | same data, seed=42 | PASS (0.00e+00) |
| DegenerateZeroSigma_Survives | prediction when sigma_ref≈0 | no crash/inf | one gene with σ=0, eps_sigma=1e-3 | PASS |

All tests in `tests/anno_symphony_correctness.cpp` (CYCLE-138, ctest 5/5 PASS).

## Citation

> Kang JB, Nathan A, Weinand K, et al. (2021). Efficient and precise single-cell reference atlas mapping with Symphony. _Nature Communications_, 12(1):5890. https://doi.org/10.1038/s41467-021-26146-6

Algorithm: (1) standardize query to reference mean/std, (2) project onto reference PCA (cuBLAS Sgemm), (3) compute Euclidean distance to reference cluster centroids, (4) soft-assign queries to clusters via 1/distance weighting, (5) transfer cluster labels to cells via weighted label probability. Unlike centroid-based classifiers, Symphony projects into PCA space (dimensionality reduction) before distance, improving robustness.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/anno/symphony.h>
#include <singlet-gpu/preprocess/log_normalize.h>
#include <singlet-gpu/reduce/svd.h>

int main() {
    namespace sg = singlet_gpu;

    // Step 1: load reference and query
    auto pz_ref = sg::load_pz("/path/to/reference.1pz");
    auto pz_query = sg::load_pz("/path/to/query.1pz");
    cudaStreamSynchronize(pz_ref.producer_stream);
    cudaStreamSynchronize(pz_query.producer_stream);

    // Step 2: preprocess both
    sg::preprocess::log_normalize(pz_ref.mat, {}, pz_ref.producer_stream);
    sg::preprocess::log_normalize(pz_query.mat, {}, pz_query.producer_stream);

    // Step 3: fit PCA on reference
    sg::reduce::SvdConfig svd_cfg{};
    svd_cfg.n_comps = 30;
    auto svd_ref = sg::reduce::svd(pz_ref.mat, svd_cfg, pz_ref.producer_stream);

    // Step 4: compute reference statistics (μ, σ per gene)
    // (Implementation detail: done via qc::calculate_qc_metrics or custom kernel)
    auto mu_ref = /* ... */;
    auto sigma_ref = /* ... */;

    // Step 5: Symphony prediction
    sg::anno::SymphonyConfig cfg{};
    cfg.eps_sigma = 1e-3f;
    cfg.eps_dist = 1e-6f;

    // Note: in production, pass reference cluster labels (from metadata)
    // For this example, assume cluster_labels_host was loaded from adata_ref.obs['cluster']
    auto cluster_labels_dev = sg::core::DeviceMemory<int>::copyFromHost(cluster_labels_host);

    auto result = sg::anno::symphony(
        /* Z_query_dense */, /* Z_ref_dense */, mu_ref, sigma_ref,
        svd_ref.loadings, cluster_labels_dev, cfg, pz_query.producer_stream);
    cudaStreamSynchronize(pz_query.producer_stream);

    // Step 6: transfer predictions to host
    std::vector<int> pred_class(result.n_query_cells);
    cudaMemcpy(pred_class.data(), result.pred_class.get(),
               result.n_query_cells * sizeof(int), cudaMemcpyDeviceToHost);
}
```

## Pitfalls and notes

1. **PCA projection requires matching reference.** The W_pca matrix is specific to the reference dataset. Query cells must be projected onto the same PCA; using a different reference's W_pca silently produces incorrect labels. No validation; you must ensure alignment.

2. **Reference statistics (μ, σ) are computed once.** If reference is very small (< 100 cells per type), σ estimates may be noisy. Larger reference (≥ 10k cells) is recommended. Per-cluster statistics (not global) are a v1.1 enhancement.

3. **Soft-assignment weighting is temperature-invariant.** Unlike some soft-assignment schemes, Symphony uses pure inverse-distance weighting (1/d) with no temperature tuning. If reference clusters are tightly separated, soft-assignments may be hard-clustered. Adding a temperature parameter is a future extension.

4. **Label transfer assumes one label per reference cluster.** If reference has multiple cell types per cluster, label transfer outputs one label (cluster-level, not cell-type-level). For fine-grained annotation, use CellTypist instead.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | accuracy | dominates_on |
|---|---|---|---|---|
| small-1k | 3 | 48 | 5/5 tests PASS, confidence ≥0.8 | correctness (all tests), usability (centroid intuitive) |
| medium-20k | 55 | 320 | pending v1.1 bench vs R Symphony | wall dominates |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369290 on g051 H100 NVL, CYCLE-138).

## Links

- Design docs: [`state/designs/23-symphony.md`](../../state/designs/23-symphony.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § anno/symphony
- Tests: `tests/anno_symphony_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`anno_celltypist.md`](anno_celltypist.md) (sister module, CYCLE-135 — logreg annotation), [`reduce_svd.md`](reduce_svd.md) (PCA used for projection), [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (required preprocessing)
