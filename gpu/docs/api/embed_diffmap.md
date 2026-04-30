# `embed::diffmap`

Feature #16. Diffusion-map embedding via symmetric normalization + cuSOLVER Ssyevd eigendecomposition. Compute low-dimensional Coifman–Lafon trajectories on GPU. **All 5 correctness tests PASS (CYCLE-150, job 370271).**

Pairs with `embed/dpt` (CYCLE-142) to form the GPU diffusion-geometry trajectory toolkit. Two entry points: one computes the embedding and top eigenvalues; the other returns the full transition matrix for downstream analysis.

## C++ signature

```cpp
namespace singlet_gpu::embed {

struct DiffmapConfig {
    int   n_comps      = 15;     // number of diffusion components (excludes trivial λ=1)
    int   t            = 1;      // diffusion time (eigenvalues raised to this power)
    float eps_lambda   = 1e-6f;  // guard for very small λ
    bool  deterministic = true;  // no-op: cuSOLVER Ssyevd is deterministic at fp32
};

struct DiffmapResult {
    core::DeviceMemory<float> embedding;    // n_cells × n_comps col-major
    core::DeviceMemory<float> eigenvalues;  // n_comps (descending — largest non-trivial first)
    int n_cells, n_comps;
};

DiffmapResult diffmap(
    const graph::KnnResult& knn,
    const DiffmapConfig&    cfg    = {},
    cudaStream_t            stream = nullptr);

}  // namespace singlet_gpu::embed
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
# ... downstream: compute_knn(adata) → diffmap ...

sg.embed.diffmap(
    adata,
    n_components=15,         # top eigenvalues to retain
    t=1.0,                   # diffusion time
    use_rep="X_pca",         # input is a PCA embedding
    key_added="X_diffmap",   # adata.obsm key for embedding
    copy=False, inplace=True,
)
# → adata.obsm['X_diffmap'] : (n_cells, n_components)
# → adata.uns['diffmap']['eigenvalues'] : [λ_1, λ_2, ..., λ_k] (descending)
```

## R signature

```r
singletGpu::diffmap(adata, n_components = 15L, t = 1.0)
```

## Inputs

- **knn** — `graph::KnnResult` from `singlet_gpu::graph::compute_knn`, with both `neighbors` and `distances` populated. Must include k-nearest-neighbor distances (not just indices).
- **cfg.n_comps** — number of non-trivial eigenvectors to retain and scale. Must satisfy `1 ≤ n_comps < n_cells - 1`.
- **cfg.t** — diffusion time (power to which eigenvalues are raised). Typical: `t=1` (no scaling) or `t=2` for multi-hop random-walk emphasis. Must be `≥ 1`.

## Outputs

`DiffmapResult`:
- **embedding** `[n_cells × n_comps]` device-resident matrix, col-major. Entry `[c, k]` = λ_k^t · ψ_k(c), where ψ_k is the k-th eigenvector of the symmetric normalized transition matrix and λ_k is the k-th eigenvalue. Caller must sync stream before reading.
- **eigenvalues** `[n_comps]` on device, in descending order (largest non-trivial first). Excludes the trivial eigenvalue λ≈1.
- **n_cells**, **n_comps** host-side dimensions for downstream sizing.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k cells, k=15) | ~2.5 ms | ~16 MB (dense n²) | streaming scheduled for v1.1 |  —  |
| medium (20k cells) | ~280 ms | ~3.2 GB | chunked kNN + Lanczos | cuSOLVER dense eig is 50% of total |
| large (100k+ cells) | out-of-core via Lanczos | O(n×k) approx | CYCLE-151 target | dense matrix hits >50% GPU mem gate |

All passes are `O(nnz_knn + n²)` where nnz_knn = n·k. Memory guard: throws if the dense transition matrix T exceeds 50% free GPU memory.

## Streaming behavior

- **Current (CYCLE-150)**: in-memory only. Dense T = n² × 4 bytes is allocated and freed per call.
- **Planned (CYCLE-151)**: chunked eigensolver via cuSOLVER Lanczos (Ssyevr) for streaming large n. Requires row-major kNN input.
- The six passes (σ → W → symmetrize → normalize → eig → scale) are naturally parallelizable across column slabs.

## Determinism

Fully deterministic. cuSOLVER `Ssyevd` (symmetric eigendecomposition via QR + Householder) is deterministic at fp32 on a fixed GPU architecture. The kernel has no random operations and no atomics. Same input (same knn, same cfg, same stream) → bit-identical output (up to fp32 rounding).

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| LinearChain_PrincipalComponentMonotonic | Spearman(embedding[:, 0], cell_index) | ≥ 0.9 | 50-cell chain, k=4 | PASS |
| TwoBranches_TwoComponentsSeparate | max t-stat over top 4 comps | > 2.0 | Y-graph 60 cells, 2 branches | PASS |
| EigenvalueOrdering | descending λ_1 ≥ λ_2 ≥ ... | exact | 40-cell ring, k=4 | PASS |
| Determinism_BitIdentical | rel_err between two runs | 0.0 | 40-cell ring, k=4 | PASS (0.00e+00) |
| TPower_ScalesEmbedding | embed(t=2) / embed(t=1) = λ_k | ≤ 1e-3 | 40-cell ring, k=4, t∈{1,2} | PASS (0.00e+00) |

All tests run in `tests/embed_diffmap_correctness.cpp` (CYCLE-150, ctest 5/5 PASS).

## Citation

> Coifman, R. R., & Lafon, S. (2006). Diffusion maps. _Applied and Computational Harmonic Analysis_, 21(1), 5–30. https://doi.org/10.1016/j.acha.2006.04.006

The algorithm: (1) per-cell local bandwidth σ_i = median(k_neighbors), (2) Gaussian similarity W[i,j] = exp(−d²/(σ_i σ_j)) on kNN edges, (3) symmetrize W, (4) symmetric normalization T = D^{−1/2} W D^{−1/2} (required — cuSOLVER assumes symmetric input), (5) eigendecompose T → eigenvalues (ascending) + eigenvectors, (6) embed by scaling top n_comps eigenvectors by λ_k^t (diffusion time).

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/embed/diffmap.h>
#include <singlet-gpu/graph/knn.h>

int main() {
    namespace sg = singlet_gpu;

    // Step 1: compute kNN from expression matrix
    auto mat = sg::load_pz("/path/to/exon_counts.1pz");
    auto pca = sg::reduce::svd(mat, {.n_comps = 30}, mat.producer_stream);
    auto knn = sg::graph::compute_knn(pca, {.k = 15}, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Step 2: diffusion map
    sg::embed::DiffmapConfig cfg{};
    cfg.n_comps = 15;
    cfg.t       = 1;
    auto diffmap_res = sg::embed::diffmap(knn, cfg, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Step 3: transfer embedding to host (optional)
    std::vector<float> embedding_host(
        diffmap_res.n_cells * diffmap_res.n_comps);
    cudaMemcpy(embedding_host.data(), diffmap_res.embedding.get(),
               embedding_host.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // embedding_host[c + k * n_cells] = embedding at cell c, component k
}
```

## Pitfalls and notes

1. **Symmetric normalization is required.** The transition matrix must be normalized as `T = D^{−1/2} W D^{−1/2}`, not the asymmetric Markov `D^{−1} W`. cuSOLVER's Ssyevd assumes a symmetric input; feeding it an asymmetric matrix silently produces incorrect eigenvectors and eigenvalues. (Lesson from CYCLE-142: always validate matrix symmetry before eigensolve.)

2. **Eigenvector sign is unconstrained.** Standard eigensolvers return eigenvectors up to sign. If downstream tasks depend on the sign of an eigenvector (e.g., a cell-type annotation that checks "sign(component_k) > threshold"), canonicalize the sign in post-processing (e.g., force the first entry positive) or use absolute value for rank-based metrics (Spearman, t-statistic).

3. **Multi-branch trajectories: sweep top k components.** For Y-shaped or multi-branch geometries, the "true" branching axis may land on ψ_2 or ψ_3 instead of ψ_1, depending on the geometry's symmetry and the data distribution. Use all top k components in downstream tasks (e.g., UMAP on the full embedding, not just the first two components).

4. **Memory guard enforces a ceiling.** If dense T = n² × 4 bytes exceeds 50% of free GPU memory, the kernel throws. For n > ~5000 on a single GPU, switch to landmark diffusion or chunked Lanczos (CYCLE-151 feature).

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-1k | 2.5 | 16 | correctness (all tests PASS), usability (simple API) |
| medium-20k | 280 | 3200 | wall (cuSOLVER dominates) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 370271 on g051 H100 NVL, CYCLE-150).

## Links

- Design docs: [`state/designs/16-diffmap.md`](../../state/designs/16-diffmap.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § embed/diffmap
- Tests: `tests/embed_diffmap_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`embed_dpt.md`](embed_dpt.md) (sister module, CYCLE-142 — shares passes 1–5), [`reduce_svd.md`](reduce_svd.md) (PCA feeds kNN), [`graph_knn.md`](graph_knn.md) (kNN input)
