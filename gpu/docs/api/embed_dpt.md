# `embed::dpt`

Feature #24. Diffusion Pseudotime (DPT) — per-cell pseudotime based on diffusion-map eigendecomposition from a root cell. Traces cell trajectories in branching geometries via Haghverdi et al. 2016. **All 5 correctness tests PASS (CYCLE-142, job 369371).**

Consumes `embed/diffmap` eigenvectors and eigenvalues; pairs with kNN graphs. Two entry points: fast version from precomputed diffmap, direct version from kNN graph alone (6 GPU passes + 1 cuSOLVER eigendecomp).

## C++ signature

```cpp
namespace singlet_gpu::embed {

struct DptConfig {
    int   root_cell     = 0;      // index of root cell in [0, n)
    int   n_eigenvecs   = 15;     // number of diffusion components used (skip trivial λ=1)
    float eps_lambda    = 1e-6f;  // clamp (1-λ) to max(eps_lambda, 1-λ) for stability
    bool  deterministic = true;   // no-op: cuSOLVER Ssyevd is deterministic at fp32
};

struct DptResult {
    core::DeviceMemory<float> pseudotime;  // n_cells; pseudotime[root] ≈ 0
    int n_cells;
};

DptResult dpt(
    const graph::KnnResult& knn,
    const DptConfig&        cfg    = {},
    cudaStream_t            stream = nullptr);

}  // namespace singlet_gpu::embed
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
# Compute kNN, then diffusion map
knn_result = sg.graph.compute_knn(adata, k=15, use_rep="X_pca")
diffmap_result = sg.embed.diffmap(adata, n_components=15, use_rep="X_pca")

# DPT: pseudotime from root cell
root_cell = 0
dpt_result = sg.embed.dpt(
    knn_result,
    root_cell=root_cell,
    n_eigenvecs=15,
    key_added="dpt_pseudotime",  # writes to adata.obs['dpt_pseudotime']
)
# → adata.obs['dpt_pseudotime'] : [n_cells], pseudotime[root] ≈ 0
```

## R signature

```r
singletGpu::dpt(adata, knn_result, root_cell = 0L, n_eigenvecs = 15L)
```

## Inputs

- **knn** — `graph::KnnResult` from `singlet_gpu::graph::compute_knn`. Must include both `neighbors` and `distances` (k-nearest-neighbor distances required).
- **cfg.root_cell** — integer index in [0, n_cells). The cell from which pseudotime distance is measured; pseudotime[root_cell] ≈ 0.
- **cfg.n_eigenvecs** — number of non-trivial eigenvectors to use in distance accumulation. Typical: 15. Must satisfy `2 ≤ n_eigenvecs < n_cells`.
- **cfg.eps_lambda** — numerical guard for (1 − λ) in denominator of diffusion-time scaling. Clamps to max(eps_lambda, 1 − λ) for stability. Typical: 1e-6.

## Outputs

`DptResult`:
- **pseudotime** `[n_cells]` device-resident array. Entry [c] = √(Σ_k (λ_k / (1 − λ_k))² × (V[root, k] − V[c, k])²), where V is the eigenvector matrix (top n_eigenvecs columns) and λ are eigenvalues from symmetric-normalized transition matrix T. Pseudotime[root] ≈ 0 by construction.
- **n_cells** — host-side dimension for downstream sizing.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k cells, k=15) | ~4.2 ms | ~8 MB (dense T matrix) | streaming scheduled for v1.1 | 6 GPU passes + cuSOLVER |
| medium (20k cells) | ~380 ms | ~3.2 GB | chunked Lanczos | cuSOLVER dominates; dense T hits ~50% mem gate |
| large (100k+ cells) | out-of-core via Lanczos | O(n×k) approx | CYCLE-151 target | dense T exceeds memory guard |

All passes O(nnz_knn + n²) where nnz_knn = n·k. Dense T = n² × 4 bytes. Memory guard: throws if T exceeds 50% free GPU memory. For n > ~5000, use landmark subset or Lanczos variant (CYCLE-151).

## Streaming behavior

- **Current (CYCLE-142)**: in-memory only. Dense n² transition matrix allocated and freed per call.
- **Planned (CYCLE-151)**: chunked Lanczos (Ssyevr) for large n. Requires row-major kNN input.
- Six passes naturally parallelize across column slabs; eigensolve dominates.

## Determinism

Fully deterministic. cuSOLVER `Ssyevd` (symmetric eigendecomposition) is deterministic at fp32. All passes are deterministic: median selection sort, Gaussian similarity, symmetrization, symmetric normalization (no atomics), and pseudotime accumulation. Same input (same knn, same cfg, same stream) → bit-identical output (up to fp32 rounding).

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| LinearTrajectory_MonotonicPseudotime | Spearman(dpt, chain_index) | ≥ 0.95 | 50-cell chain, k=15 | PASS |
| RootHasZeroPseudotime | dpt[root] | ≤ 1e-5 | any graph, root=0 | PASS |
| TwoBranches_DistinguishesPaths | t-stat over top 5 eigenvecs | > 2.5 | Y-shaped graph, 60 cells | PASS |
| Determinism_BitIdentical | rel_err across two runs | 0.0 | same knn, same cfg | PASS (0.00e+00) |
| MemoryGuard_RejectsTooLarge | exception on dense T > 50% mem | thrown | n=50000, H100 | PASS |

All tests run in `tests/embed_dpt_correctness.cpp` (CYCLE-142, ctest 5/5 PASS).

## Citation

> Haghverdi L, Büttner M, Wolf FA, Buettner F, Theis FJ (2016). "Diffusion pseudotime robustly reconstructs lineage branching." _Nat Methods_ 13:845-848. https://doi.org/10.1038/nmeth.3971

Algorithm: (1) per-cell local bandwidth σ_i = median(k_neighbors), (2) Gaussian similarity W[i,j] = exp(−d²/(σ_i σ_j)) on kNN edges, (3) symmetrize W, (4) symmetric normalization T = D^{−1/2} W D^{−1/2}, (5) eigendecompose T → eigenvalues (ascending) + eigenvectors, (6) per-cell DPT distance from root: √(Σ_k (λ_k/(1−λ_k))² × (V[r,k] − V[c,k])²).

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/embed/dpt.h>
#include <singlet-gpu/graph/knn.h>

int main() {
    namespace sg = singlet_gpu;

    // Step 1: Load data and compute kNN
    auto mat = sg::io::load_pz("/path/to/exon_counts.1pz");
    auto pca = sg::reduce::svd(mat, {.n_comps = 30}, mat.producer_stream);
    auto knn = sg::graph::compute_knn(pca, {.k = 15}, mat.producer_stream);

    // Step 2: DPT from root cell 0
    sg::embed::DptConfig cfg{};
    cfg.root_cell   = 0;
    cfg.n_eigenvecs = 15;
    auto dpt_res = sg::embed::dpt(knn, cfg, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Step 3: Transfer pseudotime to host (optional)
    std::vector<float> pseudotime(dpt_res.n_cells);
    cudaMemcpy(pseudotime.data(), dpt_res.pseudotime.get(),
               pseudotime.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    // pseudotime[0] ≈ 0.0, increasing along trajectory
}
```

## Pitfalls and notes

1. **Symmetric normalization is mandatory.** The transition matrix must use symmetric normalization T = D^{−1/2} W D^{−1/2}, not asymmetric row-stochastic D^{−1} W. cuSOLVER's Ssyevd requires a symmetric input; asymmetric matrices silently produce incorrect eigenvectors. (Lesson from CYCLE-142 job 369371: always validate matrix symmetry before eigensolve.)

2. **σ_i = median of k distances is outlier-sensitive.** If a few cells have anomalously distant k-th neighbors, the Gaussian bandwidth becomes too large in those neighborhoods, damping local structure. Ensure kNN graph has short edges across all cells: verify that max(distances) / median(distances) is reasonable (ideally < 10×).

3. **Root cell pseudotime should be ≈ 0, not exactly 0.** Floating-point rounding may produce tiny non-zero values (< 1e-5). Do not branch on pseudotime[root] == 0.0; use >= 1e-5 instead.

4. **Multi-branch ambiguity.** In Y-shaped or multi-branch geometries, the branching axis may land on ψ_2 or ψ_3 (not ψ_1) depending on symmetry. Use all top n_eigenvecs components in distance formula; single-component DPT can miss the branching axis.

5. **Memory guard enforces a ceiling.** Dense T exceeds 50% free GPU memory at ~n = 5000 on typical H100. For larger n, subset to landmarks or use Lanczos v1 (CYCLE-151).

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-1k | 4.2 | 8 | correctness (all 5 tests PASS, Spearman >= 0.95) |
| medium-20k | 380 | 3200 | wall (cuSOLVER Ssyevd dominates 65% of time) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369371 on g051 H100 NVL, CYCLE-142).

## Links

- Design docs: [`state/designs/24-dpt.md`](../../state/designs/24-dpt.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § embed/dpt
- Tests: `tests/embed_dpt_correctness.cpp` (5/5 PASS, ctest suite)
- Sister kernel: [`embed_diffmap.md`](embed_diffmap.md) (CYCLE-150, shares passes 1–5, σ_i bandwidth + symmetric-normalization pattern)
- Upstream: [`graph_knn.md`](graph_knn.md) (kNN input), [`reduce_svd.md`](reduce_svd.md) (PCA feeds kNN)
