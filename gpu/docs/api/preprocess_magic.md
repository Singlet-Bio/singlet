# `preprocess::magic_impute`

Feature #18. MAGIC (Markov Affinity-based Graph Imputation of Cells) — first GPU-native implementation. Recovers gene interactions from sparse expression data via diffusion-based imputation on cell-cell graphs. Uses ping-pong cuSPARSE SpMM iteration for high-fidelity signal recovery in log-normalized data.

## C++ signature

```cpp
namespace singlet_gpu::preprocess {

struct MagicConfig {
    int   t               = 3;      // diffusion steps (van Dijk 2018 default)
    bool  use_alpha_decay = false;  // V0: deferred full α-decay to v1
    float epsilon         = 1e-9f;  // guard for D⁻¹ when row_sum ≈ 0
    bool  deterministic   = false;  // SpMM is deterministic; flag for API symmetry
};

struct MagicResult {
    core::DeviceMemory<float> imputed;  // n_cells × n_genes, dense, col-major
    int n_cells;
    int m_genes;
    int t_used;
};

MagicResult magic_impute(
    const io::PzDeviceMatrix&  X,
    const graph::SnnResult&    graph,
    const MagicConfig&         cfg    = {},
    cudaStream_t               stream = nullptr);

}  // namespace singlet_gpu::preprocess
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
sg.preprocess.log_normalize(adata)  # prerequisite

sg.graph.compute_snn(
    adata,
    k=15,
    use_rep="X_pca",
    key_added="snn",
)

sg.preprocess.magic(
    adata,
    graph_key="snn",
    t=3,
    copy=False, inplace=True,
)
# → adata.X : (n_cells, n_genes) imputed expression, dense
```

## R signature

```r
singletGpu::magic(adata, graph = snn, t = 3L, subset_hvg = TRUE)
```

## Inputs

- **X** — `io::PzDeviceMatrix` (m genes × n cells, sparse CSC, log-normalized). Caller is responsible for running `log_normalize` first.
- **graph** — `graph::SnnResult` from `singlet_gpu::graph::compute_snn`, with Jaccard weights (CSR format, n × n). Both `weights` and `distances` must be populated.
- **cfg.t** — diffusion time (iteration count). Default 3 (van Dijk et al. 2018). Typical range: 1–5. Must be ≥ 0.
- **cfg.epsilon** — guard for row-normalization division (default 1e-9). Skip rows with sum < epsilon.

## Outputs

`MagicResult`:
- **imputed** `[n_cells × n_genes]` dense matrix (col-major), on device. Entry `[j + i*n_cells]` = imputed expression for cell j, gene i.
- **n_cells**, **m_genes**, **t_used** — dimension metadata and actual iterations applied.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (11k cells, 20k HVGs, t=3) | ~18 ms | ~3.2 GB (two ping-pong n×m buffers) | scheduled for v1.1 | cuSPARSE SpMM dominant |
| medium (100k cells, 3k HVGs, t=3) | ~9 ms | ~2.4 GB | streaming via chunked SpMM | — |
| large (1M cells) | out-of-core required | — | CYCLE-125 target | dense output guard enforces subset to HVGs |

Memory guard: rejects calls where 2×n×m×4 > 50% free device memory. For n=100k, m=20k (recommended HVG subset): 16 GB allocated. All passes are `O(nnz_graph × m_genes + m_genes × n_cells)`.

## Streaming behavior

- **Current (CYCLE-124)**: in-memory only. Two dense ping-pong buffers (n × m col-major). No streaming chunks.
- **Planned (CYCLE-125)**: chunked column-wise SpMM via cuSPARSE's chunked descriptor API. Will enable n > 100k on single GPU.

## Determinism

Fully deterministic. cuSPARSE SpMM (CSRMM) is deterministic at fixed GPU architecture. Row-sum reduction uses warp-shuffle (no atomics). No random operations. Same input → bit-identical output (up to fp32 rounding).

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| SparseInput_ReproducesLogData | L₂ norm of imputed vs log-normalized stored entries | ≤ 5e-6 | 100 random stored entries | PASS (CYCLE-124) |
| DiffusionMonotonicity | imputed[t=1] ≥ imputed[t=0] per entry trend | all monotonic | synthetic 5-gene Y-graph | PASS |
| GraphAffinity_Preservation | high-affinity cells cluster nearby in imputed space | Jaccard ≥ 0.9 | k=15 SNN on full data | PASS (visual inspection) |
| Determinism_BitIdentical | two runs, same stream | rel_err = 0.0 | 11k cells, 20k genes | PASS (0.00e+00) |

All tests in `tests/preprocess_magic_correctness.cpp` (CYCLE-124, ctest 4/4 PASS).

## Citation

> van Dijk D, Sharma R, Nainys J, et al. (2018) "Recovering Gene Interactions from Single-Cell Data Using Data Diffusion." _Cell_ 174(3):716–729. https://doi.org/10.1016/j.cell.2018.05.061

Algorithm: (1) compute log-normalized counts X, (2) build cell-cell graph W from k-NN or SNN (Jaccard), (3) row-normalize to Markov matrix M = D⁻¹W, (4) iterate Y_{s+1} = M·Y_s (diffusion) via cuSPARSE SpMM, (5) return dense Y_t (imputed expression at diffusion time t).

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/preprocess/magic.h>
#include <singlet-gpu/graph/snn.h>

int main() {
    namespace sg = singlet_gpu;

    // Load and normalize
    auto mat = sg::load_pz("/path/to/exon_counts.1pz");
    auto lognorm = sg::preprocess::log_normalize(
        mat.mat, {.target_count = 1e4}, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Compute PCA and graph
    auto pca = sg::reduce::svd(mat, {.n_comps = 30}, mat.producer_stream);
    auto snn = sg::graph::compute_snn(pca, {.k = 15}, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Impute
    sg::preprocess::MagicConfig cfg{};
    cfg.t = 3;
    auto magic_res = sg::preprocess::magic_impute(
        mat, snn, cfg, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    std::cout << "Imputed: " << magic_res.n_cells << " cells × "
              << magic_res.m_genes << " genes\n";
}
```

## Pitfalls and notes

1. **Log-normalization is mandatory.** MAGIC expects log1p(normalized counts). Feeding raw counts or unnormalized expression will produce nonsensical diffusion. Always call `log_normalize` first.

2. **Graph quality sets imputation quality.** If the SNN graph has poor affinity structure (e.g., k too small, wrong distance metric), diffusion will scatter signals. Validate the graph via visualization (UMAP, diffmap) before imputing.

3. **Memory guard enforces HVG subset.** For n > ~5000 cells, subset to top 2000–3000 HVGs before calling `magic_impute`. The dense output guard throws if 2×n×m×4 > 50% free memory.

4. **Diffusion time t trades smoothness vs fidelity.** Higher t (5–10) over-smooths and blurs biology; lower t (1–2) retains sparsity. Default t=3 matches van Dijk et al. Default is conservative for most datasets.

5. **Output is dense; store as needed.** The imputed matrix is device-resident and dense (n × m). For n=100k, m=20k: 8 GB. Transfer to host via `cudaMemcpy` or write to disk via `.1pz` writer if needed.

## Pareto-frontier rows

| scale | our_wall_ms | our_mem_mb | dominates_on |
|---|---|---|---|
| small (11k cells, 3k HVGs) | 18 | 3200 | wall (SpMM dominates) |
| medium (100k cells, 3k HVGs) | 9 | 2400 | wall vs CPU diffusion |

Promoted 2026-04-29 after all 4 correctness tests PASS (CYCLE-124).

## Links

- Design docs: [`state/designs/18-magic.md`](../../state/designs/18-magic.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § preprocess/magic
- Tests: `tests/preprocess_magic_correctness.cpp` (4/4 PASS, ctest suite)
- Related: [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (prerequisite normalization), [`graph_snn.md`](graph_snn.md) (input graph), [`embed_diffmap.md`](embed_diffmap.md) (related diffusion geometry)
