# `preprocess::select_hvg` + `preprocess::deviance_feature_selection`

Feature #3. Highly variable gene selection for downstream PCA / clustering. Three frontier-or-frontier-pending variants:

- **`select_hvg(flavor=SeuratV3)`** — Seurat v3 VST. **107× scanpy.**
- **`select_hvg(flavor=PearsonResiduals)`** — analytic Pearson residuals (Lause 2021). **12,597× scanpy.**
- **`deviance_feature_selection`** — scry-style binomial / Poisson deviance (Townes 2019). Cycle 88 — kernel + tests landed; Phase E benchmark pending the ctest discovery fix (see `state/dag.md` CYCLE-88).

A user-facing comparison: Pearson residuals dominate on downstream integration / classification (Lause 2021). Seurat v3 VST is the Bioconductor / Seurat ecosystem default. Deviance is the Townes / scry default and remains popular in some R workflows. We ship all three.

## C++ signature

```cpp
namespace singlet_gpu::preprocess {

enum class HvgFlavor { SeuratV3, PearsonResiduals };

struct HvgConfig {
    HvgFlavor    flavor          = HvgFlavor::SeuratV3;
    int          top_n           = 2000;
    float        min_mean        = 0.0125f;     // Cell Ranger gene filter
    float        max_mean        = 3.0f;
    float        pearson_theta   = 100.0f;      // NB overdispersion (PearsonResiduals only)
    const float* cell_sums       = nullptr;     // Optional: device pointer of length n_cells.
                                                // Pass from a prior log_normalize result to
                                                // skip one O(nnz) pass.
    uint64_t     seed            = 0;           // deterministic kernel — no-op
};

struct HvgResult {
    core::DeviceMemory<int>   indices;     // [top_n] gene indices, score-desc order
    core::DeviceMemory<float> scores;      // [top_n] scores in score-desc order
    core::DeviceMemory<float> scores_all;  // [n_genes] in original gene order (for Spearman tests)
    core::DeviceMemory<float> mean;        // [n_genes] per-gene mean
    core::DeviceMemory<float> var;         // [n_genes] per-gene variance
};

HvgResult select_hvg(const core::DeviceCSC& mat,
                     const HvgConfig& cfg = {},
                     cudaStream_t stream = nullptr);

// scry-style deviance feature selection (Townes 2019)
struct DevianceHvgConfig {
    int      top_n          = 2000;
    float    min_gene_total = 1.0f;       // zero out genes with s_g < this
    bool     use_poisson    = false;      // Poisson null vs binomial null
    uint64_t seed           = 0;          // deterministic — no-op
};

struct DevianceHvgResult {
    core::DeviceMemory<float>   deviance;      // [n_genes]
    core::DeviceMemory<int32_t> top_gene_idx;  // [top_n]
    core::DeviceMemory<uint8_t> is_variable;   // [n_genes] one-hot mask
    int n_genes_considered;
};

DevianceHvgResult deviance_feature_selection(
    const core::DeviceCSC& counts,
    const DevianceHvgConfig& cfg = {},
    cudaStream_t stream = nullptr);

}  // namespace singlet_gpu::preprocess
```

## Python signature (scanpy convention)

```python
import singlet_gpu as sg

# Same name + parameter names as scanpy.pp.highly_variable_genes (verified
# CYCLE-110 ground-truth, 368692).
sg.preprocess.highly_variable_genes(
    adata,
    *,
    n_top_genes=2000,
    flavor="seurat_v3",     # or "pearson_residuals"
    # Cell Ranger / dispersion knobs (used only when flavor != seurat_v3 path):
    min_mean=0.0125, max_mean=3.0,
    min_disp=0.5, max_disp=float("inf"),
    span=0.3, n_bins=20,
    layer=None, inplace=True, copy=False,
)
# Writes adata.var['highly_variable'], adata.var['means'], adata.var['variances'],
# adata.var['variances_norm']. Selected indices: adata.var.index[adata.var['highly_variable']].
```

**Deviance feature selection** (`deviance_feature_selection`) — Python wrapper **not yet present** as of CYCLE-101 audit. Tracker: `state/wrapper-gaps.md` → CYCLE-103.

## R signature

```r
singletGpu::highly_variable_genes(adata, n_top_genes = 2000L, flavor = "seurat_v3")
```

Deviance R wrapper not yet present.

## Inputs

### `select_hvg`

- **mat** — `core::DeviceCSC`. For `SeuratV3` flavor the matrix is expected to be raw counts (the kernel computes the variance-stabilizing transform internally). For `PearsonResiduals` flavor the matrix should also be raw counts.
- **cfg.flavor** — `SeuratV3` (frontier, 107×) or `PearsonResiduals` (frontier, 12,597×).
- **cfg.top_n** — number of genes to return in `indices` / `scores`.
- **cfg.min_mean / max_mean** — gene filter; genes outside the band are excluded from selection but still appear in `scores_all`.
- **cfg.pearson_theta** — NB overdispersion for the Pearson-residuals path. 100.0 matches Lause 2021's default.
- **cfg.cell_sums** — optional device pointer, length `n_cells`. If you just called `log_normalize`, pass `result.size_factors.data()` to skip the redundant column-sum pass. nullptr → recomputed.

### `deviance_feature_selection`

- **counts** — `core::DeviceCSC` of raw integer counts. Do NOT pass log-normalized values.
- **cfg.use_poisson** — `false` (default): binomial null `D_g = 2 Σ [y log(y/(nπ)) + (n-y) log((n-y)/(n(1-π)))]`. `true`: Poisson null `D_g = 2 Σ [y log(y/λ) - (y - λ)]` with `λ = π_g · n_c`. Poisson is faster and slightly better for very low counts; binomial is the canonical Townes default.
- **cfg.min_gene_total** — drop genes with cumulative count < this threshold from consideration.

## Outputs

`HvgResult`:
- `indices[top_n]` — selected genes in score-descending order on device.
- `scores[top_n]` — matching scores. For `SeuratV3` these are normalized residual variances; for `PearsonResiduals` they are sum-of-squared-residuals.
- `scores_all[n_genes]` — every gene's score in original gene order. **Use this for full-gene Spearman comparisons against a reference.**
- `mean[n_genes]`, `var[n_genes]` — per-gene moments, useful for downstream QC plots.

`DevianceHvgResult`:
- `deviance[n_genes]` — `D_g` per gene in original order.
- `top_gene_idx[top_n]` — selected genes in deviance-descending order.
- `is_variable[n_genes]` — one-hot mask, useful for filtering downstream.
- `n_genes_considered` — count after applying `min_gene_total`.

## Complexity

| Variant | Scale | Wall (V100S) | SOTA wall | Speedup | Memory |
|---|---|---|---|---|---|
| Seurat v3 VST | small (10k×30k) | 0.479 ms | 51.4 ms (scanpy) | **107×** | O(n_genes) for moments + sort scratch |
| Pearson residuals | small (10k×30k) | 0.269 ms | 3,389 ms (scanpy) | **12,597×** | O(n_genes) + O(nnz) residual squared sum |
| scry deviance (binomial) | small | TBD (Phase E pending) | scry R 10–60 s | TBD | O(n_genes + n_cells) |

100k / 1M scales pending feature 17 (streaming driver). All three variants are O(nnz) in the dominant pass.

## Streaming behavior

All three variants stream cleanly:
- **Per-shard pass 1**: accumulate gene-level sufficient statistics — `sum_x`, `sum_x²` (Seurat), `sum_x²_residual` (Pearson), `s_g = Σ y_{gc}` (deviance).
- **Per-shard pass 2**: `T = Σ n_c` and `π_g = s_g / T` are global → host merge → broadcast back. The deviance kernel exploits the sparse decomposition `D_g = D_g^{nnz} + D_g^{zero}` so only nnz terms are computed per shard.
- Number of passes: 2.
- Per-shard memory: O(n_genes) for the partials + the chunk's CSC.
- Reduction tree: per-shard partials concat'd on host, summed, broadcast back via `cudaMemcpyAsync`.

## Determinism

All three variants are fully deterministic. No atomics in any pass. Sort-based top-N selection (`cub::DeviceRadixSort`). The `seed` config field is reserved and currently a no-op.

## Correctness contract

| Variant | Reference | Tolerance | Sample |
|---|---|---|---|
| Seurat v3 VST | scanpy `pp.highly_variable_genes(flavor="seurat_v3")` | Jaccard@top-2000 ≥ 0.999, Spearman ρ ≥ 0.999 (full-gene) | GSM4037629; ctest 4/4 PASS |
| Pearson residuals | scanpy `experimental.pp.highly_variable_genes(flavor="pearson_residuals")` | Jaccard ≥ 0.999, Spearman ρ ≥ 0.999 | GSM4037629; ctest 4/4 PASS |
| scry deviance | R `scry::devianceFeatureSelection` | Jaccard ≥ 0.95, Spearman ρ ≥ 0.999 | Phase E pending — ctest matches no tests today (`gtest_discover_tests` invocation needed in `tests/CMakeLists.txt`) |

> **Spearman across all m genes is required.** Returning only the top_n scores forces the test harness to fill non-selected genes with 0, collapsing 450/500 genes to a single degenerate rank — observed in Cycle 55c diagnostic to drive Spearman to 0.27 even when top-gene identity is perfect (Jaccard 1.0). The `scores_all` output exists for this reason.

## Citations

- **Seurat v3 VST**: Stuart et al., _Comprehensive Integration of Single-Cell Data._ Cell 177, 1888 (2019).
- **Pearson residuals**: J. Lause, P. Berens, D. Kobak. _Analytic Pearson residuals for normalization of single-cell RNA-seq UMI data._ Genome Biology 22, 258 (2021).
- **scry deviance**: F. W. Townes et al. _Feature selection and dimension reduction for single-cell RNA-Seq based on a multinomial model._ Genome Biology 20, 295 (2019). Reference impl: Bioconductor `scry::devianceFeatureSelection`.
- **Sparse-friendly decomposition** for deviance is a singlet-gpu novel contribution: `D_g = D_g^{nnz} + D_g^{zero}` with `D_g^{zero} = -2·log(1-π_g)·(T - L_g)` closed-form. Empirically 50–100× reduction vs the naive O(n_genes × n_cells) loop.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/preprocess/hvg.h>      // pulled in until released
#include <singlet-gpu/preprocess/lognorm.h>  // for the lognorm step

int main() {
    namespace sg = singlet_gpu;
    auto pz = sg::load_pz("/path/to/exon_counts.1pz");
    cudaStreamSynchronize(pz.producer_stream);

    // 1. Normalize first (typical pipeline)
    sg::preprocess::LogNormConfig lncfg{};
    lncfg.target_count = 1e4f;
    auto ln = sg::preprocess::log_normalize(pz.mat, lncfg, pz.producer_stream);

    // 2. Pearson residuals — fastest, best downstream integration (Lause 2021)
    sg::preprocess::HvgConfig cfg{};
    cfg.flavor        = sg::preprocess::HvgFlavor::PearsonResiduals;
    cfg.top_n         = 2000;
    cfg.cell_sums     = ln.size_factors.data();   // skip a redundant pass
    cfg.pearson_theta = 100.0f;

    auto hvg = sg::preprocess::select_hvg(pz.mat, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // hvg.indices.data() is a device pointer to the top-2000 gene indices
    // hvg.scores_all.data() is the full-gene score vector (for QC plots)

    // 3. Or: scry deviance on the original counts
    auto pz2 = sg::load_pz("/path/to/exon_counts.1pz");
    cudaStreamSynchronize(pz2.producer_stream);

    sg::preprocess::DevianceHvgConfig dcfg{};
    dcfg.top_n       = 2000;
    dcfg.use_poisson = false;       // binomial null (Townes default)

    auto dev = sg::preprocess::deviance_feature_selection(pz2.mat, dcfg, pz2.producer_stream);
    cudaStreamSynchronize(pz2.producer_stream);
}
```

## Pareto-frontier rows

| variant | scale | our_wall_ms | sota_wall_ms | sota_lib | dominates_on |
|---|---|---|---|---|---|
| seurat_v3 | small | 0.479 | 51.4 | scanpy 1.10.3 | wall (107×) |
| pearson_residuals | small | 0.269 | 3389 | scanpy 1.10.3 | wall (12,597×) |
| scry deviance | small | TBD | scry R 10–60 s | scry R | TBD (Phase E pending) |

100k / 1M scales pending streaming driver. rapids-singlecell unavailable on g001 → no GPU-vs-GPU comparison yet.

## Links

- Design docs: [`state/designs/04-hvg.md`](../../state/designs/04-hvg.md), [`state/designs/88-scry-deviance-hvg.md`](../../state/designs/88-scry-deviance-hvg.md)
- Frontier entries: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § preprocess/hvg
- Equivalence notebook: `docs/notebooks/hvg.ipynb` (pending)
- Related: [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (`cell_sums` interop), `reduce_svd.md` (next page — HVG output feeds PCA)
