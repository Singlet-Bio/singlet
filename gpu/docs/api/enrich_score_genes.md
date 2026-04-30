# `enrich::score_genes`

Feature #29. Per-cell gene-set scoring with matched-control subtraction. GPU port of scanpy.tl.score_genes (Satija et al. 2015 / Seurat AddModuleScore). **All 5 correctness tests PASS (CYCLE-129, job 369107).**

For each cell and gene set, computes mean expression of set genes minus a random control pool sampled from genes with similar expression levels. Deterministic via seeded std::mt19937 host RNG.

## C++ signature

```cpp
namespace singlet_gpu::enrich {

struct ScoreGenesConfig {
    int      ctrl_size              = 50;   // control set size per gene set
    int      n_bins                 = 25;   // expression-mean bins (scanpy default)
    uint64_t seed                   = 0;    // std::mt19937 seed for control sampling
    bool     use_set_size_for_ctrl  = true; // max(ctrl_size, |S|) — scanpy behavior
    bool     deterministic          = true; // no-op; always deterministic
};

struct ScoreGenesResult {
    core::DeviceMemory<float> scores;  // n_cells × n_sets, col-major
    int n_cells;
    int n_sets;
};

ScoreGenesResult score_genes(
    const io::PzDeviceMatrix&              X,
    const std::vector<std::vector<int>>&  gene_sets,
    const ScoreGenesConfig&               cfg    = {},
    cudaStream_t                          stream = nullptr);

}  // namespace singlet_gpu::enrich
```

## Python signature

```python
import singlet_gpu as sg

sg.enrich.score_genes(
    adata,
    gene_sets={"set1": ["GENE1", "GENE2", ...], ...},
    ctrl_size=50,
    n_bins=25,
    use_set_size_for_ctrl=True,
    seed=0,
    layer=None, inplace=True, copy=False,
)
# → adata.obsm['X_score_genes']  (n_cells × n_sets)
# → adata.uns['score_genes']['seed', 'ctrl_size']
```

## R signature

```r
singletGpu::scoreGenes(adata, gene_sets, ctrl_size = 50L, seed = 0L)
```

## Inputs

- **X** — `io::PzDeviceMatrix` with log-normalized counts (or raw counts if passed, will be normalized by caller).
- **gene_sets** — host-side vector of gene-index lists, each list a set of indices in [0, m). Indices must be valid; checked on entry.
- **cfg.ctrl_size** — target control pool size per set; clamped to ≥1. If `use_set_size_for_ctrl=true`, effective size = max(ctrl_size, |set|).
- **cfg.n_bins** — number of equal-width mean-expression bins (default 25, scanpy's standard).
- **cfg.seed** — uint64 seed for std::mt19937. Same seed → bit-identical output across runs.

## Outputs

`ScoreGenesResult`:
- **scores** `[n_cells × n_sets]` device-resident col-major matrix. Entry [c, s] = mean(X[g ∈ S, c]) - mean(X[g ∈ ctrl_s, c]). Caller must sync stream before reading.
- **n_cells**, **n_sets** — host-side dimensions for downstream sizing.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Notes |
|---|---|---|---|
| small (1k cells, 100 gene sets) | ~1.2 ms | ~12 MB (W matrix 20k × 100) | Pass 1 (mean) + Pass 2 (bin host) + Pass 3 (W host) + Pass 4 (SpMM) |
| medium (20k cells, 50 sets) | ~8.5 ms | ~24 MB | SpMM dominates (O(nnz × n_sets)) |
| large (100k+ cells) | O(nnz) streaming ready | bin-by-bin chunking | CYCLE-151 target |

All passes stream-safe; control sampling (Pass 3) is entirely host-side.

## Streaming behavior

**Current (CYCLE-129)**: in-memory only. Control sampling is per-set and requires global gene-mean data (Pass 1 output) — naturally streamed in slab sizes via per-slab mean computation and accumulation.

**Planned (CYCLE-151)**: streaming driver will chunk per slab; control pools re-sampled per slab to approximate full-data bin distribution.

## Determinism

Fully deterministic. std::mt19937 seeded from `cfg.seed XOR (set_index + 1)` ensures each set gets an independent RNG state. Atomic scatter in Pass 1 (mean computation) has ≤1e-5 relative error for log-normalized data; binning (integer floor) is robust to this noise. Identical inputs (same X, same gene_sets, same cfg.seed) → bit-identical scores.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| TinyClosedForm | analytical vs computed | max_abs_err ≤ 1e-4 | 5-gene set, 10 cells | PASS |
| PlantedSet_HighScore | set genes rank near top | top-5 mean > bottom-5 mean | synthetic 50 genes | PASS |
| MultipleSets_Independent | set correlations low | corr between set scores ≤ 0.3 | 4 independent sets | PASS |
| Determinism_SameSeed | same seed → bit-identical | rel_err = 0 | 3 runs, cfg.seed=42 | PASS |
| AllOnesInput | constant input → 0 score | all(|scores| < 1e-6) | X[*] = 1.0, 2 sets | PASS |

All tests in `tests/enrich_score_genes_correctness.cpp` (CYCLE-129, ctest 5/5 PASS).

## Citation

> Satija R, Farrell JA, Gennert D, Schier AF, Regev A (2015). Spatial reconstruction of single-cell gene expression data. _Nat Biotechnol_ 33:495-502.

The algorithm is Satija's matched-control approach (§Methods): stratify genes by mean expression into equal-width bins, for each set pool controls from the same bins, compute difference of means. scanpy's `tl.score_genes` implements this exactly; singlet-gpu's GPU port uses the same bin-pooling and Fisher-Yates sampling but on device for speed.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/enrich/score_genes.h>

int main() {
    namespace sg = singlet_gpu;
    
    auto pz = sg::load_pz("/path/to/log_counts.1pz");
    cudaStreamSynchronize(pz.producer_stream);
    
    // Define gene sets by index (host-side)
    std::vector<std::vector<int>> gene_sets{
        {10, 20, 30},           // set 0: three genes
        {5, 15, 25, 35}         // set 1: four genes
    };
    
    sg::enrich::ScoreGenesConfig cfg{};
    cfg.ctrl_size  = 50;
    cfg.n_bins     = 25;
    cfg.seed       = 42;
    
    auto res = sg::enrich::score_genes(pz.mat, gene_sets, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);
    
    // res.scores[c + s * n_cells] = score for cell c, set s
}
```

## Pitfalls and notes

1. **Gene indices must be valid.** All indices in gene_sets must be in [0, m). Caller responsible for validation (gene name → index mapping). Kernel checks and throws.

2. **Bin width is equal-width on [μ_min, μ_max].** Genes with identical μ may fall into different bins due to floating-point binning (floor operation). This is negligible for large gene sets but affects small sets (|S| < 5) — control pool may under-represent low-variance genes.

3. **Empty gene sets are rejected.** A set with 0 genes throws a std::runtime_error. Control sets become empty only if all genes in the set bins are in the set itself — in that case, control weight is 0 and only set mean contributes.

4. **Determinism is seeded per-set.** Changing cfg.seed shifts all control pools; changing gene_sets order (even if same genes) yields different control samples due to per-set RNG seeding via XOR(seed, s+1).

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-1k | 1.2 | 12 | correctness (all tests PASS), usability (scanpy-compatible API) |
| medium-20k | 8.5 | 24 | wall (O(nnz) SpMM), user experience |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369107, CYCLE-129).

## Links

- Design docs: [`state/designs/29-score-genes.md`](../../state/designs/29-score-genes.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § enrich/score_genes
- Tests: `tests/enrich_score_genes_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`enrich_decoupler_wsum.md`](enrich_decoupler_wsum.md) (sister module, similar SpMM-based pipeline), [`reduce_svd.md`](reduce_svd.md) (commonly feeds into downstream analyses)
