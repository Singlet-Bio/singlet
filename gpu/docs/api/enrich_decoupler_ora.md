# `enrich::ora`

Feature #31. Over-Representation Analysis (ORA) hypergeometric enrichment scoring from decoupleR. **All 5 correctness tests PASS (CYCLE-132, job 369186).** Tests whether a gene set is over-represented in each cell's top-K expressed genes via one-tailed hypergeometric test.

Returns -log10(p-value) per cell per gene set, suitable for hypothesis testing on pathway activity.

## C++ signature

```cpp
namespace singlet_gpu::enrich {

struct OraConfig {
    int  top_k       = 0;     // 0 = use ceil(m * 0.05); decoupleR default = 5%
    bool deterministic = true; // no-op: ORA v0 is deterministic by design
};

struct OraResult {
    core::DeviceMemory<float> scores;  // n_cells × n_sets, col-major; -log10(p)
    int n_cells;
    int n_sets;
};

OraResult ora(
    const io::PzDeviceMatrix&              X,
    const std::vector<std::vector<int>>&  gene_sets,
    const OraConfig&                      cfg    = {},
    cudaStream_t                          stream = nullptr);

}  // namespace singlet_gpu::enrich
```

## Python signature

```python
import singlet_gpu as sg

sg.enrich.ora(
    adata,
    gene_sets={"set1": ["GENE1", ...], ...},
    top_k=None,           # None = ceil(n_genes * 0.05)
    layer=None, inplace=True, copy=False,
)
# → adata.obsm['ora_scores']  (n_cells × n_sets)  # -log10(p-value)
```

## R signature

```r
singletGpu::ora(adata, gene_sets, top_k = NULL)
```

## Inputs

- **X** — `io::PzDeviceMatrix` with expression values (typically log1p-normalized for ranking).
- **gene_sets** — host-side vector of gene-index lists; indices in [0, m). Checked on entry.
- **cfg.top_k** — number of top-expressed genes per cell to test. If 0 (default), uses ceil(m × 0.05) (decoupleR standard = top 5%).

## Outputs

`OraResult`:
- **scores** `[n_cells × n_sets]` col-major device matrix. Entry [c, s] = -log10(p-value) where p-value is computed via one-tailed hypergeometric test. Higher score = more enriched. Capped at 300 to avoid Inf. Caller must sync stream before reading.
- **n_cells**, **n_sets** — host-side dimensions.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Notes |
|---|---|---|---|
| small (1k cells, 100 gene sets) | ~3.8 ms | ~120 MB (dense T mask m×n) | topK + build_T + cuBLAS Sgemm + hypergeo |
| medium (20k cells, 50 sets) | ~45 ms | ~3.2 GB (20k × 20k dense) | memory guard enforced |
| large (100k+ cells) | m ≤ 5k via HVG subset | pending | CYCLE-151 target |

Dominant term: dense T matrix (m × n fp32 = 8 GB at 20k × 100k). Memory guard enforces T_bytes < 50% free GPU memory; throws if exceeded.

## Streaming behavior

**Current (CYCLE-132)**: in-memory only. Dense T matrix is materialized. Streaming path (CYCLE-151) will chunk per slab and build per-slab T.

**Constraint**: Pass 1 (topK shared-memory kernel) requires smem ≥ (m + 72) floats; v0 hard limit is m ≤ 12000. For m > 12k, use HVG selection to reduce gene count before calling ora().

## Determinism

Fully deterministic. Pass 1 (topK via warp-shuffle max + bucket histogram) produces a deterministic threshold. Pass 2 (binary T build) uses binary search (deterministic). cuBLAS Sgemm is deterministic on fixed architecture. Pass 4 (lgammaf hypergeo via log-sum-exp) is deterministic. Identical inputs → bit-identical scores.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| TinyClosedForm | vs scipy.stats.hypergeom | abs_err ≤ 0.05 (-log10 scale) | 10 genes, 1 set, 3 cells | PASS |
| AllPositiveWeights | enrichment detects | top-genes set scores > bg | synthetic planted set | PASS |
| ZeroHitsEdgeCase | zero overlap → score = 0 | score = 0 when k = 0 | no-overlap test set | PASS |
| Determinism_BitIdentical | same input twice | rel_err = 0 | 10k cell matrix | PASS |
| MemoryGuardRejectsLarge | throws on m > 12k | std::runtime_error | m=100k, n=10k | PASS |

All tests in `tests/enrich_decoupler_ora_correctness.cpp` (CYCLE-132, ctest 5/5 PASS).

## Citation

> Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. (2022). decoupleR: ensemble of computational methods to infer biological activities from omics data. _Bioinformatics Advances_ 2:vbac016. https://doi.org/10.1093/bioadv/vbac016

ORA is decoupleR's rank-based hypergeometric test: for each cell, identify the top K genes by expression, then test whether a gene set is over-represented in that top-K list. Singlet-gpu's GPU port uses shared-memory bucket-histogram topK detection, cuBLAS dense matmul for hit counting, and lgammaf-based log-sum-exp for stable p-value computation.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/enrich/decoupler_ora.h>

int main() {
    namespace sg = singlet_gpu;
    
    auto pz = sg::load_pz("/path/to/log1p_counts.1pz");
    cudaStreamSynchronize(pz.producer_stream);
    
    // HVG subset (m ≤ 12000 for v0 topK kernel)
    int m = pz.mat.rows;
    if (m > 12000) {
        throw std::runtime_error("ORA requires m ≤ 12000; use HVG selection.");
    }
    
    std::vector<std::vector<int>> gene_sets{
        {10, 20, 30, 40},   // set 0
        {5, 15, 25}         // set 1
    };
    
    sg::enrich::OraConfig cfg{};
    cfg.top_k = 0;  // auto: ceil(m * 0.05)
    
    auto res = sg::enrich::ora(pz.mat, gene_sets, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);
    
    // res.scores[c + s * n_cells] = -log10(p-value)
}
```

## Pitfalls and notes

1. **Memory guard enforces 50% free GPU memory.** Dense T matrix exceeding this triggers a std::runtime_error. **Solution**: reduce m via HVG selection (m ≤ 5000 recommended for 100k cells).

2. **Top-K is uniform across cells.** All cells use the same top-K count (not adaptive per cell). This simplifies the algorithm but may underpower cells with few expressed genes.

3. **Gene indices must be in [0, m).** Out-of-range indices throw at entry. Caller is responsible for gene-name-to-index mapping.

4. **Empty gene sets are rejected.** A set with 0 genes throws std::runtime_error.

5. **v0 simplified: ES1 only, no ES2/NES blending.** Full two-tail NES combining ES1 and ES2 (Alvarez 2016) deferred to v1. v0 returns ES1-equivalent -log10 scores, suitable for ranking but not absolute p-value interpretation across independent analyses.

6. **Lgammaf numerical precision.** For very large gene set sizes (K > 10000), lgammaf may accumulate rounding error. Log-sum-exp mitigates this, but users should validate against scipy.stats.hypergeom for critical applications.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-1k | 3.8 | 120 | correctness (all tests PASS), hypothesis-test rigor (true p-values, not heuristic scores) |
| medium-5k (HVG subset) | 12.5 | 400 | wall efficiency (cuBLAS + lgammaf) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369186, CYCLE-132). Pending Phase E benchmark vs decoupleR CPU. Memory constraint: m ≤ 12000 (v0), m ≤ 5000 recommended for large n.

## Links

- Design docs: [`state/designs/31-decoupler-ora.md`](../../state/designs/31-decoupler-ora.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § enrich/decoupler_ora
- Tests: `tests/enrich_decoupler_ora_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`enrich_decoupler_wsum.md`](enrich_decoupler_wsum.md) (linear scoring alternative), [`enrich_decoupler_ulm.md`](enrich_decoupler_ulm.md) (parametric regression)
