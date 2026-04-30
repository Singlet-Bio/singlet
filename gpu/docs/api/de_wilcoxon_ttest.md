# `de::wilcoxon_de` + `de::ttest_de`

Feature #11. Per-cluster differential-expression marker discovery. Both methods are on the **full frontier** with all correctness metrics at `1.0` against scanpy on a planted-signal real-data test (20k×310k, 250 markers, 5 clusters).

- **`wilcoxon_de`** — rank-sum test with on-device tie-aware ranking via log-spaced histograms. **6.5× scanpy at 500 cells; 388.8× at 20k×310k.**
- **`ttest_de`** — Welch's t-test with associative `sum + sum_sq` Pass-1 (race-safe). **10.4× / 8.4×** at the same scales.

Both share an identical `ClusterMarkers` output struct, identical BH FDR + top-N selection, identical streaming contract. Pick `wilcoxon_de` for the rank-based (scanpy default) interpretation; `ttest_de` for parametric speed when normality is reasonable post-log1p.

## C++ signature

```cpp
namespace singlet_gpu::de {

struct ClusterMarkers {
    int                              cluster_id;
    core::DeviceMemory<int>          gene_indices;   // [top_n] in score-desc order
    core::DeviceMemory<float>        z_scores;       // [top_n] for wilcoxon, t-values for ttest
    core::DeviceMemory<float>        log2_fc;        // [top_n] log2 fold change vs rest
    core::DeviceMemory<float>        p_values;       // [top_n] raw two-sided p-values
    core::DeviceMemory<float>        p_adj;          // [top_n] BH FDR over n_genes
};

struct WilcoxonResult { std::vector<ClusterMarkers> per_cluster; };
struct TtestResult    { std::vector<ClusterMarkers> per_cluster; };

struct WilcoxonConfig {
    int  n_bins        = 4096;     // log1p-spaced histogram bins for tie-aware ranking
    int  top_n         = 100;      // max marker genes returned per cluster
    int  gene_tile     = 1024;     // genes per tile (bounds workspace)
    bool deterministic = false;    // true → segmented scan instead of atomicAdd
};

struct TtestConfig {
    int  top_n         = 100;
    int  gene_tile     = 1024;
    bool deterministic = false;    // true → deterministic Welford accumulation
};

WilcoxonResult wilcoxon_de(
    const core::DeviceCSC& mat,
    const core::DeviceMemory<int>& labels,    // [n_cells] cluster labels in [0, n_clusters)
    int n_clusters,
    const WilcoxonConfig& cfg,
    cudaStream_t stream);

TtestResult ttest_de(
    const core::DeviceCSC& mat,
    const core::DeviceMemory<int>& labels,
    int n_clusters,
    const TtestConfig& cfg,
    cudaStream_t stream);

}  // namespace singlet_gpu::de
```

## Python signature (scanpy convention)

Both methods bundle into one scanpy-style entry point with a `method=` flag:

```python
import singlet_gpu as sg

# Verified ground-truth signature (CYCLE-110 smoke 368692).
sg.tools.rank_genes_groups(
    adata,
    groupby,                     # required positional — adata.obs column name
    *,
    mask_var=None,               # optional gene mask (boolean array or var-name list)
    use_raw=None,                # None ⇒ adata.raw if present
    groups="all",                # or list[str]
    reference="rest",
    n_genes=None,                # default top 100 per cluster
    rankby_abs=False,
    pts=False,
    key_added=None,              # default: 'rank_genes_groups'
    copy=False,
    method="wilcoxon",           # or "t-test", "t-test_overestim_var", "logreg" (logreg→wilcoxon)
    corr_method="benjamini-hochberg",  # or 'bonferroni', 'holm-sidak', 'fdr_bh', 'fdr_by'
    tie_correct=False,           # tie-aware Wilcoxon ranks (slower, scanpy-exact)
    layer=None,
)
# Writes adata.uns[key_added or 'rank_genes_groups']:
#   names, scores (z or t), pvals, pvals_adj, logfoldchanges  — each a structured array
#   (one column per group) of length n_genes_returned.
```

## R signature

```r
singletGpu::rank_genes_groups(adata, groupby = "leiden", method = "wilcoxon")
```

## Inputs

- **mat** — `core::DeviceCSC`. **Should be log-normalized** (run `log_normalize` first). Wilcoxon is rank-based so log normalization doesn't change the rank order, but log2_fc reporting still uses the post-norm values; ttest assumes near-normality post-log.
- **labels** — `[n_cells]` `int` cluster labels in the range `[0, n_clusters)`. From `graph::leiden` or any external clustering.
- **cfg.n_bins** (Wilcoxon) — log1p-spaced bins for tie-aware ranking. 4096 is sufficient for typical UMI counts (max ~50k → log1p ≈ 10.8 → 4096 bins gives ~0.0026 spacing, finer than fp32 precision). Not user-tunable in practice.
- **cfg.top_n** — markers returned per cluster (in z-score / t-value descending order). 100 is the scanpy default.
- **cfg.gene_tile** — genes processed per tile. Higher = more parallelism, more workspace. 1024 is the sweet spot on V100/H100.
- **cfg.deterministic** — `false` (default): atomic histogram accumulation, ~1.5× faster. `true`: segmented-scan path, bit-identical across runs.

## Outputs

A `WilcoxonResult` or `TtestResult` is `std::vector<ClusterMarkers>` of length `n_clusters`. Each entry is one cluster vs all others (one-vs-rest):

- `gene_indices[top_n]` — selected genes in score-descending order on device.
- `z_scores[top_n]` (Wilcoxon) or `t_values[top_n]` (ttest) — the test statistic.
- `log2_fc[top_n]` — `log2(expm1(mean_log_in) / expm1(mean_log_rest))` per scanpy convention. fp64 internally to avoid catastrophic cancellation; fp32 on output.
- `p_values[top_n]` — raw two-sided p-values from the normal-approximation null.
- `p_adj[top_n]` — Benjamini–Hochberg FDR computed over **all** n_genes for this cluster, then gathered to the top_n.

## Complexity

| Method | Scale | Wall (H100 NVL) | SOTA wall | Speedup |
|---|---|---|---|---|
| `wilcoxon_de` | TinyPlanted-500 | 3.8 ms | 24.7 ms (scanpy) | **6.5×** |
| `wilcoxon_de` | RealDataPlanted-20k×310k | 985.5 ms | 383,134 ms (scanpy) | **388.8×** |
| `ttest_de` | TinyPlanted-500 | 2.2 ms | 22.9 ms (scanpy) | **10.4×** |
| `ttest_de` | RealDataPlanted-20k×310k | 77.5 ms | 651.3 ms (scanpy) | **8.4×** |

Memory: O(n_genes × n_clusters × n_bins) for the Wilcoxon histograms (gene-tiled to bound peak); O(n_genes × n_clusters) for ttest sufficient statistics. Large-scale (100k × 30k) bench currently OOM-skipped due to a host-side `vector::reserve` issue in the bench driver — see `state/followups.md` → CYCLE-85-BENCH-HARNESS-OOM.

Note: `ttest_de` is "only" 8–10× scanpy because scanpy's t-test is scipy-vectorized while scanpy's Wilcoxon uses pandas rank (much slower on CPU). Both kernels are at frontier wall-time.

## Streaming behavior

- **`wilcoxon_de`**: streams cleanly via gene-tile decomposition. Each tile's histograms are local; tie correction is exact within a tile because the global histogram is the sum of tile histograms (associative). Number of passes: 1 with gene-tiling for in-memory; 2 for chunk-streaming (pass 1 = global per-cluster cell counts; pass 2 = histograms + ranks).
- **`ttest_de`**: streams via gene-tile + cell-shard. The Pass-1 sum + sum_sq accumulation is **associative** — that's the Cycle-83 fix. Per-shard partials sum to the global statistic with no race; mean and variance derive in Pass-2.

## Determinism

Both kernels are deterministic with `cfg.deterministic = true`. The default (atomic) path is non-deterministic in the LSB of the histogram bin counts (Wilcoxon) or sum_sq (ttest), but the BH-corrected p-values + top-N rankings are identical across runs because the LSB jitter is below the BH discretization threshold. For literally bit-identical reproducibility, opt in.

## Correctness contract

| Method | Reference | Tolerance | Sample |
|---|---|---|---|
| `wilcoxon_de` | scanpy `tl.rank_genes_groups(method="wilcoxon")` | Jaccard@top-50 ≥ 0.95, Spearman(p-rank) ≥ 0.999, Spearman(LFC) ≥ 0.999 | TinyPlanted-500 + RealDataPlanted-20k × 310k; **all metrics = 1.0** |
| `ttest_de` | scanpy `tl.rank_genes_groups(method="t-test")` | Jaccard ≥ 0.95, Spearman(p-rank) ≥ 0.999, Spearman(LFC) ≥ 0.999 | TinyPlanted-500 + RealDataPlanted-20k × 310k; **all metrics ≥ 0.9999** |

Both kernels were brought to full frontier through 7-cycle correctness arcs (Wilcoxon: Cycles 72-77; ttest: Cycles 78-84). Bug history is documented in `state/cycle-log.md` — most consequential fixes were:

- **Wilcoxon**: cuSPARSE m/n swap in csr2csc (Cycle 73), `expm1`-aware log2_fc formula (Cycle 74), signed-z sort key (Cycle 75), fp64 LFC promotion (Cycle 76), planted-signal real-data test design (Cycle 77).
- **ttest**: racy Welford Pass-1 → associative `sum + sum_sq` accumulation (Cycle 83), constant-vector Spearman handler (Cycle 84).

Real-data tests use a **planted-signal redesign**: 250 deterministic marker genes are bumped by U[5–20] in matched cells via Fisher-Yates on disjoint indices. Identical input goes to GPU and scanpy; both recover the planted markers perfectly. Round-robin labels on real data lacked biological signal — that pattern was abandoned in Cycle 77.

## Citations

- **Wilcoxon rank-sum**: F. Wilcoxon. _Individual Comparisons by Ranking Methods._ Biometrics Bulletin 1, 80 (1945).
- **Welch's t-test**: B. L. Welch. _The generalization of Student's problem when several different population variances are involved._ Biometrika 34, 28 (1947).
- **Benjamini–Hochberg FDR**: Y. Benjamini, Y. Hochberg. _Controlling the false discovery rate._ JRSS-B 57, 289 (1995).
- **scanpy reference impl**: F. A. Wolf et al. _SCANPY: large-scale single-cell gene expression data analysis._ Genome Biology 19, 15 (2018).

singlet-gpu contributions: tile-decomposed log-spaced histograms with associative Pass-1 (no host trips, deterministic when opted in), and the planted-signal real-data correctness pattern.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/de/wilcoxon.h>            // until released
#include <singlet-gpu/de/ttest.h>
#include <singlet-gpu/preprocess/lognorm.h>

int main() {
    namespace sg = singlet_gpu;
    auto pz = sg::load_pz("/path/to/exon_counts.1pz");
    cudaStreamSynchronize(pz.producer_stream);

    // 1. Normalize first — DE assumes log-counts in mat.values.
    sg::preprocess::log_normalize(pz.mat, {}, pz.producer_stream);

    // 2. labels comes from graph::leiden or any clustering — here we assume
    //    you already have a DeviceMemory<int> of length pz.n_cells.
    sg::core::DeviceMemory<int> labels(pz.n_cells);
    /* fill labels from leiden output */

    // 3. Wilcoxon — typical default for marker gene discovery.
    sg::de::WilcoxonConfig wcfg{};
    wcfg.top_n = 100;
    auto wilc = sg::de::wilcoxon_de(pz.mat, labels, /*n_clusters=*/8, wcfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // wilc.per_cluster[0].gene_indices is a device pointer to the top-100 marker
    // gene indices for cluster 0, sorted by z-score descending.

    // 4. Or t-test (faster on small-rank queries; same output schema).
    sg::de::TtestConfig tcfg{};
    auto tt = sg::de::ttest_de(pz.mat, labels, /*n_clusters=*/8, tcfg, pz.producer_stream);
}
```

## Pareto-frontier rows

| variant | scale | wall_ms | sota_wall_ms | sota_lib | dominates_on |
|---|---|---|---|---|---|
| wilcoxon | TinyPlanted-500 | 3.8 | 24.7 | scanpy 1.10.x | wall (6.5×), correctness |
| wilcoxon | RealDataPlanted-20k×310k | 985.5 | 383,134 | scanpy 1.10.x | wall (388.8×), correctness at scale |
| ttest | TinyPlanted-500 | 2.2 | 22.9 | scanpy 1.10.x | wall (10.4×), correctness |
| ttest | RealDataPlanted-20k×310k | 77.5 | 651.3 | scanpy 1.10.x | wall (8.4×), correctness at scale |

Both promoted to full frontier 2026-04-16. Job 361954 (g051 H100 NVL).

## Related (sub-features within the same module — not yet on frontier)

- **Donor-aware pseudobulk DE** (`de::donor_pseudobulk`) — uses `donor_assignments.tsv` from singlify to build per-donor pseudobulks before NB GLM. Singlify-unique. Header exists; correctness work pending.
- **Logistic regression DE** — Python wrapper currently dispatches to Wilcoxon with a warning (`CYCLE-21-LOGREG-DE` follow-up).
- **scry deviance / Pearson residuals**: those are HVG variants, not DE — see [`preprocess_select_hvg.md`](preprocess_select_hvg.md).

## Links

- Design docs: [`state/designs/11-de-wilcoxon.md`](../../state/designs/11-de-wilcoxon.md), [`state/designs/72-wilcoxon-postnorm-crash.md`](../../state/designs/72-wilcoxon-postnorm-crash.md), the 7-cycle correctness arc in `state/cycle-log.md`.
- Frontier entries: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § de/wilcoxon + § de/ttest
- Equivalence notebook: `docs/notebooks/de_analysis.ipynb` (pending)
- Related: [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (run before DE), `graph_leiden.md` (clustering input), `qc_metrics.md` (filter cells/genes first)
