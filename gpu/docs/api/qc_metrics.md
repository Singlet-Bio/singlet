# `qc::calculate_qc_metrics` + `qc::filter_cells` + `qc::filter_genes` + `qc::doublet_score`

Feature #6. Per-cell + per-gene QC, threshold-based filtering, and Scrublet-equivalent doublet detection. **429× scanpy CPU at small scale; 74M cells/sec sustained throughput at medium scale.**

The four functions form a complete cell-quality workflow: compute metrics → drop bad cells/genes → score doublets on the PCA embedding of what remains.

## C++ signature

```cpp
namespace singlet_gpu::qc {

struct QcResult {
    core::DeviceMemory<float> n_umis;        // [n_cells]   — total UMI per cell
    core::DeviceMemory<int>   n_genes;       // [n_cells]   — nnz per cell
    core::DeviceMemory<float> pct_mt;        // [n_cells]   — % mitochondrial UMIs
    core::DeviceMemory<float> pct_ribo;      // [n_cells]   — % ribosomal UMIs
    core::DeviceMemory<float> gene_mean;     // [n_genes]   — mean across cells
    core::DeviceMemory<float> gene_var;      // [n_genes]   — variance (N-1 denom)
    core::DeviceMemory<int>   gene_n_cells;  // [n_genes]   — #cells expressing gene
    int n_cells = 0;
    int n_genes_total = 0;
};

struct FilterConfig {
    float min_genes        = 200.0f;
    float max_genes        = std::numeric_limits<float>::infinity();
    float min_umis         = 0.0f;
    float max_umis         = std::numeric_limits<float>::infinity();
    float max_pct_mt       = 100.0f;
    int   min_cells_per_gene = 1;
};

struct QcConfig {
    bool deterministic = false;   // true = two-pass cub path (~1.5× slower, bit-identical)
};

// QC stats — both per-cell and per-gene in one pass.
QcResult calculate_qc_metrics(
    const core::DeviceCSC& mat,
    const core::DeviceMemory<uint8_t>& is_mt,    // [n_genes] gene mask
    const core::DeviceMemory<uint8_t>& is_ribo,  // [n_genes] gene mask
    cudaStream_t stream = nullptr,
    QcConfig cfg = {});

// Cell filter — returns a new compacted DeviceCSC.
core::DeviceCSC filter_cells(
    const core::DeviceCSC& mat,
    const QcResult& qc,
    const FilterConfig& cfg,
    cudaStream_t stream = nullptr);

// Gene filter — returns a new DeviceCSC with rows relabeled 0..n_genes_kept-1.
core::DeviceCSC filter_genes(
    const core::DeviceCSC& mat,
    const QcResult& qc,
    const FilterConfig& cfg,
    cudaStream_t stream = nullptr);

// Doublet detection (Scrublet-equivalent).
struct DoubletScoreConfig {
    float    n_synth_frac    = 0.25f;   // N_synth = ceil(n_synth_frac * n_real)
    int      k               = 20;      // kNN neighbours in combined cloud
    float    manual_threshold = 0.0f;   // 0 = auto-threshold via knee detection
    uint64_t seed            = 0;       // Philox seed
};

struct DoubletScoreResult {
    core::DeviceMemory<float>   score;          // [n_real] in [0, 1]
    core::DeviceMemory<uint8_t> doublet_call;   // [n_real] 0/1
    float threshold_used;
    int   n_predicted_doublets;
};

// Operates on a PCA embedding (DeviceDense, n_real × n_pcs).
// Pipeline: synthetic-doublet generation (avg of 2 random cells) → kNN on
// real ∪ synth → fraction-of-synth-neighbours score → knee-detected threshold.
DoubletScoreResult doublet_score(
    const core::DeviceDense& embedding,
    const DoubletScoreConfig& cfg,
    cudaStream_t stream);

}  // namespace singlet_gpu::qc
```

## Python signature (scanpy convention)

All four functions ship in `pip install singlet-gpu` (verified end-to-end via CYCLE-110 smoke 368692). Signatures below are the **verified ground-truth** as of v0.1.0.

```python
import singlet_gpu as sg

# Per-cell + per-gene QC stats.
# qc_vars: tuple of var-name prefixes to flag. The kernel scans
# adata.var.index and builds is_mt/is_ribo masks on device.
sg.qc.calculate_qc_metrics(
    adata,
    qc_vars=("MT", "RIBO"),
    *,
    layer=None, inplace=True, copy=False,
    deterministic=False,             # True = two-pass cub path, bit-identical (~1.5x slower)
    stream=None,
)

# Cell filter — scanpy-compatible thresholds + max_pct_mt convenience.
sg.qc.filter_cells(
    adata,
    *,
    min_genes=None, max_genes=None,
    min_counts=None, max_counts=None,
    max_pct_mt=None,                 # None = no MT filter
    qc_vars=("MT", "RIBO"),          # forwarded if QC stats not yet computed
    layer=None, inplace=True, copy=False, stream=None,
)

# Gene filter.
sg.qc.filter_genes(
    adata,
    *,
    min_cells=None, min_counts=None,
    qc_vars=("MT", "RIBO"),
    layer=None, inplace=True, copy=False, stream=None,
)

# Scrublet-equivalent doublet detection on a PCA embedding.
sg.qc.run_doublet_score(
    adata,
    *,
    embedding_key="X_pca",           # AnnData obsm key
    n_synth_frac=0.25, k=20,
    manual_threshold=0.0,            # 0 = knee-detect
    obs_score_key="doublet_score",
    obs_call_key="doublet_call",
    stream=None, seed=0, copy=False,
)
```

Outputs land at scanpy-conventional locations: `adata.obs['n_umis']`, `adata.obs['n_genes']`, `adata.obs['pct_mt']`, `adata.obs['pct_ribo']`, `adata.var['gene_mean']`, `adata.var['gene_var']`, `adata.var['gene_n_cells']`. The doublet path writes `adata.obs['doublet_score']` ∈ [0, 1] and `adata.obs['doublet_call']` ∈ {0, 1}.

## R signature

```r
singletGpu::calculate_qc_metrics(adata, qc_vars = c("MT", "RIBO"))
singletGpu::filter_cells(adata, min_genes = 200L, max_pct_mt = 20.0)
singletGpu::filter_genes(adata, min_cells = 3L)
singletGpu::run_doublet_score(adata, embedding_key = "X_pca", k = 20L, seed = 42L)
```

## Inputs

### `calculate_qc_metrics`

- **mat** — `core::DeviceCSC` with raw integer counts. Not log-transformed, not scaled.
- **is_mt** — `[n_genes]` `uint8_t` mask, `1` for mitochondrial genes (e.g. genes whose name starts with `"MT-"` or `"mt-"`). Build host-side from `mat.meta.rownames`.
- **is_ribo** — `[n_genes]` mask for ribosomal genes (`"RPS"`, `"RPL"` prefixes).
- **cfg.deterministic** — `false` (default): atomicAdd-based gene scatter, ~1.5× faster but the gene_var lower-order bits depend on warp scheduling. `true`: two-pass cub path, bit-identical across runs.

### `filter_cells` / `filter_genes`

- Both consume a `QcResult` from a prior `calculate_qc_metrics` call. Cheap — no recomputation.
- Cell filter compacts CSC columns; gene filter compacts rows + relabels `0..n_genes_kept-1` with a single O(nnz) device scan.
- The host-side indptr rebuild (`n_cells+1` integers, ~4 KB at 1M cells) is a one-time D2H, not in any inner loop — Rule 9 compliant.

### `doublet_score`

- **embedding** — `core::DeviceDense` of shape `(n_cells, n_pcs)`. Typically the PCA embedding `V * d` from `reduce::svd`. Run AFTER `filter_cells` so dead cells don't poison the synthetic pool.
- **cfg.k** — kNN neighbourhood for scoring. 20 is the Scrublet default.
- **cfg.n_synth_frac** — synthetic doublet pool size. 0.25 → for every 4 real cells, one synthetic doublet (mean of 2 random reals).
- **cfg.manual_threshold** — `0.0` (default) triggers knee detection on the score histogram. Pass an explicit value to skip.
- **cfg.seed** — Philox 4x32 seed for the synthetic generator. Bit-deterministic given the same seed.

## Outputs

`QcResult`:
- Four `[n_cells]` device arrays for per-cell stats.
- Three `[n_genes]` device arrays for per-gene stats.
- `n_cells` and `n_genes_total` host-side scalars for downstream sizing.

`DoubletScoreResult`:
- `score[n_cells]` ∈ [0, 1]. Higher = more doublet-like.
- `doublet_call[n_cells]` ∈ {0, 1}.
- `threshold_used` — knee-detected or manual.
- `n_predicted_doublets` — count of `doublet_call == 1`.

## Complexity

| Function | Scale | Wall (H100 NVL) | SOTA wall | Speedup |
|---|---|---|---|---|
| `calculate_qc_metrics` | small (1k cells) | 0.082 ms | 35.3 ms (scanpy) | **429×** |
| `calculate_qc_metrics` | medium (20.8k cells) | 0.281 ms — **74M cells/sec** | scanpy ref unavailable | wall dominates |
| `filter_cells` + `filter_genes` | small | < 0.1 ms each | TBD | TBD |
| `doublet_score` | small | TBD (Phase E pending) | Scrublet ~10–60 s | TBD |

100k / 1M scales pending feature 17 (streaming driver). All functions are `O(nnz)` per pass.

## Streaming behavior

- **`calculate_qc_metrics`**: trivially streamable — per-cell stats are local to a column slab; per-gene stats merge across shards via `cub::DeviceReduce::Sum`. Two passes for the deterministic variance path; one for the atomic path.
- **`filter_cells`**: per-shard. Mask is built on the per-shard `QcResult` and the resulting compacted columns are concatenated into the next shard.
- **`filter_genes`**: requires global gene stats (n_cells per gene), so the streaming driver does pass 1 = stats, pass 2 = filter+relabel.
- **`doublet_score`**: requires global PCA embedding for the kNN — landmark approach (subsample `n_landmarks`, score real cells against landmarks ∪ synth) is the streaming path.

## Determinism

- `calculate_qc_metrics`: deterministic only with `cfg.deterministic = true` — the atomic gene scatter has warp-order dependent ordering for the variance LSB. Set the flag if you need bit-identical results across runs.
- `filter_cells` / `filter_genes`: deterministic.
- `doublet_score`: deterministic given `cfg.seed`. Uses Philox 4x32 (counter+key construction documented in the kernel comments).

## Correctness contract

| Function | Reference | Tolerance | Sample |
|---|---|---|---|
| `calculate_qc_metrics` | scanpy `pp.calculate_qc_metrics` | bit-exact integer counts; gene_var rel err ≤ 1e-5 | GSM4037629; ctest 6/6 PASS |
| `filter_cells` / `filter_genes` | scanpy `pp.filter_cells` / `pp.filter_genes` | identical kept-cell / kept-gene index sets | GSM4037629; ctest PASS |
| `doublet_score` | Scrublet (Wolock et al.) | call-Jaccard ≥ 0.95 vs Scrublet | Phase E pending bench-venv with Scrublet installed |

## Citations

- **QC metrics** — standard fields used across Scanpy, Seurat, scran. No method paper.
- **Scrublet** — S. L. Wolock, R. Lopez, A. M. Klein. _Scrublet: Computational Identification of Cell Doublets in Single-Cell Transcriptomic Data._ Cell Systems 8 (2019).
- **Philox 4x32** — J. K. Salmon et al. _Parallel random numbers: as easy as 1, 2, 3._ SC '11.
- singlet-gpu's contribution: a single-pass per-cell + per-gene QC kernel (no separate `pp.filter_genes_dispersion`) and a Scrublet-equivalent doublet path that runs entirely on the existing PCA embedding without dispatching to Python.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/qc/metrics.h>          // until released
#include <singlet-gpu/qc/doublet_score.h>
#include <singlet-gpu/preprocess/lognorm.h>
#include <singlet-gpu/reduce/svd/auto_select.h>

int main() {
    namespace sg = singlet_gpu;
    auto pz = sg::load_pz("/path/to/exon_counts.1pz",
                          /*stream=*/nullptr, /*keep_host_pinned=*/true);
    cudaStreamSynchronize(pz.producer_stream);

    // 1. Build gene masks from rownames (host-side, once, ~ms for 30k genes)
    sg::core::DeviceMemory<uint8_t> is_mt(pz.mat.rows);
    sg::core::DeviceMemory<uint8_t> is_ribo(pz.mat.rows);
    /* fill is_mt[i] = name.starts_with("MT-") ? 1 : 0; cudaMemcpy to device */

    // 2. QC pass — per-cell + per-gene in one shot
    auto qc = sg::qc::calculate_qc_metrics(pz.mat, is_mt, is_ribo, pz.producer_stream);

    // 3. Filter
    sg::qc::FilterConfig fcfg{};
    fcfg.min_genes  = 200.0f;
    fcfg.max_pct_mt = 20.0f;
    auto filt = sg::qc::filter_cells(pz.mat, qc, fcfg, pz.producer_stream);
    fcfg.min_cells_per_gene = 3;
    filt = sg::qc::filter_genes(filt, qc, fcfg, pz.producer_stream);

    // 4. Normalize → PCA → doublet score on the PCA embedding
    sg::preprocess::log_normalize(filt, {}, pz.producer_stream);
    /* ... HVG, then PCA into a DenseMatrix `pca` ... */

    sg::qc::DoubletScoreConfig dcfg{};
    dcfg.k    = 20;
    dcfg.seed = 42;
    /* auto dbl = sg::qc::doublet_score(pca, dcfg, pz.producer_stream); */
}
```

## Pareto-frontier rows

| scale | wall_ms | accuracy | sota_wall_ms | sota_lib | dominates_on |
|---|---|---|---|---|---|
| small-1k | 0.082 | bit-exact, ctest 6/6 | 35.3 | scanpy | wall (429×) |
| medium-GSM4037629-20.8k | 0.281 — 74M cells/sec throughput | bit-exact | TBD (scanpy ref unavailable) | — | wall dominates |

Promoted 2026-04-18. Bench job 363184 on g051 H100 NVL.

## Links

- Design docs: [`state/designs/06-qc-metrics.md`](../../state/designs/06-qc-metrics.md), [`state/designs/31-doublet-detection.md`](../../state/designs/31-doublet-detection.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § qc/metrics
- Equivalence notebook: `docs/notebooks/qc_metrics.ipynb` (pending)
- Related: [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (run after filter_cells), [`reduce_svd.md`](reduce_svd.md) (PCA feeds doublet_score)
