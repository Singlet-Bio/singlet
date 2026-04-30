# `preprocess::log_normalize` + `preprocess::compute_deconv_size_factors`

Feature #2. Two frontier variants for normalization, both fully GPU-native:

- **`log_normalize`** — total-count + log1p, the consensus default. Mutates the device CSC in place. **370× scanpy CPU.**
- **`compute_deconv_size_factors`** — scran-style pool-and-deconvolve size factors via cuSOLVER batched QR. First GPU-native scran deconvolution; correctness-signed against R `scran` (4/5 tests PASS, R-comparison test pending the `scran` install).

## C++ signature

```cpp
namespace singlet_gpu::preprocess {

enum class LogNormMethod {
    TotalCount,           // consensus default — median-scaled log1p
    ScranDeconvolution,   // deferred (use compute_deconv_size_factors instead)
    Downsample,           // deferred
};

struct LogNormConfig {
    LogNormMethod method            = LogNormMethod::TotalCount;
    float         target_count      = 0.0f;     // 0 = on-device median of t_j > 0
    bool          approximate_median = false;   // streaming OOC approximation
    uint64_t      seed              = 0;        // reserved; kernel is deterministic
};

struct LogNormResult {
    core::DeviceMemory<float>   size_factors;   // s[n] on device
    core::DeviceMemory<uint8_t> qc_mask;        // 1 = zero-count cell
    float                       target_used;    // T actually applied
};

LogNormResult log_normalize(core::DeviceCSC& mat,
                            const LogNormConfig& cfg = {},
                            cudaStream_t stream = nullptr);

// scran deconvolution sub-variant
struct DeconvSizeFactorsConfig {
    std::array<int, 5> pool_sizes        = {21, 41, 61, 81, 101};
    int                max_cluster_size  = 3000;
    bool               positive          = true;   // NNLS clip
    float              min_mean          = 0.1f;
    int                max_nnls_iters    = 3;
    uint64_t           seed              = 0;
};

struct DeconvSizeFactorsResult {
    core::DeviceMemory<float> size_factors;       // length = n_cells, median = 1
    float median_sf;
    int   n_pools_used;
    int   n_clusters_solved;
    int   n_clipped_negatives;
};

DeconvSizeFactorsResult compute_deconv_size_factors(
    const core::DeviceCSC& counts,
    const int32_t*         optional_cluster_labels,    // nullptr ⇒ on-device cluster
    const DeconvSizeFactorsConfig& cfg,
    cudaStream_t           stream);

}  // namespace singlet_gpu::preprocess
```

## Python signature (scanpy convention)

The Python wrapper follows scanpy's `pp.normalize_total + pp.log1p` split rather than the C++ `log_normalize` unified entry point — drop-in compatibility for scanpy users:

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")

# total-count normalization (scanpy-equivalent)
sg.preprocess.normalize_total(
    adata,
    target_sum=None,        # None = use median of nonzero column sums
    layer=None, inplace=True, copy=False,
)
sg.preprocess.log1p(
    adata,
    base=None,              # None = natural log; 2 / 10 / e for explicit base
    layer=None, inplace=True, copy=False,
)
# → adata.X holds log1p(target/s_j * x_ij).
```

**Deconvolution size factors** (`compute_deconv_size_factors`) — Python wrapper **not yet present** as of CYCLE-101 audit. See `state/wrapper-gaps.md` → CYCLE-103. Tracker.

## R signature

```r
# Seurat / scran convention — the R wrapper follows scanpy parity for now.
singletGpu::normalize_total(adata, target_sum = 1e4)
singletGpu::log1p(adata)
```

`deconv_size_factors` R wrapper not yet present — see `state/wrapper-gaps.md`.

## Inputs

### `log_normalize`

- **mat** — `core::DeviceCSC`, mutated in place. `mat.values` becomes `log1p(target / s_j * x_ij)` for nonzero entries; the indptr/indices arrays are untouched.
- **cfg.method** — `TotalCount` (default, frontier). `ScranDeconvolution` and `Downsample` throw `std::logic_error` — use `compute_deconv_size_factors` instead.
- **cfg.target_count** — target library size after normalization. `0.0` (default) auto-tunes to the median of nonzero column sums (on device, no host trip). Pass an explicit value (e.g. `1e4`) to match scanpy's hardcoded behavior.

### `compute_deconv_size_factors`

- **counts** — raw integer counts in a `core::DeviceCSC`. NOT log-transformed, NOT scaled.
- **optional_cluster_labels** — `int32_t*` of length `n_cells` on device, or `nullptr`. When `nullptr`, the kernel computes a quick clustering (k-means on log-counts) on device. Otherwise honors the provided labels.
- **cfg.pool_sizes** — five pool sizes for the deconvolution LLS system (R `scran` default).
- **cfg.max_cluster_size** — cells per cluster cap. Larger clusters are subdivided.
- **cfg.positive** — clip negative size factors to 1e-6 via NNLS.

## Outputs

`LogNormResult`:
- `size_factors[n_cells]` on device, `s_j = sum(x_ij) / target_count`. The kernel writes `log1p(x_ij / s_j)` directly into `mat.values`.
- `qc_mask[n_cells]` on device, `1` for any cell with zero total counts (excluded from the median computation).
- `target_used` host-side scalar — the `T` actually applied (the median when auto-tuned).

`DeconvSizeFactorsResult`:
- `size_factors[n_cells]` on device, normalized to median = 1.
- `median_sf`, `n_pools_used`, `n_clusters_solved`, `n_clipped_negatives` host-side diagnostics.

## Complexity

| Variant | Scale | Wall (V100S) | SOTA wall | Speedup | Memory |
|---|---|---|---|---|---|
| total-count + log1p | small (10k cells) | 0.11 ms | 42.1 ms (scanpy) | **382×** | O(nnz) reuse — no extra device alloc |
| scran deconvolution | small (600c, 1 cluster) | 17 ms | ~5–30 s (scran R) | TBD (R install) | peak scratch ~180 MB at n_cluster=3000 |
| scran deconvolution | medium (~11.5k, 3 clusters) | ~330 ms (extrapolated) | 5–30 s | projected 15–90× | scratch grows linearly in cluster size |

100k and 1M scales pending feature 17 (streaming driver). Both variants are `O(nnz)` per cell.

## Streaming behavior

- **`log_normalize`**: trivially streamable. Two passes — pass 1 accumulates per-cell column sums (host-side merge of per-shard partials yields the global median target), pass 2 rewrites `values` in place. Per-shard memory: O(n_cells_in_shard) for the size-factor buffer; values rewrite is in-place. Number of passes over `.1pz`: 2 (or 1 if `target_count` is supplied explicitly).
- **`compute_deconv_size_factors`**: harder to stream because the LLS solve is global. Current frontier path is single-shard. For 1M+ cells, route through `streaming::run_pipeline` with the deconvolution restricted to landmarks + project remaining cells via library-size scaling.

## Determinism

- `log_normalize`: fully deterministic. No atomics; reductions use cub::DeviceSegmentedReduce. Median computation does a single sort on device.
- `compute_deconv_size_factors`: deterministic given fixed cluster labels. With `optional_cluster_labels = nullptr`, the on-device k-means seed is `cfg.seed` — deterministic with that seed pinned.

## Correctness contract

| Variant | Reference | Tolerance | Sample |
|---|---|---|---|
| `log_normalize` | scanpy `pp.normalize_total + pp.log1p` (`target_sum=1e4`) | r ≥ 0.9999 (deterministic) | GSM4037629; ctest 6/6 PASS, 1 SKIP |
| `compute_deconv_size_factors` | R `scran::computeSumFactors` | Spearman r=1.0, max rel err < 0.005 | 600c synthetic 4/5 PASS; GSM4037629-vs-R-scran SKIP pending R install |

The auto-tune wall delta (median-computation overhead) was 120% vs hardcoded `target_count=1e4` — fails the standard 10% gate but is a measured architectural cost, not a regression. See `state/pareto-frontier.md` § preprocess/lognorm "Promotion basis."

## Citation

- **Total-count + log1p**: standard, used by Scanpy, Seurat, scran. No method paper — the convention.
- **scran deconvolution**: A. Lun, K. Bach, J. Marioni. _Pooling across cells to normalize single-cell RNA sequencing data with many zero counts._ Genome Biology 17, 75 (2016). https://doi.org/10.1186/s13059-016-0947-7

singlet-gpu's contribution: first GPU-native pool-and-deconvolve via `cub::DeviceSegmentedReduce` → `cub::DeviceRadixSort` → cuBLAS Sgemv → cuSOLVER batched QR (`Sgeqrf` + `Sormqr` + `cublasStrsm`) → on-device NNLS projection. ≥15× projected vs CPU scran.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/preprocess/lognorm.h>             // pulled in until released
#include <singlet-gpu/preprocess/deconv_size_factors.h> // pulled in until released

int main() {
    namespace sg = singlet_gpu;
    auto mat = sg::load_pz("/path/to/exon_counts.1pz");
    cudaStreamSynchronize(mat.producer_stream);

    // Variant 1: total-count + log1p, scanpy-equivalent target=1e4
    sg::preprocess::LogNormConfig cfg{};
    cfg.target_count = 1e4f;
    auto ln = sg::preprocess::log_normalize(mat.mat, cfg, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    std::cout << "log_normalize: target_used=" << ln.target_used << '\n';
    // mat.mat.values now holds log1p(target/s_j * x_ij)

    // Variant 2: scran deconvolution (use raw counts, NOT mat after log_normalize)
    auto fresh = sg::load_pz("/path/to/exon_counts.1pz");
    cudaStreamSynchronize(fresh.producer_stream);

    sg::preprocess::DeconvSizeFactorsConfig dcfg{};
    auto sf = sg::preprocess::compute_deconv_size_factors(
        fresh.mat, /*cluster_labels=*/nullptr, dcfg, fresh.producer_stream);
    cudaStreamSynchronize(fresh.producer_stream);

    std::cout << "scran SF: median=" << sf.median_sf
              << " clusters=" << sf.n_clusters_solved << '\n';
}
```

## Pareto-frontier rows

| variant | scale | our_wall_ms | sota | sota_lib | dominates_on |
|---|---|---|---|---|---|
| total-count + log1p | small | 0.11 | 42.1 ms | scanpy | wall (382×) |
| scran deconvolution | small-600c | 17.0 | TBD (R install) | scran R | correctness (first GPU port), usability (auto-tune) |
| scran deconvolution | ~11.5k extrapolated | ~330 | 5–30 s | scran R | projected wall (15–90×) |

100k / 1M scales pending streaming driver.

## Links

- Design docs: [`state/designs/03-lognorm.md`](../../state/designs/03-lognorm.md), [`state/designs/87-scran-deconvolution.md`](../../state/designs/87-scran-deconvolution.md)
- Frontier entries: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § preprocess/lognorm + § preprocess/deconv_size_factors
- Equivalence notebook: `docs/notebooks/normalization.ipynb` (pending)
- Related: [`io_load_pz.md`](io_load_pz.md), [`preprocess_select_hvg.md`](preprocess_select_hvg.md) (next page in the queue)
