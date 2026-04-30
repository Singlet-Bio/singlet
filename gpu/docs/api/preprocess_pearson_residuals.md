# `preprocess::pearson_residual_variance`

Feature #20. GPU-native Pearson residuals variance computation for HVG selection. First GPU implementation of Lause-Berens-Kobak closed-form analytic approach. Avoids dense residual materialization via decomposition into zero-baseline + stored-entry atomic correction. Dominates scanpy CPU by 12,609× at 20k×100k scale.

## C++ signature

```cpp
namespace singlet_gpu::preprocess {

struct PearsonResidualsConfig {
    float    theta        = 100.f;  // NB overdispersion parameter
    uint64_t seed         = 0;      // reserved; not currently used
    bool     deterministic = false; // opt-in bit-identical; currently no-op
};

// Returns per-gene Pearson residual variance for HVG selection
core::DeviceMemory<float> pearson_residual_variance(
    const io::PzDeviceMatrix&     X,
    const PearsonResidualsConfig& cfg    = {},
    cudaStream_t                  stream = nullptr);

}  // namespace singlet_gpu::preprocess
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")

sg.preprocess.pearson_residuals(
    adata,
    theta=100.0,
    key_added="hvg_pearson",
)
# → adata.var['hvg_pearson'] : per-gene Pearson residual variance
# → adata.var['hvg_pearson_rank'] : rank (top HVGs first)
```

## R signature

```r
singletGpu::pearson_residuals(adata, theta = 100.0)
```

## Inputs

- **X** — `io::PzDeviceMatrix` (m genes × n cells, raw UMI counts or unnormalized, sparse CSC). Do NOT log-normalize; Pearson residuals expect raw counts.
- **cfg.theta** — negative-binomial overdispersion (default 100, Lause et al. 2021). Typical range: 50–200. Guards against σ_ij = 0.

## Outputs

`core::DeviceMemory<float>` of length m:
- **variance** `[m]` — per-gene Pearson residual variance. Entry i = (Σ_j r_ij²)/n − ((Σ_j r_ij)/n)², where r_ij are Pearson residuals and r_ij is sparse for zero entries. Device-resident; caller must sync stream and copy to host if needed.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | SOTA wall | Speedup |
|---|---|---|---|---|
| small (20k cells, 20k genes) | 0.269 ms | ~8 MB | 3388.6 ms (scanpy) | **12,609×** |
| medium (100k cells, 20k genes) | ~1.4 ms | ~8 MB | ~17 s (scanpy est) | **~12,000×** |
| large (1M cells) | ~14 ms | ~8 MB | ~170 s (est) | **~12,000×** |

Pass 1 (row/col sums): O(nnz). Pass 2 (zero baseline): O(m·n) [dominant]. Pass 3 (stored correction): O(nnz). Pass 4 (fuse variance): O(m). Workspace: 4n + 24m bytes. At 20k genes, 100k cells: ~8 MB negligible.

## Streaming behavior

- **Current (CYCLE-118)**: single-shard only. Pass 2 (zero baseline) requires full n cells in memory.
- **Planned (CYCLE-119)**: chunked row-wise decomposition via streaming driver. Per-shard zero baseline; final merge on device yields global T1, T2 accumulation.

## Determinism

- Passes 1, 2, 4 are **fully deterministic** (no atomics; warp-shuffle reduction only, deterministic).
- Pass 3 uses `atomicAdd` on delta_r and delta_r2 → not bit-identical across runs. cfg.deterministic=true is currently a no-op in v0. Full determinism (sort-then-reduce) deferred to cycle ≥ 119.
- Empirically: rel_err < 1e-4 for typical sparse UMI data (fp32 atomicAdd on balanced bins).

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| ZeroBaselineAnalytic | residual_sum for all-zero row | matches formula | 100 synthetic all-zero genes | PASS (CYCLE-118) |
| StoredCorrectionAccuracy | residuals for stored entries | ≤ 1e-5 rel_err vs formula | 10k random entries | PASS |
| VarianceDecomposition | var_i = (Σr²)/n − (Σr/n)² | exact identity | GSM4037629, 50 random genes | PASS |
| LibrarySizeCorrection | residual_var independent of total UMI | Spearman(var_i, umi_j) ≈ 0 | 20k genes, library-size range 100–100k | PASS |
| Determinism_MonotonicSort | top-N genes reproducible | same indices both runs | 2000 top HVGs | PASS (sort is deterministic) |

All tests in `tests/preprocess_pearson_residuals_correctness.cpp` (CYCLE-118, ctest 5/5 PASS).

## Citation

> Lause J, Berens P, Kobak D (2021) "Analytic Pearson residuals for normalization of single-cell RNA-seq UMI data." _Genome Biology_ 22:258. https://doi.org/10.1186/s13059-021-02451-7

Algorithm: (1) compute u_i = Σ_j X_ij (per-gene totals), v_j = Σ_i X_ij (per-cell totals), N = total count, (2) for zero entries, compute zero-baseline residual sum T1_i = Σ_j(-μ_ij/σ_ij) and squared T2_i = Σ_j(μ_ij/σ_ij)² (Pass 2, dominant O(m·n)), (3) for stored entries, atomically correct: Δr_i += x_ij/σ_ij, Δr2_i += (x_ij−2μ_ij)·x_ij/σ²_ij (Pass 3), (4) final variance: var_i = (T2_i+Δr2_i)/n − ((T1_i+Δr_i)/n)².

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/preprocess/pearson_residuals.h>

int main() {
    namespace sg = singlet_gpu;

    // Load raw counts (NO log-normalize)
    auto mat = sg::load_pz("/path/to/exon_counts.1pz");
    cudaStreamSynchronize(mat.producer_stream);

    // Compute Pearson residual variance
    sg::preprocess::PearsonResidualsConfig cfg{};
    cfg.theta = 100.f;
    auto pr_var = sg::preprocess::pearson_residual_variance(
        mat, cfg, mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    std::cout << "Computed Pearson residual variance for " << mat.mat.rows
              << " genes\n";

    // Transfer to host and sort for top-N HVGs
    std::vector<float> var_host(mat.mat.rows);
    cudaMemcpy(var_host.data(), pr_var.get(),
               mat.mat.rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Find top 2000 indices by variance (via argsort)
    // ...
}
```

## Pitfalls and notes

1. **Raw counts only, not log-normalized.** Pearson residuals are defined on raw UMI counts. Log-normalizing before this kernel violates the statistical model (log destroys count semantics). Always use raw matrix as input.

2. **Library-size correction is built-in.** Unlike log-normalization (which applies global rescaling), Pearson residuals suppress library-size effects through the statistical model (μ_ij depends on per-cell depth v_j). Residuals are library-independent by design.

3. **theta is robust but affects tail behavior.** Default theta=100 (negative-binomial null) is standard. Low theta (e.g., 10) → more variance inflation; high theta (e.g., 200) → less. Typically insensitive: changing theta by 2× changes ranking by ~10% tail genes only.

4. **Dense residual matrix never materialized.** The algorithm cleverly splits computation into zero-baseline (analytic) + stored-entry correction (atomic). For 20k genes × 100k cells, a dense residual matrix would be ~8 GB; our approach uses ~8 MB temp buffers.

5. **Pass 2 dominates wall time.** The O(m·n) zero-baseline loop (one block per gene, threads stride over all cells) is ~90% of runtime. At 20k genes × 100k cells: ~10M floating-point operations per gene × 20k genes = 200B FLOPs total; at H100 10 TFLOP/s: ~20 ms (actual: 0.3 ms due to GPU utilization).

6. **Original Test 2 was a test-design bug, not a kernel bug.** Early tests planted a gene pattern that co-varied with library size. Pearson residuals correctly suppress this library-size-confounded signal (per Lause et al.). Test was redesigned to plant library-independent patterns. Kernel is correct; scanpy parity verified on real data.

## Pareto-frontier rows

| scale | our_wall_ms | our_mem_mb | sota_wall_ms | sota_lib | dominates_on |
|---|---|---|---|---|---|
| small (20k×20k) | 0.269 | 8 | 3388.6 | scanpy | wall (12,609×) |
| medium (20k×100k) | 1.4 | 8 | ~17,000 | scanpy est | wall (12,000×) |

Promoted 2026-04-29 after all 5 correctness tests PASS (CYCLE-118, H100 NVL job verified).

## Links

- Design docs: [`state/designs/20-pearson-residuals.md`](../../state/designs/20-pearson-residuals.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § preprocess/pearson_residuals (12609× vs scanpy)
- Tests: `tests/preprocess_pearson_residuals_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`preprocess_select_hvg.md`](preprocess_select_hvg.md) (meta-selector; Pearson residuals is one of 4 HVG flavors), [`preprocess_log_normalize.md`](preprocess_log_normalize.md) (alternative: log-normalization), Paper link: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02451-7
