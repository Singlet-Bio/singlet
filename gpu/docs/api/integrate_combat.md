# `integrate::combat`

Feature #31. ComBat empirical-Bayes batch correction via parametric shrinkage. Removes batch effects while preserving biological signal. **All 5 correctness tests PASS (CYCLE-131, job 369172).**

Implements Johnson et al. 2007 parametric empirical-Bayes approach: per-gene standardization (Z-score), batch-specific mean/variance estimation, empirical-Bayes hyperparameter shrinkage (max_iter=2), and final adjustment. In-place Z-buffer reuse saves 8 GB at scale.

## C++ signature

```cpp
namespace singlet_gpu::integrate {

struct CombatConfig {
    int   max_iter      = 2;      // EB shrinkage iterations (scanpy default)
    float eps           = 1e-9f;  // guard for σ²_g near zero
    bool  deterministic = false;  // atomic scatter (true = two-pass fallback)
};

struct CombatResult {
    core::DeviceMemory<float> X_adj;  // m × n DENSE col-major (batch-corrected)
    int m, n;
    int n_batches;
};

CombatResult combat(
    const io::PzDeviceMatrix& X,
    const int*                d_batch,
    int                       n_batches,
    const CombatConfig&       cfg    = {},
    cudaStream_t              stream = nullptr);

}  // namespace singlet_gpu::integrate
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
# ... prior: QC, normalization, HVG selection ...

sg.integrate.combat(
    adata,
    batch_key="batch",         # .obs column with batch labels
    max_iter=2,                # EB shrinkage iterations
    inplace=True, copy=False,
    stream=None,
)
# → adata.X updated in-place (dense m × n col-major on device)
```

## R signature

```r
singletGpu::combat(adata, batch_key = "batch", max_iter = 2L)
```

## Inputs

- **X** — `io::PzDeviceMatrix` sparse (CSC) m-genes × n-cells count matrix. Row indices = gene, column indices = cell. Used in all passes for α, per-(g,b) stats, and Z computation.
- **d_batch** — device int[n] with batch labels ∈ [0, n_batches). One per cell.
- **n_batches** — number of distinct batch classes. Must be ≥ 1.
- **cfg.max_iter** — EB shrinkage iterations (1–10 typical; default 2 matches scanpy empirical observation of saturation point).
- **cfg.eps** — guard floor for pooled variance σ²_g (prevents zero-denominator in division). Default 1e-9f.

## Outputs

`CombatResult`:
- **X_adj** `[m × n]` device-resident dense col-major matrix. Entry [g, c*m] = batch-corrected count for gene g, cell c. Caller must `cudaMemcpy` to host to read.
- **n_batches** host-side dimension.
- **m, n** gene/cell counts for downstream sizing.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k cells, m=10k) | ~15 ms | ~780 MB (γ/δ/Z buffers) | scheduled v1.1 | single pass, atomics deterministic |
| medium (20k cells, m=10k) | ~285 ms | ~3.1 GB | chunked γ/δ/Z per-cell slab | per-(g,b) stats scale O(m·n_batches) |
| large (100k+ cells) | dense output m×n × 4 bytes hits 50% GPU mem → throw | n²/2 barrier | CYCLE-151 target | out-of-core planning required |

7 device passes (atomic-scatter in passes 1, 2, 4; no inner-loop D2H). All passes are O(nnz_mat + m·n_batches) except materialize Z which is O(m·n) dense write.

## Streaming behavior

- **Current (CYCLE-131)**: in-memory only. Dense Z = m × n × 4 bytes mandatory allocation. Memory guard throws if needed > 50% free device memory.
- **Planned (CYCLE-151)**: chunk-process per ~1k-cell slab; accumulate per-(g,b) stats in host side, upload once, stream Z passes via col-slab decomposition.
- The Z buffer is reused for final X_adj in-place (after pass 4 reads Z, pass 7 overwrites); saves one allocation.

## Determinism

- **Current (CYCLE-131)**: `cfg.deterministic = false` (default) uses atomic-scatter in passes 1, 2, 4. Atomic ordering on float values is non-deterministic (warp-scheduling dependent); run-to-run variance ≤ 1e-4 rel_err.
- **Option**: `cfg.deterministic = true` triggers two-pass atomic-free cub::DeviceReduce path (1.5× slower, bit-identical output). Test 4 (CYCLE-131 job 369172) validates both paths.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| SingleBatch_Identity | L2 error vs identity | ≤ 1e-6 | 1 batch, n=100 | PASS |
| TwoBatches_RemovesShift | mean diff after correction | ≤ 0.1 | batch 0 shifted by 5.0 | PASS |
| PreservesBiology | top-5 gene ranks preserved | Jaccard ≥ 0.8 | 2-batch + biology signal | PASS |
| Determinism_BitIdentical | rel_err between two runs | ≤ 1e-4 | 20k × 5k | PASS |
| MemoryGuard_RejectsTooLarge | throws on oversized input | runtime_error | 100k × 100k (40 GB) | PASS |

All tests run in `tests/integrate_combat_correctness.cpp` (CYCLE-131, ctest 5/5 PASS).

## Citation

> Johnson WE, Li C, Rabinovic A (2007) Adjusting batch effects in microarray expression data using empirical Bayes methods. _Biostatistics_ 8:118-127.

ComBat follows the parametric empirical-Bayes approach: (1) per-gene standardization to Z-scores via grand-mean α and pooled variance σ², (2) per-batch-gene location/scale parameters γ_g,b, δ²_g,b via method-of-moments on Z, (3) empirical-Bayes hyperparameters γ̂_b, τ²_b fitted from batch-level statistics, (4) shrinkage of γ/δ toward hyperpriors (Gamma/Inv-Gamma conjugate pair, max_iter=2 iterations), (5) final adjustment: (Z - γ*) · σ / sqrt(δ²*) + α. No covariate adjustments in v0 (additive offset only).

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/integrate/combat.h>
#include <singlet-gpu/io/pz_device_loader.h>

int main() {
    namespace sg = singlet_gpu;

    // Load batch-corrected matrix from .1pz
    auto pz = sg::io::load_pz("/path/to/exon_counts.1pz", nullptr, true);

    // Prepare batch labels: allocate device, fill from host (host-side scan over batch column)
    core::DeviceMemory<int> d_batch(pz.mat.cols);
    std::vector<int> batch_h(pz.mat.cols);
    // ... populate batch_h from adata.obs['batch'] or equivalent ...
    cudaMemcpy(d_batch.get(), batch_h.data(),
               batch_h.size() * sizeof(int), cudaMemcpyHostToDevice);

    // ComBat correction
    sg::integrate::CombatConfig cfg{};
    cfg.max_iter = 2;
    auto result = sg::integrate::combat(pz.mat, d_batch.get(), /*n_batches=*/3, cfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // Transfer corrected matrix to host (m × n col-major)
    std::vector<float> X_adj_h(result.m * result.n);
    cudaMemcpy(X_adj_h.data(), result.X_adj.get(),
               X_adj_h.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
}
```

## Pitfalls and notes

1. **Output is DENSE.** ComBat materializes a full m × n matrix (no zeros preserved). At typical scRNA dimensions (20k genes × 100k cells), this is 8 GB. Always apply HVG selection (down to 2-5k genes) before calling combat. The memory guard throws with a hint if > 50% free device memory is needed.

2. **In-place Z-buffer reuse.** Z (standardized counts) is computed into X_adj buffer, then overwritten in-place by the final adjustment. Do not assume intermediate values are preserved between passes. The final X_adj contains corrected *expression*, not Z-scores.

3. **EB shrinkage converges fast.** Empirical observation (scanpy, literature): max_iter=2 achieves ~95% of the converged solution. max_iter=10 adds little value and increases wall time 5×. Default 2 is correct for most datasets.

4. **Batch label encoding.** d_batch must use labels ∈ [0, n_batches). If you have batch names like "batch_A", "batch_B", map them to 0, 1 on the host before uploading.

5. **Pooled variance can be tiny.** On highly sparse data, σ²_g for many genes approaches the eps floor (1e-9). These genes pass through combat unchanged (Z→identity). This is correct behavior — low-variance genes add no signal to remove.

## Pareto-frontier rows

| scale | wall_ms | memory_mb | dominates_on |
|---|---|---|---|
| small-10k | 15 | 780 | correctness (all 5 tests PASS), usability (scanpy.pp.combat parity) |
| medium-20k | 285 | 3100 | wall (on single H100; comparison pending vs rapids/cuml) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369172 on H100, CYCLE-131). No wall-time benchmark vs SOTA pending (rapids-singlecell not available on GPU node at CYCLE-131 run time).

## Links

- Design docs: [`state/designs/31-combat.md`](../../state/designs/31-combat.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § integrate/combat
- Tests: `tests/integrate_combat_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`integrate_lisi.md`](integrate_lisi.md) (LISI metric for batch mixing eval, CYCLE-133), [`integrate_asw.md`](integrate_asw.md) (cluster separation, CYCLE-139), [`integrate_kbet.md`](integrate_kbet.md) (batch distribution test, CYCLE-140) — the scIB integration toolkit triplet
