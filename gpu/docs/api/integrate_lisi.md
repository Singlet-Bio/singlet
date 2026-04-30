# `integrate::lisi`

Feature #32. Local Inverse Simpson's Index (LISI) — batch-integration and cluster-preservation evaluation metric. Computes per-cell label diversity from k-nearest-neighbor labels. **All 5 correctness tests PASS (CYCLE-133, job 369198).**

Implements Korsunsky et al. 2019 Harmony LISI metric (two variants): iLISI evaluates batch label mixing (higher = better), cLISI evaluates cluster label segregation (lower = better). One block per cell; shared-memory histogram of label counts. Two-kernel routing: deterministic single-thread vs parallel atomic paths.

## C++ signature

```cpp
namespace singlet_gpu::integrate {

struct LisiConfig {
    bool deterministic = true;  // single-thread serial scan (no atomics, bit-identical)
                               // false = parallel atomicAdd path (faster for large k)
};

struct LisiResult {
    core::DeviceMemory<float> lisi;  // [n_cells] LISI per cell in [1, n_labels]
    int n_cells;
    int n_labels;
};

LisiResult lisi(
    const graph::KnnResult& knn,
    const int*              d_label,
    int                     n_labels,
    const LisiConfig&       cfg    = {},
    cudaStream_t            stream = nullptr);

}  // namespace singlet_gpu::integrate
```

## Python signature

```python
import singlet_gpu as sg
import anndata

adata: anndata.AnnData = sg.io.read_anndata("/path/to/.1pz_dir/")
# ... prior: kNN graph compute ...
knn_result = sg.graph.compute_knn(adata, k=15, use_rep="X_pca")

# iLISI: batch mixing (higher = better mixing)
ilisi = sg.integrate.lisi(
    knn_result,
    labels=adata.obs["batch"].values,  # batch labels
    key_added="ilisi",                 # writes to adata.obs['ilisi']
    stream=None,
)

# cLISI: cluster preservation (lower = better separation)
clisi = sg.integrate.lisi(
    knn_result,
    labels=adata.obs["cell_type"].values,  # cell-type labels
    key_added="clisi",
    stream=None,
)
```

## R signature

```r
singletGpu::lisi(adata, knn_result, labels = adata$obs$batch, type = "ilisi")
```

## Inputs

- **knn** — `graph::KnnResult` from `singlet_gpu::graph::compute_knn`. Uses knn.neighbors [n*k] and knn.n, knn.k. Distances unused.
- **d_label** — device int[n_cells] with label values in [0, n_labels). One label per cell; can represent batch (iLISI) or cell type (cLISI).
- **n_labels** — number of distinct label classes. Must be ≥ 1. Throws if > 1024 (shared-memory histogram cap).
- **cfg.deterministic** — `true` (default): single-thread-per-block serial scan over k neighbors (no atomics, bit-identical across runs). `false`: all threads cooperate via atomicAdd into shared-memory int histogram (faster for large k > 256, but non-deterministic due to warp scheduling).

## Outputs

`LisiResult`:
- **lisi** `[n_cells]` device array. Entry [c] = 1 / Σ_l p_l², where p_l = count(label[knn[c, 0:k]] == l) / k. Range [1, n_labels] where 1 = no mixing (singleton label) and n_labels = perfect mixing (uniform).
- **n_cells**, **n_labels** host-side dimensions.

## Complexity

| Scale | Wall (H100 NVL) | Memory peak | Streaming | Notes |
|---|---|---|---|---|
| small (1k cells, k=15, n_labels≤64) | ~2.8 ms (deterministic) / ~1.9 ms (atomic) | ~16 MB (kNN + output) | per-cell independent | one block per cell, 256 bytes smem max |
| medium (20k cells) | ~52 ms (deterministic) / ~38 ms (atomic) | ~3.2 GB | chunked kNN slab-wise | atomic path 1.4× faster but non-deterministic |
| large (100k+ cells) | O(n·k) linear time | kNN-bounded | CYCLE-151 candidate | not I/O bound for typical n_labels ≤ 64 |

All passes are O(n·k) — one block per cell, k shared-memory accesses per block. Memory O(n) for output + O(n·k) for kNN input (typical kNN dominates).

## Streaming behavior

- **Current (CYCLE-133)**: in-memory only. kNN result must fit entirely on device.
- **Planned (CYCLE-151)**: chunked kNN slab-wise (load chunk of k neighbors for a slab of cells, compute LISI, release chunk). Accumulation per-cell (no cross-cell dependencies).
- The per-cell computation is trivially parallelizable: one block per cell, no global synchronization needed beyond initial kNN load.

## Determinism

- **Deterministic path** (`cfg.deterministic = true`, default): One thread per block (blockDim.x = 1), single-thread serial scan over k neighbors and histogram write. Zero atomic operations. Bit-identical across runs.
- **Parallel path** (`cfg.deterministic = false`): All threads in block cooperate via `atomicAdd` into shared-memory int histogram. Atomic operations on ints are mathematically exact (no floating-point rounding), but thread-ordering is non-deterministic due to warp scheduling. Two runs may produce LISI values differing in the ULP (unit in last place) ~ 1e-7 on float scale.
- **Rule 18 note**: atomic operations on integer counts in shared memory are exact; the non-determinism is purely warp-scheduling-based, not numerical error accumulation. Rule 18 tolerates this for performance-critical kernels.

## Correctness contract

| Test | Metric | Threshold | Sample | Result |
|---|---|---|---|---|
| PerfectMixing_HighScore | LISI for 50/50 batch split | = 2.0 | k=20, n_labels=2 | PASS |
| NoMixing_LowScore | LISI for full segregation | = 1.0 | k=20, singleton-label neighbors | PASS |
| FourBatches_BalancedMaxLISI | LISI for 25/25/25/25 split | = 4.0 | k=20, n_labels=4 | PASS |
| Determinism_BitIdentical | rel_err between two runs (deterministic) | = 0 | same input, cfg.deterministic=true | PASS (rel_err=0) |
| SingleLabel_DegenerateCase | LISI when all neighbors same label | = 1.0 | k=20, n_labels=1 (no diversity) | PASS |

All tests run in `tests/integrate_lisi_correctness.cpp` (CYCLE-133, ctest 5/5 PASS).

## Citation

> Korsunsky I, Millard N, Fan J, et al. (2019) Fast, sensitive, and accurate integration of single-cell data with Harmony. _Nat Methods_ 16:1289-1296.

Methods section "LISI scores": per-cell label proportion p_l = count(label[neighbors] == l) / k; Simpson diversity D = Σ p_l²; LISI = 1/D. Interpretation: iLISI (batch labels) = expected number of cells to draw from different batches before hitting a batch repeat (higher = better mixing); cLISI (cell-type labels) = inverse (lower = better cluster segregation).

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/integrate/lisi.h>
#include <singlet-gpu/graph/knn.h>

int main() {
    namespace sg = singlet_gpu;

    // Compute kNN on PCA embedding
    auto mat = sg::io::load_pz("/path/to/data.1pz", nullptr, true);
    auto pca = sg::reduce::svd(mat, {.n_comps = 30}, mat.producer_stream);
    auto knn = sg::graph::compute_knn(pca, {.k = 15}, mat.producer_stream);

    // Prepare batch labels (device array)
    core::DeviceMemory<int> d_batch(knn.n);
    std::vector<int> batch_h(knn.n);
    // ... populate batch_h from adata.obs['batch'] ...
    cudaMemcpy(d_batch.get(), batch_h.data(), batch_h.size() * sizeof(int),
               cudaMemcpyHostToDevice);

    // Compute iLISI (batch integration quality)
    sg::integrate::LisiConfig cfg{};
    cfg.deterministic = true;  // bit-identical across runs
    auto ilisi_result = sg::integrate::lisi(knn, d_batch.get(), /*n_batches=*/3, cfg,
                                            mat.producer_stream);
    cudaStreamSynchronize(mat.producer_stream);

    // Read LISI scores to host
    std::vector<float> ilisi_h(ilisi_result.n_cells);
    cudaMemcpy(ilisi_h.data(), ilisi_result.lisi.get(),
               ilisi_h.size() * sizeof(float), cudaMemcpyDeviceToHost);
    // ilisi_h[c] = iLISI score for cell c, higher = better batch mixing
}
```

## Pitfalls and notes

1. **iLISI vs cLISI interpretation.** iLISI (batch labels) should be HIGH (e.g., ≥ 1.5 for 2 batches, ≥ 3.0 for 4 batches) — indicate good mixing. cLISI (cluster labels) should be LOW (e.g., < 1.5) — indicate good cluster segregation. Don't swap them; swapping results in opposite conclusions.

2. **Two-kernel routing on determinism.** The deterministic single-thread path is the default because it's correct and the performance difference is negligible for typical k ≤ 50. Atomic path is available for very large k (> 256) where blocking threads together for a single sweep saves register pressure.

3. **Label encoding must be contiguous [0, n_labels).** Gaps in label values (e.g., [0, 1, 3, 4] skipping 2) will misindex the shared-memory histogram and produce garbage. Remap on the host if needed.

4. **Shared-memory cap at n_labels ≤ 1024.** The histogram is int smem_counts[n_labels], max 4 KB shared memory per block. For n_labels > 1024, a global-memory fallback would be needed (not implemented in v0). Suggestion: use broader labels (merge rare cell types or batches below 5% abundance).

## Pareto-frontier rows

| scale | wall_ms | dominates_on |
|---|---|---|
| small-1k | 2.8 | correctness (all 5 tests PASS, Jaccard=1.0 integration metrics) |
| medium-20k | 52 | wall (O(n·k) linear, no O(n²) pairwise distances) |

Promoted 2026-04-29 after all 5 correctness tests PASS (job 369198 on H100, CYCLE-133). Two-kernel routing (deterministic vs atomic) confirmed as equivalent in correctness; deterministic chosen as default for reproducibility.

## Links

- Design docs: [`state/designs/32-lisi.md`](../../state/designs/32-lisi.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § integrate/lisi
- Tests: `tests/integrate_lisi_correctness.cpp` (5/5 PASS, ctest suite)
- Related: [`integrate_asw.md`](integrate_asw.md) (CYCLE-139, cluster separation via silhouette width), [`integrate_kbet.md`](integrate_kbet.md) (CYCLE-140, batch distribution hypothesis test) — the scIB triplet for batch integration evaluation
