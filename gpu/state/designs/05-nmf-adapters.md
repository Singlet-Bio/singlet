---
feature: nmf_adapters
roadmap_id: 5
module: include/singlet-gpu/reduce/nmf/{types,fit,cv,chunked,init,graph}.h
status: design
tolerance: bit-identical to factornet `nmf_fit_gpu` direct call (adapter is pure marshaling). Round-trip diff via factornet CPU `nmf::fit` for cross-validation only.
target_perf: ≤2% adapter overhead vs direct factornet call. factornet itself already on the frontier (10–38× faster MU than CD on sparse MSE).
ooc_plan: streaming via factornet's `nmf_chunked_gpu` + a new `singlet_gpu::io::PzDataLoader` that implements `factornet::io::DataLoader<float>` yielding Eigen sparse CSC chunks from `.1pz`. The loader-to-Eigen path is non-zero-copy, but factornet's chunked NMF is the only streaming entry point.
---

## Algorithm

Six **adapter** headers under `reduce/nmf/`. Same pattern as cycle 5 SVD adapters — each header is ~50–80 LOC, marshaling our `PzDeviceMatrix` (with retained host pinned buffers from cycle 5 loader extension) into factornet's `nmf_fit_gpu` family.

The factornet NMF API findings from cycle 5 code-reader confirm the host-pointer pattern: `nmf_fit_gpu(const int* col_ptr, const int* row_idx, const float* values, m, n, nnz, NMFConfig, W_init, H_init)` — same shape as SVD GPU calls. Our existing `keep_host_pinned=true` plumbing on the loader handles this for free.

Three new findings shape the design:

1. **`nmf_chunked_gpu` takes `io::DataLoader<Scalar>` of Eigen sparse chunks**, not host CSC pointers. Streaming NMF uses a different interface than in-memory fit. We provide a `singlet_gpu::io::PzDataLoader` that implements `factornet::io::DataLoader<float>` and yields `Eigen::SparseMatrix<float>` chunks decoded from `.1pz`. This is the ONLY non-zero-copy path in the entire library so far — document the cost (per-chunk Eigen allocation + decompression). Streaming NMF is opt-in.
2. **Non-MSE losses use host-mediated IRLS** — KL, NB, GP, Gamma, Tweedie pass through factornet's CPU IRLS solver per column even when called via `nmf_fit_gpu`. This is factornet's behavior, not our choice. Document in the adapter header doc and `style-rules.md` §F. Update absolute rule §⛔3 to clarify: "GPU-native math kernels except where factornet's NMF explicitly host-mediates non-MSE loss IRLS — that is factornet's behavior and we accept it."
3. **`graph::FactorGraph` has no explicit shared-H multi-modal joint factorization** — each `NMFLayerNode` owns its own H; concat/add nodes merge outputs. Hierarchical NMF runs via end-to-end BCD on chain reconstruction. Our style-rules.md claim of "multi-modal joint NMF for free via FactorGraph" is INCORRECT and must be revised. Multi-modal joint NMF (RNA + ATAC + ADT shared H) requires constructing the FactorGraph manually with `ConcatNode` and careful chain composition; it is not a one-line API.

## Module layout (six headers)

### 1. `reduce/nmf/types.h` (~30 LOC)

Re-exports under `singlet_gpu::reduce::nmf`:

```cpp
using NmfConfig    = factornet::nmf::NMFConfig<float>;
using NmfResult    = factornet::nmf::NMFResult<float>;
using FactorConfig = factornet::nmf::FactorConfig<float>;        // per-factor constraints
using DenseMatrix  = factornet::DenseMatrix<float>;             // for W_init / H_init
using LossType     = factornet::nmf::LossType;                  // MSE / KL / NB / GP / Gamma / Tweedie / Huber / Robust
using SolverMode   = factornet::nmf::SolverMode;                // 0=CD, 1=Cholesky, 2=MU, 3=Auto
using InitMode     = factornet::nmf::InitMode;                  // 0=Random, 1=Lanczos, 2=IRLBA
using LazySpeckledMask = factornet::nmf::LazySpeckledMask<float>;
```

### 2. `reduce/nmf/fit.h` (~80 LOC)

```cpp
namespace singlet_gpu::reduce::nmf {

inline NmfResult fit(const PzDeviceMatrix& m,
                     const NmfConfig& cfg,
                     const DenseMatrix* W_init = nullptr,
                     const DenseMatrix* H_init = nullptr) {
    require_host_retained(m, "nmf::fit");
    return factornet::nmf::nmf_fit_gpu<float>(
        m.host_indptr.get(), m.host_indices.get(), m.host_values.get(),
        m.mat.rows, m.mat.cols, m.mat.nnz,
        cfg, W_init, H_init);
}

}  // namespace
```

Documented behavior:
- For MSE loss + Tier-1 constraints: fully GPU-resident.
- For non-MSE loss (KL, NB, GP, Gamma, Tweedie): factornet host-mediates IRLS per column. The kernel is GPU but the solver step is on CPU per column. This is a factornet limitation, not ours.

### 3. `reduce/nmf/cv.h` (~80 LOC)

Wraps `factornet::nmf::nmf_cv_fit_gpu<float>` for held-out CV. Same shape as `fit.h` but the result has additional fields (`train_loss`, `test_loss`, `best_test_loss`, `best_iter`). Used for auto-rank determination via the orchestrator running multiple `cv_fit` calls with increasing `cfg.k`.

### 4. `reduce/nmf/chunked.h` (~120 LOC)

This is the streaming entry point. It does NOT take `PzDeviceMatrix` because factornet's chunked NMF takes a `DataLoader<float>`. Instead, it takes a `singlet_gpu::io::PzDataLoader&` reference (defined in a new file under `streaming/` — see below).

```cpp
inline NmfResult chunked_fit(singlet_gpu::io::PzDataLoader& loader,
                             const NmfConfig& cfg,
                             const DenseMatrix* W_init = nullptr,
                             const DenseMatrix* H_init = nullptr) {
    return factornet::nmf::nmf_chunked_gpu<float>(loader, cfg, W_init, H_init);
}
```

The new file `singlet-gpu/include/singlet-gpu/streaming/pz_data_loader.h` (~150 LOC) implements the `factornet::io::DataLoader<float>` interface for `.1pz` files:

```cpp
namespace singlet_gpu::io {

class PzDataLoader : public factornet::io::DataLoader<float> {
public:
    PzDataLoader(const std::string& path, int chunk_cols);

    // factornet::io::DataLoader interface:
    Eigen::SparseMatrix<float> next_forward() override;     // returns one chunk as Eigen CSC
    Eigen::SparseMatrix<float> next_transpose() override;   // for the transposed pass (if factornet needs it)
    int num_forward_chunks() const override;
    int num_transpose_chunks() const override;
    void reset_forward() override;
    void reset_transpose() override;

private:
    PzChunkIterator iter_;  // from cycle 2 loader
    int chunk_cols_;
};

}  // namespace
```

**Cost note**: the per-chunk decode is fast, but the conversion from our `PzDeviceMatrix` to `Eigen::SparseMatrix<float>` requires a host-side rebuild of `outer/inner/values`. This is the only place in singlet-gpu where we materialize Eigen — document the per-chunk overhead in benchmarks.

### 5. `reduce/nmf/init.h` (~60 LOC)

Re-exports the three init strategies:

```cpp
inline DenseMatrix init_random(int m, int n, int k, uint64_t seed);
inline std::pair<DenseMatrix, DenseMatrix>
       init_lanczos(const PzDeviceMatrix& mat, int k, const NmfConfig& cfg);
inline std::pair<DenseMatrix, DenseMatrix>
       init_irlba  (const PzDeviceMatrix& mat, int k, const NmfConfig& cfg);
```

The Lanczos and IRLBA init paths reuse our cycle-5 SVD adapters — they call `singlet_gpu::reduce::svd::lanczos(mat, ...)` or `irlba_factornet(mat, ...)` to produce W_init/H_init from the absolute values of U√Σ and V√Σ.

### 6. `reduce/nmf/graph.h` (~120 LOC)

Adapter for `factornet::graph::FactorGraph<float>` and `factornet::graph::fit<float>`. Provides:

- A re-export of the node types (`InputNode`, `NMFLayerNode`, `SVDLayerNode`, `ConcatNode`, `AddNode`, `ConditionNode`).
- A `FactorGraph` typedef.
- A `fit_graph(net, m)` adapter that calls `factornet::graph::fit(net, eigen_view_of(m))`.

Multi-modal joint NMF (RNA + ATAC + ADT) is constructed manually:

```cpp
auto rna  = io::load_pz(rna_path);
auto atac = io::load_pz(atac_path);
auto adt  = io::load_pz(adt_path);

InputNode<float> rna_in(rna_eigen,  "rna");
InputNode<float> atac_in(atac_eigen, "atac");
InputNode<float> adt_in(adt_eigen,  "adt");

ConcatNode<float> concat({&rna_in, &atac_in, &adt_in}, /*axis=*/0, "concat");
NMFLayerNode<float> joint(&concat, /*k=*/20, "joint");

FactorGraph<float> net({&rna_in, &atac_in, &adt_in}, &joint);
```

This is NOT a "shared-H" formulation in the strict sense — it is concatenation along the gene/feature axis with a single `H` over cells. For true paired multi-modal where each modality has its own gene set but shares cell identity, this is the right pattern. Document the user-facing example in `style-rules.md` §F.

The graph adapter requires Eigen views of the input matrices because `NMFLayerNode::input` is templated on Eigen::SparseMatrix. We construct these on the fly from `PzDeviceMatrix.host_indptr/indices/values` via `Eigen::Map<Eigen::SparseMatrix<float>>`.

## Numerical stability

Owned by factornet. We pass `cfg.tol`, `cfg.max_iter`, `cfg.cd_tol`, `cfg.cd_max_iter`, `cfg.irls_max_iter` through. fp32 default.

## Memory layout

- Input: `PzDeviceMatrix` with `host_retained = true` (same as cycle 5).
- Workspace: factornet's internal NMF workspace + `W` (m×k) + `H` (k×n) + `d` (k) + `loss_history`.
- For chunked: `PzDataLoader` holds one chunk at a time as `Eigen::SparseMatrix<float>`.
- Output: `NmfResult` with `W`, `d`, `H`, `iterations`, `converged`, `loss_history`, `test_loss_history` (cv variant).

## Streams

factornet creates its own GPU context; we do NOT pass a `cudaStream_t`. Same as cycle 5 SVD adapters. Document the limitation.

## Out-of-core

`chunked_fit` is the OOC entry point. `PzDataLoader` is the bridge. Per-chunk Eigen allocation is the only non-zero-copy step in the library — document the cost.

Future cycle (post-cycle-6): submit a PR to factornet adding a `nmf_chunked_gpu` overload that takes a custom loader interface returning device pointers directly, skipping Eigen. File as `CYCLE-6-FOLLOWUP-FACTORNET-DEVICE-LOADER`.

## Determinism

factornet's NMF init mode 0 (random) takes a seed via `cfg.seed`. Modes 1/2 are deterministic given the SVD's seed. The MU/CD solvers are deterministic given fixed reduction order on a fixed GPU architecture. Document.

## Correctness test spec

Test file: `tests/reduce_nmf_correctness.cpp`.

1. **Round-trip via direct factornet call**: load GSM4037629 with `keep_host_pinned=true`, call `singlet_gpu::reduce::nmf::fit(m, cfg)` and ALSO `factornet::nmf::nmf_fit_gpu<float>(m.host_indptr.get(), ..., cfg, nullptr, nullptr)`. Compare `W`, `H`, `d`, `iterations`, `loss_history` element-wise — bit-identical (rel_err == 0).
2. **CV round-trip** for `cv.h`.
3. **Init re-export tests**: confirm `init_random` produces a non-negative matrix; `init_lanczos` and `init_irlba` reuse cycle-5 SVD adapters (call them and confirm W/H shape).
4. **`PzDataLoader` interface**: confirm `next_forward()` produces an `Eigen::SparseMatrix<float>` with the expected dimensions and nnz from a known `.1pz`. Use a tiny synthetic.
5. **`chunked_fit` smoke**: tiny synthetic concat of 3 .1pz files; confirm `nmf_chunked_gpu` runs and converges.
6. **`graph::FactorGraph` smoke**: build a 2-input + ConcatNode + NMFLayerNode graph from two tiny .1pz files; confirm `fit_graph` returns a result.
7. **Loss flavor test**: run `nmf::fit` with `cfg.loss.type = LossType::KL` on tiny — confirm result is non-trivial. (Will host-mediate IRLS; document the runtime difference vs MSE.)

Tolerance:
- Round-trip: bit-identical.
- Init / loader / graph smoke tests: non-empty + correct shape.
- KL loss test: convergence (loss strictly decreasing for first 5 iters).

## Target performance

| Scale | Cells | k | Loss | Backend | factornet wall (target) | adapter overhead | our wall |
|---|---|---|---|---|---|---|---|
| 10k | 11,560 | 20 | MSE | nmf_fit_gpu | ~80ms | <1ms | <81ms |
| 100k | ~120k | 20 | MSE | nmf_fit_gpu | ~600ms | <1ms | <601ms |
| 1M | ~1M | 20 | MSE | nmf_chunked_gpu | ~5s | ~50ms (per-chunk Eigen) | <5.05s |
| 10k | 11,560 | 20 | KL | nmf_fit_gpu (host IRLS) | ~5s | <1ms | <5.001s |

The KL/NB row is intentionally bad — host-mediated IRLS is slow. Document this as a known cost and a candidate optimization target for a future cycle.

## Implementation notes (for cycle 6 kernel-dev dispatch)

- Six adapter headers under `include/singlet-gpu/reduce/nmf/`.
- One streaming bridge under `include/singlet-gpu/streaming/pz_data_loader.h` (~150 LOC).
- Total module: ~600–700 LOC.
- Each adapter ≤120 LOC.
- Build flag: `FACTORNET_HAS_GPU=1`.
- Dependencies: cycle 5 (SVD adapters, loader extension), cycle 2 (loader), cycle 1 (core types).
- `// SPDX-License-Identifier: GPL-2.0-or-later` first line on every file.
- `// integrates: factornet/nmf/{file}` second comment.
- Use `Eigen::Map<Eigen::SparseMatrix<float>>` for the graph adapter (no copy if the host buffers are aligned correctly).

## Risks

1. **Eigen alignment for `Eigen::Map`**: `Eigen::SparseMatrix<float>` requires 16-byte alignment for the inner/outer arrays. Our `cudaMallocHost` pinned buffers may or may not satisfy this. Test on the GPU node; if alignment fails, fall back to per-chunk `Eigen::SparseMatrix<float>` construction (a host copy).
2. **Non-MSE loss runtime**: host-mediated IRLS is 50–100× slower than MSE. Users must be aware. Document.
3. **`nmf::fit` (the CPU+GPU dispatcher) vs `nmf_fit_gpu` (direct GPU)**: we wrap the direct GPU function. Users who want CPU fallback (e.g., when GPU is unavailable) should use the dispatcher in a future cycle. File `CYCLE-6-FOLLOWUP-FALLBACK-DISPATCHER`.
4. **Graph adapter user-facing complexity**: building a multi-modal `FactorGraph` is a multi-step API. We document the example but do not wrap it in a one-liner — that would hide too much.
