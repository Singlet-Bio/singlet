# `reduce::nmf`

Feature #5. Non-negative matrix factorization via factornet's GPU NMF, with two singlet-gpu adapter rules tuned for H100/V100 latency. **1.82–8.66× sklearn NMF across k ∈ {10, 20, 50, 100}.**

The factornet auto-solver picks between Multiplicative Update (MU) and Coordinate Descent (CD). On GPU at k ≥ 32, CD's O(k² × cd_max_iter) inner loop is ~40× slower than MU's O(k × nnz) — Cycle 86 added a `FitConfig` shim that forces MU at high rank and lowers the default CD inner-iteration count. **Net result: k=50 went from a 3× regression to an 8.66× win.**

## C++ signature

```cpp
namespace singlet_gpu::reduce::nmf {

// Re-exports from factornet — same types factornet uses, no re-wrapping.
using NmfConfig    = ::factornet::NMFConfig<float>;
using NmfResult    = ::factornet::NMFResult<float>;
using FactorConfig = ::factornet::FactorConfig<float>;     // per-factor regularization
using DenseMatrix  = ::factornet::DenseMatrix<float>;
using DenseVector  = ::factornet::DenseVector<float>;
using LossType     = ::factornet::LossType;
using LossConfig   = ::factornet::LossConfig<float>;
using LazySpeckledMask = ::factornet::nmf::LazySpeckledMask<float>;

// singlet-gpu adapter shim (Cycle 86):
struct FitConfig {
    int k_cd_cutoff = 32;     // ranks ≥ this in auto mode → forced to MU
    int cd_max_iter = 10;     // applied only when NmfConfig.cd_max_iter is at the
                              // factornet default (100); explicit caller settings honored
};

// In-memory GPU NMF.
//
// Precondition: m must have been loaded with keep_host_pinned=true.
// factornet's GPU NMF takes host pointers and stages internally; the pinned
// host buffers held by `m` avoid a redundant allocation.
NmfResult fit(const io::PzDeviceMatrix& m,
              const NmfConfig& cfg,
              const FitConfig& fit_cfg = FitConfig{},
              const DenseMatrix* W_init = nullptr,    // optional warm start
              const DenseMatrix* H_init = nullptr);

// Backward-compatible 4-arg overload (no FitConfig).
NmfResult fit(const io::PzDeviceMatrix& m,
              const NmfConfig& cfg,
              const DenseMatrix* W_init,
              const DenseMatrix* H_init);

}  // namespace singlet_gpu::reduce::nmf
```

`NmfResult` exposes `W` (`m × k`), `d` (k diagonal), `H` (`k × n`), plus iteration count + per-iter loss history.

## Python signature

```python
import singlet_gpu as sg

# AnnData-native; auto-routing applies the Cycle-86 FitConfig shim under the hood.
sg.reduce.nmf.nmf(
    adata,
    n_factors=50,            # k
    loss="MSE",              # or "KL", "IS", ...
    solver_mode=3,           # 3=auto (→ MU at k>=32), 2=MU, 0=CD
    init_mode=2,             # 2=warm-start, 1=random, 0=nnsvd
    max_iter=100, tol=1e-5, seed=0,
    layer=None, inplace=True, copy=False,
)
# Writes adata.obsm['X_nmf'] (W^T · sqrt(d) — n_cells × k), adata.varm['NMF_components']
# (H scaled — n_genes × k), adata.uns['nmf']['loss'].

# Streaming (multi-file 1M+ cell):
sg.reduce.nmf.nmf_chunked(paths=[*1pz_paths], n_factors=50, chunk_cols=100_000)

# Multi-modal (CITE-seq / multiome shared-H factorization):
sg.reduce.nmf.nmf_graph_factorize(
    modalities={"rna": rna_adata, "adt": adt_adata},
    n_factors=20, shared_h=True,
)
```

## R signature

```r
singletGpu::nmf(adata, n_factors = 50L)
```

## Inputs

- **m** — `io::PzDeviceMatrix` loaded with `keep_host_pinned=true`. NMF requires raw counts or normalized counts (NOT log-transformed). For typical workflows, run NMF on the raw `.1pz` directly.
- **cfg.rank** — k. Field is named `rank` (not `k` or `k_max`) per factornet convention.
- **cfg.solver_mode** — `2` = MU, `0` = CD, `3` = auto. Default `3`. The shim (`fit_cfg.k_cd_cutoff`) overrides auto → MU when `cfg.rank ≥ 32`.
- **cfg.init_mode** — `0` = nnsvd, `1` = random, `2` = warm-start (W_init / H_init). int field, no enum.
- **cfg.loss** — `LossType::MSE` (default), KL, IS, etc. via `factornet::LossType`.
- **cfg.W**, **cfg.H** — `FactorConfig` per-factor regularization: L1, L2, sparsity, graph regularization. Applied during fit.
- **W_init / H_init** — optional warm-start factors. `nullptr` ⇒ random init seeded by `cfg.seed`.

## Outputs

`NmfResult { W, d, H, loss, n_iter, status }`:
- `W[m, k]` on device — gene loadings (non-negative).
- `d[k]` — column scaling, useful when re-orthogonalizing.
- `H[k, n]` on device — cell scores (non-negative).
- `loss` — host vector, one entry per iteration.
- `n_iter`, `status` — convergence diagnostics.

## Adapter rules (Cycle 86, OPTIM-NMF-K50)

| Rule | When | What | Why |
|---|---|---|---|
| 1 | `cfg.solver_mode == 3 && cfg.rank ≥ fit_cfg.k_cd_cutoff` | Force `solver_mode = 2` (MU) | CD's O(k² × cd_max_iter) inner loop dominates at k ≥ 32 on H100; MU's O(k × nnz) wins by ~40× |
| 2 | `cfg.solver_mode == 0 && cfg.cd_max_iter == 100` | Lower to `fit_cfg.cd_max_iter` (default 10) | factornet's default 100 was CPU-calibrated; on GPU 10 iters give the same accuracy at 10× the speed |

Callers can opt out by setting `solver_mode` explicitly (any value ≠ 3) or by customizing `FitConfig`.

## Complexity

| Scale | k | Wall (H100 NVL) | SOTA wall (sklearn) | Speedup |
|---|---|---|---|---|
| small | 10 | 37.2 ms | 67.5 ms | **1.82×** (CD path) |
| small | 20 | 110.3 ms | 272.2 ms | **2.47×** (CD path) |
| small | 50 | 38.0 ms | 329.1 ms | **8.66×** (MU path, post-fix) |
| small | 100 | 143.8 ms | 363.6 ms | **2.53×** (MU path) |
| medium-GSM4037629 (~20.8k) | 10 / 20 / 50 / 100 | 290.7 / 444.5 / 234.9 / 410.6 ms | TBD (sklearn ref unavailable, MTX cleaned) | wall dominates; ref pending |

Memory: factornet manages scratch. The `device_memory_mb` column reads 0.0 on H100 due to `cudaMemGetInfo` driver pooling — see `state/blockers.md` → INFRA-MEM-TRACKING-H100.

## Streaming behavior

Two streaming entry points:
- **`reduce::nmf::chunked_fit`** — wraps `factornet::nmf::fit_chunked_gpu`, takes a `factornet::io::loader<float>` interface. We implement that interface for `.1pz` chunks via `streaming::PzShardIterator`. One pass over the file per outer NMF iteration.
- **`reduce::nmf::fit_streaming_spz`** — factornet's `.spz` streaming path. Unused for our `.1pz` workflow.

For matrices that fit on one device, the in-memory `fit` path is faster.

## Determinism

Stochastic. `cfg.seed` (uint64) seeds the random init and any internal sampling. Identical seeds + identical inputs → identical `W`, `H` to fp32 reduction-order tolerance (≤ 1e-5 element-wise after convergence).

The factornet kernels use `atomicAdd` in some reductions; truly bit-identical determinism requires `cfg.deterministic = true` (segmented-scan path), at a small perf cost.

## Correctness contract

| Reference | Tolerance | Sample |
|---|---|---|
| sklearn `decomposition.NMF` (auto solver) | reconstruction-error ratio within 1.05× of sklearn at convergence | small synthetic; ctest 13/13 PASS |
| factornet CPU `nmf::fit` | bit-equivalent within fp32 reduction-order | small synthetic |

Cycle 86 fix verified: k=50 was a 3× regression vs sklearn before the `k_cd_cutoff` shim landed; now it's an 8.66× win, and 13/13 ctests still pass.

## Citations

- **NMF general**: D. D. Lee, H. S. Seung. _Algorithms for non-negative matrix factorization._ NIPS (2001).
- **Coordinate Descent**: C.-J. Hsieh, I. S. Dhillon. _Fast coordinate descent methods with variable selection for non-negative matrix factorization._ KDD (2011).
- **factornet GPU NMF backend**: Zach DeBruine, factornet 2026 (GPL-2.0).
- **Speckled-mask CV** for auto-rank: factornet's `nmf::speckled_cv` — exposed via `reduce::nmf::cv` adapter (separate header).

singlet-gpu's adapter contribution: the `FitConfig` solver-routing shim that turns the high-rank CPU-default regression into a frontier kernel.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/reduce/nmf/fit.h>           // until released
#include <singlet-gpu/reduce/nmf/types.h>

int main() {
    namespace sg = singlet_gpu;

    auto pz = sg::load_pz("/path/to/exon_counts.1pz",
                          /*stream=*/nullptr,
                          /*keep_host_pinned=*/true);
    cudaStreamSynchronize(pz.producer_stream);

    sg::reduce::nmf::NmfConfig cfg{};
    cfg.rank        = 50;             // k
    cfg.solver_mode = 3;              // auto — adapter shim will force MU at k>=32
    cfg.seed        = 42;
    cfg.loss.type   = sg::reduce::nmf::LossType::MSE;

    auto res = sg::reduce::nmf::fit(pz, cfg);

    // res.W : DenseMatrix(m × 50) — gene loadings
    // res.H : DenseMatrix(50 × n) — cell scores
    // res.loss.back() — final reconstruction error
}
```

## Pareto-frontier rows

| scale | k | wall_ms | sota_wall_ms (sklearn) | speedup |
|---|---|---|---|---|
| small | 10 (CD) | 37.2 | 67.5 | 1.82× |
| small | 20 (CD) | 110.3 | 272.2 | 2.47× |
| small | 50 (MU) | 38.0 | 329.1 | **8.66×** |
| small | 100 (MU) | 143.8 | 363.6 | 2.53× |
| medium-GSM4037629 | 10–100 | 234.9–444.5 | ref unavailable | wall dominates |

Promoted 2026-04-18 after the Cycle 86 `k_cd_cutoff=32` fix.

## Links

- Design docs: [`state/designs/06-nmf-adapter.md`](../../state/designs/06-nmf-adapter.md), [`state/designs/86-nmf-k50-solver-fix.md`](../../state/designs/86-nmf-k50-solver-fix.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § reduce/nmf
- Equivalence notebook: `docs/notebooks/nmf.ipynb` (pending)
- Related: [`reduce_svd.md`](reduce_svd.md) (sister module — both consume `PzDeviceMatrix` with `keep_host_pinned=true`)
