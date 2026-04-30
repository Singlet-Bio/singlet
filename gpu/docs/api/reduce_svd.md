# `reduce::svd`

Feature #4. Truncated SVD for PCA via factornet's GPU backend. Five backends were benchmarked at three scales × four ranks; **deflation** won at every k, **randomized** kept as a fallback. Lanczos / IRLBA / Krylov-constrained were removed (Rule 32 — adopt the winner). **27× scanpy CPU at k=50.**

## C++ signature

```cpp
namespace singlet_gpu::reduce::svd {

// Re-exports of the factornet types — these are the SAME types factornet uses,
// no re-wrapping. Setting a field on SvdConfig drives factornet's behavior directly.
using SvdConfig = ::factornet::SVDConfig<float>;
using SvdResult = ::factornet::SVDResult<float>;
using SvdMethod = ::factornet::svd::SVDMethod;

// Inspect a config for active constraints. Preserved for caller convenience;
// no longer changes routing (deflation handles all constraint types).
bool has_constraints(const SvdConfig& cfg) noexcept;

// auto_select — primary entry point. Routes to deflation; falls back to
// randomized only if deflation produces a degenerate result on a pathological
// input.
SvdResult auto_select(const io::PzDeviceMatrix& m, int k, const SvdConfig& cfg);

// Direct backend calls. Use auto_select unless you have a specific reason.
SvdResult deflation(const io::PzDeviceMatrix& m, const SvdConfig& cfg);
SvdResult randomized(const io::PzDeviceMatrix& m, const SvdConfig& cfg);

}  // namespace singlet_gpu::reduce::svd
```

`SvdResult` exposes `U` (`m × k` `DenseMatrix<float>` on device), `d` (`k` singular values, `DenseVector`), `V` (`n × k`), plus a small status block.

## Python signature (scanpy convention)

```python
import singlet_gpu as sg

# Same name + parameter names as scanpy.tl.pca / pp.pca
sg.reduce.svd.pca(
    adata,
    n_comps=50,
    backend="auto",          # → routes to deflation winner; legacy backend strings warn
    layer=None,
    center=True,
    scale=False,
    seed=0,
    inplace=True,
)
# Writes adata.obsm['X_pca'] (n_cells × n_comps), adata.varm['PCs'] (n_genes × n_comps),
# adata.uns['pca']['variance_ratio'].
```

> **Stale exports**: `sg.reduce.svd.svd_lanczos`, `svd_irlba`, `svd_krylov` exist but their underlying C++ backends were removed in Cycle 61. Call `pca(..., backend="auto")` instead. Tracked in `state/wrapper-gaps.md` → CYCLE-104.

## R signature

```r
singletGpu::pca(adata, n_comps = 50L, backend = "auto")
```

## Inputs

- **m** — `io::PzDeviceMatrix`. **Must have been loaded with `keep_host_pinned=true`** because factornet's GPU SVD takes host pointers and stages internally. Use `sg::load_pz(path, /*stream=*/nullptr, /*keep_host_pinned=*/true)`.
- **k** — desired rank. `auto_select` overwrites `cfg.k_max` with this value before routing.
- **cfg** — `SvdConfig`. Pass `{}` for defaults; populate `nonneg_u`, `L1_u`, `L2_v`, `upper_bound_u`, `graph_u`, etc. only when you need a constrained decomposition. Constraint flags route through deflation transparently.

## Outputs

`SvdResult { U, d, V, status }`:
- `U[m, k]` on device — left singular vectors (gene loadings).
- `d[k]` — singular values, descending.
- `V[n, k]` on device — right singular vectors (cell scores). Multiply by `d` for the standard "PCA embedding" matrix.
- `status` — convergence flag + iteration count.

## Adopt-winner consolidation

Cycle 61 ran the head-to-head: 5 SVD backends × k ∈ {10, 30, 50, 100} × 3 scales. **Deflation wins at every k** (k=50 small: 28 ms vs randomized 148 ms vs Lanczos / IRLBA / Krylov OOM-or-slow). Kept:

| Backend | Status | Notes |
|---|---|---|
| `deflation` | **Primary** | k-independent ~28 ms at small scale; constraint-aware |
| `randomized` | Fallback | only invoked when deflation returns empty (degenerate input) |

Removed (cycle 61): `lanczos`, `irlba_factornet`, `krylov_constrained`. Their factornet headers remain available, but the singlet-gpu adapter does not expose them.

## Complexity

| Scale | k | Wall (V100S) | SOTA wall | Speedup |
|---|---|---|---|---|
| small (10k×30k) | 50 | 28.2 ms | 758.7 ms (scanpy) / 1348.2 ms (factornet CPU) | **27× scanpy** |
| 100k | 30/50/100 | TBD | TBD | TBD |
| 1M | 30/50/100 | TBD | TBD | TBD |

Memory: factornet manages its own scratch; peak usage scales with `m × k + n × k + iter scratch`. cuml unavailable on g001 → no GPU-vs-GPU comparison filed.

## Streaming behavior

Streaming SVD is provided by `factornet::svd::streaming_matvec` paired with our `streaming::PzShardIterator`. Two passes:
1. **Power-method warm-up**: each shard contributes a partial `A^T A v` product; host reduction sums.
2. **Refinement**: deflation operates on the warmed Krylov subspace.

The streaming driver lives in `streaming/` (feature 17). For matrices that fit on one device, the in-memory path is faster.

## Determinism

The deflation backend is **deterministic given a fixed `cfg.seed`**. The randomized fallback path uses `cfg.seed` for the random sketch. Identical seeds + identical inputs → bit-identical output up to fp32 reduction-order tolerance (≤ 1e-6 element-wise on `d`).

## Correctness contract

| Reference | Tolerance | Sample |
|---|---|---|
| scanpy `pp.pca` (svd_solver="arpack") | singular-value relative error ≤ 1e-3 | GSM4037629; ctest 10/10 PASS |
| factornet CPU (`svd::lanczos<float>`) | bit-equivalent to fp32 reduction-order | small synthetic |

Stochastic-mode tolerance for the `randomized` fallback: r ≥ 0.999 on `U @ diag(d)` against the `arpack` reference.

## Citations

- **Randomized SVD**: Halko, Martinsson, Tropp. _Finding structure with randomness._ SIAM Review 53, 217 (2011).
- **Deflation**: standard subspace iteration with successive rank-1 deflation; factornet's implementation (Zach DeBruine, factornet 2026).
- **Adopt-winner consolidation** is a singlet-gpu policy decision (Rule 32 in `agents/singlet-gpu-orchestrator.md`).

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/reduce/svd/auto_select.h>     // until released

int main() {
    namespace sg = singlet_gpu;

    // keep_host_pinned=true is required by the SVD adapter
    auto pz = sg::load_pz("/path/to/exon_counts.1pz",
                          /*stream=*/nullptr,
                          /*keep_host_pinned=*/true);
    cudaStreamSynchronize(pz.producer_stream);

    sg::reduce::svd::SvdConfig cfg{};
    auto svd = sg::reduce::svd::auto_select(pz, /*k=*/50, cfg);

    // svd.U : DenseMatrix(m × 50) — gene loadings
    // svd.d : DenseVector(50) — singular values, descending
    // svd.V : DenseMatrix(n × 50) — cell scores (multiply by d for PCA embedding)
}
```

## Pareto-frontier row

| scale | wall_ms | accuracy | sota_wall_ms | sota_lib | dominates_on |
|---|---|---|---|---|---|
| small-k50 (10k cells) | 28.2 | SV rel err ≤ 1e-3 | 758.7 (scanpy) / 1348.2 (factornet CPU) | scanpy + factornet_cpu | wall (27× vs scanpy_pca) |

Promoted 2026-04-15. Rule-32 consolidation applied 2026-04-16 (Cycle 61): kept deflation + randomized; removed lanczos / irlba / krylov.

## Links

- Design doc: [`state/designs/05-svd-adapter.md`](../../state/designs/05-svd-adapter.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § reduce/svd
- Equivalence notebook: `docs/notebooks/pca.ipynb` (pending)
- Related: [`preprocess_select_hvg.md`](preprocess_select_hvg.md) (HVG output feeds PCA), [`reduce_nmf.md`](reduce_nmf.md) (sister module)
