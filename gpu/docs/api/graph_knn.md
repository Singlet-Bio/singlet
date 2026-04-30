# `graph::compute_knn` + `graph::compute_snn`

Feature #8. k-nearest neighbour graph on a PCA embedding, plus shared-nearest-neighbour Jaccard pruning. **2.1× sklearn brute-force at small scale.**

Two backends: **Exact** (warp-level top-k via `cub::DeviceSegmentedRadixSort`) for ≤ ~50k cells, and **CAGRA** (RAFT/cuVS proximity-graph index) for 50k+. CAGRA is currently **blocked on cuVS install** (`state/blockers.md` → INFRA-CUVS-CUGRAPH-INSTALL); Exact is shippable today.

## C++ signature

```cpp
namespace singlet_gpu::graph {

enum class KnnBackend { Auto, Exact, Cagra };
enum class DistanceMetric { L2, Cosine, Inner };

struct KnnConfig {
    int            k                 = 15;
    KnnBackend     backend           = KnnBackend::Auto;   // Auto routes by n
    DistanceMetric metric            = DistanceMetric::L2;
    bool           return_squared    = false;              // skip sqrtf — caller gets d²
    uint64_t       seed              = 0;                  // forwarded to CAGRA builder
    // CAGRA-only knobs (ignored for Exact backend)
    int            cagra_graph_degree              = 64;
    int            cagra_intermediate_graph_degree = 128;
    int            cagra_search_width              = 1;
    int            cagra_itopk                     = 0;    // 0 = auto: min(k*5, 512)
};

struct KnnResult {
    core::DeviceMemory<int>   row_offsets;   // [n+1], uniform: row_offsets[i] = i*k
    core::DeviceMemory<int>   neighbors;     // [n*k], ascending by distance per row
    core::DeviceMemory<float> distances;     // [n*k], L2 / cosine / -dot per cfg
    int n;
    int k;
    KnnBackend backend_used;
};

KnnResult compute_knn(const core::DeviceDense& embedding,
                      const KnnConfig& cfg,
                      cudaStream_t stream);

// SNN (shared-nearest-neighbour) Jaccard pruning on top of compute_knn output.
struct SnnConfig {
    float prune_threshold = 1.0f / 15.0f;   // drop edges with Jaccard < this
};

struct SnnResult {
    core::DeviceMemory<int>   row_offsets;   // [n+1] CSR (variable per row after pruning)
    core::DeviceMemory<int>   neighbors;
    core::DeviceMemory<float> jaccard;
    int n;
};

SnnResult compute_snn(const KnnResult& knn,
                    const SnnConfig& cfg,
                    cudaStream_t stream);

}  // namespace singlet_gpu::graph
```

## Python signature (scanpy convention)

The Python wrapper bundles `compute_knn` + `compute_snn` into a single scanpy-style entry point. The result lives in `adata.obsp['connectivities']` and `adata.obsp['distances']`, exactly as scanpy users expect:

```python
import singlet_gpu as sg

# Same name + parameter names as scanpy.pp.neighbors
sg.pp.neighbors(
    adata,
    n_neighbors=15,
    n_pcs=None,                  # use all components in adata.obsm[use_rep]
    use_rep=None,                # default: 'X_pca'
    knn=True,
    method="umap",
    metric="euclidean",
    rng=None,
    key_added=None,
)
# Writes adata.obsp['distances'] (sparse n×n) + adata.obsp['connectivities']
# (Jaccard-pruned SNN graph) + adata.uns['neighbors'].
```

## R signature

```r
singletGpu::neighbors(adata, n_neighbors = 15L, use_rep = "X_pca")
```

## Inputs

- **embedding** — `core::DeviceDense`, row-major, shape `(n_cells, n_pcs)`. Typically the output of `reduce::svd::auto_select` after multiplying `V * d`.
- **cfg.k** — neighbours per cell. 15 is the scanpy default; 30 is common for finer-grained Leiden.
- **cfg.backend**:
  - `Auto` (default): routes by n. Exact for ≤ 50k cells; CAGRA above. Currently routes only to Exact until cuVS lands — explicit `Cagra` requests trigger a `static_assert` until INFRA-CUVS-CUGRAPH-INSTALL resolves (`state/blockers.md`).
  - `Exact`: warp-per-row top-k via `cub::DeviceSegmentedRadixSort`. FLT_MAX self-exclusion eliminates the post-sort filter — zero host↔device traffic in the hot loop (Cycle 49a fix).
  - `Cagra`: RAFT / cuVS proximity-graph index. Build O(n × graph_degree); search O(n × itopk). Order of magnitude faster than Exact above ~50k.
- **cfg.metric** — L2 (default), Cosine (normalize then L2), Inner (negate the dot product so smaller is better; useful for similarity searches).
- **cfg.return_squared** — skip the final `sqrtf` in the L2 path. Faster; caller gets `d²` instead of `d`. Cosine is unaffected.

## Outputs

`KnnResult`:
- `row_offsets[n+1]` — uniform CSR: `row_offsets[i] = i*k`. Stored explicitly for downstream consumers that expect a generic CSR adjacency.
- `neighbors[n*k]` — neighbour indices, ascending by distance within each row.
- `distances[n*k]` — distances. L2 (or L2² if `return_squared`), or Cosine (1 - cos similarity), or `-dot(x_i, x_j)` for Inner.
- `n`, `k`, `backend_used` — host-side scalars; `backend_used` reports the actual route Auto picked.

`SnnResult`:
- Variable-row-length CSR after Jaccard pruning. Rows with fewer than 1 surviving neighbour are not pre-allocated (variable nnz).

## Complexity

| Backend | Scale | Wall (V100S) | SOTA wall | Speedup |
|---|---|---|---|---|
| Exact | small-11k | 59.9 ms | 125.8 ms (sklearn BruteForce) | **2.1×** |
| CAGRA | 100k | TBD | TBD | TBD (cuVS install needed) |
| CAGRA | 1M | TBD | TBD | TBD |

Memory: O(n × k) for the result, plus O(n × n) for the Exact distance matrix on small scales (V100S 32 GB → safe up to ~50k cells at fp32). Above 50k, CAGRA's proximity graph is O(n × graph_degree) — orders of magnitude smaller.

## Streaming behavior

- **Exact**: not streamable in the strict sense — kNN of a cell against shards depends on all distances. Streaming approach: subsample landmarks, compute kNN among landmarks, then approximate kNN of remaining cells via projection. Currently single-shard.
- **CAGRA**: build the proximity graph from a landmark sample, then search remaining cells against it. Pending cuVS install.

## Determinism

- **Exact**: deterministic (cub radix sort is stable). Two runs on the same embedding produce bit-identical results.
- **CAGRA**: best-effort given `cfg.seed`. RAFT/cuVS does not guarantee bit-identical builds across runs even with a fixed seed.

## Correctness contract

| Backend | Reference | Tolerance | Sample |
|---|---|---|---|
| Exact | sklearn `NearestNeighbors(algorithm='brute')` | identical neighbour set; identical distances within fp32 epsilon | small-11k; ctest 9/12 (3 CAGRA tests skipped pending cuVS) |
| CAGRA | cuml `NearestNeighbors` | recall@10 ≥ 0.99 | TBD |

## Citations

- **CAGRA**: H. Ootomo et al. _CAGRA: Highly parallel graph construction and approximate nearest neighbor search for GPUs._ ICDE 2024.
- **SNN Jaccard pruning**: standard Seurat `FindNeighbors` step. R. Stuart et al. _Comprehensive integration of single-cell data._ Cell 177 (2019).
- singlet-gpu's contribution (Cycle 49a): `cub::DeviceSegmentedRadixSort` Exact path with FLT_MAX self-exclusion — eliminates the host-side `partial_sort` round-trip the Cycle-8 implementation had violated.

## Example

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <singlet-gpu/graph/knn.h>             // until released
#include <singlet-gpu/graph/snn.h>
#include <singlet-gpu/reduce/svd/auto_select.h>

int main() {
    namespace sg = singlet_gpu;
    auto pz = sg::load_pz("/path/to/exon_counts.1pz",
                          /*stream=*/nullptr, /*keep_host_pinned=*/true);
    cudaStreamSynchronize(pz.producer_stream);

    /* ... lognorm + HVG ... */

    auto svd = sg::reduce::svd::auto_select(pz, /*k=*/50, {});
    /* construct DeviceDense embedding from svd.V * diag(svd.d) — typical PCA scores */

    sg::graph::KnnConfig kcfg{};
    kcfg.k       = 15;
    kcfg.backend = sg::graph::KnnBackend::Auto;
    kcfg.metric  = sg::graph::DistanceMetric::L2;
    auto knn = sg::graph::compute_knn(/* embedding */ {}, kcfg, pz.producer_stream);

    sg::graph::SnnConfig scfg{};
    scfg.prune_threshold = 1.0f / 15.0f;
    auto snn = sg::graph::compute_snn(knn, scfg, pz.producer_stream);
    cudaStreamSynchronize(pz.producer_stream);

    // snn.row_offsets / .neighbors / .jaccard form the input to graph::leiden.
}
```

## Pareto-frontier row

| backend | scale | wall_ms | sota_wall_ms | sota_lib | dominates_on |
|---|---|---|---|---|---|
| Exact | small-11k | 59.9 | 125.8 | sklearn BruteForce | wall (2.1×) |
| CAGRA | 100k+ | TBD | TBD | cuml / RAFT | TBD (blocked on cuVS) |

Promoted (Exact only) 2026-04-16. Tier-2 CAGRA pending unblock.

## Links

- Design doc: [`state/designs/08-knn.md`](../../state/designs/08-knn.md)
- Frontier entry: [`state/pareto-frontier.md`](../../state/pareto-frontier.md) § graph/knn
- Equivalence notebook: `docs/notebooks/knn_graph.ipynb` (pending)
- Blocker: [`state/blockers.md`](../../state/blockers.md) → INFRA-CUVS-CUGRAPH-INSTALL
- Related: [`reduce_svd.md`](reduce_svd.md) (PCA feeds kNN), `graph_leiden.md` (clusters built on SNN output, pending)
