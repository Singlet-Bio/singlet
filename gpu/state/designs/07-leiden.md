---
feature: leiden
roadmap_id: 7
module: include/singlet-gpu/graph/leiden.h
status: design
tolerance: ARI ≥ 0.95 vs CPU leidenalg on standard scRNA datasets at fixed resolution + seed; modularity within 5% of leidenalg
target_perf: 1M cells × k=15 (15M edges) ≤2s on A100 (rapids-singlecell reports ~14.4s for the full clustering pipeline; the leiden step itself is ~1s); 100k cells ≤200ms; 10k cells ≤30ms
ooc_plan: not streamed — Leiden requires the full graph in memory. Defer billion-cell clustering to feature 16's streaming driver via the `compute_knn → leiden` over chunked subgraphs (deferred to cycle ≥ 12)
---

## Algorithm

`graph/leiden.h` integrates **cuGraph's Leiden** as the primary backend. Per lit-scout consensus: rapids-singlecell uses it, achieves 1958× speedup vs CPU leidenalg on 1.1M cells (7.83h → 14.4s end-to-end for the clustering pipeline). The kernel itself is C++/CUDA in cuGraph; we wrap it.

Same architectural pattern as factornet: we INTEGRATE, we don't reimplement. cuGraph is Apache 2.0, GPL-2-compatible.

Two backends:

1. **`LeidenBackend::CuGraph`** (default): cuGraph's `cugraph::leiden(graph, max_iter, resolution)`. Modularity objective (CPM not exposed in recent cuGraph releases per lit-scout). Expects a CSR graph on device with sorted neighbor lists.

2. **`LeidenBackend::CpuRefine`** (validator-only fallback): runs `leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=res, seed=seed)` in a Python subprocess for correctness validation. NOT user-facing — only the validator uses this to confirm cuGraph's output ARI ≥ 0.95.

### Multi-resolution vectorization

Per lit-scout's "novel trick" finding (rapids-singlecell does this): a single call can return multiple clusterings at different resolutions, amortizing the graph load and the layer-1 modularity computation:

```cpp
std::vector<float> resolutions = {0.3f, 0.5f, 0.7f, 1.0f, 1.5f};
auto results = leiden_multi(knn_result, resolutions, cfg);
// results[0].labels = clustering at res=0.3
// results[1].labels = clustering at res=0.5
// ...
```

The implementation calls cuGraph's leiden once per resolution but reuses the device-resident graph across calls (no re-upload). For 5 resolutions on 100k cells: ~5× the single-resolution runtime, NOT 5× the load time.

### Input

A `singlet_gpu::graph::KnnResult` from cycle 8 (CSR with `row_offsets[n+1]`, `neighbors[n*k]`, `distances[n*k]`). cuGraph wants a weighted graph — we use `weights[i] = exp(-distances[i]² / 2σ²)` where `σ = median(distances)` per lit-scout's standard scRNA convention. Or: pass-through `1.0 / (1.0 + distances²)` for connectivity-style weights.

The conversion from kNN distances to edge weights is exposed via `cfg.weight_function` (enum: `Gaussian`, `Connectivity`, `Inverse`, or `Custom` with a callback).

## Numerical stability

- Modularity computation in fp64 internally (cuGraph's choice). Output labels are `int32`.
- Distance-to-weight transformation in fp32; tied weights resolved by `(weight, neighbor_id)` lexicographic order to make Leiden's tie-breaking deterministic.
- `cfg.seed` forwarded to cuGraph for reproducible move ordering.

## Memory layout

- Input: `KnnResult` from cycle 8 (CSR, fp32 distances, int32 indices).
- Workspace: cuGraph's internal Leiden workspace (~10 × n bytes for the move queue + community labels).
- Output:
  - `labels[n]` int32 — community ID per cell.
  - `n_clusters` int — number of communities.
  - `modularity` float — final modularity score.
- For multi-resolution: an array of N `LeidenResult` structs, sharing the input graph.

Total: ~12n bytes overhead beyond the input graph. For 1M cells: ~12 MB. Negligible.

## Streams

cuGraph creates its own GPU context per call (same pattern as factornet SVD). We do NOT pass a `cudaStream_t`. Document.

## Out-of-core

Not streamed in cycle 9. Leiden requires the full graph in device memory. For the billion-cell case, the workflow is:
1. PCA on a subsample (cycle 7's streaming pipeline + in-memory PCA fallback).
2. kNN on the embedding (cycle 8).
3. Leiden on the resulting graph (this cycle).
4. Project the remaining cells to the nearest cluster centroid (a separate "label assignment" feature, deferred).

This is the rapids-singlecell pattern. Document.

## Determinism

cuGraph Leiden is deterministic given a fixed seed and a sorted CSR input. We sort neighbor lists by `(weight, neighbor_id)` before passing to cuGraph to break ties consistently.

Note: lit-scout pitfall #4 — cuGraph Leiden may **skip the refinement phase** on "well-clusterable" graphs (heuristic), leading to 3.5% lower modularity than CPU leidenalg. We document this as a known limitation and offer the `CpuRefine` validator path for audits.

## Correctness test spec

Test file: `tests/graph_leiden_correctness.cpp`.

1. **Tiny synthetic**: a 200-node graph with 4 known communities of 50 nodes each (block-stochastic graph). Run `leiden(..., resolution=1.0, seed=42)`. ARI vs ground truth ≥ 0.95.
2. **Real PCA + kNN**: load GSM4037629 → lognorm → HVG → PCA (k=50) → kNN (k=15) → Leiden. Compare to scanpy `sc.tl.leiden(adata, resolution=1.0, random_state=42)` on the same input. ARI ≥ 0.90 (allowing for cuGraph's refinement-skip heuristic).
3. **Multi-resolution test**: `leiden_multi(knn_result, {0.3, 0.5, 1.0}, cfg)`. Confirm 3 results returned, each with monotonically decreasing `n_clusters` as resolution decreases.
4. **Deterministic test**: run leiden twice with the same seed. Bit-identical labels.
5. **`cfg.weight_function` test**: confirm `Gaussian` and `Connectivity` produce different but valid clusterings on the same input.

Tolerance:
- Tiny synthetic ARI ≥ 0.95
- Real data ARI ≥ 0.90 (vs scanpy CPU leidenalg)
- Multi-resolution: shape correctness
- Determinism: bit-identical
- Modularity: within 5% of leidenalg

Reference implementation: scanpy's `sc.tl.leiden` (which calls leidenalg) in a Python subprocess. Subprocess command:
```
python -c "
import scanpy as sc, anndata as ad, numpy as np
adata = sc.read_h5ad('input.h5ad')
sc.pp.neighbors(adata, n_neighbors=15)
sc.tl.leiden(adata, resolution=1.0, random_state=42)
np.save('expected_labels.npy', adata.obs['leiden'].cat.codes.values)
"
```

Validator dumps our cluster labels and compares via ARI.

## Target performance

| Scale | Cells | k | Edges | Resolution | Backend | Target wall | SOTA |
|---|---|---|---|---|---|---|---|
| tiny | 200 | 5 | 1k | 1.0 | CuGraph | <1ms | n/a |
| 10k | 11,560 | 15 | 173k | 1.0 | CuGraph | <30ms | scanpy ~5s |
| 100k | ~120k | 15 | 1.8M | 1.0 | CuGraph | <200ms | scanpy ~80s |
| 1M | ~1M | 15 | 15M | 1.0 | CuGraph | <2s | scanpy ~7.8h |
| 1M | ~1M | 15 | 15M | 5 res's | CuGraph | <8s | n/a (we batch) |

Memory: ≤12n bytes overhead.

## API corrections from cycle 8 code-reader

The cuGraph C++ API (branch-25.10) is more involved than originally sketched:

- Function signature (flattened clustering version, simpler for our adapter):
  ```cpp
  template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
  std::pair<size_t, weight_t> cugraph::leiden(
      raft::handle_t const& handle,
      raft::random::RngState& rng_state,            // MUTABLE reference, required
      graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
      std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
      vertex_t* clustering,                          // output, caller-allocated
      size_t max_level = 100,
      weight_t resolution = 1,
      weight_t theta = 1);
  ```
- **Edge weights are MANDATORY in leiden** (the function treats `std::nullopt` as an error) — our adapter must always provide uniform weights if the user did not specify a weight function.
- **No multi-resolution single call** in cuGraph — `leiden_multi` is a singlet-gpu-side loop over `resolutions`, calling `cugraph::leiden` once per resolution. The trick is to keep the `graph_view_t` constructed once and reuse it across calls (which we already document).
- **Random state**: caller constructs a `raft::random::RngState rng(cfg.seed)` and passes it as a mutable reference. Each call advances the RNG.
- **Graph type** is `cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu>` with `raft::device_span<>` row offsets + col indices. Constructing this from our `KnnResult` requires wrapping the device buffers in `raft::device_span<int const>` and `raft::device_span<int const>` for the CSR offsets/neighbors. Non-trivial; defer the construction details to kernel-dev.
- **Output**: `vertex_t* clustering` is caller-allocated. We pre-allocate `core::DeviceMemory<int> labels(n)` and pass `labels.get()`.
- **Return**: `std::pair<size_t, weight_t>` = `(n_levels, modularity)`. Number of clusters is found by computing `*max_element(labels) + 1` post-call.
- **CMake target**: `cugraph::cugraph` (single-GPU).

The `LeidenBackend::CpuRefine` path (validator-only) uses Python `leidenalg` in a subprocess. No change.

## Implementation notes (for cycle 9 kernel-dev dispatch)

- Header path: `include/singlet-gpu/graph/leiden.h`.
- API:
  ```cpp
  namespace singlet_gpu::graph {
      enum class LeidenBackend { CuGraph, CpuRefine };
      enum class LeidenWeight { Gaussian, Connectivity, Inverse };
      struct LeidenConfig {
          float resolution = 1.0f;
          int max_iter = 100;
          uint64_t seed = 0;
          LeidenBackend backend = LeidenBackend::CuGraph;
          LeidenWeight weight_function = LeidenWeight::Connectivity;
          float gaussian_sigma = 0.0f;  // 0 = use median
      };
      struct LeidenResult {
          singlet_gpu::core::DeviceMemory<int> labels;  // n
          int n_clusters;
          float modularity;
          int iterations;
      };
      LeidenResult leiden(
          const singlet_gpu::graph::KnnResult& graph,
          const LeidenConfig& cfg = {},
          cudaStream_t stream = nullptr);

      std::vector<LeidenResult> leiden_multi(
          const singlet_gpu::graph::KnnResult& graph,
          const std::vector<float>& resolutions,
          const LeidenConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- Build flag: `FACTORNET_HAS_GPU=1`. cuGraph is an additional dep — `find_package(cugraph)` in CMakeLists.
- Dependencies: cycle 8 (kNN graph).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: cugraph::leiden + cugraph::louvain` first comment.

## Risks

1. **cuGraph dependency**: must be installed on the GPU dev node. `find_package(cugraph)` in CMake. If unavailable, the leiden backend errors at compile time (no fallback to CPU — clustering at our target scale is unreasonably slow on CPU).
2. **CPM objective unavailable**: cuGraph defaults to modularity. Users wanting CPM must use the `CpuRefine` validator path (slow, audit-only).
3. **Refinement-skip heuristic** in cuGraph Leiden may produce slightly lower-quality clusterings vs leidenalg. Document.
4. **Tied modularity gains** cause minor NMI variance (±2–5% per lit-scout pitfall #1). Set seed for reproducibility.
5. **Weight function choice** matters for downstream interpretability. `Connectivity` (1/(1+d²)) is the scanpy default; document.
