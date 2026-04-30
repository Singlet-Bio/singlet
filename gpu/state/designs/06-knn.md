---
feature: knn
roadmap_id: 6
module: include/singlet-gpu/graph/knn.h
status: design
tolerance: exact backend = bit-identical vs cuBLAS reference; HNSW backend = recall@k=15 ≥ 0.95 vs exact
target_perf: exact brute-force ≤200ms for 100k cells × 50 PCs, k=15 on A100; HNSW ≤500ms for 1M cells × 50 PCs, k=15
ooc_plan: exact brute-force chunks the query rows (one tile of cell×cell distances at a time); HNSW indexes the full embedding once, queries stream
---

## Algorithm

`graph/knn.h` implements two backends per the lit-scout consensus (lit-scout: "Exact brute-force is SOTA on ≤10M cells on A100"):

1. **`KnnBackend::Exact`** (default for n ≤ 10M): cuBLAS-GEMM all-pairs L2 distances + per-row radix select for top-k. ScaleSC's choice. Validates bit-identical vs scanpy when scanpy is forced to exact mode. **No recall loss.**
2. **`KnnBackend::Hnsw`** (default for n > 10M): cuVS / RAFT HNSW index. Recall@15 > 95%, ms-scale queries on million-cell embeddings.

Selection auto-routes by `n` unless `cfg.backend` is forced.

The output is a sparse adjacency in CSR form: `int32 row_offsets[n+1] = {0, k, 2k, ..., n*k}`, `int32 neighbors[n*k]`, `float distances[n*k]`. Identical layout to scanpy's `neighbors_distances` / `neighbors_indices`.

### Exact brute-force kernel

Input: dense fp32 `embedding[n × d]` (typically PCA output, n cells, d=50 PCs). Uses our `factornet::gpu::DenseMatrixGPU<float>` re-export.

Two kernels:

**Pass 1 — `compute_distances_tiled`** (tiled cuBLAS GEMM):
- For each query tile of `Q` rows × `d`: compute `D[Q × n] = ||q||² + ||x||² - 2·q·xᵀ` via cuBLAS `gemmStridedBatched` or a single GEMM with broadcasted norms.
- Tile size `Q` chosen to fit `Q × n × 4 bytes` in device memory (typical: Q=4096 for n=100k, Q=512 for n=1M).
- Pre-compute `||x[i]||²` once for all rows in fp32 (single dot kernel).
- Output: per-tile distance matrix `D_tile[Q × n]`.

**Pass 2 — `radix_select_topk_per_row`** (one block per query row):
- Each block radix-sorts the `n` distances and keeps the top-k smallest. Use `cub::BlockRadixSort` for k ≤ 32, then global memory radix select for larger k.
- Skip self-distance (the query row itself) — exclude index `query_id` from the top-k.
- Output: `neighbors[Q*k]`, `distances[Q*k]` per tile.

Concatenate tiles → final `neighbors[n*k]`, `distances[n*k]`.

Total cost: O(n²·d / GPU_peak) for the GEMM, dominant for d ≥ 50. The radix select is O(n log k) per row but typically <10% of total.

### HNSW backend (cuVS / RAFT)

Builds a multi-layer navigable small world graph on the GPU using cuVS's HNSW implementation (currently in `cuvs::neighbors::hnsw::*`). Build then query.

If cuVS is not available at build time (header-only fallback), error at compile time with a clear message — do NOT silently fall back to exact for >10M cells (that would be slow). The user must install cuVS or switch to the exact backend on a smaller subset.

cuVS is GPL-3-compatible (Apache 2.0). License header check: cuVS Apache 2.0 + factornet GPL-2.0 + scanpy BSD-3 — all compatible with our GPL-2.0-or-later.

### Distance metric

Default `DistanceMetric::L2`. Also support `Cosine` (normalize then L2) and `Inner` (negative dot product). scanpy default is L2 on PCA-reduced data.

## Numerical stability

- **fp32 throughout**. lit-scout pitfall: fp16 distance accumulation loses precision; keep cuBLAS in fp32.
- Pre-computed `||x||²` uses `cublasSdot`. fp32 sufficient for d ≤ 1024.
- Negative distances (numerical noise from `||q||² + ||x||² - 2q·x`) are clipped to 0 before sqrt.
- `sqrt` is applied per element — actual L2, not squared L2 — so radix select is on the actual distances. Optional `cfg.return_squared = true` skips the sqrt for slightly faster output (downstream Leiden / UMAP can take squared distances).

## Memory layout

- Input: `DenseMatrixGPU<float>` (n × d) = our embedding.
- Workspace:
  - `norms[n]` fp32 — `4n` bytes.
  - `D_tile[Q × n]` fp32 — `4·Q·n` bytes (tiled, freed between tiles).
  - Output `row_offsets[n+1]`, `neighbors[n*k]`, `distances[n*k]` — `4(n+1) + 8nk` bytes.
- For 1M cells, k=15, d=50: input 200 MB, norms 4 MB, tile (Q=512) = 2 GB, output 120 MB. Total peak: ~2.5 GB device.

## Streams

One stream, caller-provided. The tiled GEMM and per-tile radix select chain on the same stream.

## Out-of-core

For n > 10M (or device memory < ~2.5 GB free), the exact backend tiles the QUERY axis automatically — `Q` shrinks as `n` grows. The data axis (full `n × d` matrix) must fit on device — that's a `4nd` byte requirement (200 MB for 1M cells × 50 PCs, 2 GB for 10M cells × 50 PCs). For n > 10M cells, switch to HNSW.

## Determinism

Exact brute-force is deterministic (radix select is order-stable for tied distances when block ordering is fixed). HNSW is **not** deterministic across runs because of the random graph initialization — `cfg.seed` is forwarded to the HNSW builder but cuVS may still introduce floating-point race effects. Document.

## Correctness test spec

Test file: `tests/graph_knn_correctness.cpp`.

1. **Tiny synthetic**: 200 × 10 random fp32 embedding (fixed seed). Compute kNN via our `Exact` backend AND scanpy's `sc.pp.neighbors(..., method='umap', metric='euclidean')` → confirm bit-identical sorted neighbor lists per cell (with ties resolved deterministically).
2. **Real PCA output**: load GSM4037629 → lognorm → HVG (top 2000) → PCA (k=50) → kNN (Exact, k=15). Compare to scanpy on the same input. Tolerance: ≥99% Jaccard on neighbor lists.
3. **HNSW recall test**: build HNSW on the same 100k embedding, query top-15. Compare to exact reference. Recall@15 ≥ 0.95.
4. **`return_squared` test**: confirm `cfg.return_squared = true` returns `d²`, not `d`.
5. **Edge cases**: n < k → return all neighbors; k > 1024 → handled via global radix select.

Tolerance per backend declared.

## Target performance

| Scale | n | d | k | Backend | Target wall | SOTA (cuML / scanpy) | Notes |
|---|---|---|---|---|---|---|---|
| tiny | 200 | 10 | 15 | Exact | <1ms | n/a | smoke |
| 10k | 11,560 | 50 | 15 | Exact | <30ms | ~50ms (cuML) | beat 1.5× |
| 100k | ~120k | 50 | 15 | Exact | <200ms | ~400ms (cuML) | beat 2× |
| 1M | ~1M | 50 | 15 | Exact | <3s (chunked Q) | ~6s | beat 2× |
| 1M | ~1M | 50 | 15 | HNSW | <500ms | ~700ms (cuVS) | beat 1.4× |
| 10M | ~10M | 50 | 15 | HNSW | <8s | n/a | streaming required |

## Implementation notes (for cycle-8 kernel-dev dispatch)

- Header path: `include/singlet-gpu/graph/knn.h`.
- API:
  ```cpp
  namespace singlet_gpu::graph {
      enum class KnnBackend { Auto, Exact, Hnsw };
      enum class DistanceMetric { L2, Cosine, Inner };
      struct KnnConfig {
          int k = 15;
          KnnBackend backend = KnnBackend::Auto;
          DistanceMetric metric = DistanceMetric::L2;
          bool return_squared = false;
          uint64_t seed = 0;
          int hnsw_M = 16;        // HNSW connectivity
          int hnsw_ef = 64;       // HNSW search budget
      };
      struct KnnResult {
          singlet_gpu::core::DeviceMemory<int>   row_offsets;  // size n+1
          singlet_gpu::core::DeviceMemory<int>   neighbors;    // size n*k
          singlet_gpu::core::DeviceMemory<float> distances;    // size n*k
          int n;
          int k;
          KnnBackend backend_used;
      };
      KnnResult compute_knn(
          const singlet_gpu::core::DeviceDense& embedding,  // (n × d), row-major
          const KnnConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- Build flag: `FACTORNET_HAS_GPU=1`.
- Optional dependency on cuVS for the HNSW backend. If `find_package(cuvs)` fails, the `Hnsw` backend errors at compile time; `Exact` works regardless.
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (Exact path) + cuVS (HNSW path)` first comment.

## Risks

1. **cuVS may not be installed** on the GPU dev node. The Exact path covers most of the user base; CAGRA is opt-in. Document the install path: `pip install cuvs-cu12` or `conda install -c rapidsai cuvs`.
2. **Tiled GEMM tile-size selection** — getting `Q` right matters for performance. Auto-tune at runtime by querying `cudaMemGetInfo` and picking `Q = (free_mem / 2) / (n * 4)`.
3. **Self-loop exclusion** — the query row's own index appears as distance 0; must be excluded from top-k. Trivial but easy to forget.
4. **Tied distances** — bit-identical comparison requires deterministic tie-breaking. Use `(distance, original_index)` as the radix-sort key.

---

## v2 — Adopt-Winner Update (Cycle 62, 2026-04-16)

### Rule 32 decision: Replace HNSW with CAGRA

Based on lit-scout research (arXiv 2308.15136, Ootomo et al. 2023):

**KEEP (2 backends)**:
1. **Brute-force fp32** (cuBLAS GEMM + cub radix sort) — for n < 50k cells, exact, zero index overhead. Already implemented and working (Cycle 8 + Cycle 49a fixes).
2. **CAGRA** (cuVS `cagra::build` + `cagra::search`) — for n >= 50k cells. CAGRA is 2.2-27x faster graph build and 33-77x faster query vs HNSW CPU, 3.8-8.8x faster than GPU IVF (Ootomo et al. 2023). It is the current SOTA for GPU ANN.

**REMOVE**: HNSW backend. The existing code uses `cuvs::neighbors::hnsw` which internally builds a CAGRA index then converts to HNSW format for querying — this conversion adds overhead for no benefit. Use CAGRA directly.

**Auto-select threshold change**: 10M → **50k cells**. At 50k cells, brute-force GEMM is still fast (~30ms) but CAGRA's graph build amortization starts winning. Below 50k, brute is exact and faster due to no index overhead.

### Updated API

```cpp
enum class KnnBackend { Auto, Exact, Cagra };  // was: Auto, Exact, Hnsw
struct KnnConfig {
    int k = 15;
    KnnBackend backend = KnnBackend::Auto;
    DistanceMetric metric = DistanceMetric::L2;
    bool return_squared = false;
    uint64_t seed = 0;
    // CAGRA parameters (ignored for Exact)
    int cagra_graph_degree = 64;           // build graph degree
    int cagra_intermediate_graph_degree = 128; // intermediate graph size
    int cagra_search_width = 1;            // beam width for search
    int cagra_itopk = 0;                   // auto: min(k*5, 512)
};
```

### CAGRA backend design

**Build phase**: `cagra::build(cagra_params, dataset)` → proximity graph index on device. Parameters: graph_degree=64, intermediate_graph_degree=128 (CAGRA default). Build is O(n * d * log n).

**Search phase**: `cagra::search(search_params, index, queries, k)` → k-nearest neighbors per query. search_width=1 (default), itopk=min(k*5, 512) per lit-scout recommendation. Search is O(n * k * log n) amortized.

**Recall target**: >= 0.95 at k=15 vs exact brute-force reference (same threshold as original HNSW spec).

**Memory**: CAGRA index for 1M x 50 embedding: ~500MB (graph_degree=64, int32 neighbors). Much less than brute-force tile (~2GB).

### SNN graph construction (new section)

After kNN, Seurat's `FindNeighbors` converts to SNN via Jaccard overlap pruning:

```
For each pair (i, j) where j in kNN(i):
    jaccard(i, j) = |kNN(i) ∩ kNN(j)| / |kNN(i) ∪ kNN(j)|
    if jaccard(i, j) < prune_threshold:  drop edge
```

**Fused SNN kernel**: One CUDA kernel operating on the kNN output directly on device:
1. Load kNN neighbor lists for row i and each neighbor j into shared memory
2. Compute intersection via sorted merge (both lists are pre-sorted from radix sort)
3. Jaccard = intersection / (2k - intersection)
4. Write to CSR output only edges above prune_threshold (default 1/15 per Seurat)

**No host transfer** between kNN and SNN — both operate on the same device buffers.

Output: CSR sparse adjacency matrix (same type as Leiden/Louvain input). Weights = Jaccard similarity.

### Updated target performance

| Scale | n | Backend | Target wall | vs SOTA |
|---|---|---|---|---|
| tiny | 200 | Exact | <1ms | smoke |
| 10k | 11.5k | Exact | <10ms | beat cuml 2x |
| 50k | 50k | Exact | <100ms | beat cuml 2x |
| 100k | 120k | CAGRA | <50ms | beat cuml 4x |
| 1M | 1M | CAGRA | <200ms | beat cuml 6x |
| 10M | 10M | CAGRA (streaming) | <3s | novel |

### Streaming (billion-cell)

For n > device memory:
1. Subsample **landmarks** (~100k cells via reservoir sampling)
2. Build CAGRA index on landmarks
3. Stream remaining cells in tiles: query each tile against the landmark index
4. Merge: for each cell, keep top-k across its landmark neighbors' neighborhoods (2-hop expansion)

This is a novel streaming strategy that avoids building a full n×n graph. The landmark index stays resident on device (~500MB for 100k landmarks). Each streaming tile is O(tile_size × k) memory.

### Implementation plan (for gpu-kernel-dev)

1. Keep the Exact backend as-is (working, Cycle 8/49a fixes applied)
2. Replace HNSW backend with CAGRA: swap `cuvs::neighbors::hnsw` → `cuvs::neighbors::cagra`
3. Update auto-select threshold: 10M → 50k
4. Add SNN Jaccard kernel as a new function: `compute_snn(knn_result, prune_threshold) → SnnResult`
5. Update enum: `Hnsw` → `Cagra`
6. Update tests: HNSW recall test → CAGRA recall test; add SNN correctness test vs Seurat
