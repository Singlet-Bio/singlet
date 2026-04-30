---
feature: umap
roadmap_id: 8
module: include/singlet-gpu/embed/umap.h
status: design
tolerance: trustworthiness ≥ 0.95 vs umap-learn CPU on the same kNN graph; kNN-preservation@15 ≥ 0.85
target_perf: 1M cells from precomputed kNN (k=15) → 2D embedding ≤30s on A100 (rapids-singlecell reports 25s for 1.3M cells)
ooc_plan: cuML UMAP loads the full graph and embedding on device; for >10M cells, batch via k-means partitioning (lit-scout's "batched partitioning" trick — defer to cycle ≥ 12)
---

## Algorithm

`embed/umap.h` integrates **cuML UMAP** as the primary backend. Per lit-scout consensus: rapids-singlecell uses it, achieves 350× speedup on 1.3M cells vs Scanpy CPU (52 min → 25s).

Same architectural pattern as cycles 5/6/9: we INTEGRATE, we don't reimplement. cuML is Apache 2.0, GPL-2 compatible.

The adapter takes a `singlet_gpu::graph::KnnResult` (from cycle 8) and produces a 2D (or N-D) embedding. cuML's UMAP API does NOT take CSR directly — it takes `(indices[n*k], distances[n*k])` tuple format. Per lit-scout pitfall #3: "cuML UMAP does NOT take CSR kNN directly in current API—requires (indices, distances) tuple or dense." Our adapter does the trivial conversion (drop `row_offsets`, keep `neighbors[]` and `distances[]` as paired arrays).

### Init mode

- **`UmapInit::Random`** (default per lit-scout pitfall #1: spectral unstable with fixed seed in cuML).
- **`UmapInit::Spectral`** (offered but documented as non-deterministic across runs).

### N components

Default 2 (visualization). Configurable to 3 for 3D plots.

## Numerical stability

- fp32 throughout. lit-scout: "no reported numerical issues vs umap-learn on 1M+ cells."
- cuML's internal optimizer uses fp32 SGD with negative sampling. We pass through unchanged.

## Memory layout

- Input: `KnnResult` (CSR) — `row_offsets[n+1]`, `neighbors[n*k]`, `distances[n*k]` on device.
- cuML allocates internal workspace for the fuzzy 1-skeleton graph and the embedding.
- Output: `core::DeviceMemory<float> embedding(n * n_components)`.
- Total: `4 * n * n_components` bytes for output (1M cells × 2D × 4B = 8 MB) + cuML's internal workspace (~10× input).

## Streams

cuML creates its own GPU context per call. We do NOT pass a `cudaStream_t`. Same pattern as factornet SVD and cuGraph leiden.

## Out-of-core

cuML UMAP loads the full input on device. For >10M cells, the lit-scout cited NVIDIA's batched partitioning via k-means subsampling (153 GB on 80 GB H100). Defer to cycle ≥ 12.

## Determinism

- **Random init + fixed seed**: deterministic across runs.
- **Spectral init**: NOT deterministic per lit-scout pitfall #1 (cuML GitHub issue #6696). Document.
- The SGD optimizer uses random negative sampling — the seed controls this; same seed gives the same trajectory.

## Correctness test spec

Test file: `tests/embed_umap_correctness.cpp`.

1. **Tiny synthetic**: 200 × 10 random fp32 embedding (fixed seed). Run `compute_knn` (Exact, k=15), then `umap(knn, {n_components=2, init=Random, seed=42})`. Compute trustworthiness vs the input embedding using the kNN-preservation metric: for each point, what fraction of its top-15 input neighbors are also in its top-15 output neighbors? Average ≥ 0.85.
2. **Comparison vs umap-learn**: same input. In a Python subprocess, run `umap.UMAP(n_neighbors=15, n_components=2, init='random', random_state=42).fit_transform(embedding)`. Compute trustworthiness on both outputs against the input. Confirm both are ≥ 0.95.
3. **Determinism (random init)**: run our UMAP twice with the same seed. Confirm bit-identical output.
4. **3D embedding**: run with `n_components=3`. Confirm output shape `[n, 3]`.
5. **Edge cases**:
   - n=20 cells, k=5 → small graph, should still produce a valid embedding.
   - n_components=1 → confirm 1D output.
   - n_components > 50 → cuML may error; confirm clean error.

Reference: `umap-learn` Python via subprocess (`pip install umap-learn`).

Tolerance:
- Trustworthiness ≥ 0.95
- kNN-preservation@15 ≥ 0.85
- Determinism: bit-identical with random init + fixed seed

## Target performance

| Scale | Cells | k | Backend | Target wall | SOTA (rapids-singlecell) |
|---|---|---|---|---|---|
| tiny | 200 | 15 | cuML | <50ms | n/a |
| 10k | 11,560 | 15 | cuML | <500ms | ~1s |
| 100k | ~120k | 15 | cuML | <3s | ~5s |
| 1M | ~1M | 15 | cuML | <25s | 25s (matches) |

We will not beat cuml dramatically — UMAP is well-optimized in cuml. Goal: match cuml within 10%.

## Implementation notes (for cycle 10 kernel-dev dispatch)

- Header path: `include/singlet-gpu/embed/umap.h`.
- API:
  ```cpp
  namespace singlet_gpu::embed {
      enum class UmapInit { Random, Spectral };
      struct UmapConfig {
          int n_components = 2;
          int n_epochs = 0;            // 0 = cuml default (200 for ≤10k, 500 for >10k)
          float min_dist = 0.5f;
          float spread = 1.0f;
          float learning_rate = 1.0f;
          UmapInit init = UmapInit::Random;
          uint64_t seed = 0;
          int negative_sample_rate = 5;
      };
      struct UmapResult {
          singlet_gpu::core::DeviceMemory<float> embedding;  // n × n_components
          int n;
          int n_components;
      };
      UmapResult umap(
          const singlet_gpu::graph::KnnResult& knn,
          const UmapConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- Build flag: `FACTORNET_HAS_GPU=1`. cuML dependency: `find_package(cuml)` in CMakeLists.
- Dependencies: cycle 8 (kNN graph).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: cuml::manifold::UMAP` first comment.

## Risks

1. **cuML dependency**: must be installed on the GPU dev node. `find_package(cuml)` in CMake.
2. **cuML UMAP C++ API may differ from Python**: cycle 10's code-reader must verify the exact C++ signature. The Python API takes `(indices, distances)`; the C++ API may want a `raft::handle_t` and explicit device pointers.
3. **Spectral init non-determinism**: documented but not fixed (cuML upstream issue).
4. **`n_epochs` defaults**: cuML auto-selects based on dataset size; we expose the field but pass 0 to use cuml's default.
