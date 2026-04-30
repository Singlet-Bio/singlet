---
feature: integration
roadmap_id: 12
module: include/singlet-gpu/integrate/harmony.h, include/singlet-gpu/integrate/bbknn.h
status: design
tolerance: scIB iLISI ≥ 0.85, cLISI ≥ 0.90 vs CPU harmonypy on the same input; ASW ≥ 0.5
target_perf: 1M cells × 50 PCs × 10 batches Harmony ≤25s on A100 (matches rapids-singlecell); BBKNN graph rebuild ≤500ms
ooc_plan: Harmony fits on the dense embedding (n × n_pcs); for >2M cells the streaming PCA workaround applies. BBKNN modifies the kNN graph in place — bounded by n*k edges
---

## Algorithm

Two methods in this cycle (scVI-lite deferred to a future cycle since it requires a full neural-network training infrastructure):

1. **`integrate/harmony.h`** — Harmony (Korsunsky et al. 2019), the consensus default. Iteratively projects PCA cells toward batch-mean centroids while preserving cluster structure via fuzzy k-means soft assignments.
2. **`integrate/bbknn.h`** — Batch-Balanced kNN (Polański et al. 2020). Modifies the kNN graph instead of the embedding: each cell's neighbors come `k_within / n_batches` from each batch separately, balancing cell counts.

scVI-lite is deferred — it requires either wrapping PyTorch (heavyweight) or implementing a small VAE in CUDA (out of scope for one cycle). Add to roadmap as a future feature.

### Harmony (`integrate/harmony.h`)

Per the lit-scout consensus + rapids-singlecell + ScaleSC patterns:

Algorithm (Korsunsky 2019, simplified):

1. **Initial fuzzy clustering**: soft k-means on the input embedding `Z[n × d]`. K cluster centroids in d-dim space. Output: `R[n × K]` soft assignments.
2. **Iterative correction loop** (typical 10 iters):
   a. **Compute batch-mean centroids per cluster**: `centroid[k][b] = Σ_j R[j][k] * (b_j == b) * Z_j / Σ_j R[j][k] * (b_j == b)`.
   b. **Compute global cluster centroid**: `global[k] = Σ_j R[j][k] * Z_j / Σ_j R[j][k]`.
   c. **Correction term per cell**: `correction[j] = Σ_k R[j][k] * (centroid[k][b_j] - global[k])`.
   d. **Apply**: `Z_j ← Z_j - correction[j]` (or with a learning rate).
   e. **Update soft assignments** `R` by re-running soft k-means on the corrected `Z`.
   f. **Convergence check**: if max relative change in `R` < tol, stop.
3. **Output**: corrected embedding `Z'[n × d]`, ready for downstream kNN/Leiden/UMAP.

GPU implementation:
- Soft k-means via cuBLAS `sgemm` for distances + custom softmax kernel.
- Centroid computation via `cub::DeviceSegmentedReduce` keyed by (cluster, batch) pairs — NOT atomicAdd (cycle 8 lesson).
- Correction kernel: one block per cell.
- Convergence via `cub::DeviceReduce::Max` on the per-cell `|R_new - R_old|`.

Lit-scout's "label-vector encoding" trick: instead of one-hot `n × n_batches` matrix (sparse but big), store `int batch_id[n]` and use it as an index in the centroid kernel. Saves O(n × n_batches) memory.

### BBKNN (`integrate/bbknn.h`)

Per Polański et al. 2020:

Algorithm:
1. Take the input kNN graph (from cycle 8, k=15) — but it was built on the full embedding without batch awareness.
2. For each cell j with batch b_j:
   - Replace its k neighbors with: `k_within / n_batches` neighbors from each batch separately.
   - Use the same Exact brute-force kNN backend, but masked per batch.
3. Output: a NEW `KnnResult` with the rebalanced graph.

Implementation:
- Per-batch sub-call to `compute_knn` masking out cells from other batches.
- Concatenate per-batch neighbor lists per cell.
- The cuBLAS GEMM tile structure from cycle 8 is reused; just per-batch.

Cost: `n_batches × cost(per_batch_knn)`. For 10 batches, ~10× the single-batch kNN cost. Manageable at 100k cells.

## Numerical stability

- Harmony: fp32 throughout. Soft k-means with stable softmax (max-subtraction). Convergence criterion uses max relative change in soft assignments.
- BBKNN: same fp32 issues as cycle 8 kNN.
- Both deterministic given fixed seeds (Harmony's k-means init takes a seed).

## Memory layout

- Harmony: `Z[n × d]` (in-place corrected) + `R[n × K]` (soft assignments) + `centroid[K × n_batches × d]` (per-cluster per-batch centroids) + `global[K × d]`.
  - For 1M cells × 50 PCs × K=20 clusters × 10 batches: 200 MB embedding + 80 MB R + 40 KB centroids + 4 KB global. Fine.
- BBKNN: same as cycle 8 kNN but per-batch.

## Streams

One stream, caller-provided.

## Out-of-core

Harmony fits on the full embedding. For n_cells > 2M the same in-memory PCA fallback applies (cycle 7 streaming pipeline). BBKNN per-batch sub-calls are independent — could stream one batch at a time.

## Determinism

- Harmony: deterministic with fixed `cfg.seed` for the k-means init.
- BBKNN: inherits cycle 8 kNN determinism.

## Correctness test spec

Tests:
- `tests/integrate_harmony_correctness.cpp`
- `tests/integrate_bbknn_correctness.cpp`

Reference: harmonypy + bbknn Python packages via subprocess.

Test cases (Harmony):
1. **`Harmony_TwoBatch_iLISI_Improvement`**: synthetic dataset with 2 batches and known biology. Compute iLISI on the input and on the Harmony-corrected output. Confirm iLISI improves (closer to 1.0 = perfect mixing).
2. **`Harmony_VsHarmonypy`**: same input, run our Harmony and harmonypy. Compare corrected embeddings via canonical correlation analysis (CCA score ≥ 0.95).
3. **`Harmony_PreservesBiology_cLISI`**: confirm cLISI (cell-type local inverse Simpson) does NOT degrade by more than 5% post-correction.
4. **`Harmony_Convergence_FixedSeed`**: same seed → bit-identical (or very close) corrected embedding.
5. **`Harmony_OvercorrectionGuard`**: massively imbalanced batches (90% / 10%). Confirm the correction does NOT collapse the minority batch into the majority.

Test cases (BBKNN):
1. **`BBKNN_TwoBatch_NeighborBalance`**: confirm each cell's k neighbors are balanced across batches (`k / n_batches` per batch).
2. **`BBKNN_VsPython`**: compare to bbknn Python.
3. **`BBKNN_LeidenDownstream`**: run Leiden on the BBKNN graph and confirm cluster ARI ≥ 0.80 vs the unintegrated Leiden (which is biased by batches).

Tolerances:
- iLISI improvement ≥ 0.1
- cLISI degradation ≤ 5%
- CCA score with harmonypy ≥ 0.95
- BBKNN balance: each cell has `k/n_batches ± 1` neighbors per batch
- Determinism: bit-identical with fixed seed

## Target performance

| Scale | Cells | PCs | Batches | Method | Target wall | SOTA |
|---|---|---|---|---|---|---|
| 10k | 11,560 | 50 | 5 | Harmony | <300ms | rapids ~500ms |
| 100k | ~120k | 50 | 5 | Harmony | <2s | rapids ~3s |
| 1M | ~1M | 50 | 10 | Harmony | <25s | rapids ~25s (matches) |
| 100k | ~120k | 50 | 5 | BBKNN | <500ms | bbknn Python ~5s |

## Implementation notes

- Headers: `integrate/harmony.h` (~500 LOC) + `integrate/bbknn.h` (~300 LOC) + `integrate/types.h` (~50 LOC).
- API:
  ```cpp
  namespace singlet_gpu::integrate {
      struct HarmonyConfig {
          int n_clusters = 20;          // soft k-means K
          int max_iter = 10;
          float tol = 1e-4f;
          float lambda = 1.0f;          // ridge regularization
          int kmeans_init_iter = 10;
          uint64_t seed = 0;
      };
      struct HarmonyResult {
          singlet_gpu::core::DeviceMemory<float> corrected;  // n × d, in-place possible
          int n_iters_used;
          float final_obj;
      };
      HarmonyResult harmony(
          const singlet_gpu::core::DeviceDense& embedding,
          const singlet_gpu::core::DeviceMemory<int>& batch_labels,
          int n_batches,
          const HarmonyConfig& cfg = {},
          cudaStream_t stream = nullptr);

      struct BbknnConfig {
          int k_within = 3;             // neighbors per batch (total k = k_within * n_batches)
          int approx_threshold = 100000; // n above which to use HNSW
      };
      singlet_gpu::graph::KnnResult bbknn(
          const singlet_gpu::core::DeviceDense& embedding,
          const singlet_gpu::core::DeviceMemory<int>& batch_labels,
          int n_batches,
          const BbknnConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- Build flag: `FACTORNET_HAS_GPU=1`.
- Dependencies: cycle 1 (core), cycle 8 (kNN), cycle 4 (PCA — for the embedding input).

## Risks

1. **Harmony convergence on extreme batches**: imbalanced batches may not converge. Apply a max_iter cap and document.
2. **Soft k-means init sensitivity**: deterministic seeded init helps but the local optimum may differ from harmonypy's.
3. **scIB metric implementations**: iLISI/cLISI/ASW/kBET need reference implementations. Use the Python `scib` package via subprocess for the metrics, not in our C++ test.
4. **scVI-lite deferral**: document why and add to roadmap.
