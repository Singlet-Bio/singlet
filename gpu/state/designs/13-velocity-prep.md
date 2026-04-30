---
feature: velocity_prep
roadmap_id: 13
module: include/singlet-gpu/preprocess/velocity_prep.h
status: design
tolerance: per-gene γ Spearman ρ ≥ 0.95 vs scVelo CPU on the same input; smoothed moments rel_err ≤ 1e-3
target_perf: 100k cells × 30k genes (full velocity prep) ≤2s on A100; 1M cells ≤20s
ooc_plan: per-cell moments stream via PzDataLoader chunks; gene-level γ fit runs once at the end
---

## Why this is unique to us

This is the **first** of three roadmap features (13, 14, 15) that exploit singlify outputs no other library reads:

- singlify emits both `exon_counts.1pz` (spliced) AND `intron_counts.1pz` (unspliced) per cell directly from the alignment, **no extra STARsolo --soloFeatures Gene Velocyto step required**. Every other velocity pipeline currently re-runs the alignment with the velocyto/STARsolo flag to get those matrices. We get them for free.
- This is why feature 13 is unique: the prep is **zero-friction**. Users plug a single `.1pz` directory into our adapter and get full velocity inputs out, ready to feed into scVelo, veloVI, or DeepVelo.
- The cycle 16 (`anno/mt_lineage.h`) and cycle 17 (`de/donor_pseudobulk.h`) features have the same property: they exploit `mt_alleles.1pz` and `donor_assignments.tsv` respectively — outputs only singlify writes.

## Algorithm

`preprocess/velocity_prep.h` computes the prerequisites for any downstream velocity model (scVelo / veloVI / DeepVelo / Cell2fate):

1. **Load both matrices**: spliced `S[m × n]` and unspliced `U[m × n]` from `.1pz`. Both via `pz_device_loader`.
2. **Filter genes**: per lit-scout pitfall, filter out genes with `total_S < min_S_count` (default 10) or `total_U < min_U_count` (default 5). These are too noisy for γ fitting.
3. **kNN moments smoothing**: for each cell, replace its spliced/unspliced values with the k-NN-weighted average across its neighbors. This is scVelo's standard preprocessing — uses the same kNN graph from cycle 8.
4. **Steady-state γ regression** per gene: fit `U_smoothed = γ_g * S_smoothed + ε` per gene via simple linear least squares (with intercept = 0 per scVelo convention). The fit uses the top-N highest-expression cells per gene to avoid dropout bias.
5. **Compute velocity vectors**: `v_g = U_smoothed - γ_g * S_smoothed` per (gene, cell). Optional output.
6. **Output a `VelocityPrep` struct** with all the components.

### Steady-state γ kernel

For each gene g (one warp per gene):
- Sort cells by `S[g, j]` descending.
- Take top-N (default N = 5% of cells with `S > 0`).
- Compute `γ_g = sum(S * U) / sum(S²)` over those top-N cells (closed-form linear regression through origin).
- Variance estimate: residual `Σ(U - γ S)² / (N - 1)` for downstream uncertainty.

### kNN moments smoothing kernel

Input: kNN graph from cycle 8 (CSR), original `S[m × n]`, `U[m × n]`.

For each cell j:
- Read its k neighbors from `KnnResult.neighbors[j*k .. j*k+k]`.
- Compute Gaussian weights from the cycle-8 distances: `w_i = exp(-d²/2σ²)`.
- Weighted average: `S_smoothed[g, j] = Σ_i w_i * S[g, i] / Σ_i w_i`. Same for `U`.

This is sparse-times-dense: walk each cell column in CSC, gather neighbor columns. Cost: O(nnz_cell × k). Per cell parallelized.

### Velocity vectors

`v_g = U_smoothed - γ_g * S_smoothed` per (gene, cell). Stored as a dense `m × n` matrix (or sparse, if most are near zero — start with dense).

## Numerical stability

- fp32 throughout. Lit-scout: standard for velocity prep.
- Min-count filter prevents division by zero in γ fit.
- Top-N filtering prevents dropout cells from biasing the regression.
- Welford for `S²` accumulation (cycle 4 pattern).

## Memory layout

- Inputs: `S` and `U` (both CSC, m × n).
- Workspace:
  - `S_smoothed[m × n]` and `U_smoothed[m × n]` — DENSE (scVelo's smoothed moments are dense). For 1M cells × 30k genes: 240 GB. **TOO BIG**.
  - **Tile by genes**: process gene chunks of `g_tile = 1024` rows at a time. Per tile: 1024 × 1M × 4 bytes = 4 GB. Still too big.
  - **Tile by both**: `g_tile = 1024`, `c_tile = 100k`. Per double-tile: 1024 × 100k × 4 = 400 MB. Acceptable.
  - Per-gene γ + variance + filter mask: `4 * m * 4` bytes = 480 KB at m=30k. Tiny.
- Output: `VelocityPrep` struct with γ, γ_se, S_means, U_means, filter_mask. Optional: `v[m × n]` velocity matrix (gene × cell, dense).

## Streams

One stream, caller-provided. Tile loop runs sequentially.

## Out-of-core

`PzDataLoader` chunks: per chunk, accumulate per-gene `(sum_S, sum_U, sum_S², sum_S*U, n_cells_with_S>0)` partial moments. Welford-merge across chunks. Final γ fit runs once on the global per-gene moments.

The kNN smoothing is harder to stream — it requires neighbors which span chunks. Two options:
(a) Build the full kNN graph first (cycle 7's in-memory PCA fallback), then stream the smoothing.
(b) Skip smoothing for streaming mode, use raw moments. Less accurate but simpler.

Default: option (a). Document option (b) as a fast path.

## Determinism

Deterministic by construction. Welford merge across chunks is deterministic given fixed chunk order.

## Correctness test spec

Test: `tests/preprocess_velocity_prep_correctness.cpp`.

Reference: scVelo Python (`pip install scvelo anndata scanpy`) via subprocess.

Test cases:
1. **`VelocityPrep_TinyPlanted_VsScvelo`**: synthetic 200 cells × 100 genes with planted γ values (some genes high γ, some low). Run our `velocity_prep` and `scvelo.pp.moments` + `scvelo.tl.velocity` on the same input. Compare per-gene γ: Spearman ρ ≥ 0.95.
2. **`VelocityPrep_GSM4037629_RealData`**: load GSM4037629 spliced + unspliced. Run prep. Confirm: γ values are mostly in [0, 5], filter_mask is meaningful, no NaNs, top genes by γ are biologically sensible (mitochondrial genes typically have high γ).
3. **`VelocityPrep_MinCountFilter_FiltersLowExpression`**: gene with `total_S < 10` is filtered out (`filter_mask[g] = false`).
4. **`VelocityPrep_MomentsSmoothing_PreservesShape`**: confirm smoothed S/U matrices have the same shape as input.
5. **`VelocityPrep_VelocityVectorsSign`**: cells with high U/S ratio have positive velocity; cells with low have negative. Spot-check a few.
6. **`VelocityPrep_Determinism_BitIdentical`**.

Tolerance:
- γ Spearman ρ ≥ 0.95
- moments rel_err ≤ 1e-3
- shape correctness
- determinism: bit-identical

## Target performance

| Scale | Cells | Genes | Wall (target) | scVelo CPU |
|---|---|---|---|---|
| 10k | 11,560 | 30k | <100ms | ~30s |
| 100k | ~120k | 30k | <2s | ~5min |
| 1M | ~1M | 30k | <20s | OOM / hours |

## Implementation notes

- Header path: `include/singlet-gpu/preprocess/velocity_prep.h` (~600 LOC budget).
- API:
  ```cpp
  namespace singlet_gpu::preprocess {
      struct VelocityPrepConfig {
          int min_S_count = 10;
          int min_U_count = 5;
          int top_n_quantile = 5;     // top 5% cells per gene for γ fit
          bool smooth_moments = true;
          float gaussian_sigma = 0.0f;  // 0 = use median distance
          bool compute_velocity = true;
          int gene_tile = 1024;
          int cell_tile = 100000;
      };
      struct VelocityPrepResult {
          singlet_gpu::core::DeviceMemory<float> gamma;          // m
          singlet_gpu::core::DeviceMemory<float> gamma_se;       // m
          singlet_gpu::core::DeviceMemory<float> S_mean;         // m
          singlet_gpu::core::DeviceMemory<float> U_mean;         // m
          singlet_gpu::core::DeviceMemory<uint8_t> filter_mask;  // m (1 = passes filter)
          singlet_gpu::core::DeviceMemory<float> velocity;       // m × n, optional
          int n_genes_passing_filter;
      };
      VelocityPrepResult velocity_prep(
          const singlet_gpu::core::DeviceCSC& spliced,
          const singlet_gpu::core::DeviceCSC& unspliced,
          const singlet_gpu::graph::KnnResult* knn_for_smoothing = nullptr,
          const VelocityPrepConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- Build flag: `FACTORNET_HAS_GPU=1`.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 8 (kNN for smoothing).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (intron-aware velocity prep — first to exploit singlify's exon+intron output)` first comment.

## Risks

1. **Memory tile sizing**: 240 GB at full scale forces double tiling (gene + cell). Get the tile dimensions right.
2. **Smoothing kernel cost**: O(nnz × k) is expensive for dense smoothing. Use shared-memory tiles for the per-cell neighbor gather.
3. **scVelo comparison may diverge**: scVelo uses different defaults (n_neighbors=30, perplexity, etc.). Match its defaults in the test config.
4. **Velocity vector storage**: dense `m × n` is 240 GB at 1M × 30k × 4. Store sparse-bytemap or compress on the fly. Alternatively, emit per-cell only the top-K most variable velocity components.
