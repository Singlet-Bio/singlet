---
feature: mt_lineage
roadmap_id: 14
module: include/singlet-gpu/anno/mt_lineage.h
status: design
tolerance: clone label ARI ≥ 0.85 vs MQuad on the Lareau 2019 dataset; per-site VAF rel_err ≤ 1e-4 vs maegatk
target_perf: 100k cells × 16569 MT positions × 5 clones in ≤500ms on A100; 1M cells in ≤5s
ooc_plan: per-site VAF computation streams via PzDataLoader; clone clustering runs on the small per-cell heteroplasmy summary (n × n_informative_sites)
---

## Why this is unique to us (and unique to the field)

This is the SECOND "unique to us" feature exploiting singlify's `mt_alleles.1pz` output, AND the SECOND "first GPU implementation" in singlet-gpu (after cycle 13 GSEA). Per lit-scout: **no GPU implementation of MT heteroplasmy clone calling exists**. MQuad, mgatk/maegatk, MAESTER are all CPU/Python.

The "unique to us" angle: singlify writes `mt_alleles.1pz` directly during alignment via the pileup pass (per the catalog orchestrator's outputs). Every other MT lineage pipeline requires re-running the BAM through `mgatk` or `maegatk`, which takes hours per sample. We get the matrix for free.

## Algorithm

`anno/mt_lineage.h` implements binomial mixture clone calling on heteroplasmy profiles per cycle 15 lit-scout's MQuad consensus.

Pipeline:

1. **Load `mt_alleles.1pz`** via `pz_device_loader`. Shape: `n_cells × n_mt_sites_max` (where `n_mt_sites_max = 16569` for human mtDNA). Values are alt-allele counts. We also need depth per site per cell — this comes either from a paired `mt_depth.1pz` (if singlify writes it) or computed from a homozygous-reference call.
2. **Per-site filter**:
   - Depth filter: keep sites with median depth ≥ `min_depth` (default 10 reads).
   - VAF filter: keep sites where at least `min_cells_alt` (default 5) cells have VAF > `min_vaf` (default 1%).
   - Strand concordance filter: per lit-scout pitfall #2, skip sites where alt reads are strand-imbalanced. (Requires per-strand counts — if `.1pz` only stores combined counts, skip this filter and document.)
3. **Compute per-cell per-site VAF**: `vaf[i, j] = alt[i, j] / (alt[i, j] + ref[i, j])`. Sparse — most cells × site pairs are zero.
4. **Binomial mixture per site** (the MQuad core): for each informative site, fit a 2-component binomial mixture (`k=clone_with_variant` vs `k=other`). The mixture parameters are `(p_alt | clone), (p_alt | other), π_clone`. Estimate via EM.
5. **Cell-level clone assignment**: Gaussian mixture on the per-cell heteroplasmy profile (n_cells × n_informative_sites). The number of clones K is auto-detected via BIC over k=2..10.
6. **Output**: per-cell clone label + clone consensus heteroplasmy profile + clone-defining sites.

### EM kernel for binomial mixture

Per site (one block per site, parallel across sites):
- E-step: compute responsibilities `γ[i] = π_clone * P(x_i | p_clone) / (π_clone * P(x_i | p_clone) + (1-π_clone) * P(x_i | p_other))`.
- M-step: update `p_clone = Σ_i γ[i] * x_i / Σ_i γ[i] * depth_i`, similarly for `p_other`.
- Convergence: relative change in log-likelihood < 1e-5.

Each site's EM is independent → embarrassingly parallel across sites.

### Gaussian mixture clustering (cell-level)

The per-cell heteroplasmy profile is `n × n_informative_sites` (often n_informative_sites < 100, so it's a small dense matrix even at 1M cells = 400 MB).

GMM via EM:
- E-step: per cell, compute responsibility for each cluster k.
- M-step: update cluster means + covariances.
- Convergence: log-likelihood relative change.

For K candidates {2..10}, run separately. Pick the K with the lowest BIC.

### Determinism and seeds

Both EM (binomial and GMM) are deterministic given a fixed init. Use a seeded k-means++ init for the GMM, seeded random init for the binomial.

## Numerical stability

- fp32 throughout.
- Log-likelihood in fp64 to avoid underflow on very small probabilities.
- Beta-Binomial overdispersion is NOT modeled (lit-scout note: standard binomial is the MQuad default). Add as a future cycle if accuracy is insufficient.
- Underflow handling: use the log-sum-exp trick for posterior normalization.

## Memory layout

- Input: `DeviceCSC` for alt counts (n_cells × n_mt_sites). Add a paired `DeviceMemory<int>` for total depth per cell per site if available.
- Per-site VAF: dense `n_informative_sites × n_cells` (typical: 50 × 1M = 200 MB).
- Per-cell heteroplasmy profile (post-filter): `n_cells × n_informative_sites` = 200 MB.
- GMM workspace: `n_cells × K` responsibilities (4 MB at 1M × 10).
- Output: `int clone_label[n_cells]` + clone consensus profiles.

## Streams

One stream, caller-provided. Per-site EM kernels run in parallel; cell GMM runs sequentially.

## Out-of-core

Per-site VAF computation streams via `PzDataLoader` chunks (per chunk, accumulate per-site sums). Clone clustering runs once on the full per-cell heteroplasmy profile (small enough to fit in memory: 200 MB at 1M cells × 50 sites).

## Determinism

Deterministic with fixed `cfg.seed`. Both EM steps are deterministic given fixed init.

## Correctness test spec

Test: `tests/anno_mt_lineage_correctness.cpp`.

Reference: MQuad Python (`pip install MQuad`) via subprocess.

Test cases:
1. **`MtLineage_TinySynthetic_VsMQuad`**: synthetic mixture of 3 clones with planted heteroplasmy patterns. Run our `mt_lineage` and MQuad. Compare clone label ARI ≥ 0.85.
2. **`MtLineage_GSM4037629_RealData`**: load GSM4037629 mt_alleles.1pz. Run prep. Confirm: clone count ∈ [1, 10], no NaNs. (Real data may not have multi-clone signal — accept K=1 as valid.)
3. **`MtLineage_DepthFilter_RemovesLowCoverage`**: site with depth < 10 → filtered.
4. **`MtLineage_BICAutoK_PicksKnownK`**: synthetic with 4 clones. BIC picks K=4 ± 1.
5. **`MtLineage_Determinism_BitIdentical`**: bit-identical with fixed seed.

Tolerances:
- Clone ARI ≥ 0.85
- VAF rel_err ≤ 1e-4
- BIC K-selection ± 1 of true K
- Determinism: bit-identical

## Target performance

| Scale | Cells | MT sites | Clones | Wall (target) |
|---|---|---|---|---|
| 10k | 11,560 | ~50 informative | 5 | <50ms |
| 100k | ~120k | ~50 | 5 | <500ms |
| 1M | ~1M | ~50 | 5 | <5s |

vs MQuad CPU on the Lareau 2019 dataset: ~30 minutes for 10k cells. Target speedup: 200×–1000×.

## Implementation notes

- Header path: `include/singlet-gpu/anno/mt_lineage.h` (~600 LOC budget).
- API:
  ```cpp
  namespace singlet_gpu::anno {
      struct MtLineageConfig {
          int min_depth = 10;
          int min_cells_alt = 5;
          float min_vaf = 0.01f;
          int max_em_iters = 100;
          float em_tol = 1e-5f;
          int min_K = 2;
          int max_K = 10;
          uint64_t seed = 0;
      };
      struct ClonePrediction {
          singlet_gpu::core::DeviceMemory<int>   labels;          // n_cells
          int n_clones;
          singlet_gpu::core::DeviceMemory<int>   informative_sites;  // n_informative_sites
          singlet_gpu::core::DeviceMemory<float> clone_profiles;     // n_clones × n_informative_sites
          singlet_gpu::core::DeviceMemory<float> per_cell_heteroplasmy; // n_cells × n_informative_sites
      };
      ClonePrediction call_clones(
          const singlet_gpu::core::DeviceCSC& alt_counts,
          const singlet_gpu::core::DeviceCSC& depth_counts,  // total depth per (cell, site)
          const MtLineageConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- Build flag: `FACTORNET_HAS_GPU=1`.
- Dependencies: cycle 1 (core), cycle 2 (loader).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU MT lineage tracing — exploits singlify mt_alleles.1pz directly)` first comment.

## Risks

1. **Depth matrix may not exist**: singlify might only write `mt_alleles.1pz` (alt counts) without a paired depth file. If so, depth must be computed from a separate `mt_depth.1pz` or inferred from the global pileup. Check at first GPU dispatch.
2. **Strand concordance filter requires per-strand counts** which singlify may not split out. Skip if unavailable, document.
3. **MQuad reference subprocess install** is non-trivial (depends on cellsnp-lite). May fail on the test node.
4. **K-selection robustness**: BIC can prefer over-fitting. Document and possibly add a min cluster size constraint (`≥ 50 cells per clone`).
5. **Real GSM4037629 may show K=1** (no clonal signal in a healthy donor PBMC). That's valid; the test must accept K=1.
