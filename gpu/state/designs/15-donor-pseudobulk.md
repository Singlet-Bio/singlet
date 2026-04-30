---
feature: donor_pseudobulk
roadmap_id: 15
module: include/singlet-gpu/de/donor_pseudobulk.h
status: design
tolerance: per-gene LFC Spearman ρ ≥ 0.97 vs DESeq2 on the same pseudobulk; q-value rank Spearman ρ ≥ 0.95
target_perf: 1M cells × 30k genes × 10 clusters × 50 donors pseudobulk + NB GLM ≤2s on A100 (DESeq2 CPU is ~5min for the same)
ooc_plan: pseudobulk aggregation streams via PzDataLoader; NB GLM runs on the small dense pseudobulk count matrix (n_genes × n_clusters × n_donors)
---

## Why this is unique to us

This is the **THIRD and FINAL "unique to us"** feature, exploiting singlify's `donor_assignments.tsv` output. No other pipeline integrates donor demultiplexing into the DE step automatically — they all require a separate vireoSNP / souporcell / freemuxlet run with a fragile glue script. We get the donor labels for free from the singlify SNP-based demultiplexing pass.

Per cycle 16 lit-scout: NO GPU NB-GLM with random effects exists. dreamlet (CPU LMM) is the consensus. Our cycle 17 contribution is the **first GPU NB-GLM for donor-aware pseudobulk DE**, third in the singlet-gpu "first GPU" series after cycle 13 GSEA and cycle 16 MT lineage.

## Algorithm

`de/donor_pseudobulk.h` implements the muscat / dreamlet pseudobulk pattern, GPU-accelerated:

1. **Load** the count matrix + cluster labels (cycle 9 Leiden output) + donor labels (from singlify `donor_assignments.tsv` — loaded host-side and uploaded as `int donor_id[n_cells]`).
2. **Pseudobulk aggregation**: per gene, sum counts across cells in each (cluster × donor) group. Output: dense `[n_genes × (n_clusters × n_donors)]` matrix. Sum aggregation per lit-scout's recommendation (NOT mean — preserves count distributions for NB).
3. **Pseudobulk filter**: drop (cluster × donor) cells with `< min_cells_per_pseudobulk` (default 10) — too few cells to estimate dispersion.
4. **NB GLM per gene per cluster**: fit `count ~ donor + intercept` with negative binomial likelihood. Donor is the fixed effect with `n_donors - 1` parameters.
5. **Dispersion estimation**: per-gene NB dispersion via Cox-Reid adjusted profile likelihood (DESeq2's approach). Shrinkage via empirical Bayes prior fit across genes.
6. **apeglm-style LFC shrinkage**: Bayesian moderation of the per-gene log2 fold change, using a Cauchy prior fit on the unshrunk LFCs.
7. **Output**: per-gene per-cluster log2 fold change, p-value, BH-adjusted q-value.

### Pseudobulk aggregation kernel

The trick: build a per-cell "group key" `key[j] = cluster_id[j] * n_donors + donor_id[j]`, then use `cub::DeviceSegmentedReduce::Sum` over the count matrix with these keys. The segments are not contiguous in the original cell order, so first run `cub::DeviceRadixSort::SortPairs` on `(key, cell_index)` to bring same-group cells together.

After sort, use `cub::DeviceSegmentedReduce::Sum` to accumulate per-gene counts into the pseudobulk matrix. Pure device-side — no atomicAdd, no host trips.

Cost: O(nnz) for the sum + O(n log n) for the sort. Memory: O(n) for sort temp + O(m × n_clusters × n_donors) for the output. For 30k × 10 × 50: 60 MB. Fine.

### NB GLM kernel (the hard part)

For each gene g and cluster c independently (parallel across all g × c pairs):
- Extract the per-donor pseudobulk vector `y[d] = pseudobulk[g, c, d]` for d=1..n_donors.
- Fit NB GLM: `y ~ NB(μ_d, α)` where `log(μ_d) = β_0 + β_donor[d]` and `α` is the dispersion.
- Solve via IRLS (Iteratively Reweighted Least Squares):
  - Init β with method-of-moments.
  - Each IRLS iter: compute `W[d] = μ_d / (1 + α * μ_d)`, solve weighted LSQ via cuSOLVER `Sgels` for the donor effects.
  - Convergence: scalar fp32 readback of max |Δβ| (ONE per outer iter, ≤4 bytes — within the rule preamble exception).
  - Max iters: 50.
- Fit dispersion `α` via Cox-Reid profile likelihood: maximize `ℓ_CR(α) = ℓ(α | β̂(α))` via Brent's method or Newton on log(α). Each step requires one IRLS refit, but n_iters ≤ 10.
- Apply empirical Bayes shrinkage: pool dispersion estimates across genes, fit a meta-prior, and posterior-shrink each per-gene α toward the prior mean.
- Compute per-donor log2 fold change vs intercept: `lfc[d] = β_donor[d] / log(2)`.
- Apeglm shrinkage: pool LFCs across genes per cluster, fit a Cauchy prior, MAP-shrink the per-gene LFCs.
- p-value via Wald test: `z = β / SE`, `p = erfcf(|z| / sqrt(2))`.
- BH adjust per cluster.

This is a lot of compute, but the per-(gene, cluster) work is independent and embarrassingly parallel. With n_genes_passing × n_clusters work units ~ 30k × 10 = 300k independent NB GLM fits, A100's 108 SMs each running ~3000 fits in parallel = full saturation.

### Per-gene per-cluster parallelism

One block per (gene, cluster) pair. Block uses shared memory for the donor-effect Hessian (n_donors × n_donors fp32, typical 50 × 50 = 10 KB). cuSOLVER per-block solve uses block-level QR (`gels` is sequential; we do a small custom block-level Gauss-Jordan for n_donors ≤ 100).

For n_donors > 100: fall back to per-gene cuSOLVER `Sgels` with one stream-callback per gene. Document.

## Numerical stability

- fp32 throughout for counts and intermediate `μ`, `W`.
- fp64 accumulators for the per-gene Hessian and Cox-Reid log-likelihood (small, ≤k² Gram).
- Use the digamma function (`tgammaf` / device intrinsic) for the NB log-likelihood derivatives.
- Underflow on extremely overdispersed genes: clamp `α` to `[1e-6, 1e6]`.
- BH adjustment via the cycle 11 pattern.

## Memory layout

- Input: `DeviceCSC` (m × n) + `DeviceMemory<int> cluster_id(n)` + `DeviceMemory<int> donor_id(n)`.
- Pseudobulk: `DeviceMemory<float> pb(m × n_clusters × n_donors)`. For 30k × 10 × 50: 60 MB.
- Per-gene per-cluster GLM workspace: shared memory only (Hessian + RHS).
- Output: `DonorPseudobulkResult` containing `lfc[n_genes × n_clusters × (n_donors-1)]`, p-values, q-values, dispersions.

## Streams

One stream, caller-provided. The pseudobulk aggregation, GLM kernel, and BH adjustment chain on it.

## Out-of-core

Pseudobulk aggregation streams via `PzDataLoader`: per chunk, accumulate into the per-(cluster, donor) pseudobulk matrix on device. The matrix is small (60 MB) and shared across chunks — single device-resident accumulator.

NB GLM runs once at the end on the small dense pseudobulk matrix.

## Determinism

Deterministic by construction:
- Pseudobulk: `cub::DeviceSegmentedReduce::Sum` is deterministic per architecture.
- IRLS: deterministic given fixed init. We seed from method-of-moments (deterministic).
- Cox-Reid dispersion fit: Brent's method is deterministic.
- Apeglm shrinkage: deterministic given fixed prior fit.

## Correctness test spec

Test: `tests/de_donor_pseudobulk_correctness.cpp`.

Reference: DESeq2 R via Rscript subprocess (`Rscript -e 'library(DESeq2); ...'`). Subprocess runs the same pseudobulk aggregation in R via `muscat::aggregateData(method='sum')` then `DESeq2::DESeq()`.

Test cases:
1. **`DonorPseudobulk_TinyPlanted_VsDESeq2`**: synthetic 200 cells × 100 genes with 2 clusters × 3 donors and planted differential genes (some up in donor A, some up in donor B). Run our `donor_pseudobulk_de` and DESeq2. Compare per-gene LFC: Spearman ρ ≥ 0.97. Top-10 markers Jaccard ≥ 0.90.
2. **`DonorPseudobulk_GSM4037629_RealData`**: load GSM4037629 + read its `donor_assignments.tsv` from singlify (the unique output). Run pseudobulk DE. Confirm finite values, no NaNs.
3. **`DonorPseudobulk_MinCellsFilter_DropsSparseGroups`**: a (cluster, donor) cell with < 10 cells is dropped from the pseudobulk.
4. **`DonorPseudobulk_DispersionShrinkage_Convergence`**: per-gene dispersions before/after shrinkage; confirm shrinkage moves outliers toward the prior mean.
5. **`DonorPseudobulk_Determinism_BitIdentical`**.
6. **`DonorPseudobulk_FewDonorsErrors`**: < 3 donors → graceful error (insufficient replicates).

Tolerances:
- LFC Spearman ρ ≥ 0.97
- q-value rank Spearman ρ ≥ 0.95
- Top markers Jaccard ≥ 0.90
- Determinism: bit-identical

## Target performance

| Scale | Cells | Genes | Clusters | Donors | Wall (target) | DESeq2 CPU |
|---|---|---|---|---|---|---|
| 10k | 11,560 | 30k | 5 | 3 | <100ms | ~30s |
| 100k | ~120k | 30k | 10 | 10 | <500ms | ~2min |
| 1M | ~1M | 30k | 10 | 50 | <2s | ~5min |

## Implementation notes

- Header path: `include/singlet-gpu/de/donor_pseudobulk.h` (~700 LOC budget).
- API:
  ```cpp
  namespace singlet_gpu::de {
      struct DonorPseudobulkConfig {
          int min_cells_per_pseudobulk = 10;
          int max_irls_iters = 50;
          float irls_tol = 1e-5f;
          int max_dispersion_iters = 10;
          bool apeglm_shrinkage = true;
          int top_n = 100;
          uint64_t seed = 0;
      };
      struct DonorPseudobulkResult {
          singlet_gpu::core::DeviceMemory<float> log2_fc;       // m × n_clusters × (n_donors - 1)
          singlet_gpu::core::DeviceMemory<float> p_values;
          singlet_gpu::core::DeviceMemory<float> p_adj;
          singlet_gpu::core::DeviceMemory<float> dispersion;    // m × n_clusters
          singlet_gpu::core::DeviceMemory<float> pseudobulk;    // m × n_clusters × n_donors
          int n_genes_passing_filter;
      };
      DonorPseudobulkResult donor_pseudobulk_de(
          const singlet_gpu::core::DeviceCSC& mat,
          const singlet_gpu::core::DeviceMemory<int>& cluster_labels,
          int n_clusters,
          const singlet_gpu::core::DeviceMemory<int>& donor_labels,
          int n_donors,
          const DonorPseudobulkConfig& cfg = {},
          cudaStream_t stream = nullptr);
  }
  ```
- Build flag: `FACTORNET_HAS_GPU=1`. cuSOLVER + cub.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 9 (Leiden labels), cycle 11 (DE patterns reused).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU NB GLM for donor-aware pseudobulk DE — exploits singlify donor_assignments.tsv)` first comment.

## Risks

1. **n_donors > 100** breaks the shared-memory Gauss-Jordan path. Fall back to cuSOLVER per-gene `Sgels` with stream callbacks. Document.
2. **Sparse pseudobulk** (few cells per group) will trigger the filter. The remaining gene set may be small.
3. **DESeq2 reference subprocess install** is heavy. Make it optional with fallback to manual sanity checks.
4. **Apeglm shrinkage** is non-trivial. Initial implementation could skip it with a flag, fix in a follow-up.
