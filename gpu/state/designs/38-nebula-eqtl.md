---
feature: nebula_eqtl
roadmap_id: 38
module: include/singlet-gpu/eqtl/nebula.h + python/singlet_gpu/eqtl/nebula.py + r/R/eqtl.R
status: design
tolerance: per-SNP-gene beta Spearman ρ ≥ 0.95 vs NEBULA R; p-value rank Spearman ρ ≥ 0.90
target_perf: 100k cells × 1M SNPs × 20k genes × 100 donors ≤15 min on A100 (NEBULA R is 2-8 hours)
ooc_plan: per-SNP chunked inference (1M SNP → 10k batches); gene chunking at high memory pressure
---

## Why this exists

Cycle 37 lookahead: **NEBULA** (He et al. Nature Commun Biology 2021) is the leading single-cell eQTL method based on negative-binomial mixed-model inference over donors. **Entirely CPU** (R package). sc-eQTLGen consortium uses it at scale — 2-8 hours per sample at 100k cells. Zero GPU implementation exists in RAPIDS / cuML / cuGraph. 13th "first GPU" candidate. **Directly consumes singlify's snp_ad.1pz + snp_dp.1pz**, making this the first cycle that exploits singlify's SNP pileup outputs for eQTL inference (not just donor demultiplexing).

## Algorithm — NEBULA

```
Inputs:
  counts (n_cells × n_genes): spliced UMI counts (from exon_counts.1pz)
  snp_ad (n_cells × n_snps): alt-allele counts (from snp_ad.1pz)
  snp_dp (n_cells × n_snps): total depth (from snp_dp.1pz)
  donor_id (n_cells): donor label per cell (from donor_assignments.tsv)
  covariates (n_cells × n_cov): age, sex, PCs, etc.

Outputs:
  beta (n_snps × n_genes): eQTL effect size
  se (n_snps × n_genes): standard error
  p_value (n_snps × n_genes): Wald test p-value
  sigma_sq (n_snps × n_genes): random-effect variance (donor-level)
  converged (n_snps × n_genes): Fisher-scoring convergence flag

Algorithm (per SNP, per gene):
  1. Aggregate per-donor: genotype[donor] from snp_ad/snp_dp, expression[donor] from counts.
  2. Negative binomial mixed model:
       y_ij ~ NB(mu_ij, theta)
       log(mu_ij) = X_ij^T beta + u_i,  u_i ~ N(0, sigma^2)
     where i = donor, j = cell within donor.
  3. Fisher scoring for (beta, theta, sigma^2):
       a. E-step: compute expected u_i given current params (BLUP).
       b. M-step: update beta via IRLS, theta via profile likelihood, sigma^2 via REML.
       c. Iterate until convergence (Δlog-lik < 1e-5, max 50 iters).
  4. Wald test: z = beta / se, p = 2 * Phi(-|z|).
  5. Genome-wide BH FDR correction across (SNP, gene) pairs.
```

## GPU implementation strategy

Native CUDA. The key observation: **each (SNP, gene) pair is independent**, and the per-pair problem is small (O(n_donors) × Fisher scoring × 50 iters). This is embarrassingly parallel across SNPs.

### Kernels

1. **Per-donor aggregation** (`donor_aggregate_kernel`): segmented sum over `counts` and `snp_ad`/`snp_dp` grouped by `donor_id`. `cub::DeviceSegmentedReduce::Sum`.

2. **Genotype dosage** (`dosage_kernel`): elementwise `genotype_dosage[donor, snp] = 2 * snp_ad[donor, snp] / max(snp_dp[donor, snp], 1)`. Filter out donors with < 5 reads at the SNP.

3. **Batched Fisher scoring** (the hot kernel): 1 block per (SNP, gene) pair, `n_donors`-element covariate vector in shared memory. Each block runs the full 50-iter Fisher loop in shared memory, producing (beta, se, sigma², p_value) at block exit.
   - Block layout: 128 threads per block, 1 block per (SNP, gene) pair.
   - Shared memory budget: `n_donors × 16 bytes × 3 = 4.8 KB` for covariates + genotype + expression (at n_donors=100).
   - Each iter: compute fitted means via `expf`, update beta via 2×2 Newton step (analytic inverse), update theta via profile likelihood, update sigma² via REML.
   - Convergence check: shared-memory scalar, no device-wide sync.

4. **BH FDR correction** (`cub::DeviceSegmentedSort` + per-gene cummin), reuses cycle 11 fgsea pattern.

5. **Per-SNP chunked outer loop**: 1M SNPs / CHUNK=10k = 100 iterations of the Fisher kernel. Host-side loop over chunks, all device work inside.

## Numerical stability

- **fp32 hot path** for Fisher scoring. The per-pair problem is small enough that fp32 is stable.
- **fp64 accumulator** in the 2×2 Hessian inverse (compute Hessian in fp32, promote to fp64 for the analytic inverse, demote result).
- **Log-space likelihood**: `lgammaf`, `expf`, `logf` throughout. No underflow.
- **Ridge regularization**: add `1e-6 * I` to the Hessian when the minimum eigenvalue is below threshold (detect via 2×2 determinant sign).

## Memory layout

- Input: `counts` CSC, `snp_ad` CSC, `snp_dp` CSC — all via cycle 0 loader.
- `donor_id`: int[n_cells], uploaded once.
- Per-donor aggregates: `dense (n_donors × n_genes) × 4 bytes`. For 100 × 20k: 8 MB.
- Per-donor genotype: `dense (n_donors × n_snps) × 2 bytes (uint16_t dosage). For 100 × 1M: 200 MB.
- `beta, se, p_value`: (n_snps × n_genes) × 4 bytes. For 1M × 20k: 80 GB × 3 = 240 GB. **TOO BIG** — requires chunking.
- **Chunked workspace**: at CHUNK_SNPS=10k and CHUNK_GENES=500, result buffer = 10k × 500 × 12 bytes = 60 MB. Host-side loop over (snp_chunk, gene_chunk). Total wall dominated by SpMM I/O.
- Total per-chunk: ~300 MB.

## Streams

Two streams for overlap: stream 0 runs Fisher scoring, stream 1 prefetches the next (snp_chunk, gene_chunk) aggregation.

## Out-of-core

Naturally chunked: 1M SNPs × 20k genes = 20G (SNP, gene) pairs, chunked into 10k × 500 = 5M pairs per kernel launch, 4000 launches total. Each launch ~0.2s = 800s total at the target perf.

## Determinism

No stochasticity in Fisher scoring. Result is bit-identical across runs given the same inputs. `cfg.deterministic=true` is a no-op.

## Correctness test spec

Test: `tests/eqtl_nebula_correctness.cpp`.

Reference: NEBULA R via Rscript subprocess (`library(nebula)`).

5 test cases:
1. **`Nebula_TinySynthetic_VsR`**: 200 cells × 50 genes × 100 SNPs × 5 donors synthetic with planted eQTL effects. Spearman ρ ≥ 0.90 on beta and p-value rank vs NEBULA R.
2. **`Nebula_GSM_RealMultisample`**: concat 3 real scRNA samples (cycle 21 pattern) with shared donor pool. Confirm finite results, non-zero converged count.
3. **`Nebula_FisherScoring_Convergence`**: under the null (random SNPs), >95% of (SNP, gene) pairs converge within 50 iters.
4. **`Nebula_PvalueCalibration_Uniform`**: under the null, p-values are uniformly distributed (KS tolerance 0.10).
5. **`Nebula_Determinism_BitIdentical`**: bit-identical across two runs on the same inputs.

## Target performance

| Scale | Cells | SNPs | Genes | Donors | Wall (target) | NEBULA R |
|---|---|---|---|---|---|---|
| tiny | 200 | 100 | 50 | 5 | <100ms | ~30s |
| 10k | 11,560 | 100k | 20k | 20 | <30s | ~20 min |
| 100k | ~120k | 1M | 20k | 100 | <15 min | ~4 hours |

NEBULA R baseline: 2-8 hours at 100k scale → target ≥15× speedup.

## Implementation notes

- Header path: `include/singlet-gpu/eqtl/nebula.h` (~1100 LOC, under 1500 cap).
- New module path `singlet-gpu/eqtl/` (NEW).
- Python wrapper: `python/singlet_gpu/eqtl/nebula.py` (~150 LOC).
- R wrapper: `r/R/eqtl.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cuRAND + cub + cuSPARSE.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 15 (donor pseudobulk pattern).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU NEBULA-style single-cell eQTL mapping)` first comment.
- Use factornet field-access style.

## Risks

1. **Shared memory pressure**: at n_donors > 200, the per-block shared memory budget overflows 48 KB limit. Mitigation: fall back to global-memory per-donor buffers with fp16 covariates, or cap n_donors at 200 with a warning.
2. **Fisher scoring non-convergence** at low-depth SNPs: 50-iter cap with a `converged` flag output. Filter downstream.
3. **NEBULA R install** is heavy (Rcpp + RcppArmadillo). Skip cleanly (exit code 2 GTEST_SKIP).
4. **1M SNP × 20k gene result buffer** = 240 GB. **Must** chunk; never materialize the full result matrix. Document in API.
5. **Covariate handling**: minimal MVP supports only age + sex + PC1-5. Document.
