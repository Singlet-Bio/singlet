---
feature: daesc_ase
roadmap_id: 40
module: include/singlet-gpu/ase/daesc.h + python/singlet_gpu/ase/daesc.py + r/R/ase.R
status: design
tolerance: per-SNP allelic imbalance p-value rank Spearman ρ ≥ 0.95 vs DAESC R; beta estimate Spearman ρ ≥ 0.95
target_perf: 100k cells × 1M SNP sites × 10 cell types ≤30s on A100 (DAESC R is ~30-60 min)
ooc_plan: per-SNP batched MLE (1M SNPs / 10k batches); cell-type batched per-SNP aggregation
---

## Why this exists

Cycle 39 lookahead: **DAESC** (Differential Allele-Specific Expression in Single Cells, Hu et al. 2023) and its successor DAESC+ (bioRxiv 2025) are the leading beta-binomial regression tools for allele-specific expression in single-cell data. Entirely CPU. **Second cycle to exploit singlify's unique `snp_ad.1pz`/`snp_dp.1pz` outputs** (NEBULA cycle 38 was first). ASE and eQTL are complementary: ASE asks "at this SNP site, is one allele expressed more than the other within each cell?"; eQTL asks "does genotype at this SNP predict expression at that gene across donors?". 15th "first GPU" candidate.

Critical biological use cases: imprinting detection, cis-eQTL refinement, X-chromosome inactivation (XCI) escape mapping, tumor clonal ASE analysis.

## Algorithm — DAESC

```
Inputs:
  snp_ad (n_cells × n_snps): alt-allele read counts (from snp_ad.1pz)
  snp_dp (n_cells × n_snps): total depth at SNP sites (from snp_dp.1pz)
  (optional) cell_type (n_cells): cluster labels for stratified ASE
  (optional) covariates (n_cells × n_cov): age, sex, ancestry, PCs

Outputs:
  beta (n_snps): allelic bias log-odds
  dispersion (n_snps): beta-binomial overdispersion phi
  se (n_snps): standard error
  p_value (n_snps): Wald test p-value of beta=0 (balanced null)
  per_celltype (n_snps × n_types): stratified beta if cell_type provided

Algorithm (per SNP):
  1. Filter cells with dp >= MIN_DEPTH (default 5).
  2. Beta-binomial MLE:
       y_i ~ BetaBin(dp_i, p_i, phi)
       logit(p_i) = beta + X_i^T gamma
     where X_i are optional covariates.
  3. Fisher scoring for (beta, gamma, phi):
       a. Compute fitted p_i via logistic link.
       b. Score and Fisher information via beta-binomial digamma derivatives.
       c. Iterate until |delta_beta| < 1e-5 or 50 iters.
  4. Wald test: z = beta / se, p = 2 * Phi(-|z|).
  5. Optional stratified variant: repeat step 2 per cell type.
  6. Genome-wide BH FDR correction.
```

## GPU implementation strategy

Native CUDA. Each SNP's MLE is independent → embarrassingly parallel across SNPs. Much smaller per-SNP problem than NEBULA (no random effect), so we can fit more SNPs per kernel launch.

### Kernels

1. **Per-SNP cell aggregation** (`snp_cell_gather_kernel`): for each SNP, gather the (dp, ad) pairs from cells where `dp >= MIN_DEPTH`. cuSPARSE CSC traversal with predicate. Output: sparse `(snp, cell_idx, dp, ad)` tuples.

2. **Batched beta-binomial MLE** (the hot kernel): 1 block per SNP, 128 threads. Each block iterates its cells, computes Fisher scores and information in shared memory, runs 50-iter Fisher loop to convergence, writes (beta, se, phi, pvalue) at exit.
   - Shared memory budget: `n_cells_per_snp × 8 bytes` for (dp, ad) pairs — but `n_cells_per_snp` varies wildly. Use `shared_cache_size=2 KB` hard cap; overflow SNPs spill (dp, ad) to global memory and stream through in chunks.
   - Each iter: `digammaf`, `lgammaf` per cell (fp32 CUDA math lib). Beta-binomial log-likelihood is numerically stable via `lbeta` tricks.
   - Fp64 accumulator in the 2×2 to 3×3 Fisher information matrix for the analytic inverse.

3. **Stratified ASE (optional)**: same kernel with cell_type-masked aggregation. Run n_types times.

4. **BH FDR correction** via `cub::DeviceSegmentedSort` + per-group cummin (cycle 11/36 pattern).

## Numerical stability

- fp32 hot path; fp64 Hessian inverse (small).
- `lgammaf` / `digammaf` from CUDA math library (sufficient precision for beta-binomial).
- Overflow guard: if fitted `p_i` saturates near 0 or 1, cap at `[1e-6, 1 - 1e-6]`.
- Dispersion constrained `phi ∈ [1e-4, 1e4]`; clamp each Fisher iter.

## Memory layout

- Input: `snp_ad` CSC + `snp_dp` CSC via cycle 0 loader.
- Per-SNP aggregated cell list: `n_snps × (avg n_cells_per_snp × 8 bytes)`. At 1M SNPs × 50 cells avg: 400 MB.
- Output: `n_snps × (beta, se, phi, pvalue) × 4 bytes = 16 MB`.
- Stratified: × n_types, so 160 MB at n_types=10.
- cub temps: ~20 MB.
- Total: ~500-600 MB.

## Streams

Two streams: stream 0 runs MLE kernel on SNP chunk N; stream 1 aggregates cells for SNP chunk N+1.

## Out-of-core

Per-SNP chunking at CHUNK_SNPS=10k. 1M SNPs → 100 chunks. All cell-level data streamed from CSC without densification.

## Determinism

No stochasticity. Bit-identical across runs.

## Correctness test spec

Test: `tests/ase_daesc_correctness.cpp`.

Reference: DAESC R via Rscript subprocess (CRAN). Fallback: pure-R beta-binomial MLE via `VGAM::betabinomial.ab()` — always produces a reference even when DAESC absent.

5 test cases:
1. **`Daesc_TinySynthetic_VsR`**: 200 cells × 100 SNPs × 3 cell types, planted allelic imbalance at known SNPs. Spearman ρ ≥ 0.95 on beta and p-value rank vs R.
2. **`Daesc_GSM_RealData`**: load a real sample with snp_ad.1pz + snp_dp.1pz. Confirm finite results, non-zero significant SNP count.
3. **`Daesc_FisherConvergence`**: >98% of MLEs converge within 50 iters on the tiny synthetic (well-specified data).
4. **`Daesc_PvalueCalibration_Uniform`**: under the null (balanced allele counts), p-values uniform (KS tol 0.10).
5. **`Daesc_Determinism_BitIdentical`**: bit-identical across two runs.

## Target performance

| Scale | Cells | SNPs | Wall (target) | DAESC R |
|---|---|---|---|---|
| tiny | 200 | 100 | <100ms | ~5s |
| 10k | 11,560 | 100k | <5s | ~3 min |
| 100k | ~120k | 1M | <30s | ~30-60 min |

## Implementation notes

- Header path: `include/singlet-gpu/ase/daesc.h` (~700 LOC, well under 1500 cap).
- New module path `singlet-gpu/ase/` (NEW).
- Python wrapper: `python/singlet_gpu/ase/daesc.py` (~150 LOC).
- R wrapper: `r/R/ase.R` (~120 LOC).
- Build flag: `FACTORNET_HAS_GPU=1`. cuBLAS + cub + cuSPARSE.
- Dependencies: cycle 1 (core), cycle 2 (loader), cycle 38 (NEBULA for eQTL complementarity — optional integration point).
- `// SPDX-License-Identifier: GPL-2.0-or-later`.
- `// integrates: original (first GPU DAESC-style beta-binomial ASE calling)` first comment.
- Use factornet field-access style.

## Risks

1. **Shared memory variable per-SNP size**: 2 KB cap forces spill-to-global path for SNPs with many cells. Handle both code paths.
2. **Beta-binomial digamma/lgamma precision**: fp32 CUDA `digammaf`/`lgammaf` is good to ~5 decimal places. For SNPs with `dp > 10000`, promote to fp64 (rare but defensible).
3. **DAESC R install** is heavy (VGAM + Rcpp). Pure-R VGAM fallback always runs.
4. **Covariates** deferred from MVP: support only intercept-only model in v1. Document.
5. **Stratified ASE**: keep optional (n_types iterations). Cell-type input from cycle 10 marker scoring or cycle 7 leiden.
